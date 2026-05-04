"""RealSense Loader for Boxer
静止画またはRealSenseキャプチャ済みデータ (rgb.png + depth.npy + intrinsics.json) を
Boxer が期待するフォーマットに変換するローダー。

カメラポーズは単位行列 (原点固定) として扱う。
重力方向はデフォルトで (0, 0, -9.81) とする（カメラが水平を向いている想定）。
"""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from loaders.base_loader import BaseLoader
from utils.tw.camera import CameraTW, get_pinhole_camera
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


class RealSenseLoader(BaseLoader):
    """Load a single RealSense capture (or a directory of captures) for Boxer."""

    camera = "rgb"
    device_name = "RealSense D435i"

    def __init__(self, data_dir, max_frames=1, resize=None):
        """
        Args:
            data_dir: Path to directory containing rgb.png, depth.npy, intrinsics.json
                      OR path directly to rgb.png
            max_frames: Number of frames to generate (1 = single shot, >1 = repeat)
            resize: Optional (W, H) tuple to resize images
        """
        self.resize = resize

        # Support passing rgb.png directly or a directory
        if os.path.isfile(data_dir) and data_dir.endswith(".png"):
            self.rgb_path = data_dir
            data_dir = os.path.dirname(data_dir)
        else:
            self.rgb_path = os.path.join(data_dir, "rgb.png")

        self.depth_path = os.path.join(data_dir, "depth.npy")
        self.intrinsics_path = os.path.join(data_dir, "intrinsics.json")

        # Load intrinsics
        with open(self.intrinsics_path) as f:
            intr = json.load(f)
        self.fx = float(intr["fx"])
        self.fy = float(intr["fy"])
        self.cx = float(intr["cx"])
        self.cy = float(intr["cy"])
        self.width = int(intr["width"])
        self.height = int(intr["height"])
        self.depth_scale = float(intr.get("depth_scale", 0.001))

        # Load RGB
        rgb_img = Image.open(self.rgb_path).convert("RGB")
        self.rgb_np = np.array(rgb_img)  # (H, W, 3) uint8

        # Load depth (meters)
        if os.path.exists(self.depth_path):
            depth_raw = np.load(self.depth_path)  # float32 or uint16
            if depth_raw.dtype == np.uint16:
                self.depth_m = depth_raw.astype(np.float32) * self.depth_scale
            else:
                self.depth_m = depth_raw.astype(np.float32)
        else:
            # Try depth.png (uint16 mm)
            depth_png = os.path.join(os.path.dirname(self.rgb_path), "depth.png")
            if os.path.exists(depth_png):
                depth_raw = np.array(Image.open(depth_png)).astype(np.float32)
                self.depth_m = depth_raw * self.depth_scale
            else:
                self.depth_m = None
                print("[RealSenseLoader] Warning: No depth found, sdp_w will be empty")

        self.length = max_frames
        self.index = 0
        print(f"[RealSenseLoader] Loaded: {self.rgb_path}")
        print(f"  RGB: {self.rgb_np.shape}, depth: {'available' if self.depth_m is not None else 'none'}")
        print(f"  Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} cx={self.cx:.1f} cy={self.cy:.1f}")
        self._init_prefetch()

    def load(self, idx):
        W, H = self.width, self.height

        # Resize if needed
        rgb = self.rgb_np.copy()
        depth = self.depth_m.copy() if self.depth_m is not None else None
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        if self.resize is not None:
            rW, rH = (self.resize, self.resize) if isinstance(self.resize, int) else self.resize
            sx, sy = rW / W, rH / H
            rgb = cv2.resize(rgb, (rW, rH))
            if depth is not None:
                depth = cv2.resize(depth, (rW, rH), interpolation=cv2.INTER_NEAREST)
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
            W, H = rW, rH

        # RGB → (1, 3, H, W) float32 [0, 1]
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        img_t = img_t.unsqueeze(0)  # (1, 3, H, W)

        # Camera intrinsics (Pinhole model: [fx, fy, cx, cy])
        params = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)
        cam = get_pinhole_camera(params=params, width=W, height=H)

        # Camera pose: Y-Z swap rotation (camera +Y down, +Z fwd → world +Z up)
        # R = [[1,0,0],[0,0,1],[0,-1,0]]  (same as omni_loader for SUN-RGBD/ScanNet)
        T_wr_data = torch.tensor(
            [1, 0, 0,
             0, 0, 1,
             0, -1, 0,
             0, 0, 0],
            dtype=torch.float32,
        )
        T_world_rig = PoseTW(T_wr_data)
        R_wc = T_wr_data[:9].reshape(3, 3).numpy()
        t_wc = T_wr_data[9:].numpy()

        # Semi-dense points from depth
        sdp_w = BaseLoader.sdp_from_depth(
            depth, fx, fy, cx, cy, R_wc, t_wc, num_samples=10000
        )

        # Gravity direction in world frame (Z-up after Y-Z swap)
        gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32)

        datum = {
            "img0": img_t.float(),
            "cam0": cam.float(),
            "T_world_rig0": T_world_rig.float(),
            "sdp_w": sdp_w.float(),
            "gravity": gravity,
            "time_ns0": idx * 100_000_000,  # 100ms per frame (dummy)
            "rotated0": torch.tensor([False]),
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "obbs": ObbTW(torch.zeros(0, 165)),
        }
        return datum

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        # Wait for prefetch or load directly
        if self._prefetch_result is not None:
            datum = self._prefetch_result
            self._prefetch_result = None
        else:
            datum = self.load(self.index)
        self.index += 1
        self._start_prefetch()
        return datum
