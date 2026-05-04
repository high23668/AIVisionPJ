#!/usr/bin/env python3
"""RealSense単枚・複数枚画像でBoxerを実行するスクリプト。

使い方:
  # カップ画像で3D OBB検出
  python run_realsense.py --data /home/vr01/AIVisionPJ/data/realsense/

  # くまのぬいぐるみ画像
  python run_realsense.py --data /home/vr01/AIVisionPJ/data/realsense_bear/

  # カスタムラベルで検出（デフォルト: lvisplus の1220クラス全部）
  python run_realsense.py --data /home/vr01/AIVisionPJ/data/realsense/ --labels cup,mug,bottle
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from boxernet.boxernet import BoxerNet
from loaders.realsense_loader import RealSenseLoader
from utils.demo_utils import CKPT_PATH, CudaTimer
from utils.file_io import ObbCsvWriter2
from utils.image import draw_bb3s, put_text, render_bb2, torch2cv2
from utils.taxonomy import load_text_labels
from utils.tw.tensor_utils import pad_string, string2tensor


def main():
    parser = argparse.ArgumentParser(description="Run Boxer on RealSense data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to directory with rgb.png + depth.npy + intrinsics.json")
    parser.add_argument("--labels", type=str, default="lvisplus",
                        help="Comma-separated labels or 'lvisplus' for all 1220 classes")
    parser.add_argument("--thresh2d", type=float, default=0.25)
    parser.add_argument("--thresh3d", type=float, default=0.4)
    parser.add_argument("--output_dir", type=str, default="output/realsense")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==> Using device: {device}")

    # ─── Labels ───────────────────────────────────────────────────────────────
    if args.labels == "lvisplus":
        labels = load_text_labels("lvisplus")
    else:
        labels = [l.strip() for l in args.labels.split(",")]
    print(f"==> Labels ({len(labels)}): {labels[:10]}{'...' if len(labels) > 10 else ''}")

    label_tensors = [string2tensor(pad_string(l, max_len=128)) for l in labels]
    label_t = torch.stack(label_tensors).to(device)

    # ─── Model ────────────────────────────────────────────────────────────────
    model = BoxerNet(labels, device=device, ckpt=args.ckpt)

    # ─── Loader ───────────────────────────────────────────────────────────────
    loader = RealSenseLoader(args.data, max_frames=1)

    # ─── Inference ────────────────────────────────────────────────────────────
    for datum in loader:
        img_t    = datum["img0"].to(device)           # (1, 3, H, W)
        cam      = datum["cam0"].to(device)
        T_wr     = datum["T_world_rig0"].to(device)
        sdp_w    = datum["sdp_w"].to(device)
        gravity  = datum["gravity"].to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            result = model.forward_single(
                img=img_t,
                cam=cam,
                T_world_rig=T_wr,
                sdp_w=sdp_w,
                gravity=gravity,
                thresh2d=args.thresh2d,
                thresh3d=args.thresh3d,
                labels=labels,
            )
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"==> 推論完了: {elapsed:.0f}ms")

        bb2d = result.get("bb2d", None)
        bb3d = result.get("bb3d", None)

        if bb2d is not None:
            print(f"  2D検出: {len(bb2d)} 物体")
        if bb3d is not None:
            print(f"  3D検出: {len(bb3d)} 物体")
            for i, obb in enumerate(bb3d):
                name = obb.get("name", "?")
                prob = obb.get("prob", 0.0)
                pos  = obb.get("pos", [0, 0, 0])
                sz   = obb.get("scale", [0, 0, 0])
                print(f"    [{i}] {name:20s} score={prob:.2f}  pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m  size=({sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f})m")

        # ─── Visualization ────────────────────────────────────────────────────
        img_cv = torch2cv2(img_t[0])  # (H, W, 3) BGR uint8

        # Draw 2D BBs
        if bb2d is not None and len(bb2d) > 0:
            for det in bb2d:
                x1, y1, x2, y2 = [int(v) for v in det["box"]]
                score = det.get("score", 0.0)
                name  = det.get("name", "?")
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_cv, f"{name} {score:.2f}", (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out_path = os.path.join(args.output_dir, "boxer_result.jpg")
        cv2.imwrite(out_path, img_cv)
        print(f"==> 結果画像保存: {out_path}")


if __name__ == "__main__":
    main()
