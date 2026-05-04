"""
RealSense D435i キャプチャスクリプト
- RGB画像 + 深度画像を保存
- カメラ内部パラメータ (intrinsics) も保存

使い方 (システムPythonで実行):
  python3 scripts/capture_realsense.py
  python3 scripts/capture_realsense.py --output data/realsense --frames 1
  python3 scripts/capture_realsense.py --output data/realsense --frames 5  # 5枚連続

出力ファイル:
  data/realsense/rgb.png          - RGB画像 (640x480)
  data/realsense/depth.png        - 深度画像 (可視化用, 16bit PNG)
  data/realsense/depth.npy        - 深度データ (meters, float32)
  data/realsense/intrinsics.json  - カメラ内部パラメータ
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def capture(output_dir, n_frames=1, warmup=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # 深度スケール取得
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # カメラ内部パラメータ取得
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    K = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "width": intrinsics.width,
        "height": intrinsics.height,
        "depth_scale": depth_scale,
    }

    # align depth to color
    align = rs.align(rs.stream.color)

    print(f"ウォームアップ中 ({warmup}フレーム)...")
    for _ in range(warmup):
        pipeline.wait_for_frames()

    saved = []
    for i in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            print(f"  フレーム{i}: 取得失敗")
            continue

        color_img = np.asanyarray(color_frame.get_data())  # BGR
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16, mm units
        depth_meters = depth_raw * depth_scale              # float32, meters

        suffix = f"_{i:03d}" if n_frames > 1 else ""
        rgb_path = out / f"rgb{suffix}.png"
        depth_png_path = out / f"depth{suffix}.png"
        depth_npy_path = out / f"depth{suffix}.npy"

        # BGR→RGB変換してPILで保存
        Image.fromarray(color_img[:, :, ::-1]).save(str(rgb_path))
        Image.fromarray(depth_raw).save(str(depth_png_path))  # 16bit PNG
        np.save(str(depth_npy_path), depth_meters.astype(np.float32))

        print(f"  [{i+1}/{n_frames}] Saved: {rgb_path.name}, {depth_png_path.name}")
        print(f"    深度範囲: {depth_meters[depth_meters > 0].min():.3f}m ~ "
              f"{depth_meters.max():.3f}m")

        saved.append(str(rgb_path))

        if n_frames > 1 and i < n_frames - 1:
            time.sleep(0.5)

    pipeline.stop()

    # intrinsics保存
    intr_path = out / "intrinsics.json"
    with open(intr_path, "w") as f:
        json.dump(K, f, indent=2)
    print(f"  Intrinsics: {intr_path}")
    print(f"  K matrix: fx={K['fx']:.1f}, fy={K['fy']:.1f}, "
          f"cx={K['cx']:.1f}, cy={K['cy']:.1f}")

    return saved, K


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense D435i キャプチャ")
    parser.add_argument("--output", default="data/realsense",
                        help="出力ディレクトリ")
    parser.add_argument("--frames", type=int, default=1,
                        help="撮影枚数 (1=単発, N=連続)")
    parser.add_argument("--warmup", type=int, default=30,
                        help="ウォームアップフレーム数")
    args = parser.parse_args()

    print(f"RealSense D435i キャプチャ開始")
    print(f"出力先: {args.output}")
    saved, K = capture(args.output, args.frames, args.warmup)
    print(f"\n完了: {len(saved)}枚保存")
