"""
参照画像セット撮影スクリプト (FoundationPose model-free 用)
- カップ等を手で回しながら複数角度で撮影
- 各撮影: Enter押下 → RealSense 1フレーム取得 → 保存

使い方 (.venv で実行):
  source /home/vr01/AIVisionPJ/.venv/bin/activate
  python scripts/capture_ref_images.py --output data/realsense/ref --n 15

出力:
  data/realsense/ref/rgb_000.png ~ rgb_014.png
  data/realsense/ref/depth_000.npy ~ depth_014.npy
  data/realsense/ref/intrinsics.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def main(output_dir, n_frames, warmup=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    K = {
        "fx": intrinsics.fx, "fy": intrinsics.fy,
        "cx": intrinsics.ppx, "cy": intrinsics.ppy,
        "width": intrinsics.width, "height": intrinsics.height,
        "depth_scale": depth_scale,
    }

    align = rs.align(rs.stream.color)

    print(f"ウォームアップ中 ({warmup}フレーム)...")
    for _ in range(warmup):
        pipeline.wait_for_frames()

    print(f"\n参照画像撮影モード ({n_frames}枚)")
    print("操作方法:")
    print("  Enter    → 撮影")
    print("  s + Enter → スキップ")
    print("  q + Enter → 終了\n")

    saved = 0
    i = 0
    while saved < n_frames:
        cmd = input(f"  [{saved+1}/{n_frames}] Enterで撮影 / s=スキップ / q=終了: ").strip()
        if cmd.lower() == "q":
            break
        if cmd.lower() == "s":
            continue

        # 最新フレームを取得 (バッファをフラッシュ)
        for _ in range(5):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            print("  フレーム取得失敗、再試行してください")
            continue

        color_img = np.asanyarray(color_frame.get_data())  # BGR
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_meters = depth_raw * depth_scale

        rgb_path = out / f"rgb_{saved:03d}.png"
        depth_npy_path = out / f"depth_{saved:03d}.npy"

        Image.fromarray(color_img[:, :, ::-1]).save(str(rgb_path))
        np.save(str(depth_npy_path), depth_meters.astype(np.float32))

        depth_valid = depth_meters[depth_meters > 0]
        if len(depth_valid) > 0:
            print(f"  保存: {rgb_path.name} | 深度: {depth_valid.min():.3f}m ~ {depth_valid.max():.3f}m")
        else:
            print(f"  保存: {rgb_path.name} | 深度データなし")
        saved += 1

    pipeline.stop()

    intr_path = out / "intrinsics.json"
    with open(intr_path, "w") as f:
        json.dump(K, f, indent=2)

    print(f"\n完了: {saved}枚保存 → {out}/")
    print(f"次のステップ: SAM3でマスク生成")
    print(f"  source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate")
    print(f"  python scripts/generate_sam3_mask.py --dir {out} --prompt \"cup\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="参照画像セット撮影")
    parser.add_argument("--output", default="data/realsense/ref",
                        help="出力ディレクトリ (デフォルト: data/realsense/ref)")
    parser.add_argument("--n", type=int, default=15,
                        help="撮影枚数 (デフォルト: 15)")
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    main(args.output, args.n, args.warmup)
