"""
RealSense D435i ステレオキャプチャスクリプト
- RGB + IR左右 + RealSense内蔵depth を保存
- IRカメラのintrinsicsとbaselineも保存（Foundation-Stereo用）

使い方:
  python3 scripts/capture_realsense_stereo.py --output data/stereo_simple/
  python3 scripts/capture_realsense_stereo.py --output data/stereo_simple/ --no_emitter

出力ファイル:
  rgb.png         - RGB (640x480)
  left_ir.png     - 左IR (640x480, mono)
  right_ir.png    - 右IR (640x480, mono)
  depth.png       - RealSense内蔵depth (16bit, mm)
  depth.npy       - RealSense内蔵depth (float32, meters)
  intrinsics.json - RGB + IR intrinsics + baseline
  K_ir.txt        - Foundation-Stereo用 (3x3行列 + baseline)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def capture(output_dir, warmup=30, disable_emitter=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # Infrared streams: 1=left, 2=right (y8 monochrome)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # IR emitter制御：有効にするとactive stereo depthが改善するが、
    # IR画像にdotパターンが映り込みFoundation-Stereoには悪影響
    # → デフォルトは無効 (Foundation-Stereo優先)
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(
            rs.option.emitter_enabled,
            0 if disable_emitter else 1,
        )
        emitter_state = "OFF" if disable_emitter else "ON"
        print(f"  IR Emitter: {emitter_state}")

    # intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

    K_rgb = color_stream.get_intrinsics()
    K_left = left_stream.get_intrinsics()
    K_right = right_stream.get_intrinsics()

    # Baseline: left→right カメラの並進 (単位: m)
    extrinsics = left_stream.get_extrinsics_to(right_stream)
    # translation[0] は x軸方向（水平）の並進 = baseline (負の値になることがある)
    baseline = abs(extrinsics.translation[0])

    print(f"  RGB intrinsics : fx={K_rgb.fx:.1f} fy={K_rgb.fy:.1f} cx={K_rgb.ppx:.1f} cy={K_rgb.ppy:.1f}")
    print(f"  Left IR        : fx={K_left.fx:.1f} fy={K_left.fy:.1f} cx={K_left.ppx:.1f} cy={K_left.ppy:.1f}")
    print(f"  Right IR       : fx={K_right.fx:.1f} fy={K_right.fy:.1f} cx={K_right.ppx:.1f} cy={K_right.ppy:.1f}")
    print(f"  Baseline       : {baseline:.4f} m ({baseline*1000:.1f} mm)")

    # align depth to color
    align = rs.align(rs.stream.color)

    print(f"ウォームアップ中 ({warmup}フレーム)...")
    for _ in range(warmup):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()

    # IR左右はalignしない（rectified済み）
    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)

    # depth・RGBはalign
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    if not all([left_frame, right_frame, color_frame, depth_frame]):
        raise RuntimeError("フレーム取得失敗")

    color_img = np.asanyarray(color_frame.get_data())  # BGR
    depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 mm
    left_img = np.asanyarray(left_frame.get_data())    # uint8 mono
    right_img = np.asanyarray(right_frame.get_data())  # uint8 mono

    depth_meters = depth_raw * depth_scale  # float32 meters

    # Save files
    Image.fromarray(color_img[:, :, ::-1]).save(str(out / "rgb.png"))
    Image.fromarray(left_img).save(str(out / "left_ir.png"))
    Image.fromarray(right_img).save(str(out / "right_ir.png"))
    Image.fromarray(depth_raw).save(str(out / "depth.png"))
    np.save(str(out / "depth.npy"), depth_meters.astype(np.float32))

    # Intrinsics JSON
    intrinsics_data = {
        "rgb": {
            "fx": K_rgb.fx, "fy": K_rgb.fy, "cx": K_rgb.ppx, "cy": K_rgb.ppy,
            "width": K_rgb.width, "height": K_rgb.height,
        },
        "left_ir": {
            "fx": K_left.fx, "fy": K_left.fy, "cx": K_left.ppx, "cy": K_left.ppy,
            "width": K_left.width, "height": K_left.height,
        },
        "right_ir": {
            "fx": K_right.fx, "fy": K_right.fy, "cx": K_right.ppx, "cy": K_right.ppy,
            "width": K_right.width, "height": K_right.height,
        },
        "baseline_m": baseline,
        "depth_scale": depth_scale,
        "emitter_disabled": disable_emitter,
    }
    with open(out / "intrinsics.json", "w") as f:
        json.dump(intrinsics_data, f, indent=2)

    # Foundation-Stereo用 K.txt (IRの左カメラ intrinsics + baseline)
    # Format: line 1 = 9 values (3x3 matrix), line 2 = baseline (m)
    K_mat = [K_left.fx, 0.0, K_left.ppx,
             0.0, K_left.fy, K_left.ppy,
             0.0, 0.0, 1.0]
    with open(out / "K_ir.txt", "w") as f:
        f.write(" ".join(f"{v}" for v in K_mat) + "\n")
        f.write(f"{baseline}\n")

    print(f"\n完了:")
    print(f"  RGB depth範囲: {depth_meters[depth_meters > 0].min():.3f}m ~ {depth_meters.max():.3f}m")
    invalid_pct = 100 * (1 - (depth_meters > 0.1).sum() / depth_meters.size)
    print(f"  RealSense depth 欠損率: {invalid_pct:.1f}%")

    pipeline.stop()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--no_emitter", action="store_true",
                        help="IR emitterを無効化 (Foundation-Stereo用推奨)")
    args = parser.parse_args()

    print(f"RealSense D435i ステレオキャプチャ")
    print(f"出力先: {args.output}")
    capture(args.output, args.warmup, disable_emitter=args.no_emitter)
