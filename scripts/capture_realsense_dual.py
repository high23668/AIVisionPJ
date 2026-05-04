"""
RealSense D435i デュアルモード撮影スクリプト
- emitter ON で RGB + RealSense depth を取得（能動照明で精度◎）
- emitter OFF で 左右IR を取得（Foundation-Stereo用、ドットパターンなし）
- 連続取得（<1秒）でシーンが変化しない前提

使い方:
  python3 scripts/capture_realsense_dual.py --output data/3glass_simple/

出力ファイル:
  rgb.png           - RGB (640x480)
  depth.png         - RealSense depth (emitter ON, uint16 mm)
  depth.npy         - RealSense depth (float32 meters)
  left_ir.png       - 左IR (emitter OFF, 3ch化済み)
  right_ir.png      - 右IR (emitter OFF, 3ch化済み)
  intrinsics.json   - RGB + IR intrinsics + baseline + extrinsics
  K_ir.txt          - Foundation-Stereo用
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def capture(output_dir, warmup=30):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    profile = pipeline.start(config)
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Intrinsics + extrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    K_rgb = color_stream.get_intrinsics()
    K_left = left_stream.get_intrinsics()
    K_right = right_stream.get_intrinsics()
    baseline = abs(left_stream.get_extrinsics_to(right_stream).translation[0])

    # IR left → RGB extrinsics (Phase 4ワープで使用)
    ir_to_rgb = left_stream.get_extrinsics_to(color_stream)
    R_ir2rgb = np.array(ir_to_rgb.rotation).reshape(3, 3).tolist()
    t_ir2rgb = list(ir_to_rgb.translation)

    print(f"  RGB K    : fx={K_rgb.fx:.1f} fy={K_rgb.fy:.1f} cx={K_rgb.ppx:.1f} cy={K_rgb.ppy:.1f}")
    print(f"  Left IR K: fx={K_left.fx:.1f} fy={K_left.fy:.1f} cx={K_left.ppx:.1f} cy={K_left.ppy:.1f}")
    print(f"  Baseline : {baseline*1000:.1f} mm")

    # depth alignment to RGB
    align = rs.align(rs.stream.color)

    # ── Phase A: Emitter ON で RGB + depth ──
    print(f"\n[Phase A] Emitter ON, warmup {warmup} frames...")
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    for _ in range(warmup):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    color_img = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_meters = depth_raw * depth_scale

    # ── Phase B: Emitter OFF で 左右 IR（同じシーン） ──
    print(f"[Phase B] Emitter OFF, settle 10 frames...")
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    for _ in range(10):  # emitter切替後の安定化
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)
    left_img = np.asanyarray(left_frame.get_data())
    right_img = np.asanyarray(right_frame.get_data())

    # Emitter元に戻す
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

    # 3ch化 (Foundation-Stereo 要求仕様)
    left_img3 = np.stack([left_img] * 3, axis=-1)
    right_img3 = np.stack([right_img] * 3, axis=-1)

    # Save
    Image.fromarray(color_img[:, :, ::-1]).save(str(out / "rgb.png"))
    Image.fromarray(depth_raw).save(str(out / "depth.png"))
    np.save(str(out / "depth.npy"), depth_meters.astype(np.float32))
    Image.fromarray(left_img3).save(str(out / "left_ir.png"))
    Image.fromarray(right_img3).save(str(out / "right_ir.png"))

    # Metadata
    intrinsics_data = {
        "rgb": {"fx": K_rgb.fx, "fy": K_rgb.fy, "cx": K_rgb.ppx, "cy": K_rgb.ppy,
                "width": K_rgb.width, "height": K_rgb.height},
        "left_ir": {"fx": K_left.fx, "fy": K_left.fy, "cx": K_left.ppx, "cy": K_left.ppy,
                    "width": K_left.width, "height": K_left.height},
        "right_ir": {"fx": K_right.fx, "fy": K_right.fy, "cx": K_right.ppx, "cy": K_right.ppy,
                     "width": K_right.width, "height": K_right.height},
        "baseline_m": baseline,
        "depth_scale": depth_scale,
        "ir_to_rgb_R": R_ir2rgb,
        "ir_to_rgb_t": t_ir2rgb,
        "notes": "depth captured with emitter ON; IR stereo captured with emitter OFF",
    }
    with open(out / "intrinsics.json", "w") as f:
        json.dump(intrinsics_data, f, indent=2)

    # K_ir.txt for Foundation-Stereo
    K_mat = [K_left.fx, 0.0, K_left.ppx, 0.0, K_left.fy, K_left.ppy, 0.0, 0.0, 1.0]
    with open(out / "K_ir.txt", "w") as f:
        f.write(" ".join(str(v) for v in K_mat) + "\n")
        f.write(f"{baseline}\n")

    invalid_pct = 100 * (1 - (depth_meters > 0.1).sum() / depth_meters.size)
    print(f"\n完了:")
    print(f"  RGB depth範囲 (emitter ON): {depth_meters[depth_meters > 0].min():.3f}m ~ {depth_meters.max():.3f}m")
    print(f"  RealSense depth 欠損率: {invalid_pct:.1f}%")
    print(f"  左右IR (emitter OFF) 3ch化済み")

    pipeline.stop()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="出力ディレクトリ")
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    print(f"RealSense D435i デュアルモード撮影")
    print(f"出力先: {args.output}")
    capture(args.output, args.warmup)
