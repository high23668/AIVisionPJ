"""
第2段実験 RealSense D435i 撮影スクリプト

2モードで実行する:
  --mode paper     : カラー紙貼り付け状態を撮影 → rgb_paper.png, depth_paper.npy
  --mode clear     : 透明状態を撮影 → rgb.png, depth.npy, left_ir.png, right_ir.png

Phase 2 改善点:
  - RS depth を --avg-frames (default 30) フレーム平均 → 壁面 scale 校正アンカーを安定化
  - measurements.json テンプレートを自動生成 (未入力フィールドは手動で埋める)

使い方:
  # 1) まずカラー紙を貼った状態で撮影
  python scripts/exp2/capture_exp2.py --scene data/exp2/glass_front --mode paper --paper-color red

  # 2) 紙を剥がして透明状態を撮影
  python scripts/exp2/capture_exp2.py --scene data/exp2/glass_front --mode clear

出力ファイル:
  paper モード:
    rgb_paper.png          RGB (emitter ON)
    depth_paper.npy        RealSense depth float32 meters (平均後)
    depth_paper.png        uint16 mm (可視化・FP 入力用)
  clear モード:
    rgb.png                RGB (emitter ON)
    depth.npy              float32 meters (平均後)
    depth.png              uint16 mm
    left_ir.png            emitter OFF 左 IR (3ch)
    right_ir.png           emitter OFF 右 IR (3ch)
    intrinsics.json        カメラ内部パラメータ (clear 初回のみ上書き)
  両モード共通:
    measurements.json      未入力フィールドは手動で埋める (初回 paper モードで生成)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from PIL import Image


def average_depth(pipeline, align, depth_scale, n_frames: int) -> np.ndarray:
    """n_frames の depth を平均して float32 meters を返す。"""
    acc = None
    count = None
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        d = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32)
        valid = d > 0
        if acc is None:
            acc = np.zeros_like(d)
            count = np.zeros_like(d)
        acc += np.where(valid, d, 0)
        count += valid.astype(np.float32)
    avg_raw = np.where(count > 0, acc / np.maximum(count, 1), 0)
    return (avg_raw * depth_scale).astype(np.float32)


def capture_paper(out: Path, paper_color: str, warmup: int, avg_frames: int):
    """カラー紙貼り付け状態を撮影して GT depth を取得する。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)

    align = rs.align(rs.stream.color)

    print(f"[paper] Emitter ON, warmup {warmup} frames...")
    for _ in range(warmup):
        pipeline.wait_for_frames()

    # RGB 1 frame
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_img = np.asanyarray(aligned.get_color_frame().get_data())

    # depth 平均
    print(f"[paper] Averaging {avg_frames} depth frames...")
    depth_m = average_depth(pipeline, align, depth_scale, avg_frames)

    pipeline.stop()

    # 保存
    Image.fromarray(color_img[:, :, ::-1]).save(str(out / "rgb_paper.png"))
    np.save(str(out / "depth_paper.npy"), depth_m)
    depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(str(out / "depth_paper.png"))

    valid = depth_m[depth_m > 0.05]
    print(f"[paper] 完了: depth 有効画素={len(valid)}, "
          f"median={np.median(valid):.3f}m, range=[{valid.min():.3f}, {valid.max():.3f}]m")

    # measurements.json テンプレート (初回のみ)
    meas_path = out / "measurements.json"
    if not meas_path.exists():
        meas = {
            "scene": out.name,
            "camera_to_object_front_mm": 0,
            "object_thickness_mm": 0,
            "object_to_wall_mm": 0,
            "object_tilt_deg": 0,
            "paper_color": paper_color,
            "gt_z_center_mm": None,
            "notes": ""
        }
        meas_path.write_text(json.dumps(meas, indent=2, ensure_ascii=False))
        print(f"[paper] measurements.json テンプレートを生成 → 数値を手動で入力してください")


def capture_clear(out: Path, warmup: int, avg_frames: int):
    """透明状態を撮影する。emitter ON で RGB+depth、emitter OFF で左右 IR。"""
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

    # Intrinsics
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    left_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    K_rgb = color_stream.get_intrinsics()
    K_left = left_stream.get_intrinsics()
    K_right = right_stream.get_intrinsics()
    baseline = abs(left_stream.get_extrinsics_to(right_stream).translation[0])
    ir_to_rgb = left_stream.get_extrinsics_to(color_stream)

    align = rs.align(rs.stream.color)

    # ── Phase A: Emitter ON ──
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    print(f"[clear] Emitter ON, warmup {warmup} frames...")
    for _ in range(warmup):
        pipeline.wait_for_frames()

    # RGB 1 frame
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_img = np.asanyarray(aligned.get_color_frame().get_data())

    # depth 平均
    print(f"[clear] Averaging {avg_frames} depth frames...")
    depth_m = average_depth(pipeline, align, depth_scale, avg_frames)

    # ── Phase B: Emitter OFF で左右 IR ──
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    print("[clear] Emitter OFF, settling 10 frames...")
    for _ in range(10):
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    left_img = np.asanyarray(frames.get_infrared_frame(1).get_data())
    right_img = np.asanyarray(frames.get_infrared_frame(2).get_data())

    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    pipeline.stop()

    # 保存
    Image.fromarray(color_img[:, :, ::-1]).save(str(out / "rgb.png"))
    np.save(str(out / "depth.npy"), depth_m)
    depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(str(out / "depth.png"))
    Image.fromarray(np.stack([left_img] * 3, axis=-1)).save(str(out / "left_ir.png"))
    Image.fromarray(np.stack([right_img] * 3, axis=-1)).save(str(out / "right_ir.png"))

    # intrinsics.json
    intrinsics_data = {
        "rgb": {"fx": K_rgb.fx, "fy": K_rgb.fy, "cx": K_rgb.ppx, "cy": K_rgb.ppy,
                "width": K_rgb.width, "height": K_rgb.height},
        "left_ir": {"fx": K_left.fx, "fy": K_left.fy, "cx": K_left.ppx, "cy": K_left.ppy,
                    "width": K_left.width, "height": K_left.height},
        "right_ir": {"fx": K_right.fx, "fy": K_right.fy, "cx": K_right.ppx, "cy": K_right.ppy,
                     "width": K_right.width, "height": K_right.height},
        "baseline_m": baseline,
        "depth_scale": depth_scale,
        "ir_to_rgb_R": np.array(ir_to_rgb.rotation).reshape(3, 3).tolist(),
        "ir_to_rgb_t": list(ir_to_rgb.translation),
        "avg_frames": avg_frames,
        "notes": "depth averaged over avg_frames; IR stereo with emitter OFF",
    }
    (out / "intrinsics.json").write_text(json.dumps(intrinsics_data, indent=2))

    # K_ir.txt for Foundation-Stereo
    K_mat = [K_left.fx, 0.0, K_left.ppx, 0.0, K_left.fy, K_left.ppy, 0.0, 0.0, 1.0]
    with open(out / "K_ir.txt", "w") as f:
        f.write(" ".join(str(v) for v in K_mat) + "\n")
        f.write(f"{baseline}\n")

    invalid_pct = 100 * (1 - (depth_m > 0.1).sum() / depth_m.size)
    print(f"[clear] 完了: depth 欠損率 {invalid_pct:.1f}%, "
          f"range=[{depth_m[depth_m > 0].min():.3f}, {depth_m.max():.3f}]m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="出力ディレクトリ (例: data/exp2/glass_front)")
    parser.add_argument("--mode", choices=["paper", "clear"], required=True,
                        help="paper: カラー紙貼り付け撮影 / clear: 透明状態撮影")
    parser.add_argument("--paper-color", default="red", choices=["red", "blue"],
                        help="カラー紙の色 (paper モード用)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--avg-frames", type=int, default=30, help="depth 平均フレーム数")
    args = parser.parse_args()

    out = Path(args.scene)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "paper":
        capture_paper(out, args.paper_color, args.warmup, args.avg_frames)
    else:
        capture_clear(out, args.warmup, args.avg_frames)
