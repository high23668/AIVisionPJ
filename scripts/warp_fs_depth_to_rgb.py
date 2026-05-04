"""
Foundation-Stereo depth (IR左カメラ視点) を RGB カメラ視点に warp する。

使い方:
  python scripts/warp_fs_depth_to_rgb.py \
    --fs_depth results/transparent/fs_simple_vits/depth_meter.npy \
    --intrinsics data/stereo_simple/intrinsics.json \
    --output data/stereo_simple/depth_fs_aligned.npy

出力: RGB カメラ視点の depth.npy (meters, float32)
手順: IR 深度 → 3D 点群 (IR frame) → RGB frame に変換 → RGB K で投影
"""
import argparse
import json
import numpy as np
import pyrealsense2 as rs


def get_ir_to_rgb_extrinsics():
    """RealSenseデバイスから IR_left → RGB の外部パラメータを取得"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    profile = pipeline.start(config)
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        left_ir_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        extr = left_ir_stream.get_extrinsics_to(color_stream)
        R = np.array(extr.rotation).reshape(3, 3)
        t = np.array(extr.translation)  # meters
    finally:
        pipeline.stop()
    return R, t


def warp_depth(fs_depth, K_ir, K_rgb, R_ir2rgb, t_ir2rgb, out_hw=(480, 640)):
    """
    FS depth を RGB 座標系に warp する。
    forward warping: IR の各ピクセルを 3D に持ち上げ、RGB に再投影。
    """
    H, W = fs_depth.shape
    out_H, out_W = out_hw

    # IR ピクセル座標グリッド
    us, vs = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
    valid = fs_depth > 0.01  # 0.01m以上を有効

    # IR 3D 点群 (camera frame)
    fx, fy = K_ir[0, 0], K_ir[1, 1]
    cx, cy = K_ir[0, 2], K_ir[1, 2]
    x_ir = (us - cx) / fx * fs_depth  # (H, W)
    y_ir = (vs - cy) / fy * fs_depth
    z_ir = fs_depth

    # 有効点を抽出
    X_ir = np.stack([x_ir[valid], y_ir[valid], z_ir[valid]], axis=0)  # (3, N)

    # IR → RGB 変換
    X_rgb = R_ir2rgb @ X_ir + t_ir2rgb[:, None]  # (3, N)

    # RGB 投影
    fx_r, fy_r = K_rgb[0, 0], K_rgb[1, 1]
    cx_r, cy_r = K_rgb[0, 2], K_rgb[1, 2]

    u_rgb = (X_rgb[0] * fx_r / X_rgb[2] + cx_r).astype(np.int32)
    v_rgb = (X_rgb[1] * fy_r / X_rgb[2] + cy_r).astype(np.int32)
    z_rgb = X_rgb[2]

    # RGB frame 内の有効点 (2x2 splatting のため-1マージン)
    in_frame = (u_rgb >= 0) & (u_rgb < out_W - 1) & (v_rgb >= 0) & (v_rgb < out_H - 1) & (z_rgb > 0)
    u_rgb, v_rgb, z_rgb = u_rgb[in_frame], v_rgb[in_frame], z_rgb[in_frame]

    # Z-buffer + 2x2 splatting で穴を埋める
    # RGBがIRより高解像度（fx比~1.6倍）なので各IRピクセルを2x2領域に書き込む
    aligned = np.full((out_H, out_W), np.inf, dtype=np.float32)
    order = np.argsort(-z_rgb)  # 遠い順（後書き＝近いが優先）
    u_ord, v_ord, z_ord = u_rgb[order], v_rgb[order], z_rgb[order]

    for du in (0, 1):
        for dv in (0, 1):
            vv = v_ord + dv
            uu = u_ord + du
            # 既存値より近いものだけ更新
            mask = z_ord < aligned[vv, uu]
            aligned[vv[mask], uu[mask]] = z_ord[mask]

    aligned[aligned == np.inf] = 0.0
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs_depth", required=True, help="Foundation-Stereo depth_meter.npy")
    parser.add_argument("--intrinsics", required=True,
                        help="intrinsics.json (rgb / left_ir 両方含む)")
    parser.add_argument("--output", required=True, help="出力 depth.npy (RGB-aligned)")
    parser.add_argument("--use_cached_extr", default=None,
                        help="既存の extrinsics.npz があれば再利用 (RealSense接続不要)")
    parser.add_argument("--save_extr", default=None,
                        help="取得した extrinsics を保存しておくパス")
    args = parser.parse_args()

    fs_depth = np.load(args.fs_depth).astype(np.float32)
    with open(args.intrinsics) as f:
        intr = json.load(f)

    K_ir = np.array([
        [intr["left_ir"]["fx"], 0.0, intr["left_ir"]["cx"]],
        [0.0, intr["left_ir"]["fy"], intr["left_ir"]["cy"]],
        [0.0, 0.0, 1.0],
    ])
    K_rgb = np.array([
        [intr["rgb"]["fx"], 0.0, intr["rgb"]["cx"]],
        [0.0, intr["rgb"]["fy"], intr["rgb"]["cy"]],
        [0.0, 0.0, 1.0],
    ])

    # Extrinsics
    if args.use_cached_extr is not None:
        data = np.load(args.use_cached_extr)
        R, t = data["R"], data["t"]
        print(f"  Loaded extrinsics from {args.use_cached_extr}")
    else:
        print("  Connecting to RealSense to fetch IR->RGB extrinsics...")
        R, t = get_ir_to_rgb_extrinsics()
        print(f"  t (IR left -> RGB) = {t} m")
        if args.save_extr:
            np.savez(args.save_extr, R=R, t=t)
            print(f"  Cached to {args.save_extr}")

    print(f"  FS depth shape={fs_depth.shape}, valid={int((fs_depth>0.01).sum())}")
    aligned = warp_depth(fs_depth, K_ir, K_rgb, R, t,
                         out_hw=(intr["rgb"]["height"], intr["rgb"]["width"]))
    valid_pct = 100 * (aligned > 0.01).sum() / aligned.size
    print(f"  Aligned depth: valid={valid_pct:.1f}%, range={aligned[aligned>0].min():.3f}~{aligned.max():.3f}m")
    np.save(args.output, aligned)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
