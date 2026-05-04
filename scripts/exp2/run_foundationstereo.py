"""
第2段実験 Foundation-Stereo 実行スクリプト

入力: data/exp2/{scene}/left_ir.png + right_ir.png + K_ir.txt + mask.png + measurements.json
出力: results/exp2/depth/fs_{scene}_wall.npy   壁基準 scale 校正
      results/exp2/depth/fs_{scene}_gt.npy    GT affine 校正
      results/exp2/depth/fs_{scene}_vis.jpg   可視化
      results/exp2/depth/fs_{scene}_metrics.json

実行環境: .venv_stereo
  .venv_stereo/bin/python scripts/exp2/run_foundationstereo.py
  .venv_stereo/bin/python scripts/exp2/run_foundationstereo.py --scenes glass_front

depth = fx_ir * baseline / disparity  (stereo の定式)
→ Foundation-Stereo は metric depth を直接出力するため
  GT affine 校正が scale ~1.0 になることが期待される

モデル重みは 23-51-11 を使用 (ViT-L)。
"""
import json
import sys
import time
from pathlib import Path

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent.parent
FS_ROOT = PROJ / "models/foundation_stereo/FoundationStereo"
sys.path.insert(0, str(FS_ROOT))

DATA = PROJ / "data/exp2"
OUT  = PROJ / "results/exp2/depth"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_DIR = FS_ROOT / "pretrained_models/23-51-11"
CKPT     = CKPT_DIR / "model_best_bp2.pth"
SCENES   = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]
MODEL_NAME = "fs"


def load_model():
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    from Utils import set_seed

    set_seed(0)
    cfg = OmegaConf.load(CKPT_DIR / "cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg.valid_iters = 32
    cfg.hiera = 0

    model = FoundationStereo(cfg)
    ckpt = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"  ckpt step={ckpt['global_step']}  epoch={ckpt['epoch']}")
    return model, cfg


def predict(model, cfg, left_img, right_img, fx, baseline):
    """ステレオ画像 → depth (m)"""
    from core.utils.utils import InputPadder

    H, W = left_img.shape[:2]
    # 3ch float tensor
    img0 = torch.as_tensor(left_img).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right_img).cuda().float()[None].permute(0, 3, 1, 2)

    padder = InputPadder(img0.shape, divis_by=32)
    img0, img1 = padder.pad(img0, img1)

    import torch.cuda.amp as amp
    with torch.no_grad(), amp.autocast(True):
        disp = model.forward(img0, img1, iters=cfg.valid_iters, test_mode=True)

    disp = padder.unpad(disp.float())
    disp_np = disp.data.cpu().numpy().reshape(H, W)  # pixels

    # disparity → depth
    disp_safe = np.where(disp_np > 0.5, disp_np, np.nan)
    depth = (fx * baseline) / disp_safe
    return depth.astype(np.float32), disp_np


def calibrate_wall(depth_raw, rs_depth, mask):
    bg = (~mask) & (rs_depth > 0.05) & (rs_depth < 5.0) \
       & np.isfinite(depth_raw) & (depth_raw > 0)
    if bg.sum() < 500:
        return None, None
    s = float(np.median(rs_depth[bg]) / np.median(depth_raw[bg]))
    return (depth_raw * s).astype(np.float32), s


def calibrate_gt_affine(depth_raw, rs_depth, mask, gt_z_m):
    bg    = (~mask) & (rs_depth > 0.05) & (rs_depth < 5.0) \
          & np.isfinite(depth_raw) & (depth_raw > 0)
    plate = mask & np.isfinite(depth_raw) & (depth_raw > 0)
    if bg.sum() < 500 or plate.sum() < 50:
        return None, None, None
    x1 = float(np.median(depth_raw[bg]))
    y1 = float(np.median(rs_depth[bg]))
    x2 = float(np.median(depth_raw[plate]))
    y2 = gt_z_m
    if abs(x1 - x2) < 1e-6:
        return None, None, None
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    d = a * depth_raw + b
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    return d.astype(np.float32), a, b


def compute_metrics(depth_cal, gt_z_m, mask):
    plate = mask & np.isfinite(depth_cal) & (depth_cal > 0)
    if plate.sum() < 10:
        return {}
    pred = depth_cal[plate]
    err  = pred - gt_z_m
    return {
        "rmse_m":     float(np.sqrt(np.mean(err ** 2))),
        "mae_m":      float(np.mean(np.abs(err))),
        "inlier_5cm": float((np.abs(err) < 0.05).mean()),
        "inlier_2cm": float((np.abs(err) < 0.02).mean()),
        "plate_median_m": float(np.median(pred)),
        "gt_z_m": gt_z_m,
        "n_pixels": int(plate.sum()),
    }


def visualize(rgb_bgr, disp, rs_depth, mask, depth_wall, depth_gt,
              gt_z_m, scene, out_path, vrange):
    vmin, vmax = vrange
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(rgb_bgr[:, :, ::-1]); axes[0, 0].set_title(f"RGB ({scene})"); axes[0, 0].axis("off")

    rs_vis = rs_depth.copy(); rs_vis[rs_vis == 0] = np.nan
    im = axes[0, 1].imshow(rs_vis, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("RealSense depth"); axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(disp, cmap="plasma",
                            vmin=np.nanpercentile(disp, 5),
                            vmax=np.nanpercentile(disp, 95))
    axes[0, 2].set_title("Foundation-Stereo disparity"); axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    if depth_wall is not None:
        im = axes[1, 0].imshow(depth_wall, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("FS wall_cal depth (m)"); axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].axis("off")

    if depth_gt is not None:
        im = axes[1, 1].imshow(depth_gt, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1, 1].set_title("FS GT affine_cal (m)"); axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

        err = depth_gt - gt_z_m
        err_vis = np.where(mask & np.isfinite(err), err, np.nan)
        im2 = axes[1, 2].imshow(err_vis, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
        axes[1, 2].set_title("Error map (GT cal) ±50mm"); axes[1, 2].axis("off")
        plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 1].text(0.5, 0.5, "GT Z not available", ha="center", va="center"); axes[1, 1].axis("off")
        axes[1, 2].axis("off")

    fig.suptitle(f"Foundation-Stereo — {scene}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def run_scene(model, cfg, scene_name):
    scene_dir = DATA / scene_name
    left  = cv2.imread(str(scene_dir / "left_ir.png"))
    right = cv2.imread(str(scene_dir / "right_ir.png"))
    rgb   = cv2.imread(str(scene_dir / "rgb.png"))
    rs_depth = np.load(str(scene_dir / "depth.npy")).astype(np.float32)
    mask  = cv2.imread(str(scene_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    meas  = json.loads((scene_dir / "measurements.json").read_text())
    intr  = json.loads((scene_dir / "intrinsics.json").read_text())

    # K_ir.txt から fx と baseline を取得
    k_lines = (scene_dir / "K_ir.txt").read_text().strip().splitlines()
    K_vals  = list(map(float, k_lines[0].split()))
    fx_ir   = K_vals[0]
    baseline = float(k_lines[1])

    gt_z_m = meas.get("gt_z_center_mm")
    if gt_z_m is not None:
        gt_z_m /= 1000.0

    print(f"  fx_ir={fx_ir:.1f}  baseline={baseline*1000:.1f}mm")

    t0 = time.time()
    depth_raw, disp = predict(model, cfg, left, right, fx_ir, baseline)
    elapsed = time.time() - t0
    print(f"  推論: {elapsed:.1f}s")

    depth_wall, s_wall     = calibrate_wall(depth_raw, rs_depth, mask)
    depth_gt, a_gt, b_gt   = (None, None, None)
    if gt_z_m is not None:
        depth_gt, a_gt, b_gt = calibrate_gt_affine(depth_raw, rs_depth, mask, gt_z_m)

    metrics = {"scene": scene_name, "model": MODEL_NAME, "infer_sec": elapsed,
               "s_wall": s_wall, "a_gt": a_gt, "b_gt": b_gt,
               "fx_ir": fx_ir, "baseline_m": baseline}
    if depth_wall is not None:
        metrics["wall"] = compute_metrics(depth_wall, gt_z_m, mask) if gt_z_m else {}
    if depth_gt is not None:
        metrics["gt"]   = compute_metrics(depth_gt, gt_z_m, mask)

    tag = f"{MODEL_NAME}_{scene_name}"
    if depth_wall is not None:
        np.save(OUT / f"{tag}_wall.npy", depth_wall)
    if depth_gt is not None:
        np.save(OUT / f"{tag}_gt.npy",   depth_gt)

    cam_mm = meas.get("camera_to_object_front_mm", 400)
    vmin = max(0.05, cam_mm / 1000.0 * 0.5)
    vmax = (cam_mm + meas.get("object_to_wall_mm", 200)) / 1000.0 * 1.2
    visualize(rgb, disp, rs_depth, mask, depth_wall, depth_gt,
              gt_z_m, scene_name, OUT / f"{tag}_vis.jpg", (vmin, vmax))

    (OUT / f"{tag}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    if "gt" in metrics and metrics["gt"]:
        m = metrics["gt"]
        print(f"  GT RMSE={m['rmse_m']*1000:.1f}mm  MAE={m['mae_m']*1000:.1f}mm  "
              f"Inlier@5cm={m['inlier_5cm']*100:.0f}%  Inlier@2cm={m['inlier_2cm']*100:.0f}%")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()

    print("[Foundation-Stereo] モデルロード中...")
    model, cfg = load_model()

    all_metrics = []
    for scene in args.scenes:
        scene_dir = DATA / scene
        if not (scene_dir / "left_ir.png").exists():
            print(f"  SKIP: {scene} (left_ir.png なし)")
            continue
        print(f"\n[{scene}]")
        try:
            m = run_scene(model, cfg, scene)
            all_metrics.append(m)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    summary = OUT / "fs_summary.json"
    summary.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"\n完了 → {summary}")


if __name__ == "__main__":
    main()
