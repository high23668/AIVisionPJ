"""
第2段実験 depth モデル一括実行スクリプト

対象モデル (6本):
  depthpro  Depth Pro (Apple)           metric
  moge2     MoGe-2 ViT-L (Microsoft)   affine-invariant
  unidepth  UniDepth V2 ViT-L           metric
  metric3d  Metric3D v2 ViT-Large       metric
  da2       Depth Anything V2 Indoor    metric
  marigold  Marigold LCM v1.0           affine-invariant

校正方式:
  wall_cal : depth_cal = s * depth_raw          (壁基準 scale のみ)
  gt_cal   : depth_cal = a * depth_raw + b      (壁 + GT Z の2点 affine)

使い方:
  .venv/bin/python scripts/exp2/run_all_depth_models.py --model depthpro
  .venv/bin/python scripts/exp2/run_all_depth_models.py --model all
  .venv/bin/python scripts/exp2/run_all_depth_models.py --model depthpro moge2 --scenes glass_front

出力:
  results/exp2/depth/{model}_{scene}_wall.npy     壁校正後 depth (m, float32)
  results/exp2/depth/{model}_{scene}_gt.npy       GT affine 校正後 depth (m, float32)
  results/exp2/depth/{model}_{scene}_vis.jpg      6パネル可視化
  results/exp2/depth/{model}_{scene}_metrics.json 板面 RMSE/MAE/Inlier (速報値)
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent.parent
DATA = PROJ / "data/exp2"
OUT  = PROJ / "results/exp2/depth"
OUT.mkdir(parents=True, exist_ok=True)

SCENES = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]
ALL_MODELS = ["depthpro", "moge2", "unidepth", "metric3d", "da2", "marigold"]


# ── モデル ロード関数 ──────────────────────────────────────────────

def load_depthpro():
    import depth_pro
    cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    cfg.checkpoint_uri = str(PROJ / "models/depth_pro/checkpoints/depth_pro.pt")
    model, transform = depth_pro.create_model_and_transforms(
        config=cfg, device=torch.device("cuda"), precision=torch.float16)
    model.eval()
    return model, transform


def load_moge2():
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    model.cuda().eval()
    return model


def load_unidepth():
    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    model.cuda().eval()
    return model


def load_metric3d():
    model = torch.hub.load("yvanyin/metric3d", "metric3d_vit_large",
                           pretrain=True, trust_repo=True)
    model.cuda().eval()
    return model


def load_da2():
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        device=0,
    )
    return pipe


def load_marigold():
    from diffusers import MarigoldDepthPipeline
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0", torch_dtype=torch.float16
    ).to("cuda")
    return pipe


# ── 推論関数 ──────────────────────────────────────────────────────

def predict_depthpro(model_data, rgb_bgr, intr):
    model, transform = model_data
    rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    fx = intr["rgb"]["fx"]
    x = transform(rgb)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to("cuda", dtype=torch.float16)
    with torch.no_grad():
        out = model.infer(x, f_px=torch.tensor([fx], device="cuda"))
    d = out["depth"]
    if d.dim() == 3:
        d = d.squeeze(0)
    d_np = d.float().cpu().numpy()
    if d_np.shape != (h, w):
        d_np = cv2.resize(d_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return d_np.astype(np.float32)


def predict_moge2(model, rgb_bgr, intr):
    rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    fx = intr["rgb"]["fx"]
    fov_x_deg = math.degrees(2.0 * math.atan(w / (2.0 * fx)))
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().cuda() / 255.0
    with torch.no_grad():
        out = model.infer(x, fov_x=fov_x_deg, resolution_level=9, apply_mask=True)
    d = out["depth"].squeeze().cpu().numpy()
    mask = out.get("mask")
    if mask is not None:
        d = np.where(mask.squeeze().cpu().numpy().astype(bool), d, np.nan)
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32)


def predict_unidepth(model, rgb_bgr, intr):
    from unidepth.utils.camera import Pinhole
    rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    K = intr["rgb"]
    K_t = torch.tensor([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]],
                        dtype=torch.float32, device="cuda")
    cam = Pinhole(K=K_t.unsqueeze(0))
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).cuda()
    with torch.no_grad():
        out = model.infer(x, camera=cam)
    d = out["depth"].squeeze().cpu().numpy().astype(np.float32)
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d


def predict_metric3d(model, rgb_bgr, intr):
    rgb = rgb_bgr[:, :, ::-1]
    h0, w0 = rgb.shape[:2]
    K = intr["rgb"]
    intrinsic = [K["fx"], K["fy"], K["cx"], K["cy"]]
    input_size = (616, 1064)
    scale = min(input_size[0] / h0, input_size[1] / w0)
    rgb_r = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)))
    intr_s = [v * scale for v in intrinsic]
    pad_h = input_size[0] - rgb_r.shape[0]
    pad_w = input_size[1] - rgb_r.shape[1]
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    mean_val = [123.675, 116.28, 103.53]
    rgb_pad = cv2.copyMakeBorder(rgb_r, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=mean_val)
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std  = torch.tensor([58.395,  57.12,  57.375]).float()[:, None, None]
    x = (torch.from_numpy(rgb_pad.transpose(2, 0, 1)).float() - mean) / std
    x = x[None].cuda()
    with torch.no_grad():
        pred, _, _ = model.inference({"input": x})
    pred = pred.squeeze()
    pred = pred[pt: pred.shape[0] - pb, pl: pred.shape[1] - pr]
    pred = torch.nn.functional.interpolate(pred[None, None], (h0, w0), mode="bilinear").squeeze()
    canon_to_real = intr_s[0] / 1000.0
    return (pred * canon_to_real).cpu().numpy().astype(np.float32)


def predict_da2(pipe, rgb_bgr, intr):
    from PIL import Image as PILImage
    rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    pil = PILImage.fromarray(rgb)
    out = pipe(pil)
    d = np.array(out["depth"])
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32)


def predict_marigold(pipe, rgb_bgr, intr):
    from PIL import Image as PILImage
    rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    pil = PILImage.fromarray(rgb)
    with torch.no_grad():
        out = pipe(pil, num_inference_steps=4, ensemble_size=5)
    d = out.prediction.squeeze()
    if d.shape != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    return d.astype(np.float32)


# ── 校正関数 ──────────────────────────────────────────────────────

def calibrate_wall(depth_raw, rs_depth, mask):
    """壁基準 scale 校正。"""
    bg = (~mask) & (rs_depth > 0.05) & (rs_depth < 5.0) \
       & np.isfinite(depth_raw) & (depth_raw > 0)
    if bg.sum() < 500:
        return None, None
    s = float(np.median(rs_depth[bg]) / np.median(depth_raw[bg]))
    return (depth_raw * s).astype(np.float32), s


def calibrate_gt_affine(depth_raw, rs_depth, mask, gt_z_m):
    """壁 + GT Z の2点で affine 校正: depth_cal = a * depth_raw + b"""
    bg = (~mask) & (rs_depth > 0.05) & (rs_depth < 5.0) \
       & np.isfinite(depth_raw) & (depth_raw > 0)
    plate = mask & np.isfinite(depth_raw) & (depth_raw > 0)
    if bg.sum() < 500 or plate.sum() < 50:
        return None, None, None
    x1 = float(np.median(depth_raw[bg]))      # モデル壁median
    y1 = float(np.median(rs_depth[bg]))        # RS壁median (実距離)
    x2 = float(np.median(depth_raw[plate]))    # モデル板median
    y2 = gt_z_m                                # GT Z (実距離)
    if abs(x1 - x2) < 1e-6:
        return None, None, None
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    d = a * depth_raw + b
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    return d.astype(np.float32), a, b


# ── 評価指標 ──────────────────────────────────────────────────────

def compute_metrics(depth_cal, gt_z_m, mask):
    """マスク内の RMSE/MAE/Inlier を計算する。"""
    plate = mask & np.isfinite(depth_cal) & (depth_cal > 0)
    if plate.sum() < 10:
        return {}
    pred = depth_cal[plate]
    gt   = gt_z_m
    err  = pred - gt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    in5  = float((np.abs(err) < 0.05).mean())
    in2  = float((np.abs(err) < 0.02).mean())
    med  = float(np.median(pred))
    return {"rmse_m": rmse, "mae_m": mae, "inlier_5cm": in5, "inlier_2cm": in2,
            "plate_median_m": med, "gt_z_m": gt_z_m, "n_pixels": int(plate.sum())}


# ── 可視化 ────────────────────────────────────────────────────────

def visualize(rgb_bgr, rs_depth, mask, depth_wall, depth_gt,
              gt_z_m, model_name, scene, out_path, vrange):
    vmin, vmax = vrange
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 0
    axes[0, 0].imshow(rgb_bgr[:, :, ::-1]); axes[0, 0].set_title(f"RGB ({scene})"); axes[0, 0].axis("off")

    rs_vis = rs_depth.copy(); rs_vis[rs_vis == 0] = np.nan
    im = axes[0, 1].imshow(rs_vis, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("RealSense depth (glass transparent→wall)"); axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    mask_overlay = rgb_bgr[:, :, ::-1].copy()
    mask_overlay[mask] = (mask_overlay[mask] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
    axes[0, 2].imshow(mask_overlay); axes[0, 2].set_title("Plate mask (GT)"); axes[0, 2].axis("off")

    # Row 1
    if depth_wall is not None:
        im = axes[1, 0].imshow(depth_wall, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f"{model_name} wall_cal (m)"); axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[1, 0].axis("off")

    if depth_gt is not None:
        im = axes[1, 1].imshow(depth_gt, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f"{model_name} GT affine_cal (m)"); axes[1, 1].axis("off")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        # Error map
        err = depth_gt - gt_z_m
        err_vis = np.where(mask & np.isfinite(err), err, np.nan)
        im2 = axes[1, 2].imshow(err_vis, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
        axes[1, 2].set_title(f"Error map (GT cal) ±50mm"); axes[1, 2].axis("off")
        plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 1].text(0.5, 0.5, "GT Z not available", ha="center", va="center"); axes[1, 1].axis("off")
        axes[1, 2].text(0.5, 0.5, "GT Z not available", ha="center", va="center"); axes[1, 2].axis("off")

    fig.suptitle(f"{model_name} — {scene}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ── シーン処理 ────────────────────────────────────────────────────

def run_scene(model_name, predict_fn, scene_name):
    scene_dir = DATA / scene_name
    rgb      = cv2.imread(str(scene_dir / "rgb.png"))
    rs_depth = np.load(str(scene_dir / "depth.npy")).astype(np.float32)
    intr     = json.loads((scene_dir / "intrinsics.json").read_text())
    mask     = cv2.imread(str(scene_dir / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    meas     = json.loads((scene_dir / "measurements.json").read_text())
    gt_z_m   = meas.get("gt_z_center_mm")
    if gt_z_m is not None:
        gt_z_m = gt_z_m / 1000.0

    # depth 推定
    t0 = time.time()
    depth_raw = predict_fn(rgb, intr)
    elapsed = time.time() - t0
    print(f"    推論: {elapsed:.1f}s")

    # 壁校正
    depth_wall, s_wall = calibrate_wall(depth_raw, rs_depth, mask)

    # GT affine 校正
    depth_gt, a_gt, b_gt = (None, None, None)
    if gt_z_m is not None:
        depth_gt, a_gt, b_gt = calibrate_gt_affine(depth_raw, rs_depth, mask, gt_z_m)

    # メトリクス
    metrics = {"scene": scene_name, "model": model_name, "infer_sec": elapsed,
               "s_wall": s_wall, "a_gt": a_gt, "b_gt": b_gt}
    if depth_wall is not None:
        metrics["wall"] = compute_metrics(depth_wall, gt_z_m, mask) if gt_z_m else {}
    if depth_gt is not None:
        metrics["gt"]   = compute_metrics(depth_gt,   gt_z_m, mask)

    # 保存
    tag = f"{model_name}_{scene_name}"
    if depth_wall is not None:
        np.save(OUT / f"{tag}_wall.npy", depth_wall)
    if depth_gt is not None:
        np.save(OUT / f"{tag}_gt.npy",   depth_gt)

    # 可視化 vrange はシーンのカメラ距離に合わせる
    cam_mm = meas.get("camera_to_object_front_mm", 400)
    vmin = max(0.05, cam_mm / 1000.0 * 0.5)
    vmax = (meas.get("camera_to_object_front_mm", 0) +
            meas.get("object_to_wall_mm", 200)) / 1000.0 * 1.2
    visualize(rgb, rs_depth, mask, depth_wall, depth_gt,
              gt_z_m, model_name, scene_name,
              OUT / f"{tag}_vis.jpg", (vmin, vmax))

    (OUT / f"{tag}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    # 速報
    if "gt" in metrics and metrics["gt"]:
        m = metrics["gt"]
        print(f"    GT RMSE={m['rmse_m']*1000:.1f}mm  MAE={m['mae_m']*1000:.1f}mm  "
              f"Inlier@5cm={m['inlier_5cm']*100:.0f}%  Inlier@2cm={m['inlier_2cm']*100:.0f}%")
    return metrics


# ── エントリポイント ──────────────────────────────────────────────

LOADERS = {
    "depthpro": load_depthpro,
    "moge2":    load_moge2,
    "unidepth": load_unidepth,
    "metric3d": load_metric3d,
    "da2":      load_da2,
    "marigold": load_marigold,
}

PREDICTORS = {
    "depthpro": predict_depthpro,
    "moge2":    predict_moge2,
    "unidepth": predict_unidepth,
    "metric3d": predict_metric3d,
    "da2":      predict_da2,
    "marigold": predict_marigold,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["depthpro"],
                    help=f"モデル名 or 'all'. 選択肢: {ALL_MODELS}")
    ap.add_argument("--scenes", nargs="+", default=SCENES,
                    help="シーン名リスト")
    args = ap.parse_args()

    models = ALL_MODELS if "all" in args.model else args.model

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"[{model_name}] モデルロード中...")
        model = LOADERS[model_name]()
        predict_raw = PREDICTORS[model_name]

        # モデルを引数に束縛したラムダ
        def make_fn(m):
            return lambda rgb, intr: predict_raw(m, rgb, intr)
        predict_fn = make_fn(model)

        all_metrics = []
        for scene in args.scenes:
            scene_dir = DATA / scene
            if not (scene_dir / "rgb.png").exists():
                print(f"  SKIP: {scene} (rgb.png なし)")
                continue
            print(f"  [{scene}]")
            try:
                m = run_scene(model_name, predict_fn, scene)
                all_metrics.append(m)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

        # モデル単位サマリ
        summary_path = OUT / f"{model_name}_summary.json"
        summary_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
        print(f"  → {summary_path}")

        # GPU 解放
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()

    print("\n全て完了")


if __name__ == "__main__":
    main()
