"""Phase 7.4 UniDepth V2 ガラス板 depth 推定 + RS壁面で scale 校正 + FP scene 出力."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("/home/vr01/AIVisionPJ")


def load_unidepth(repo: str = "lpiccinelli/unidepth-v2-vitl14"):
    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained(repo)
    model.cuda().eval()
    return model


def predict_depth(model, rgb_bgr: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    from unidepth.utils.camera import Pinhole
    rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).cuda()
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")
    cam = Pinhole(K=K.unsqueeze(0))
    with torch.no_grad():
        out = model.infer(rgb_t, camera=cam)
    depth = out["depth"].squeeze().cpu().numpy().astype(np.float32)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    confidence = out.get("confidence")
    conf_np = None
    if confidence is not None:
        conf_np = confidence.squeeze().cpu().numpy().astype(np.float32)
        if conf_np.shape != (h, w):
            conf_np = cv2.resize(conf_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth, conf_np


def calibrate_scale(model_depth, rs_depth, plate_mask):
    bg = (~plate_mask) & (rs_depth > 0.05) & (rs_depth < 3.0) & np.isfinite(model_depth) & (model_depth > 0.05)
    if bg.sum() < 1000:
        return 1.0, bg
    s = float(np.median(rs_depth[bg]) / np.median(model_depth[bg]))
    return s, bg


def visualize(rgb_bgr, depth_raw, depth_cal, conf, rs_depth, plate_mask, out_path: Path, model_name: str, scene: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(rgb_bgr[:, :, ::-1]); axes[0, 0].set_title(f"RGB ({scene})"); axes[0, 0].axis("off")

    rs_vis = rs_depth.copy(); rs_vis[rs_vis == 0] = np.nan
    im = axes[0, 1].imshow(rs_vis, cmap="turbo", vmin=0.2, vmax=1.5); axes[0, 1].set_title("RealSense depth (m)"); axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    if conf is not None:
        im = axes[0, 2].imshow(conf, cmap="viridis"); axes[0, 2].set_title(f"{model_name} confidence"); axes[0, 2].axis("off")
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    else:
        axes[0, 2].imshow(plate_mask, cmap="gray"); axes[0, 2].set_title("Plate mask"); axes[0, 2].axis("off")

    raw_v = depth_raw[np.isfinite(depth_raw)]
    raw_vmin = float(np.percentile(raw_v, 5)) if raw_v.size else 0
    raw_vmax = float(np.percentile(raw_v, 95)) if raw_v.size else 1
    im = axes[1, 0].imshow(depth_raw, cmap="turbo", vmin=raw_vmin, vmax=raw_vmax)
    axes[1, 0].set_title(f"{model_name} raw depth (m, auto-scale)"); axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(depth_cal, cmap="turbo", vmin=0.2, vmax=1.5)
    axes[1, 1].set_title(f"{model_name} calibrated (m)"); axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    plate = plate_mask & np.isfinite(depth_cal) & (depth_cal > 0)
    rs_plate_valid = plate_mask & (rs_depth > 0)
    plate_vals = depth_cal[plate] if plate.sum() else np.array([0])
    rs_plate = rs_depth[rs_plate_valid] if rs_plate_valid.sum() else np.array([0])
    axes[1, 2].hist(plate_vals, bins=40, alpha=0.6, label=f"{model_name} on plate (n={plate.sum()})", color="tab:orange")
    axes[1, 2].hist(rs_plate, bins=40, alpha=0.6, label=f"RS on plate (n={rs_plate_valid.sum()})", color="tab:blue")
    axes[1, 2].set_xlabel("depth (m)"); axes[1, 2].set_ylabel("freq"); axes[1, 2].set_title("plate-region depth histogram"); axes[1, 2].legend()

    fig.suptitle(f"Phase 7.4 {model_name} ({scene})", fontsize=14)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def export_fp_scene(scene_in: Path, scene_out: Path, depth_m: np.ndarray, intrinsics: dict):
    scene_out.mkdir(parents=True, exist_ok=True)
    (scene_out / "rgb").mkdir(exist_ok=True); (scene_out / "depth").mkdir(exist_ok=True); (scene_out / "masks").mkdir(exist_ok=True)
    shutil.copy(scene_in / "rgb.png", scene_out / "rgb" / "000000.png")
    mask_src = scene_in / "mask_plate.png"
    if not mask_src.exists():
        mask_src = scene_in / "mask.png"
    shutil.copy(mask_src, scene_out / "masks" / "000000.png")
    fx, fy = intrinsics["rgb"]["fx"], intrinsics["rgb"]["fy"]
    cx, cy = intrinsics["rgb"]["cx"], intrinsics["rgb"]["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    np.savetxt(scene_out / "cam_K.txt", K, fmt="%.6f")
    d = np.where(np.isfinite(depth_m), depth_m, 0.0)
    depth_mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(scene_out / "depth" / "000000.png"), depth_mm)


def run_scene(model, scene: str, model_tag: str, out_results: Path):
    scene_dir = ROOT / "data" / f"glassboard_{scene}"
    rgb = cv2.imread(str(scene_dir / "rgb.png"))
    intr = json.loads((scene_dir / "intrinsics.json").read_text())
    fx, fy = intr["rgb"]["fx"], intr["rgb"]["fy"]
    cx, cy = intr["rgb"]["cx"], intr["rgb"]["cy"]
    rs_depth = np.load(scene_dir / "depth.npy").astype(float)
    mask_file = scene_dir / "mask_plate.png"
    if not mask_file.exists():
        mask_file = scene_dir / "mask.png"
    plate_mask = (cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0)
    print(f"  mask: {mask_file.name}")

    depth_raw, conf = predict_depth(model, rgb, fx, fy, cx, cy)
    np.save(scene_dir / f"depth_{model_tag}.npy", depth_raw)
    if conf is not None:
        np.save(scene_dir / f"depth_{model_tag}_conf.npy", conf)

    s, bg_mask = calibrate_scale(depth_raw, rs_depth, plate_mask)
    depth_cal = depth_raw * s
    visualize(rgb, depth_raw, depth_cal, conf, rs_depth, plate_mask, out_results / f"{model_tag}_{scene}.jpg", model_tag, scene)
    fp_scene = ROOT / "data" / f"glassboard_{scene}_fp_{model_tag}"
    export_fp_scene(scene_dir, fp_scene, depth_cal, intr)
    plate = plate_mask & np.isfinite(depth_cal) & (depth_cal > 0)
    rs_plate_valid = plate_mask & (rs_depth > 0)
    return {
        "scene": scene, "scale": s, "bg_pixels": int(bg_mask.sum()),
        "plate_depth_median_m": float(np.median(depth_cal[plate])) if plate.sum() else None,
        "plate_depth_p10_p90_m": [float(np.percentile(depth_cal[plate], 10)), float(np.percentile(depth_cal[plate], 90))] if plate.sum() else None,
        "rs_plate_median_m": float(np.median(rs_depth[rs_plate_valid])) if rs_plate_valid.sum() else None,
        "plate_confidence_median": float(np.nanmedian(conf[plate_mask])) if conf is not None and plate_mask.sum() else None,
        "fp_scene_dir": str(fp_scene),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["simple", "complex"])
    ap.add_argument("--repo", default="lpiccinelli/unidepth-v2-vitl14")
    ap.add_argument("--tag", default="unidepth")
    args = ap.parse_args()
    out_results = ROOT / "results" / "glassboard"
    out_results.mkdir(parents=True, exist_ok=True)
    print(f"[Phase 7.4] Loading {args.repo} ...")
    model = load_unidepth(args.repo)
    summaries = []
    for scene in args.scenes:
        print(f"[Phase 7.4] Running scene={scene} ...")
        summaries.append(run_scene(model, scene, args.tag, out_results))
        torch.cuda.empty_cache()
    out = out_results / f"{args.tag}_summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[Phase 7.4] Wrote {out}")
    for s in summaries: print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
