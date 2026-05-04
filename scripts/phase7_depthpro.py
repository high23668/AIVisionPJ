"""Phase 7.3 Depth Pro (Apple) ガラス板 depth 推定 + RS壁面で scale 校正 + FP scene 出力."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("/home/vr01/AIVisionPJ")


def load_depthpro():
    import depth_pro
    cfg = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    cfg.checkpoint_uri = str(ROOT / "models" / "depth_pro" / "checkpoints" / "depth_pro.pt")
    model, transform = depth_pro.create_model_and_transforms(config=cfg, device=torch.device("cuda"), precision=torch.float16)
    model.eval()
    return model, transform


def predict_depth(model, transform, rgb_bgr: np.ndarray, fx: float):
    rgb_rgb = rgb_bgr[:, :, ::-1].copy()
    h, w = rgb_rgb.shape[:2]
    x = transform(rgb_rgb)  # already returns CHW tensor in [-1,1]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to("cuda", dtype=torch.float16)
    with torch.no_grad():
        out = model.infer(x, f_px=torch.tensor([fx], device="cuda"))
    depth = out["depth"]
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    depth_np = depth.float().cpu().numpy()
    if depth_np.shape != (h, w):
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth_np.astype(np.float32)


def calibrate_scale(model_depth, rs_depth, plate_mask):
    bg = (~plate_mask) & (rs_depth > 0.05) & (rs_depth < 3.0) & np.isfinite(model_depth) & (model_depth > 0.05)
    if bg.sum() < 1000:
        return 1.0, bg
    s = float(np.median(rs_depth[bg]) / np.median(model_depth[bg]))
    return s, bg


def visualize(rgb_bgr, depth_raw, depth_cal, rs_depth, plate_mask, out_path: Path, model_name: str, scene: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(rgb_bgr[:, :, ::-1]); axes[0, 0].set_title(f"RGB ({scene})"); axes[0, 0].axis("off")

    rs_vis = rs_depth.copy(); rs_vis[rs_vis == 0] = np.nan
    im = axes[0, 1].imshow(rs_vis, cmap="turbo", vmin=0.2, vmax=1.5); axes[0, 1].set_title("RealSense depth (m)"); axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

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

    fig.suptitle(f"Phase 7.3 {model_name} ({scene})", fontsize=14)
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


def run_scene(model, transform, scene: str, model_tag: str, out_results: Path):
    scene_dir = ROOT / "data" / f"glassboard_{scene}"
    rgb = cv2.imread(str(scene_dir / "rgb.png"))
    intr = json.loads((scene_dir / "intrinsics.json").read_text())
    fx = intr["rgb"]["fx"]
    rs_depth = np.load(scene_dir / "depth.npy").astype(float)
    mask_file = scene_dir / "mask_plate.png"
    if not mask_file.exists():
        mask_file = scene_dir / "mask.png"
    plate_mask = (cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0)
    print(f"  mask: {mask_file.name}")

    depth_raw = predict_depth(model, transform, rgb, fx)
    np.save(scene_dir / f"depth_{model_tag}.npy", depth_raw)

    s, bg_mask = calibrate_scale(depth_raw, rs_depth, plate_mask)
    depth_cal = depth_raw * s
    visualize(rgb, depth_raw, depth_cal, rs_depth, plate_mask, out_results / f"{model_tag}_{scene}.jpg", model_tag, scene)
    fp_scene = ROOT / "data" / f"glassboard_{scene}_fp_{model_tag}"
    export_fp_scene(scene_dir, fp_scene, depth_cal, intr)
    plate = plate_mask & np.isfinite(depth_cal) & (depth_cal > 0)
    rs_plate_valid = plate_mask & (rs_depth > 0)
    return {
        "scene": scene, "scale": s, "bg_pixels": int(bg_mask.sum()),
        "plate_depth_median_m": float(np.median(depth_cal[plate])) if plate.sum() else None,
        "plate_depth_p10_p90_m": [float(np.percentile(depth_cal[plate], 10)), float(np.percentile(depth_cal[plate], 90))] if plate.sum() else None,
        "rs_plate_median_m": float(np.median(rs_depth[rs_plate_valid])) if rs_plate_valid.sum() else None,
        "fp_scene_dir": str(fp_scene),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["simple", "complex"])
    ap.add_argument("--tag", default="depthpro")
    args = ap.parse_args()
    out_results = ROOT / "results" / "glassboard"
    out_results.mkdir(parents=True, exist_ok=True)
    print("[Phase 7.3] Loading Depth Pro ...")
    model, transform = load_depthpro()
    summaries = []
    for scene in args.scenes:
        print(f"[Phase 7.3] Running scene={scene} ...")
        summaries.append(run_scene(model, transform, scene, args.tag, out_results))
        torch.cuda.empty_cache()
    out = out_results / f"{args.tag}_summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[Phase 7.3] Wrote {out}")
    for s in summaries: print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
