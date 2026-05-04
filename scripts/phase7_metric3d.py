"""Phase 7.1 Metric3D v2 (ViT-Large) ガラス板 depth 推定 + RS壁面で scale 校正 + FP scene 出力."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("/home/vr01/AIVisionPJ")


def load_metric3d(model_name: str = "metric3d_vit_large"):
    model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True, trust_repo=True)
    model.cuda().eval()
    return model


def predict_depth(model, rgb_bgr: np.ndarray, intrinsic: list[float]) -> np.ndarray:
    rgb = rgb_bgr[:, :, ::-1]  # to RGB
    h0, w0 = rgb.shape[:2]
    input_size = (616, 1064)  # ViT
    scale = min(input_size[0] / h0, input_size[1] / w0)
    rgb_resized = cv2.resize(rgb, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
    intr_s = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

    pad_h = input_size[0] - rgb_resized.shape[0]
    pad_w = input_size[1] - rgb_resized.shape[1]
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    padding_mean = [123.675, 116.28, 103.53]
    rgb_pad = cv2.copyMakeBorder(rgb_resized, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=padding_mean)

    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    x = torch.from_numpy(rgb_pad.transpose((2, 0, 1))).float()
    x = (x - mean) / std
    x = x[None].cuda()

    with torch.no_grad():
        pred, conf, _ = model.inference({"input": x})

    pred = pred.squeeze()
    pred = pred[pt : pred.shape[0] - pb, pl : pred.shape[1] - pr]
    pred = torch.nn.functional.interpolate(pred[None, None], (h0, w0), mode="bilinear").squeeze()

    canon_to_real = intr_s[0] / 1000.0
    pred = pred * canon_to_real
    return pred.cpu().numpy()


def calibrate_scale(model_depth: np.ndarray, rs_depth: np.ndarray, plate_mask: np.ndarray):
    """RS depth と model depth から、ガラス板を除外した領域で scale 係数を算出."""
    bg = (~plate_mask) & (rs_depth > 0.05) & (rs_depth < 3.0) & (model_depth > 0.05) & np.isfinite(model_depth)
    if bg.sum() < 1000:
        return 1.0, bg
    s = float(np.median(rs_depth[bg]) / np.median(model_depth[bg]))
    return s, bg


def visualize(rgb_bgr, depth_raw, depth_cal, rs_depth, plate_mask, out_path: Path, model_name: str, scene: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(rgb_bgr[:, :, ::-1])
    axes[0, 0].set_title(f"RGB ({scene})")
    axes[0, 0].axis("off")

    rs_vis = rs_depth.copy()
    rs_vis[rs_vis == 0] = np.nan
    im = axes[0, 1].imshow(rs_vis, cmap="turbo", vmin=0.2, vmax=1.5)
    axes[0, 1].set_title("RealSense depth (m)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(plate_mask, cmap="gray")
    axes[0, 2].set_title("Plate mask (SAM3)")
    axes[0, 2].axis("off")

    raw_vmin = float(np.nanpercentile(depth_raw, 5))
    raw_vmax = float(np.nanpercentile(depth_raw, 95))
    im = axes[1, 0].imshow(depth_raw, cmap="turbo", vmin=raw_vmin, vmax=raw_vmax)
    axes[1, 0].set_title(f"{model_name} raw depth (m, auto-scale)")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(depth_cal, cmap="turbo", vmin=0.2, vmax=1.5)
    axes[1, 1].set_title(f"{model_name} calibrated (m)")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    plate = plate_mask & (depth_cal > 0)
    plate_vals = depth_cal[plate] if plate.sum() > 0 else np.array([0])
    rs_plate = rs_depth[plate_mask & (rs_depth > 0)] if (plate_mask & (rs_depth > 0)).sum() > 0 else np.array([0])
    axes[1, 2].hist(plate_vals, bins=40, alpha=0.6, label=f"{model_name} on plate (n={plate.sum()})", color="tab:orange")
    axes[1, 2].hist(rs_plate, bins=40, alpha=0.6, label=f"RS on plate (n={(plate_mask & (rs_depth > 0)).sum()})", color="tab:blue")
    axes[1, 2].set_xlabel("depth (m)"); axes[1, 2].set_ylabel("freq"); axes[1, 2].set_title("plate-region depth histogram"); axes[1, 2].legend()

    fig.suptitle(f"Phase 7.1 {model_name} ({scene})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def export_fp_scene(scene_in: Path, scene_out: Path, depth_m: np.ndarray, intrinsics: dict):
    scene_out.mkdir(parents=True, exist_ok=True)
    (scene_out / "rgb").mkdir(exist_ok=True)
    (scene_out / "depth").mkdir(exist_ok=True)
    (scene_out / "masks").mkdir(exist_ok=True)
    shutil.copy(scene_in / "rgb.png", scene_out / "rgb" / "000000.png")
    mask_src = scene_in / "mask_plate.png"
    if not mask_src.exists():
        mask_src = scene_in / "mask.png"
    shutil.copy(mask_src, scene_out / "masks" / "000000.png")
    fx, fy = intrinsics["rgb"]["fx"], intrinsics["rgb"]["fy"]
    cx, cy = intrinsics["rgb"]["cx"], intrinsics["rgb"]["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    np.savetxt(scene_out / "cam_K.txt", K, fmt="%.6f")
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(scene_out / "depth" / "000000.png"), depth_mm)


def run_scene(model, scene: str, model_tag: str, out_results: Path):
    scene_dir = ROOT / "data" / f"glassboard_{scene}"
    rgb = cv2.imread(str(scene_dir / "rgb.png"))
    intr = json.loads((scene_dir / "intrinsics.json").read_text())
    intrinsic = [intr["rgb"]["fx"], intr["rgb"]["fy"], intr["rgb"]["cx"], intr["rgb"]["cy"]]
    rs_depth = np.load(scene_dir / "depth.npy").astype(float)
    mask_file = scene_dir / "mask_plate.png"
    if not mask_file.exists():
        mask_file = scene_dir / "mask.png"
    plate_mask = (cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0)
    print(f"  mask: {mask_file.name}")

    depth_raw = predict_depth(model, rgb, intrinsic)
    np.save(scene_dir / f"depth_{model_tag}.npy", depth_raw.astype(np.float32))

    s, bg_mask = calibrate_scale(depth_raw, rs_depth, plate_mask)
    depth_cal = depth_raw * s

    visualize(rgb, depth_raw, depth_cal, rs_depth, plate_mask, out_results / f"{model_tag}_{scene}.jpg", model_tag, scene)

    fp_scene = ROOT / "data" / f"glassboard_{scene}_fp_{model_tag}"
    export_fp_scene(scene_dir, fp_scene, depth_cal, intr)

    plate = plate_mask & (depth_cal > 0)
    rs_plate_valid = plate_mask & (rs_depth > 0)
    summary = {
        "scene": scene,
        "scale": s,
        "bg_pixels": int(bg_mask.sum()),
        "plate_depth_median_m": float(np.median(depth_cal[plate])) if plate.sum() else None,
        "plate_depth_p10_p90_m": [float(np.percentile(depth_cal[plate], 10)), float(np.percentile(depth_cal[plate], 90))] if plate.sum() else None,
        "rs_plate_median_m": float(np.median(rs_depth[rs_plate_valid])) if rs_plate_valid.sum() else None,
        "fp_scene_dir": str(fp_scene),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["simple", "complex"])
    ap.add_argument("--model_name", default="metric3d_vit_large")
    ap.add_argument("--tag", default="metric3d")
    args = ap.parse_args()

    out_results = ROOT / "results" / "glassboard"
    out_results.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 7.1] Loading {args.model_name} ...")
    model = load_metric3d(args.model_name)
    summaries = []
    for scene in args.scenes:
        print(f"[Phase 7.1] Running scene={scene} ...")
        summaries.append(run_scene(model, scene, args.tag, out_results))
        torch.cuda.empty_cache()

    out = out_results / f"{args.tag}_summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[Phase 7.1] Wrote {out}")
    for s in summaries:
        print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
