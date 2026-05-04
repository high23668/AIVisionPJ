"""
Phase 7 再校正: スタンド(近点) + 壁(遠点) の 2 点で affine 校正し
FoundationPose scene を作り直す。対象: depthpro / moge2 / unidepth
(Metric3D は板を透過するため対象外)
"""
import json, shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path("/home/vr01/AIVisionPJ")


def get_stand_anchor(rs_depth: np.ndarray, mask: np.ndarray):
    """板マスク下辺 (y 85%ile 以上) で RS が有効な画素 = スタンド候補。
    RS < 0.38m に絞って wall 混入を除去し (median, median) を返す。"""
    h = rs_depth.shape[0]
    ys, _ = np.where(mask)
    y85 = int(np.percentile(ys, 85))
    cand = mask & (rs_depth > 0.05) & (rs_depth < 0.38) & (np.arange(h)[:, None] >= y85)
    if cand.sum() < 100:
        return None, cand
    return float(np.median(rs_depth[cand])), cand


def calibrate_affine(model_depth: np.ndarray, rs_depth: np.ndarray, mask: np.ndarray):
    """
    2 点 (stand_near, wall_far) で linear fit:
        rs = a * model + b
    stand: マスク下辺の RS 有効画素 (近点)
    wall:  マスク外の RS 有効画素 (遠点)
    """
    rs_stand_val, stand_mask = get_stand_anchor(rs_depth, mask)
    wall_mask = (~mask) & (rs_depth > 0.05) & (rs_depth < 3.0) & np.isfinite(model_depth) & (model_depth > 0.1)

    if rs_stand_val is None or wall_mask.sum() < 500:
        s = float(np.median(rs_depth[wall_mask]) / np.median(model_depth[wall_mask]))
        return s, 0.0, {"method": "scale_only_fallback"}

    dp_stand = float(np.median(model_depth[stand_mask & np.isfinite(model_depth)]))
    rs_wall   = float(np.median(rs_depth[wall_mask]))
    dp_wall   = float(np.median(model_depth[wall_mask]))

    # 2-point linear: rs = a * dp + b
    a = (rs_wall - rs_stand_val) / (dp_wall - dp_stand)
    b = rs_stand_val - a * dp_stand

    info = {
        "method": "affine_2point",
        "stand": {"rs": rs_stand_val, "dp": dp_stand, "n_px": int(stand_mask.sum())},
        "wall":  {"rs": rs_wall,      "dp": dp_wall,  "n_px": int(wall_mask.sum())},
        "a": a, "b": b,
    }
    return a, b, info


def apply_affine(model_depth: np.ndarray, a: float, b: float):
    d = np.where(np.isfinite(model_depth), a * model_depth + b, np.nan)
    return d.astype(np.float32)


def visualize(rgb_bgr, depth_cal_old, depth_cal_new, rs_depth, plate_mask,
              info, out_path: Path, tag: str, scene: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(rgb_bgr[:, :, ::-1]); axes[0, 0].set_title(f"RGB ({scene})"); axes[0, 0].axis("off")

    rs_vis = rs_depth.copy(); rs_vis[rs_vis == 0] = np.nan
    im = axes[0, 1].imshow(rs_vis, cmap="turbo", vmin=0.2, vmax=0.6)
    axes[0, 1].set_title("RealSense depth (m)"); axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(depth_cal_old, cmap="turbo", vmin=0.2, vmax=0.6)
    axes[0, 2].set_title(f"{tag} old (wall-only scale) (m)"); axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(depth_cal_new, cmap="turbo", vmin=0.2, vmax=0.6)
    axes[1, 0].set_title(f"{tag} new (affine stand+wall) (m)"); axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    plate = plate_mask & np.isfinite(depth_cal_new) & (depth_cal_new > 0)
    rs_plate = plate_mask & (rs_depth > 0)
    axes[1, 1].hist(depth_cal_old[plate_mask & np.isfinite(depth_cal_old) & (depth_cal_old > 0)],
                    bins=40, alpha=0.5, color="tab:orange", label="old plate")
    axes[1, 1].hist(depth_cal_new[plate] if plate.sum() else [0],
                    bins=40, alpha=0.5, color="tab:green", label="new plate")
    axes[1, 1].hist(rs_depth[rs_plate] if rs_plate.sum() else [0],
                    bins=40, alpha=0.4, color="tab:blue", label="RS wall (through glass)")
    axes[1, 1].axvline(info.get("stand", {}).get("rs", 0), color="lime", ls="--", label="stand RS")
    axes[1, 1].set_xlabel("depth (m)"); axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_title("plate-region depth histogram")

    s = info
    txt = (f"method: {s['method']}\n"
           f"a={s.get('a',1):.4f}, b={s.get('b',0):.4f}\n"
           f"stand: RS={s.get('stand',{}).get('rs','?'):.3f}m  DP={s.get('stand',{}).get('dp','?'):.3f}m\n"
           f"wall:  RS={s.get('wall', {}).get('rs','?'):.3f}m  DP={s.get('wall', {}).get('dp','?'):.3f}m\n"
           f"plate new median: {float(np.median(depth_cal_new[plate])) if plate.sum() else 0:.3f}m")
    axes[1, 2].text(0.05, 0.95, txt, transform=axes[1, 2].transAxes,
                    va="top", fontfamily="monospace", fontsize=9)
    axes[1, 2].axis("off"); axes[1, 2].set_title("calibration info")

    fig.suptitle(f"Phase 7 affine recalib: {tag} ({scene})", fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def export_fp_scene(scene_in: Path, scene_out: Path, depth_m: np.ndarray, intrinsics: dict):
    scene_out.mkdir(parents=True, exist_ok=True)
    for sub in ("rgb", "depth", "masks"):
        (scene_out / sub).mkdir(exist_ok=True)
    shutil.copy(scene_in / "rgb.png", scene_out / "rgb" / "000000.png")
    mask_src = scene_in / "mask_plate.png"
    if not mask_src.exists():
        mask_src = scene_in / "mask.png"
    shutil.copy(mask_src, scene_out / "masks" / "000000.png")
    fx, fy = intrinsics["rgb"]["fx"], intrinsics["rgb"]["fy"]
    cx, cy = intrinsics["rgb"]["cx"], intrinsics["rgb"]["cy"]
    np.savetxt(scene_out / "cam_K.txt",
               np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), fmt="%.6f")
    d = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
    cv2.imwrite(str(scene_out / "depth" / "000000.png"),
                np.clip(d * 1000, 0, 65535).astype(np.uint16))


def process(tag: str, scene: str, out_results: Path):
    scene_dir = ROOT / "data" / f"glassboard_{scene}"
    rgb = cv2.imread(str(scene_dir / "rgb.png"))
    intr = json.loads((scene_dir / "intrinsics.json").read_text())
    rs = np.load(scene_dir / "depth.npy").astype(float)
    mask_file = scene_dir / "mask_plate.png"
    if not mask_file.exists():
        mask_file = scene_dir / "mask.png"
    plate_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 0
    depth_raw = np.load(scene_dir / f"depth_{tag}.npy").astype(float)

    # old calibration (wall only, for comparison)
    bg = (~plate_mask) & (rs > 0.05) & (rs < 3.0) & np.isfinite(depth_raw) & (depth_raw > 0.05)
    s_old = float(np.median(rs[bg]) / np.median(depth_raw[bg]))
    depth_old = depth_raw * s_old

    # new affine calibration
    a, b, info = calibrate_affine(depth_raw, rs, plate_mask)
    depth_new = apply_affine(depth_raw, a, b)

    np.save(scene_dir / f"depth_{tag}_affine.npy", depth_new)
    print(f"[{tag} {scene}] old scale={s_old:.4f} | affine a={a:.4f} b={b:.4f}")
    print(f"  stand: RS={info.get('stand',{}).get('rs','—')} DP={info.get('stand',{}).get('dp','—')}")
    plate = plate_mask & np.isfinite(depth_new) & (depth_new > 0)
    if plate.sum():
        print(f"  plate depth: old={np.median(depth_old[plate_mask & np.isfinite(depth_old)]):.3f}m "
              f"→ new={np.median(depth_new[plate]):.3f}m")

    vis_path = out_results / f"{tag}_affine_{scene}.jpg"
    visualize(rgb, depth_old, depth_new, rs, plate_mask, info, vis_path, tag, scene)

    fp_out = ROOT / "data" / f"glassboard_{scene}_fp_{tag}_affine"
    export_fp_scene(scene_dir, fp_out, depth_new, intr)
    return {"tag": tag, "scene": scene, "a": a, "b": b, "info": info,
            "plate_median_new": float(np.median(depth_new[plate])) if plate.sum() else None,
            "fp_scene": str(fp_out)}


def main():
    out = ROOT / "results" / "glassboard"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for tag in ["depthpro", "moge2", "unidepth"]:
        for scene in ["simple", "complex"]:
            try:
                results.append(process(tag, scene, out))
            except Exception as e:
                print(f"ERROR {tag} {scene}: {e}")
    (out / "affine_calibration_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("Done.")


if __name__ == "__main__":
    main()
