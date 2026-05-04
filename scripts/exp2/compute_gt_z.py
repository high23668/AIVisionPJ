"""
GT Z 計算スクリプト

depth_paper.npy (カラー紙貼り付け時の RealSense depth) と mask.png から
板面中央の GT 奥行きを計算し、measurements.json の gt_z_center_mm を更新する。

使い方:
  python scripts/exp2/compute_gt_z.py --scene data/exp2/glass_front

  # 全4シーン一括
  python scripts/exp2/compute_gt_z.py --all

出力:
  measurements.json の gt_z_center_mm を更新
  depth_paper_gt_stats.json を保存 (median/mean/std/inlier_count)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJ = Path(__file__).resolve().parent.parent.parent
SCENES = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]


def compute(scene_dir: Path) -> dict | None:
    depth_path = scene_dir / "depth_paper.npy"
    mask_path = scene_dir / "mask.png"

    if not depth_path.exists():
        print(f"  SKIP: depth_paper.npy が見つかりません: {scene_dir}")
        return None
    if not mask_path.exists():
        print(f"  SKIP: mask.png が見つかりません: {scene_dir}")
        return None

    depth_m = np.load(str(depth_path))
    mask = np.array(Image.open(mask_path).convert("L")) > 127

    d_masked = depth_m[mask]
    d_valid = d_masked[d_masked > 0.05]

    if len(d_valid) < 100:
        print(f"  WARNING: 有効深度画素が少なすぎます ({len(d_valid)} px): {scene_dir}")
        return None

    gt_z_m = float(np.median(d_valid))
    stats = {
        "gt_z_center_m": gt_z_m,
        "gt_z_center_mm": gt_z_m * 1000,
        "mean_m": float(np.mean(d_valid)),
        "std_m": float(np.std(d_valid)),
        "inlier_count": int(len(d_valid)),
        "mask_coverage_pct": float(mask.mean() * 100),
    }

    print(f"[{scene_dir.name}] GT Z = {gt_z_m*1000:.1f} mm  "
          f"(std={stats['std_m']*1000:.1f}mm, N={stats['inlier_count']})")

    # measurements.json 更新
    meas_path = scene_dir / "measurements.json"
    if meas_path.exists():
        meas = json.loads(meas_path.read_text())
    else:
        meas = {"scene": scene_dir.name}
    meas["gt_z_center_mm"] = stats["gt_z_center_mm"]
    meas_path.write_text(json.dumps(meas, indent=2, ensure_ascii=False))
    print(f"  measurements.json 更新: gt_z_center_mm = {stats['gt_z_center_mm']:.1f}")

    # 詳細統計保存
    stats_path = scene_dir / "depth_paper_gt_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  depth_paper_gt_stats.json → {stats_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", help="シーンディレクトリ (例: data/exp2/glass_front)")
    group.add_argument("--all", action="store_true", help="4シーン全て処理")
    args = parser.parse_args()

    if args.all:
        results = []
        for name in SCENES:
            d = PROJ / "data/exp2" / name
            r = compute(d)
            results.append((name, r))
        print("\n── GT Z サマリ ──")
        for name, r in results:
            if r:
                print(f"  {name:20s}  {r['gt_z_center_mm']:.1f} mm")
            else:
                print(f"  {name:20s}  --")
        sys.exit(0 if all(r for _, r in results) else 1)
    else:
        r = compute(Path(args.scene))
        sys.exit(0 if r else 1)
