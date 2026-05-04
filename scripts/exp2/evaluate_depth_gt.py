"""
第2段実験 depth 評価スクリプト

run_all_depth_models.py / run_foundationstereo.py が出力した *_metrics.json を集約して
results/exp2/metrics/depth_errors.csv を生成する。

追加評価:
  - SAM3 IoU (mask.png vs mask_transparent.png が存在する場合)
  - 壁校正 vs GT校正の誤差比較 (本番運用校正精度の見積もり)

使い方:
  .venv/bin/python scripts/exp2/evaluate_depth_gt.py
  .venv/bin/python scripts/exp2/evaluate_depth_gt.py --show    # 表をターミナル表示
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

PROJ    = Path(__file__).resolve().parent.parent.parent
DEPTH_R = PROJ / "results/exp2/depth"
DATA    = PROJ / "data/exp2"
OUT_DIR = PROJ / "results/exp2/metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENES    = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]
MODELS    = ["depthpro", "moge2", "unidepth", "metric3d", "da2", "marigold", "fs"]
CAL_TYPES = ["wall", "gt"]


def load_all_metrics():
    rows = []
    for f in sorted(DEPTH_R.glob("*_metrics.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        model = d.get("model", "?")
        scene = d.get("scene", "?")
        base  = {"model": model, "scene": scene, "infer_sec": d.get("infer_sec")}

        for cal in CAL_TYPES:
            m = d.get(cal, {})
            if not m:
                continue
            row = base.copy()
            row["cal"] = cal
            row["rmse_mm"]     = round(m.get("rmse_m",  float("nan")) * 1000, 2)
            row["mae_mm"]      = round(m.get("mae_m",   float("nan")) * 1000, 2)
            row["inlier_5cm"]  = round(m.get("inlier_5cm", float("nan")) * 100, 1)
            row["inlier_2cm"]  = round(m.get("inlier_2cm", float("nan")) * 100, 1)
            row["plate_med_mm"]= round(m.get("plate_median_m", float("nan")) * 1000, 1)
            row["gt_z_mm"]     = round(m.get("gt_z_m",  float("nan")) * 1000, 1)
            row["n_pixels"]    = m.get("n_pixels", 0)
            row["a_gt"]        = round(d.get("a_gt") or float("nan"), 4)
            row["b_gt_mm"]     = round((d.get("b_gt") or float("nan")) * 1000, 2)
            rows.append(row)
    return rows


def compute_sam3_iou():
    """各シーンの SAM3 IoU を計算する (mask.png vs mask_transparent.png)。"""
    results = {}
    for scene in SCENES:
        gt_path   = DATA / scene / "mask.png"
        pred_path = DATA / scene / "mask_transparent.png"
        if not gt_path.exists() or not pred_path.exists():
            results[scene] = None
            continue
        try:
            from PIL import Image
            gt   = np.array(Image.open(gt_path).convert("L"))   > 127
            pred = np.array(Image.open(pred_path).convert("L")) > 127
            inter = (gt & pred).sum()
            union = (gt | pred).sum()
            results[scene] = round(float(inter / union), 3) if union > 0 else 0.0
        except Exception:
            results[scene] = None
    return results


def write_csv(rows, path):
    if not rows:
        print("  集計対象の metrics.json が見つかりませんでした")
        return
    fieldnames = ["model", "scene", "cal", "rmse_mm", "mae_mm",
                  "inlier_5cm", "inlier_2cm", "plate_med_mm", "gt_z_mm",
                  "n_pixels", "a_gt", "b_gt_mm", "infer_sec"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  depth_errors.csv → {path}  ({len(rows)} rows)")


def print_table(rows, iou_map):
    """モデル × シーン × GT校正 の RMSE を表形式で表示。"""
    print("\n── GT校正 RMSE (mm) ──────────────────────────────────")
    header = f"{'Model':12s}" + "".join(f"  {s:12s}" for s in SCENES)
    print(header)
    print("─" * len(header))
    # モデル × シーン の lookup
    lookup = {}
    for r in rows:
        if r["cal"] == "gt":
            lookup[(r["model"], r["scene"])] = r["rmse_mm"]
    for model in MODELS:
        vals = [lookup.get((model, s), float("nan")) for s in SCENES]
        line = f"{model:12s}" + "".join(
            f"  {v:>12.1f}" if not np.isnan(v) else f"  {'--':>12s}" for v in vals)
        print(line)

    print("\n── 壁校正 vs GT校正 誤差比較 (RMSE mm) ──────────────")
    for model in MODELS:
        for scene in SCENES:
            w = next((r["rmse_mm"] for r in rows
                      if r["model"] == model and r["scene"] == scene and r["cal"] == "wall"), None)
            g = next((r["rmse_mm"] for r in rows
                      if r["model"] == model and r["scene"] == scene and r["cal"] == "gt"), None)
            if w is not None and g is not None:
                print(f"  {model:12s} {scene:18s}  wall={w:6.1f}mm  gt={g:6.1f}mm  "
                      f"改善={w-g:+.1f}mm")

    print("\n── SAM3 IoU (透明物体) ────────────────────────────────")
    for scene, iou in iou_map.items():
        iou_str = f"{iou:.3f}" if iou is not None else "検出不可"
        print(f"  {scene:20s}  {iou_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="表をターミナルに表示")
    args = ap.parse_args()

    rows    = load_all_metrics()
    iou_map = compute_sam3_iou()

    csv_path = OUT_DIR / "depth_errors.csv"
    write_csv(rows, csv_path)

    # SAM3 IoU を別ファイルにも保存
    iou_path = OUT_DIR / "sam3_iou.json"
    iou_path.write_text(json.dumps(iou_map, indent=2, ensure_ascii=False))
    print(f"  sam3_iou.json  → {iou_path}")

    if args.show or not rows:
        print_table(rows, iou_map)


if __name__ == "__main__":
    main()
