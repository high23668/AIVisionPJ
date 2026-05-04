"""
第2段実験 FoundationPose シーンデータ準備スクリプト

GT affine 校正済み depth (results/exp2/depth/{model}_{scene}_gt.npy) と
data/exp2/{scene}/ の RGB・マスクを FoundationPose YcbineoatReader 形式に変換する。

出力先: data/exp2/fp/{model}_{scene}/
  rgb/000000.png      RGB
  depth/000000.png    depth uint16 mm
  masks/000000.png    マスク
  cam_K.txt           カメラ内部パラメータ

CAD の割り当て:
  glass_* → data/exp2/cad/glass_plate.obj       (148×148×2.5mm)
  resin_* → data/exp2/cad/resin_sheet_fp.obj    (70×100×1mm、FP用厚み補正版)

使い方:
  .venv/bin/python scripts/exp2/setup_fp_scenes.py
  .venv/bin/python scripts/exp2/setup_fp_scenes.py --models moge2 depthpro --scenes glass_front
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJ    = Path(__file__).resolve().parent.parent.parent
DATA    = PROJ / "data/exp2"
DEPTH_R = PROJ / "results/exp2/depth"
OUT_FP  = DATA / "fp"

SCENES = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]
MODELS = ["moge2", "depthpro", "da2"]   # 精度上位3本

CAD_MAP = {
    "glass_front":   PROJ / "data/exp2/cad/glass_plate.obj",
    "glass_oblique": PROJ / "data/exp2/cad/glass_plate.obj",
    "resin_front":   PROJ / "data/exp2/cad/resin_sheet_fp.obj",
    "resin_oblique": PROJ / "data/exp2/cad/resin_sheet_fp.obj",
}


def setup_scene(model: str, scene: str) -> bool:
    scene_dir = DATA / scene
    depth_gt  = DEPTH_R / f"{model}_{scene}_gt.npy"

    if not depth_gt.exists():
        print(f"  SKIP: GT depth なし ({depth_gt.name})")
        return False

    out = OUT_FP / f"{model}_{scene}"
    for sub in ["rgb", "depth", "masks"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # RGB
    shutil.copy(scene_dir / "rgb.png", out / "rgb" / "000000.png")

    # depth: float32 meters → uint16 mm
    depth_m = np.load(str(depth_gt)).astype(np.float32)
    # NaN / 負値を 0 に
    depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
    depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(out / "depth" / "000000.png"), depth_mm)

    # mask
    shutil.copy(scene_dir / "mask.png", out / "masks" / "000000.png")

    # cam_K.txt
    intr = json.loads((scene_dir / "intrinsics.json").read_text())
    K = intr["rgb"]
    cam_k = np.array([[K["fx"], 0, K["cx"]],
                      [0, K["fy"], K["cy"]],
                      [0, 0, 1.0]])
    np.savetxt(str(out / "cam_K.txt"), cam_k, fmt="%.4f")

    # depth 統計
    valid = depth_mm[depth_mm > 0]
    print(f"  [OK] {model}/{scene}  depth median={np.median(valid):.0f}mm  "
          f"valid_px={len(valid)}  → {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()

    ok = 0
    for model in args.models:
        for scene in args.scenes:
            print(f"{model} × {scene}")
            if setup_scene(model, scene):
                ok += 1

    print(f"\n{ok}/{len(args.models)*len(args.scenes)} シーンを準備完了")
    print(f"出力先: {OUT_FP}")

    # CAD ファイルの存在確認
    print("\nCAD ファイル確認:")
    for path in set(CAD_MAP.values()):
        status = "✅" if path.exists() else "❌ 見つかりません"
        print(f"  {status}  {path.name}")


if __name__ == "__main__":
    main()
