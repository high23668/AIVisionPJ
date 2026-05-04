"""
第2段実験 カラー紙 → SAM3 マスク生成スクリプト

rgb_paper.png (カラー紙貼り付け画像) に対して SAM3 で "red paper" / "blue paper" プロンプトを
かけて mask.png を生成する。Phase 1 の問題 (透明物体を低スコアで検出) を回避するため、
紙が貼られた不透明状態でマスクを生成する。

紙の色は measurements.json の paper_color フィールドから自動取得する。
存在しない場合は --paper-color で明示指定する。

最大面積マスクを選択 (板全体を覆う紙 = 最大面積確実)。

実行環境: .venv_sam3  DISPLAY=:0 必須
  DISPLAY=:0 .venv_sam3/bin/python scripts/exp2/generate_masks_from_paper.py \\
      --scene data/exp2/glass_front

  # 全4シーン一括
  DISPLAY=:0 .venv_sam3/bin/python scripts/exp2/generate_masks_from_paper.py --all

出力:
  {scene}/mask.png          バイナリマスク (0 or 255)
  {scene}/mask_overlay.jpg  確認用オーバーレイ
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJ = Path(__file__).resolve().parent.parent.parent
SCENES = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]


def load_model(device="cuda"):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("SAM3 モデルロード中...")
    model = build_sam3_image_model(load_from_HF=True, device=device)
    processor = Sam3Processor(model)
    print("  OK")
    return processor


def select_largest_mask(masks) -> np.ndarray:
    """最大面積マスクを返す (板全体を覆う紙が最大面積)。"""
    best, best_area = None, 0
    for m in masks:
        m_np = m.cpu().numpy().squeeze().astype(bool) if isinstance(m, torch.Tensor) \
               else np.array(m).squeeze().astype(bool)
        area = m_np.sum()
        if area > best_area:
            best, best_area = m_np, area
    return best


def run_scene(processor, scene_dir: Path, paper_color: str) -> bool:
    img_path = scene_dir / "rgb_paper.png"
    if not img_path.exists():
        print(f"  SKIP: rgb_paper.png が見つかりません: {scene_dir}")
        return False

    prompt = f"{paper_color} paper"
    print(f"[{scene_dir.name}] SAM3 prompt='{prompt}'")

    pil_img = Image.open(img_path).convert("RGB")
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    if len(masks) == 0:
        print(f"  WARNING: '{prompt}' が検出されませんでした")
        return False

    mask_np = select_largest_mask(masks)
    score_max = float(scores.max() if isinstance(scores, torch.Tensor) else np.max(scores))
    print(f"  最大面積マスク選択: coverage={mask_np.mean()*100:.1f}%, best_score={score_max:.3f}")

    # mask.png (0 or 255)
    mask_uint8 = mask_np.astype(np.uint8) * 255
    out_mask = scene_dir / "mask.png"
    cv2.imwrite(str(out_mask), mask_uint8)
    print(f"  mask.png → {out_mask}")

    # オーバーレイ確認画像
    img_bgr = cv2.imread(str(img_path))
    overlay = img_bgr.copy()
    overlay[mask_np] = (0, 255, 0)
    vis = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)
    out_overlay = scene_dir / "mask_overlay.jpg"
    cv2.imwrite(str(out_overlay), vis)
    print(f"  mask_overlay.jpg → {out_overlay}")

    return True


def get_paper_color(scene_dir: Path, fallback: str) -> str:
    meas = scene_dir / "measurements.json"
    if meas.exists():
        data = json.loads(meas.read_text())
        color = data.get("paper_color", "").strip()
        if color in ("red", "blue"):
            return color
    return fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", help="シーンディレクトリ (例: data/exp2/glass_front)")
    group.add_argument("--all", action="store_true", help="4シーン全て処理")
    parser.add_argument("--paper-color", default="red", choices=["red", "blue"],
                        help="measurements.json が無い場合のフォールバック色")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    processor = load_model(args.device)

    if args.all:
        results = []
        for name in SCENES:
            d = PROJ / "data/exp2" / name
            color = get_paper_color(d, args.paper_color)
            ok = run_scene(processor, d, color)
            results.append((name, ok))
        print("\n── 結果サマリ ──")
        for name, ok in results:
            print(f"  {'OK' if ok else 'NG'}  {name}")
        sys.exit(0 if all(r for _, r in results) else 1)
    else:
        d = Path(args.scene)
        color = get_paper_color(d, args.paper_color)
        ok = run_scene(processor, d, color)
        sys.exit(0 if ok else 1)
