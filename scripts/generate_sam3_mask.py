"""
SAM3 マスク生成スクリプト
- テキストプロンプトで物体マスクを生成し、バイナリPNGとして保存
- FoundationPose model-free の入力として使用

使い方 (.venv_sam3 で実行):
  source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate

  # 単一画像のマスク生成
  python scripts/generate_sam3_mask.py \
    --image data/realsense/rgb.png \
    --prompt "cup" \
    --output data/realsense/mask.png

  # ディレクトリ内の全rgb_*.pngにマスク生成 (参照画像用)
  python scripts/generate_sam3_mask.py \
    --dir data/realsense/ref \
    --prompt "cup"

出力:
  --output で指定したパス、または入力画像と同じディレクトリに mask.png / mask_NNN.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_model(device="cuda"):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("SAM3 モデルロード中...")
    model = build_sam3_image_model(load_from_HF=True, device=device)
    processor = Sam3Processor(model)
    print("  OK")
    return model, processor


def generate_mask(processor, image_path, prompt):
    """Generate binary mask for the best-scoring segment."""
    pil_img = Image.open(image_path).convert("RGB")

    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    if len(masks) == 0:
        print(f"  WARNING: '{prompt}' が検出されませんでした: {image_path}")
        return None

    # スコアが最も高いマスクを選択
    best_idx = int(torch.argmax(scores) if isinstance(scores, torch.Tensor)
                   else np.argmax(scores))
    mask = masks[best_idx]

    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().squeeze().astype(bool)
    else:
        mask_np = np.array(mask).squeeze().astype(bool)

    score = float(scores[best_idx])
    print(f"  '{prompt}': score={score:.3f}, mask coverage={mask_np.mean()*100:.1f}%")
    return mask_np


def save_mask(mask_np, output_path):
    """Save boolean mask as binary uint8 PNG (0 or 255)."""
    mask_uint8 = (mask_np.astype(np.uint8)) * 255
    cv2.imwrite(str(output_path), mask_uint8)
    print(f"  Saved mask: {output_path}")


def save_overlay(image_path, mask_np, output_path):
    """Save visualization: original image with mask overlay."""
    img_bgr = cv2.imread(str(image_path))
    overlay = img_bgr.copy()
    overlay[mask_np] = (0, 255, 0)
    vis = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)
    cv2.imwrite(str(output_path), vis)
    print(f"  Saved overlay: {output_path}")


def process_single(processor, image_path, prompt, output_path=None):
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.parent / "mask.png"
    else:
        output_path = Path(output_path)

    print(f"処理中: {image_path.name}")
    mask_np = generate_mask(processor, image_path, prompt)
    if mask_np is None:
        return False

    save_mask(mask_np, output_path)
    overlay_path = output_path.parent / (output_path.stem + "_overlay.jpg")
    save_overlay(image_path, mask_np, overlay_path)
    return True


def process_directory(processor, dir_path, prompt):
    """Process all rgb_*.png files in a directory."""
    dir_path = Path(dir_path)
    images = sorted(dir_path.glob("rgb_*.png")) + sorted(dir_path.glob("rgb.png"))
    images = sorted(set(images))  # deduplicate

    if not images:
        print(f"ERROR: {dir_path} に rgb_*.png が見つかりません")
        return

    print(f"{len(images)}枚の画像を処理します (prompt='{prompt}')")
    success = 0
    for img_path in images:
        # rgb_000.png → mask_000.png
        if img_path.stem == "rgb":
            mask_path = dir_path / "mask.png"
        else:
            suffix = img_path.stem.replace("rgb", "")  # "_000"
            mask_path = dir_path / f"mask{suffix}.png"

        ok = process_single(processor, img_path, prompt, mask_path)
        if ok:
            success += 1

    print(f"\n完了: {success}/{len(images)} 枚のマスクを生成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 マスク生成")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="入力画像パス (単一画像モード)")
    group.add_argument("--dir", help="入力ディレクトリ (rgb_*.png を一括処理)")
    parser.add_argument("--prompt", required=True, help="テキストプロンプト (例: cup)")
    parser.add_argument("--output", default=None,
                        help="出力マスクパス (--image モード時, デフォルト: 同ディレクトリの mask.png)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, processor = load_model(args.device)

    if args.image:
        ok = process_single(processor, args.image, args.prompt, args.output)
        sys.exit(0 if ok else 1)
    else:
        process_directory(processor, args.dir, args.prompt)
