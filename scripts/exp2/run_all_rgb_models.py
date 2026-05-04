"""
第2段実験 RGB 系モデル一括実行スクリプト

対象モデル:
  yolo    YOLO26             - 2D 物体検出
  dinov3  DINOv3 ViT-L      - PCA 特徴可視化
  qwen    Qwen3-VL 4B        - 透明物体 VQA (言語識別)

※ SAM3 は .venv_sam3 + DISPLAY=:0 が必要なため別コマンドで実行済み
   (mask.png / mask_transparent.png / IoU は evaluate_depth_gt.py で集計)

実行環境: .venv
  .venv/bin/python scripts/exp2/run_all_rgb_models.py --model all
  .venv/bin/python scripts/exp2/run_all_rgb_models.py --model yolo dinov3

出力:
  results/exp2/rgb_models/yolo_{scene}.jpg         bbox 可視化
  results/exp2/rgb_models/yolo_{scene}.json        検出結果
  results/exp2/rgb_models/dinov3_{scene}_pca.jpg   PCA 特徴マップ
  results/exp2/rgb_models/qwen_{scene}.jpg         VQA 結果オーバーレイ
  results/exp2/rgb_models/qwen_{scene}.json        応答テキスト
  results/exp2/rgb_models/rgb_summary.json         全モデル結果サマリ
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ    = Path(__file__).resolve().parent.parent.parent
DATA    = PROJ / "data/exp2"
OUT     = PROJ / "results/exp2/rgb_models"
OUT.mkdir(parents=True, exist_ok=True)

SCENES     = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]
ALL_MODELS = ["yolo", "dinov3", "qwen"]

# 各シーンで Qwen に問いかけるプロンプト
QWEN_PROMPT = (
    "Look at this image carefully. "
    "Answer the following questions in one sentence each:\n"
    "1. Is there a transparent glass plate visible?\n"
    "2. Is there a transparent plastic or resin sheet visible?\n"
    "3. What transparent objects can you see, if any?"
)


# ── YOLO26 ────────────────────────────────────────────────────────

def run_yolo(scenes):
    from ultralytics import YOLO
    print("[YOLO26] モデルロード中...")
    model = YOLO("yolo26l.pt")
    results_all = {}

    for scene in scenes:
        rgb_path = DATA / scene / "rgb.png"
        if not rgb_path.exists():
            continue
        print(f"  [{scene}]")
        t0 = time.time()
        res = model(str(rgb_path), conf=0.25, verbose=False)
        elapsed = time.time() - t0

        detections = []
        img_vis = cv2.imread(str(rgb_path))
        for r in res:
            if r.boxes is not None:
                for box in r.boxes:
                    cls  = model.names[int(box.cls)]
                    conf = float(box.conf)
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    detections.append({"class": cls, "confidence": round(conf, 3), "bbox": xyxy})
                    cv2.rectangle(img_vis, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(img_vis, f"{cls} {conf:.2f}",
                                (xyxy[0], xyxy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 未検出の場合も記録
        if not detections:
            cv2.putText(img_vis, "No detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        out_img = OUT / f"yolo_{scene}.jpg"
        cv2.imwrite(str(out_img), img_vis)

        result = {"scene": scene, "model": "yolo26l", "infer_sec": round(elapsed, 3),
                  "n_detections": len(detections), "detections": detections}
        (OUT / f"yolo_{scene}.json").write_text(json.dumps(result, indent=2))
        results_all[scene] = result
        print(f"    検出数: {len(detections)}  ({elapsed:.2f}s)")
        for d in detections:
            print(f"      {d['class']:20s} conf={d['confidence']:.3f}")

    del model; torch.cuda.empty_cache()
    return results_all


# ── DINOv3 ────────────────────────────────────────────────────────

def run_dinov3(scenes):
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image as PILImage
    from sklearn.decomposition import PCA

    print("[DINOv3] モデルロード中...")
    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).cuda().eval()
        print("  DINOv3 ViT-L loaded")
    except Exception:
        print("  DINOv3 not available, falling back to DINOv2")
        model_id = "facebook/dinov2-large"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).cuda().eval()

    results_all = {}
    for scene in scenes:
        rgb_path = DATA / scene / "rgb.png"
        if not rgb_path.exists():
            continue
        print(f"  [{scene}]")
        pil = PILImage.open(rgb_path).convert("RGB")
        h, w = pil.size[1], pil.size[0]

        # 高解像度処理: patch_size=16 で 480×640 → 30×40=1200 patches
        inputs = processor(images=pil, return_tensors="pt",
                           size={"height": h, "width": w})
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t0 = time.time()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=False)
        elapsed = time.time() - t0

        # patch tokens のみ取得 (CLS + register tokens を除く)
        patch_size = model.config.patch_size
        proc_h = inputs["pixel_values"].shape[2]
        proc_w = inputs["pixel_values"].shape[3]
        patch_h = proc_h // patch_size
        patch_w = proc_w // patch_size
        n_patch = patch_h * patch_w
        feats = out.last_hidden_state[0, 1:1 + n_patch].float().cpu().numpy()
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # PCA 3成分 → RGB 可視化 (bilinear でスムーズにアップスケール)
        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(feats)
        for i in range(3):
            pca_feats[:, i] = (pca_feats[:, i] - pca_feats[:, i].min()) / \
                              (pca_feats[:, i].max() - pca_feats[:, i].min() + 1e-8)
        pca_img = (pca_feats.reshape(patch_h, patch_w, 3) * 255).astype(np.uint8)
        pca_img = cv2.resize(pca_img, (w, h), interpolation=cv2.INTER_LINEAR)

        rgb_bgr = cv2.imread(str(rgb_path))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(rgb_bgr[:, :, ::-1]); axes[0].set_title(f"RGB ({scene})"); axes[0].axis("off")
        axes[1].imshow(pca_img); axes[1].set_title("DINOv3 PCA (3 components)"); axes[1].axis("off")
        fig.suptitle(f"DINOv3 feature PCA — {scene}", fontsize=13)
        fig.tight_layout()
        out_path = OUT / f"dinov3_{scene}_pca.jpg"
        fig.savefig(out_path, dpi=100); plt.close(fig)

        result = {"scene": scene, "model": "dinov3-vitl", "infer_sec": round(elapsed, 3),
                  "pca_explained_ratio": [round(float(r), 3) for r in pca.explained_variance_ratio_]}
        (OUT / f"dinov3_{scene}.json").write_text(json.dumps(result, indent=2))
        results_all[scene] = result
        print(f"    PCA 説明率: {[f'{r:.1%}' for r in pca.explained_variance_ratio_]}  ({elapsed:.2f}s)")

    del model; torch.cuda.empty_cache()
    return results_all


# ── Qwen3-VL ──────────────────────────────────────────────────────

def run_qwen(scenes):
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from PIL import Image as PILImage

    print("[Qwen3-VL 4B] モデルロード中...")
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        low_cpu_mem_usage=True,
    ).eval()

    results_all = {}
    for scene in scenes:
        rgb_path = DATA / scene / "rgb.png"
        if not rgb_path.exists():
            continue
        print(f"  [{scene}]")
        pil = PILImage.open(rgb_path).convert("RGB")
        W, H = pil.size

        # ── VQA ──────────────────────────────────────────────────
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text": QWEN_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        elapsed_vqa = time.time() - t0
        response = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"    VQA ({elapsed_vqa:.1f}s): {response[:80]}...")

        # ── Grounding ─────────────────────────────────────────────
        grounding_prompt = (
            f"Locate all transparent objects (glass plate or plastic sheet) in the image "
            f"and output their bounding boxes as a JSON array. "
            f"Format: [{{\"label\": \"name\", \"bbox\": [x1, y1, x2, y2]}}] "
            f"where coordinates are pixel values (image is {W}x{H}). "
            f"Output only the JSON array, nothing else."
        )
        messages_g = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text": grounding_prompt},
        ]}]
        text_g = processor.apply_chat_template(messages_g, tokenize=False, add_generation_prompt=True)
        inputs_g = processor(text=[text_g], images=[pil], return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            ids_g = model.generate(**inputs_g, max_new_tokens=256, do_sample=False)
        elapsed_g = time.time() - t0
        raw_g = processor.decode(ids_g[0][inputs_g["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        print(f"    Grounding ({elapsed_g:.1f}s): {raw_g[:120]}")

        # JSON から bbox を抽出
        import re, json as _json
        boxes = []
        m = re.search(r'\[.*\]', raw_g, re.DOTALL)
        if m:
            try:
                for item in _json.loads(m.group()):
                    b = item.get("bbox", [])
                    if len(b) == 4:
                        # Qwen3-VL は 0-1000 正規化座標 → 実ピクセルにスケール
                        x1 = int(b[0] / 1000 * W)
                        y1 = int(b[1] / 1000 * H)
                        x2 = int(b[2] / 1000 * W)
                        y2 = int(b[3] / 1000 * H)
                        boxes.append({"label": item.get("label", "transparent"),
                                      "bbox": [x1, y1, x2, y2]})
            except Exception:
                pass

        # ── 可視化: RGB + bbox + VQA テキスト ────────────────────
        rgb_bgr = cv2.imread(str(rgb_path))
        vis = rgb_bgr.copy()
        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 0)]
        for i, det in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = colors[i % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, det["label"], (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if not boxes:
            cv2.putText(vis, "No grounding", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(vis[:, :, ::-1]); axes[0].set_title(f"Grounding ({scene})"); axes[0].axis("off")
        axes[1].axis("off")
        axes[1].text(0.05, 0.95,
                     f"VQA Response:\n{response}\n\nGrounding raw:\n{raw_g[:300]}",
                     transform=axes[1].transAxes, va="top", ha="left",
                     fontsize=8, wrap=True,
                     bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        fig.suptitle(f"Qwen3-VL 4B — {scene}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT / f"qwen_{scene}.jpg", dpi=100); plt.close(fig)

        result = {"scene": scene, "model": "qwen3-vl-4b",
                  "infer_sec_vqa": round(elapsed_vqa, 3), "infer_sec_grounding": round(elapsed_g, 3),
                  "response": response, "grounding_raw": raw_g, "boxes": boxes}
        (OUT / f"qwen_{scene}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results_all[scene] = result

    del model; torch.cuda.empty_cache()
    return results_all


# ── エントリポイント ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["all"],
                    help=f"モデル名 or 'all'. 選択肢: {ALL_MODELS}")
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    args = ap.parse_args()

    models  = ALL_MODELS if "all" in args.model else args.model
    scenes  = [s for s in args.scenes if (DATA / s / "rgb.png").exists()]
    summary = {}

    if "yolo" in models:
        summary["yolo"] = run_yolo(scenes)
    if "dinov3" in models:
        summary["dinov3"] = run_dinov3(scenes)
    if "qwen" in models:
        summary["qwen"] = run_qwen(scenes)
    if "sam3" in models:
        print("\n[SAM3] 別コマンドで実行してください:")
        print("  DISPLAY=:0 .venv_sam3/bin/python scripts/exp2/generate_masks_from_paper.py --all")
        print("  (mask.png / mask_transparent.png は撮影時に既に生成済み)")

    out_summary = OUT / "rgb_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n完了 → {out_summary}")


if __name__ == "__main__":
    main()
