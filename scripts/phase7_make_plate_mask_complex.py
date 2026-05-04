"""Phase 7.0: Complex シーン用の板マスクを Qwen3-VL bbox + SAM3 box-prompt で生成."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/vr01/AIVisionPJ")
QWEN_PY = ROOT / ".venv/bin/python"
SAM3_PY = ROOT / ".venv_sam3/bin/python"

QWEN_BBOX_SCRIPT = ROOT / "scripts" / "_phase7_qwen_bbox.py"
SAM3_FROM_BBOX_SCRIPT = ROOT / "scripts" / "_phase7_sam3_from_bbox.py"


QWEN_BBOX_CODE = '''
"""Qwen3-VL grounding to get a single bbox for glass plate."""
import json, re, sys
from pathlib import Path
from PIL import Image
import torch
sys.path.insert(0, str(Path("/home/vr01/AIVisionPJ/models")))
from qwen3vl.qwen3vl_demo import load_model

img_path = sys.argv[1]
out_json = sys.argv[2]

model, processor = load_model(variant="4b", load_in_4bit=True)
image = Image.open(img_path).convert("RGB")
W, H = image.size
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": (
            f"Locate the transparent rectangular glass plate (a flat sheet of glass standing upright) "
            f"and output its bounding box. Note: it is NOT the bottle, the spray container, or the pen jar. "
            f"It is the flat transparent panel behind/around them. "
            f"Format as JSON: {{\\"bbox\\": [x1, y1, x2, y2]}} pixel coords (image is {W}x{H}). "
            f"Output only JSON, no extra text."
        )},
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
gen = out_ids[:, inputs.input_ids.shape[1]:]
raw = processor.batch_decode(gen, skip_special_tokens=False)[0].strip()
print("RAW:", raw[:400])

m = re.search(r"\\{[^}]*\\"bbox\\"[^}]*\\}", raw)
if m:
    obj = json.loads(m.group())
    bbox = obj["bbox"]
else:
    arr = re.findall(r"\\[(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)\\]", raw)
    if not arr:
        raise SystemExit(f"No bbox parsed from: {raw}")
    bbox = [int(v) for v in arr[0]]
Path(out_json).write_text(json.dumps({"bbox": bbox, "W": W, "H": H, "raw": raw}))
print("BBOX:", bbox)
'''

SAM3_FROM_BBOX_CODE = '''
"""Run SAM3 with a box prompt to segment the plate."""
import sys, json
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

img_path = sys.argv[1]
bbox_json = sys.argv[2]
out_mask = sys.argv[3]
overlay = sys.argv[4]

bbox_meta = json.loads(Path(bbox_json).read_text())
x1, y1, x2, y2 = bbox_meta["bbox"]
W, H = bbox_meta["W"], bbox_meta["H"]

model = build_sam3_image_model(load_from_HF=True, device="cuda")
processor = Sam3Processor(model)
pil = Image.open(img_path).convert("RGB")
state = processor.set_image(pil)

# SAM3 box format: cxcywh normalized
cx = ((x1 + x2) / 2) / W
cy = ((y1 + y2) / 2) / H
w = (x2 - x1) / W
h = (y2 - y1) / H
out = processor.add_geometric_prompt(box=[cx, cy, w, h], label=True, state=state)
masks = out["masks"]
scores = out["scores"]
print("SAM3 masks:", len(masks), "scores:", scores)
if len(masks) == 0:
    raise SystemExit("SAM3 returned no mask for box prompt")
best = int(torch.argmax(scores) if isinstance(scores, torch.Tensor) else np.argmax(scores))
mask = masks[best]
mask_np = mask.cpu().numpy().squeeze().astype(bool) if isinstance(mask, torch.Tensor) else np.array(mask).squeeze().astype(bool)
cv2.imwrite(out_mask, (mask_np.astype(np.uint8) * 255))
img_bgr = cv2.imread(img_path)
ovl = img_bgr.copy()
ovl[mask_np] = (0, 255, 0)
vis = cv2.addWeighted(img_bgr, 0.5, ovl, 0.5, 0)
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite(overlay, vis)
print("score:", float(scores[best]) if hasattr(scores, "__len__") else float(scores))
print("coverage:", f"{mask_np.mean()*100:.1f}%")
print("saved:", out_mask)
'''


def write_helper_scripts():
    QWEN_BBOX_SCRIPT.write_text(QWEN_BBOX_CODE)
    SAM3_FROM_BBOX_SCRIPT.write_text(SAM3_FROM_BBOX_CODE)


def run_qwen(img_path: Path, out_json: Path):
    print(f"[Phase 7.0] Qwen3-VL grounding on {img_path}...")
    cmd = [str(QWEN_PY), str(QWEN_BBOX_SCRIPT), str(img_path), str(out_json)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    r.check_returncode()


def run_sam3(img_path: Path, bbox_json: Path, out_mask: Path, overlay: Path):
    print(f"[Phase 7.0] SAM3 box-prompt on {img_path}...")
    env = {"DISPLAY": ":0", "PATH": "/usr/bin:/bin"}
    import os
    full_env = os.environ.copy(); full_env.update(env)
    cmd = [str(SAM3_PY), str(SAM3_FROM_BBOX_SCRIPT), str(img_path), str(bbox_json), str(out_mask), str(overlay)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    r.check_returncode()


def main():
    write_helper_scripts()
    scene_dir = ROOT / "data" / "glassboard_complex"
    img = scene_dir / "rgb.png"
    bbox_json = scene_dir / "qwen_bbox_plate.json"
    out_mask = scene_dir / "mask_plate.png"
    overlay = scene_dir / "mask_plate_overlay.jpg"

    run_qwen(img, bbox_json)
    run_sam3(img, bbox_json, out_mask, overlay)
    print(f"\n[Phase 7.0] Done.\n  mask: {out_mask}\n  overlay: {overlay}")


if __name__ == "__main__":
    main()
