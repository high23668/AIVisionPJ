
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
            f"Format as JSON: {{\"bbox\": [x1, y1, x2, y2]}} pixel coords (image is {W}x{H}). "
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

m = re.search(r"\{[^}]*\"bbox\"[^}]*\}", raw)
if m:
    obj = json.loads(m.group())
    bbox = obj["bbox"]
else:
    arr = re.findall(r"\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", raw)
    if not arr:
        raise SystemExit(f"No bbox parsed from: {raw}")
    bbox = [int(v) for v in arr[0]]
Path(out_json).write_text(json.dumps({"bbox": bbox, "W": W, "H": H, "raw": raw}))
print("BBOX:", bbox)
