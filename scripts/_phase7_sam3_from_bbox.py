
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
