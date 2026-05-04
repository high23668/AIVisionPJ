"""SAM3 with text prompt → pick the LARGEST mask (assume it's the plate)."""
import sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

img_path = sys.argv[1]
prompt = sys.argv[2]
out_mask = sys.argv[3]
overlay = sys.argv[4]

model = build_sam3_image_model(load_from_HF=True, device="cuda")
processor = Sam3Processor(model)
pil = Image.open(img_path).convert("RGB")
state = processor.set_image(pil)
out = processor.set_text_prompt(state=state, prompt=prompt)
masks = out["masks"]; scores = out["scores"]
print(f"SAM3 returned {len(masks)} masks for '{prompt}'")
if len(masks) == 0:
    raise SystemExit("no masks")

areas = []
mask_arrays = []
for i, m in enumerate(masks):
    arr = m.cpu().numpy().squeeze().astype(bool) if isinstance(m, torch.Tensor) else np.array(m).squeeze().astype(bool)
    mask_arrays.append(arr)
    areas.append(arr.sum())
    s = float(scores[i]) if hasattr(scores, "__len__") else float(scores)
    print(f"  mask {i}: area={arr.sum()} score={s:.3f}")

best = int(np.argmax(areas))
mask_np = mask_arrays[best]
print(f"  picked mask {best} (largest area = {areas[best]} px, {mask_np.mean()*100:.1f}% coverage)")
cv2.imwrite(out_mask, (mask_np.astype(np.uint8) * 255))
img_bgr = cv2.imread(img_path)
ovl = img_bgr.copy(); ovl[mask_np] = (0, 255, 0)
vis = cv2.addWeighted(img_bgr, 0.5, ovl, 0.5, 0)
cv2.imwrite(overlay, vis)
print("saved:", out_mask)
