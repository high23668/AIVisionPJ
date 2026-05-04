#!/bin/bash
# 第2段実験 Any6D (SF3D + FoundationPose model-free) 実行スクリプト
#
# パイプライン:
#   Step 1: SF3D でアンカー画像からメッシュ生成 (.venv_3d)
#           アンカー: rgb_paper.png + mask.png (青画用紙 GT マスク)
#   Step 2: Any6D Docker でポーズ推定
#           クエリ: rgb.png + MoGe-2 GT depth + mask_transparent.png
#
# 使い方:
#   bash scripts/exp2/run_any6d.sh glass_front
#   bash scripts/exp2/run_any6d.sh  (全シーン)

set -e
ROOT=/home/vr01/AIVisionPJ
OUT=${ROOT}/results/exp2/any6d
mkdir -p "${OUT}"

SCENES=${@:-"glass_front glass_oblique resin_front resin_oblique"}
DEPTH_MODEL="moge2"  # GT depth に使用するモデル

for scene in ${SCENES}; do
  echo ""
  echo "=== Any6D: ${scene} ==="
  DATA=${ROOT}/data/exp2/${scene}
  DEPTH_GT=${ROOT}/results/exp2/depth/${DEPTH_MODEL}_${scene}_gt.npy
  MESH_OUT=${OUT}/${scene}_mesh.obj
  SCENE_OUT=${OUT}/${scene}
  mkdir -p "${SCENE_OUT}"

  # ── Step 1: SF3D でアンカー画像からメッシュ生成 ──────────────────
  echo "[Step 1] SF3D mesh 生成中..."
  ANCHOR_IMG=${DATA}/rgb_paper.png
  ANCHOR_MASK=${DATA}/mask.png
  if [ ! -f "${ANCHOR_IMG}" ]; then
    echo "  SKIP: rgb_paper.png なし"
    continue
  fi

  ${ROOT}/.venv_3d/bin/python ${ROOT}/scripts/any6d_generate_mesh.py \
    --image "${ANCHOR_IMG}" \
    --mask  "${ANCHOR_MASK}" \
    --output "${MESH_OUT}" \
    2>&1 | grep -E "OK|error|Error|完了|mesh|saved|output" | head -10
  echo "  mesh → ${MESH_OUT}"

  # ── Step 2: depth.npy (GT) → uint16 PNG ─────────────────────────
  echo "[Step 2] GT depth → uint16 PNG"
  DEPTH_PNG=${SCENE_OUT}/depth_gt.png
  ${ROOT}/.venv/bin/python -c "
import numpy as np, cv2
d = np.load('${DEPTH_GT}').astype(float)
import numpy as np
d = np.where(np.isfinite(d) & (d > 0), d, 0.0)
d_mm = (d * 1000).clip(0, 65535).astype('uint16')
cv2.imwrite('${DEPTH_PNG}', d_mm)
print(f'  depth range: {d_mm[d_mm>0].min() if (d_mm>0).any() else 0}~{d_mm.max()} mm')
"

  # マスク選択: glass は mask_transparent.png、resin は mask.png
  if [[ "${scene}" == glass_* ]]; then
    MASK=${DATA}/mask_transparent.png
    [ -f "${MASK}" ] || MASK=${DATA}/mask.png
  else
    MASK=${DATA}/mask.png
  fi
  echo "  mask → $(basename ${MASK})"

  # intrinsics: nested → flat 形式に変換
  INTR_FLAT=${SCENE_OUT}/intrinsics_flat.json
  ${ROOT}/.venv/bin/python -c "
import json
from pathlib import Path
intr = json.loads(Path('${DATA}/intrinsics.json').read_text())
K = intr['rgb']
flat = {'fx': K['fx'], 'fy': K['fy'], 'cx': K['cx'], 'cy': K['cy'],
        'width': K['width'], 'height': K['height'], 'depth_scale': 0.001}
Path('${INTR_FLAT}').write_text(json.dumps(flat, indent=2))
"

  # ── Step 3: Any6D Docker でポーズ推定 ───────────────────────────
  echo "[Step 3] Any6D 推論中..."
  docker run --rm --gpus all \
    -v ${ROOT}:${ROOT} \
    -v ${ROOT}/models/foundationpose/FoundationPose/weights:/foundationpose/weights \
    -v ${ROOT}/models/any6d/run_realsense.py:/run_realsense.py \
    any6d:latest \
    python /run_realsense.py \
      --rgb         "${DATA}/rgb.png" \
      --depth       "${DEPTH_PNG}" \
      --mask        "${MASK}" \
      --mesh        "${MESH_OUT}" \
      --intrinsics  "${INTR_FLAT}" \
      --output      "${SCENE_OUT}" \
    2>&1 | grep -E "推定ポーズ|pred_pose|score|error|Error|Done|完了|translation|Z=|結果保存" || true

  # 可視化コピー
  VIS=$(ls ${SCENE_OUT}/pose_overlay.png 2>/dev/null)
  if [ -n "${VIS}" ]; then
    cp "${VIS}" "${OUT}/any6d_${scene}.jpg"
    echo "  → ${OUT}/any6d_${scene}.jpg"
  fi
done

echo ""
echo "=== 完了 → ${OUT} ==="