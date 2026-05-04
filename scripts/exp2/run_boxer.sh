#!/bin/bash
# 第2段実験 Boxer 一括実行スクリプト
#
# 使い方:
#   bash scripts/exp2/run_boxer.sh
#   bash scripts/exp2/run_boxer.sh glass_front glass_oblique

set -e
PROJ=/home/vr01/AIVisionPJ
VENV=${PROJ}/models/boxer/.venv/bin/python
SCRIPT=${PROJ}/models/boxer/run_boxer.py
OUT=${PROJ}/results/exp2/rgb_models
LABELS="glass plate,transparent glass,glass,plastic sheet,transparent object"

SCENES=${@:-"glass_front glass_oblique resin_front resin_oblique"}

# Step 1: 一時ディレクトリにflat形式のintrinsics.jsonを生成
echo "=== intrinsics.json 変換中 ==="
${VENV} -c "
import json, shutil
from pathlib import Path
PROJ = Path('${PROJ}')
for scene in '${SCENES}'.split():
    src = PROJ / 'data/exp2' / scene
    dst = PROJ / 'data/exp2/boxer_tmp' / scene
    dst.mkdir(parents=True, exist_ok=True)
    for f in ['rgb.png', 'depth.npy']:
        if (src/f).exists() and not (dst/f).exists():
            shutil.copy2(src/f, dst/f)
    intr = json.loads((src/'intrinsics.json').read_text())
    K = intr['rgb']
    flat = {'fx': K['fx'], 'fy': K['fy'], 'cx': K['cx'], 'cy': K['cy'],
            'width': K['width'], 'height': K['height'],
            'depth_scale': intr.get('depth_scale', 0.001)}
    (dst/'intrinsics.json').write_text(json.dumps(flat, indent=2))
    print(f'  OK: {scene}')
"

# Step 2: 各シーンでBoxerを実行
for scene in ${SCENES}; do
  TMP_DIR=${PROJ}/data/exp2/boxer_tmp/${scene}
  SCENE_OUT=${OUT}/boxer_${scene}
  echo ""
  echo "=== Boxer: ${scene} ==="

  ${VENV} ${SCRIPT} \
    --input "${TMP_DIR}" \
    --labels "${LABELS}" \
    --thresh2d 0.1 \
    --thresh3d 0.2 \
    --max_n 1 \
    --output_dir "${OUT}/boxer_debug" \
    --no_csv 2>&1 | grep -E "detect|3D|2D|推論|完了|Error|Traceback|score|glass|label|Labels|Using"

  # 可視化画像をコピー
  VIS=$(find ${OUT}/boxer_debug/${scene} -name "*.jpg" 2>/dev/null | head -1)
  if [ -n "${VIS}" ]; then
    cp "${VIS}" "${OUT}/boxer_${scene}.jpg"
    echo "  → ${OUT}/boxer_${scene}.jpg"
  else
    echo "  WARN: 可視化画像なし"
  fi
done

echo ""
echo "=== 完了 ==="