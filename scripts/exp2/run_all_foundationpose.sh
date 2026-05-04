#!/bin/bash
# 第2段実験 FoundationPose × CAD 一括実行スクリプト
#
# 前提:
#   setup_fp_scenes.py を実行済みで data/exp2/fp/{model}_{scene}/ が存在すること
#   Docker コンテナ foundationpose_build が起動済みであること
#
# 使い方:
#   docker start foundationpose_build
#   bash scripts/exp2/run_all_foundationpose.sh
#   bash scripts/exp2/run_all_foundationpose.sh moge2   # モデル絞り込み

set -e

ROOT=/home/vr01/AIVisionPJ
OUT_DIR=${ROOT}/results/exp2/foundationpose

MODELS=${1:-"moge2 depthpro da2"}
SCENES="glass_front glass_oblique resin_front resin_oblique"

CAD_glass=${ROOT}/data/exp2/cad/glass_plate.obj
CAD_resin=${ROOT}/data/exp2/cad/resin_sheet_fp.obj

mkdir -p "${OUT_DIR}"

run_fp() {
  local model=$1
  local scene=$2
  local fp_scene=${ROOT}/data/exp2/fp/${model}_${scene}
  local debug=${OUT_DIR}/debug_${model}_${scene}

  # CAD 選択
  if [[ "${scene}" == glass_* ]]; then
    local cad=${CAD_glass}
  else
    local cad=${CAD_resin}
  fi

  if [ ! -d "${fp_scene}" ]; then
    echo "  SKIP: ${fp_scene} が見つかりません (setup_fp_scenes.py を先に実行)"
    return
  fi

  echo "=== FP: model=${model} scene=${scene} cad=$(basename ${cad}) ==="
  docker exec foundationpose_build bash -c "rm -rf ${debug}; mkdir -p ${debug}"

  docker exec foundationpose_build bash -c "
    cd /home/vr01/AIVisionPJ/models/foundationpose/FoundationPose &&
    source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
    export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:\$LD_LIBRARY_PATH &&
    FP_SKIP_IMSHOW=1 python run_demo.py \
      --mesh_file ${cad} \
      --test_scene_dir ${fp_scene} \
      --debug 3 --debug_dir ${debug}
  "

  # 可視化画像をコピー
  local vis_src=${debug}/track_vis/000000.png
  local vis_dst=${OUT_DIR}/fp_${model}_${scene}.jpg
  if [ -f "${vis_src}" ]; then
    cp "${vis_src}" "${vis_dst}"
    echo "  → ${vis_dst}"
  else
    echo "  WARN: 可視化画像が見つかりません (${vis_src})"
  fi

  # pose CSV をコピー
  local pose_src=${debug}/ob_in_cam/000000.txt
  local pose_dst=${OUT_DIR}/pose_${model}_${scene}.txt
  if [ -f "${pose_src}" ]; then
    cp "${pose_src}" "${pose_dst}"
  fi
}

for model in ${MODELS}; do
  for scene in ${SCENES}; do
    run_fp "${model}" "${scene}"
  done
done

echo ""
echo "=== 完了 ==="
echo "結果: ${OUT_DIR}"
ls "${OUT_DIR}"/*.jpg 2>/dev/null | wc -l | xargs -I{} echo "{} 枚の可視化画像を生成"
