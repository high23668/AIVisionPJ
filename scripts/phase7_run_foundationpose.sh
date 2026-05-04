#!/bin/bash
# Phase 7.5: FoundationPose × parametric CAD on Phase 7 depth scenes
set -e

ROOT=/home/vr01/AIVisionPJ
CAD=/home/vr01/AIVisionPJ/data/glassboard_cad/glass_plate.obj

run_fp() {
  local scene=$1   # simple | complex
  local tag=$2     # metric3d | moge2 | depthpro | unidepth
  local scene_dir=${ROOT}/data/glassboard_${scene}_fp_${tag}
  local debug=${ROOT}/results/glassboard/fp_${scene}_${tag}

  echo "=== FP: scene=${scene} depth=${tag} ==="
  docker exec foundationpose_build bash -c "rm -rf ${debug}; mkdir -p ${debug}"

  docker exec foundationpose_build bash -c "
    cd /home/vr01/AIVisionPJ/models/foundationpose/FoundationPose &&
    source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
    export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:\$LD_LIBRARY_PATH &&
    FP_SKIP_IMSHOW=1 python run_demo.py \
      --mesh_file ${CAD} \
      --test_scene_dir ${scene_dir} \
      --debug 1 --debug_dir ${debug}
  "

  local vis_src=${debug}/track_vis/000000.png
  local vis_dst=${ROOT}/results/glassboard/foundationpose_${tag}_${scene}.jpg
  if [ -f "${vis_src}" ]; then
    cp "${vis_src}" "${vis_dst}"
    echo "  saved viz -> ${vis_dst}"
  else
    echo "  WARN: ${vis_src} not found"
  fi
}

for tag in metric3d moge2 depthpro unidepth; do
  for scene in simple complex; do
    run_fp "${scene}" "${tag}"
  done
done

echo "=== Phase 7.5 done ==="
