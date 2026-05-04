#!/bin/bash
set -e
ROOT=/home/vr01/AIVisionPJ
CAD=${ROOT}/data/glassboard_cad/glass_plate.obj

run_fp() {
  local scene=$1; local tag=$2
  local scene_dir=${ROOT}/data/glassboard_${scene}_fp_${tag}_affine
  local debug=${ROOT}/results/glassboard/fp_${scene}_${tag}_affine
  echo "=== FP affine: scene=${scene} depth=${tag} ==="
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
  local vis=${debug}/track_vis/000000.png
  local dst=${ROOT}/results/glassboard/foundationpose_${tag}_affine_${scene}.jpg
  [ -f "${vis}" ] && cp "${vis}" "${dst}" && echo "  saved -> ${dst}"
}

for tag in depthpro moge2 unidepth; do
  for scene in simple complex; do
    run_fp "${scene}" "${tag}"
  done
done
echo "=== affine FP done ==="
