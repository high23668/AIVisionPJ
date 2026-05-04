#!/bin/bash
# 明日の再開スクリプト
# 実行: bash /home/vr01/AIVisionPJ/log/resume_tomorrow.sh

set -e

PROJ_DIR="/home/vr01/AIVisionPJ"
VENV="$PROJ_DIR/.venv"

echo "======================================"
echo "AIVisionPJ 再開チェック"
echo "======================================"

# GPU確認
echo ""
echo "[1/5] GPU確認"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Python仮想環境確認
echo ""
echo "[2/5] Python仮想環境"
if [ -d "$VENV" ]; then
    echo "  OK - $VENV"
    source "$VENV/bin/activate"
    python -c "import torch; print(f'  PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
    python -c "import ultralytics; print(f'  Ultralytics {ultralytics.__version__}')"
    python -c "import transformers; print(f'  Transformers {transformers.__version__}')"
else
    echo "  ERROR: .venv not found. Run: cd $PROJ_DIR && uv venv .venv && source .venv/bin/activate"
    exit 1
fi

# Dockerコンテナ確認
echo ""
echo "[3/5] FoundationPose Docker"
if docker ps | grep -q foundationpose_build; then
    echo "  OK - foundationpose_build コンテナ稼働中"
else
    echo "  INFO: コンテナ停止中。再起動します..."
    REPO_DIR="$PROJ_DIR/models/foundationpose/FoundationPose"
    docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host \
        --name foundationpose_build \
        -v ${REPO_DIR}:${REPO_DIR} \
        -v ${PROJ_DIR}:${PROJ_DIR} \
        -v /home:/home \
        -d foundationpose:latest bash -c "sleep 7200" 2>/dev/null || \
    docker start foundationpose_build
    echo "  OK - foundationpose_build 起動完了"
fi

# RealSense確認
echo ""
echo "[4/5] RealSense カメラ"
if command -v rs-enumerate-devices &> /dev/null; then
    rs-enumerate-devices 2>/dev/null | head -5 || echo "  INFO: カメラが接続されていないか、認識されていません"
else
    echo "  INFO: librealsense未インストール"
    echo "  インストール: sudo apt install ros-humble-librealsense2 または pip install pyrealsense2"
fi

# データフォルダ確認
echo ""
echo "[5/5] データフォルダ"
mkdir -p "$PROJ_DIR/data/sample_images"
mkdir -p "$PROJ_DIR/data/sample_depth"
mkdir -p "$PROJ_DIR/data/cad_models"
echo "  OK - data/ フォルダ確認済み"

echo ""
echo "======================================"
echo "準備完了。以下のテストを実行できます:"
echo ""
echo "# YOLO26 (Webカメラ: --source 0, RealSenseは --source 4 など番号確認)"
echo "python $PROJ_DIR/models/yolo26/yolo26_demo.py --task detect --source 0"
echo "#   -> Ultralyticsがリアルタイムウィンドウを自動表示 (Qキーで終了)"
echo ""
echo "# DINOv3 特徴可視化"
echo "python $PROJ_DIR/models/dinov3/dinov3_demo.py --image <画像パス> --task visualize"
echo ""
echo "# FoundationPose (Dockerコンテナ内)"
echo "docker exec -it foundationpose_build bash"
echo "  -> source /opt/conda/etc/profile.d/conda.sh && conda activate my"
echo "  -> export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo "  -> python $PROJ_DIR/models/foundationpose/fp_demo.py --mode verify"
echo ""
echo "# Qwen3-VL VQA"
echo "python $PROJ_DIR/models/qwen3vl/qwen3vl_demo.py --image <画像パス> --task describe"
echo "======================================"
