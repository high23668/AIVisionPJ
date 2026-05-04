#!/bin/bash
# Any6D パイプライン (SF3D → FoundationPose)
#
# Step 1 (ホスト): SF3Dでアンカー画像からメッシュ生成
# Step 2 (Docker): Any6Dでクエリ画像のポーズ推定
#
# 使い方:
#   bash scripts/any6d_pipeline.sh --anchor data/cup.png --scene demo_data
#
# 前提:
#   - .venv_3d が有効 (SF3D用)
#   - Dockerイメージ any6d:latest がビルド済み
#   - FoundationPose weights が models/foundationpose/FoundationPose/weights/ にある

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FP_WEIGHTS="$PROJECT_DIR/models/foundationpose/FoundationPose/weights"
ANY6D_DIR="$PROJECT_DIR/models/any6d"
VENV_PYTHON="$PROJECT_DIR/.venv_3d/bin/python"

# --- 引数パース ---
ANCHOR_IMAGE=""
SCENE_DIR="demo_data"
OUTPUT_DIR="$PROJECT_DIR/results/any6d"

while [[ $# -gt 0 ]]; do
    case $1 in
        --anchor) ANCHOR_IMAGE="$2"; shift 2 ;;
        --scene)  SCENE_DIR="$2";   shift 2 ;;
        --output) OUTPUT_DIR="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$ANCHOR_IMAGE" ]; then
    echo "Usage: $0 --anchor <image_path> [--scene <demo_data_dir>] [--output <output_dir>]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
MESH_PATH="$OUTPUT_DIR/generated_mesh.obj"

echo "============================================"
echo "Step 1: SF3D でメッシュ生成 (ホスト)"
echo "  入力: $ANCHOR_IMAGE"
echo "  出力: $MESH_PATH"
echo "============================================"
"$VENV_PYTHON" "$PROJECT_DIR/scripts/any6d_generate_mesh.py" \
    --image "$ANCHOR_IMAGE" \
    --output "$MESH_PATH"

echo ""
echo "============================================"
echo "Step 2: Any6D ポーズ推定 (Docker)"
echo "  メッシュ: $MESH_PATH"
echo "  シーン:   $ANY6D_DIR/$SCENE_DIR"
echo "============================================"
docker run --rm --gpus all \
    -v "$FP_WEIGHTS:/foundationpose/weights" \
    -v "$OUTPUT_DIR:/results_out" \
    -v "$MESH_PATH:/input_mesh.obj" \
    any6d:latest bash -c "
        source /opt/conda/etc/profile.d/conda.sh && conda activate Any6D
        python run_demo.py --mesh /input_mesh.obj
        cp -r /results/* /results_out/ 2>/dev/null || true
    "

echo ""
echo "完了: 結果は $OUTPUT_DIR に保存されました"
