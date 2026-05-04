"""
FoundationPose シーンデータ準備スクリプト
- data/realsense/ の画像・マスクを FoundationPose の YcbineoatReader 形式に整理
- cam_K.txt を生成

使い方 (.venv で実行):
  source /home/vr01/AIVisionPJ/.venv/bin/activate
  python scripts/setup_fp_scene.py

出力ディレクトリ: data/fp_scene/
  rgb/000000.png       ← クエリ RGB
  depth/000000.png     ← クエリ深度 (uint16 mm)
  masks/000000.png     ← クエリマスク (SAM3出力)
  cam_K.txt            ← 内部パラメータ
  ref/                 ← 参照フレーム (mesh生成用、Dockerスクリプトが使用)
    rgb_000.png, depth_000.npy, mask_000.png, ...
"""
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    REALSENSE_DIR = Path("data/realsense")
    REF_DIR = Path("data/realsense/ref")
    SCENE_DIR = Path("data/fp_scene")

    # 必要ファイルの確認
    required = [
        REALSENSE_DIR / "rgb.png",
        REALSENSE_DIR / "depth.png",
        REALSENSE_DIR / "mask.png",
        REALSENSE_DIR / "intrinsics.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR: 以下のファイルが見つかりません:")
        for f in missing:
            print(f"  {f}")
        print()
        print("実行順序:")
        print("  1. python scripts/capture_realsense.py")
        print("  2. source .venv_sam3/bin/activate && python scripts/generate_sam3_mask.py \\")
        print("       --image data/realsense/rgb.png --prompt \"cup\"")
        return False

    # シーンディレクトリ作成
    for d in ["rgb", "depth", "masks", "ref"]:
        (SCENE_DIR / d).mkdir(parents=True, exist_ok=True)

    # クエリフレームをコピー
    shutil.copy(REALSENSE_DIR / "rgb.png", SCENE_DIR / "rgb" / "000000.png")
    shutil.copy(REALSENSE_DIR / "depth.png", SCENE_DIR / "depth" / "000000.png")
    shutil.copy(REALSENSE_DIR / "mask.png", SCENE_DIR / "masks" / "000000.png")
    print("クエリフレームをコピー:")
    print(f"  {SCENE_DIR}/rgb/000000.png")
    print(f"  {SCENE_DIR}/depth/000000.png")
    print(f"  {SCENE_DIR}/masks/000000.png")

    # cam_K.txt 生成
    with open(REALSENSE_DIR / "intrinsics.json") as f:
        K = json.load(f)
    cam_k = np.array([
        [K["fx"], 0, K["cx"]],
        [0, K["fy"], K["cy"]],
        [0, 0, 1.0]
    ])
    np.savetxt(str(SCENE_DIR / "cam_K.txt"), cam_k, fmt="%.4f")
    print(f"  {SCENE_DIR}/cam_K.txt  (fx={K['fx']:.1f}, fy={K['fy']:.1f})")

    # 参照フレームをコピー (mesh生成用)
    ref_count = 0
    if REF_DIR.exists():
        for rgb_file in sorted(REF_DIR.glob("rgb_*.png")):
            suffix = rgb_file.stem.replace("rgb", "")  # "_000"
            depth_file = REF_DIR / f"depth{suffix}.npy"
            mask_file = REF_DIR / f"mask{suffix}.png"
            if depth_file.exists() and mask_file.exists():
                shutil.copy(rgb_file, SCENE_DIR / "ref" / rgb_file.name)
                shutil.copy(depth_file, SCENE_DIR / "ref" / depth_file.name)
                shutil.copy(mask_file, SCENE_DIR / "ref" / mask_file.name)
                ref_count += 1
        shutil.copy(REALSENSE_DIR / "intrinsics.json", SCENE_DIR / "ref" / "intrinsics.json")
        print(f"  {SCENE_DIR}/ref/ に {ref_count}フレームの参照データをコピー")

    print(f"\n完了: {SCENE_DIR}/")
    print("\n次のステップ (Docker内でMesh生成 + FoundationPose実行):")
    print("  docker exec -it foundationpose_build bash -c \\")
    print("    'source /opt/conda/etc/profile.d/conda.sh && conda activate my && \\")
    print(f"     export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH && \\")
    print(f"     cd /home/vr01/AIVisionPJ && \\")
    print(f"     python scripts/fp_create_mesh_and_run.py --scene_dir {SCENE_DIR}'")
    return True


if __name__ == "__main__":
    main()
