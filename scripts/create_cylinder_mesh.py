"""
カップ用円柱メッシュ生成スクリプト
- 参照フレームの点群からカップの高さ・半径を推定
- trimesh で円柱メッシュを作成し .obj で保存
- Docker内で実行

使い方 (Docker内):
  docker exec -it foundationpose_build bash -c '
    source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
    cd /home/vr01/AIVisionPJ &&
    python scripts/create_cylinder_mesh.py
  '
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


def estimate_cup_dimensions(ref_dir, K_dict):
    fx, fy, cx, cy = K_dict["fx"], K_dict["fy"], K_dict["cx"], K_dict["cy"]

    ref_dir = Path(ref_dir)
    mask_files = sorted(ref_dir.glob("mask_*.png"))

    all_pts = []
    for mask_file in mask_files:
        suffix = mask_file.stem.replace("mask", "")
        depth_file = ref_dir / f"depth{suffix}.npy"
        if not depth_file.exists():
            continue

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) > 127
        depth = np.load(str(depth_file)).astype(np.float32)

        valid = mask & (depth > 0.05) & (depth < 2.0)
        if valid.sum() < 100:
            continue

        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        Z = depth[valid]
        X = (u[valid] - cx) * Z / fx
        Y = (v[valid] - cy) * Z / fy
        all_pts.append(np.stack([X, Y, Z], axis=1))

    if not all_pts:
        return None, None

    pts = np.concatenate(all_pts, axis=0)

    # 外れ値除去 (1〜99パーセンタイル)
    for i in range(3):
        lo, hi = np.percentile(pts[:, i], [1, 99])
        pts = pts[(pts[:, i] >= lo) & (pts[:, i] <= hi)]

    height = float(pts[:, 1].max() - pts[:, 1].min())  # Y軸 = 高さ
    width_x = float(pts[:, 0].max() - pts[:, 0].min())
    width_z = float(pts[:, 2].max() - pts[:, 2].min())
    radius = float(max(width_x, width_z)) / 2.0

    print(f"  推定高さ: {height*100:.1f}cm")
    print(f"  推定半径: {radius*100:.1f}cm")
    print(f"  (X幅={width_x*100:.1f}cm, Z幅={width_z*100:.1f}cm)")

    return height, radius


def main():
    SCENE_DIR = Path("data/fp_scene")
    REF_DIR = SCENE_DIR / "ref"
    MESH_PATH = SCENE_DIR / "mesh" / "cup_cylinder.obj"
    MESH_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REF_DIR / "intrinsics.json") as f:
        K_dict = json.load(f)

    print("=== カップ寸法推定 ===")
    height, radius = estimate_cup_dimensions(REF_DIR, K_dict)

    if height is None:
        print("ERROR: 点群データが不足しています")
        sys.exit(1)

    # 推定値の妥当性チェック・クランプ
    height = np.clip(height, 0.05, 0.30)  # 5cm〜30cm
    radius = np.clip(radius, 0.02, 0.10)  # 2cm〜10cm

    print(f"\n  → 使用値: 高さ={height*100:.1f}cm, 半径={radius*100:.1f}cm")

    # trimesh で円柱生成 (Y軸が高さ方向, 重心が原点)
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=64)

    # FoundationPose: Z軸が奥行き方向なので Y→Z に変換 (90度回転)
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    cylinder.apply_transform(rot)

    cylinder.export(str(MESH_PATH))
    print(f"\n保存: {MESH_PATH}")
    print(f"  頂点数: {len(cylinder.vertices)}, 面数: {len(cylinder.faces)}")

    print("\n=== FoundationPose 再実行 ===")
    print("docker exec -it foundationpose_build bash -c '\\")
    print("  source /opt/conda/etc/profile.d/conda.sh && conda activate my &&\\")
    print("  export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH &&\\")
    print("  cd /home/vr01/AIVisionPJ &&\\")
    print(f"  python scripts/fp_create_mesh_and_run.py --scene_dir data/fp_scene --mesh_path {MESH_PATH}'")


if __name__ == "__main__":
    main()
