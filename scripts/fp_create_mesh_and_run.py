"""
FoundationPose 実行スクリプト (Docker内で実行)
1. 参照フレームの深度+マスクからOpen3Dで物体メッシュを生成
2. FoundationPose で6DoF姿勢推定

使い方 (Docker内):
  docker exec -it foundationpose_build bash -c '
    source /opt/conda/etc/profile.d/conda.sh && conda activate my &&
    export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH &&
    cd /home/vr01/AIVisionPJ &&
    python scripts/fp_create_mesh_and_run.py --scene_dir data/fp_scene
  '

メッシュ生成:
  - 参照フレームのうちマスク面積が最も大きいフレームを選択
  - 深度 + マスクから物体領域のPointCloudを生成
  - Poisson Surface Reconstructionでメッシュ化
  - 重心をoriginに移動して .obj 保存

FoundationPose実行:
  - YcbineoatReader でシーンを読み込み (rgb/, depth/, masks/, cam_K.txt)
  - 1フレームのみ pose.register() → 4x4行列を出力
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def create_mesh_from_ref(scene_dir, mesh_path):
    """Create object mesh from reference frames using Open3D."""
    import open3d as o3d

    ref_dir = Path(scene_dir) / "ref"
    with open(ref_dir / "intrinsics.json") as f:
        K_dict = json.load(f)

    fx = K_dict["fx"]
    fy = K_dict["fy"]
    cx = K_dict["cx"]
    cy = K_dict["cy"]

    # マスク面積が最大の参照フレームを選択
    best_idx = None
    best_area = 0
    mask_files = sorted(ref_dir.glob("mask_*.png"))
    if not mask_files:
        print("ERROR: 参照マスクが見つかりません: ref/mask_*.png")
        return False

    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        area = (mask > 127).sum()
        if area > best_area:
            best_area = area
            best_idx = mask_file.stem.replace("mask", "")  # "_007" など

    print(f"  最良参照フレーム: {best_idx} (マスク面積={best_area}px)")

    # 深度・マスク読み込み
    depth_npy = ref_dir / f"depth{best_idx}.npy"
    mask_png = ref_dir / f"mask{best_idx}.png"
    rgb_png = ref_dir / f"rgb{best_idx}.png"

    depth = np.load(str(depth_npy)).astype(np.float32)  # meters
    mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE) > 127
    rgb = cv2.imread(str(rgb_png))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # マスク領域の点群生成
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = mask & (depth > 0.05) & (depth < 2.0)

    Z = depth[valid]
    X = (u[valid] - cx) * Z / fx
    Y = (v[valid] - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=1)
    colors = rgb[valid]

    # 外れ値除去
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  PointCloud: {len(pcd.points)} points (after outlier removal)")

    # 重心を原点に移動
    centroid = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-centroid)

    # 法線推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    # Poisson Surface Reconstruction
    print("  Poisson Surface Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, scale=1.1, linear_fit=False)

    # 低密度頂点を削除 (ノイズ除去)
    densities = np.asarray(densities)
    threshold = np.percentile(densities, 10)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    print(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # .obj として保存
    mesh_path = Path(mesh_path)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"  Saved: {mesh_path}")
    return True


def run_foundationpose(scene_dir, mesh_path, debug_dir, est_refine_iter=5):
    """Run FoundationPose on the scene."""
    FP_REPO = Path(__file__).parent.parent / "models/foundationpose/FoundationPose"
    sys.path.insert(0, str(FP_REPO))

    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from datareader import YcbineoatReader
    import nvdiffrast.torch as dr
    import trimesh
    import torch

    print("\nFoundationPose 初期化中...")
    mesh = trimesh.load(str(mesh_path))
    print(f"  Mesh loaded: {len(mesh.vertices)} vertices")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(f"{debug_dir}/ob_in_cam", exist_ok=True)

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=1,
        glctx=glctx,
    )
    print("  OK")

    reader = YcbineoatReader(video_dir=str(scene_dir), shorter_side=None, zfar=np.inf)
    print(f"  フレーム数: {len(reader)}")

    print("\n姿勢推定中...")
    t0 = time.perf_counter()

    color = reader.get_color(0)
    depth = reader.get_depth(0)
    mask = reader.get_mask(0).astype(bool)

    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                        iteration=est_refine_iter)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  推論時間: {elapsed_ms:.1f}ms")

    np.savetxt(f"{debug_dir}/ob_in_cam/000000.txt", pose.reshape(4, 4))

    print("\n========== 結果 ==========")
    print(f"Translation (XYZ): {pose[:3, 3]}")
    print(f"距離: {np.linalg.norm(pose[:3, 3]):.3f}m")
    print(f"Rotation:\n{pose[:3, :3]}")
    print(f"\n4x4 Pose matrix:\n{pose}")
    print(f"\n保存先: {debug_dir}/ob_in_cam/000000.txt")

    # 可視化 (debug>=1 の場合 run_demo.py のように描画)
    try:
        from Utils import draw_posed_3d_box, draw_xyz_axis
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.05, K=reader.K,
                            thickness=3, transparency=0, is_input_rgb=True)
        out_vis = f"{debug_dir}/result_vis.png"
        cv2.imwrite(out_vis, vis[..., ::-1])
        print(f"可視化: {out_vis}")
    except Exception as e:
        print(f"可視化スキップ: {e}")

    return pose


def main():
    parser = argparse.ArgumentParser(description="FoundationPose mesh生成 + 実行 (Docker内)")
    parser.add_argument("--scene_dir", default="data/fp_scene",
                        help="シーンデータディレクトリ")
    parser.add_argument("--mesh_path", default=None,
                        help="既存meshファイル (省略時は参照フレームから生成)")
    parser.add_argument("--debug_dir", default="results/fp_debug",
                        help="デバッグ出力ディレクトリ")
    parser.add_argument("--iter", type=int, default=5,
                        help="est_refine_iter (デフォルト=5)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    # Meshパス決定
    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if not mesh_path.exists():
            print(f"ERROR: mesh not found: {mesh_path}")
            sys.exit(1)
    else:
        mesh_path = scene_dir / "mesh" / "object.obj"

    # Mesh生成 (まだ存在しない場合)
    if not mesh_path.exists():
        print("=== Step 1: 参照フレームからMesh生成 ===")
        ok = create_mesh_from_ref(scene_dir, mesh_path)
        if not ok:
            sys.exit(1)
    else:
        print(f"既存Meshを使用: {mesh_path}")

    # FoundationPose 実行
    print("\n=== Step 2: FoundationPose 実行 ===")
    pose = run_foundationpose(scene_dir, mesh_path, args.debug_dir, args.iter)
    print("\n完了")


if __name__ == "__main__":
    main()
