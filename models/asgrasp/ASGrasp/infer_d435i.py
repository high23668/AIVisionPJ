"""
ASGrasp 推論スクリプト D435i 対応版
- 入力: 3glass_*/ ディレクトリ (rgb.png + left_ir.png + right_ir.png + intrinsics.json)
- 入力解像度: D435i 640×480 → 中央 640×360 に top/bottom-crop（モデル想定解像度に合わせ）
- intrinsics: 我々の RealSense D435i 値を上書き使用
- 出力: 点群PLY + grasp poses + RGB上の grasp 投影画像
"""
import os, sys, json, argparse
import numpy as np
import torch
import cv2
sys.path.append("/asgrasp")
sys.path.append("/asgrasp/gsnet/pointnet2")
sys.path.append("/asgrasp/gsnet/utils")
sys.path.append("/asgrasp/gsnet/dataset")
sys.path.append("/asgrasp/src/core_multilayers")
from graspnetAPI.graspnet_eval import GraspGroup
from gsnet.models.graspnet import GraspNet, pred_decode
from raft_mvs_multilayers import RAFTMVS_2Layer
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
import open3d as o3d


def load_image_crop(path, crop_top=60, crop_bottom=60):
    """640×480 → 640×360 (top/bottom 60rows crop)"""
    img = cv2.imread(path, 1)
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape[:2]
    if H == 480:
        img = img[crop_top:H - crop_bottom]  # 360 rows
    img = img[:, :, ::-1].copy()  # BGR→RGB
    t = torch.from_numpy(img).permute(2, 0, 1).float()[None].cuda()
    return t, img  # tensor + numpy(rgb)


def build_proj_matrices(intr, baseline_m):
    """3視点のprojection matrixを作る (RGB は世界基準, IR_left は -baseline/2 シフト的に近似)"""
    # 我々のD435iではRGB→IR_leftの並進は約15mm、IR_left→IR_right=baseline=49.9mm
    # ASGrasp元コードはRGBを原点として ir1 = -15mm, ir2 = -70.1mm
    # ここでは "intr" 内の ir_to_rgb_t を使う
    fx_rgb, fy_rgb = intr["rgb"]["fx"], intr["rgb"]["fy"]
    cx_rgb = intr["rgb"]["cx"]
    cy_rgb = intr["rgb"]["cy"] - 60  # 上60pxクロップ

    fx_ir, fy_ir = intr["left_ir"]["fx"], intr["left_ir"]["fy"]
    cx_ir = intr["left_ir"]["cx"]
    cy_ir = intr["left_ir"]["cy"] - 60

    K_rgb = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]], dtype=np.float32)
    K_ir = np.array([[fx_ir, 0, cx_ir], [0, fy_ir, cy_ir], [0, 0, 1]], dtype=np.float32)

    # ir_to_rgb_t: IR_left→RGBの並進 (m)、ASGrasp期待は RGB→IR pose
    t_ir2rgb = np.array(intr["ir_to_rgb_t"], dtype=np.float32)  # ~[+15mm, ~0, ~0]
    # RGB→IR_left = -t_ir2rgb (近似、回転は恒等扱い)
    pose_rgb = np.eye(4, dtype=np.float32)
    pose_ir1 = np.eye(4, dtype=np.float32)
    pose_ir1[:3, 3] = -t_ir2rgb
    # IR_left→IR_right は X軸 -baseline_m
    pose_ir2 = np.eye(4, dtype=np.float32)
    pose_ir2[:3, 3] = -t_ir2rgb + np.array([-baseline_m, 0, 0], dtype=np.float32)

    poses = np.stack([pose_rgb, pose_ir1, pose_ir2], 0)
    intrinsics = np.stack([K_rgb, K_ir, K_ir], 0)
    poses_t = torch.from_numpy(poses)
    intrs_t = torch.from_numpy(intrinsics)
    proj = poses_t.clone()
    proj[:, :3, :4] = torch.matmul(intrs_t, poses_t[:, :3, :4])
    return proj[None].cuda(), K_rgb, K_ir


class ASGraspD435i:
    def __init__(self, cfgs, intr, baseline):
        self.args = cfgs
        net = torch.nn.DataParallel(GraspNet(seed_feat_dim=cfgs.seed_feat_dim,
                                             graspness_threshold=cfgs.graspness_threshold,
                                             is_training=False))
        net.cuda()
        ckpt = torch.load(cfgs.checkpoint_path)
        net.load_state_dict(ckpt["model_state_dict"])
        self.gsnet = net.module.eval()

        mvs = torch.nn.DataParallel(RAFTMVS_2Layer(cfgs)).cuda()
        mvs.load_state_dict(torch.load(cfgs.restore_ckpt))
        self.mvs_net = mvs.module.eval()

        self.proj, K_rgb, K_ir = build_proj_matrices(intr, baseline)
        self.K_rgb = K_rgb
        # CameraInfo for point cloud creation expects fx, fy, cx, cy at the image resolution we use (640x360)
        self.camera = CameraInfo(640.0, 360.0,
                                 K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2], 1.0)
        self.dmin = torch.tensor(0.2)[None].cuda()
        self.dmax = torch.tensor(1.5)[None].cuda()

    def infer(self, rgb_path, ir1_path, ir2_path):
        with torch.no_grad():
            color, color_np = load_image_crop(rgb_path)
            ir1, _ = load_image_crop(ir1_path)
            ir2, _ = load_image_crop(ir2_path)

            depth_up = self.mvs_net(color, ir1, ir2, self.proj.clone(),
                                    self.dmin, self.dmax,
                                    iters=self.args.valid_iters, test_mode=True).squeeze()
            depth_2 = depth_up.detach().cpu().numpy().squeeze()
            depth = depth_2[0]; depth1 = depth_2[1]

            cloud = create_point_cloud_from_depth_image(depth, self.camera, organized=True)
            depth_mask = (depth > 0.15) & (depth < 1.0)
            seg = np.ones(depth.shape)
            ws_mask = get_workspace_mask(cloud, seg, organized=True, outlier=0.02)
            mask = depth_mask & ws_mask
            cloud_m = cloud[mask]
            color_t = color.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            color_m = color_t[mask]

            n = len(cloud_m)
            if n == 0:
                print("ERROR: No valid depth points")
                return None, None, depth, depth1
            idx = np.random.choice(n, min(25000, n), replace=(n < 25000))
            cs = cloud_m[idx]; cls_color = color_m[idx]

            # 2nd layer
            cloud1 = create_point_cloud_from_depth_image(depth1, self.camera, organized=True)
            dm1 = (depth1 > 0.15) & (depth1 < 1.0) & (depth1 - depth > 0.01)
            mask1 = dm1 & ws_mask
            cloud_m1 = cloud1[mask1]
            print(f"  layer0 pts={n}, layer1 (背面) pts={len(cloud_m1)}")
            # GraspNet入力を作る前に、各レイヤを保存用に分離保持
            self._layer0_cloud = cs.copy()
            self._layer0_color = cls_color.copy()
            self._layer1_cloud = None
            if len(cloud_m1) > 0:
                k = min(10000, len(cloud_m1))
                idx1 = np.random.choice(len(cloud_m1), k, replace=(len(cloud_m1) < k))
                cs1 = cloud_m1[idx1]
                self._layer1_cloud = cs1.copy()
                cs = np.vstack([cs, cs1])
                cls_color = np.vstack([cls_color, np.zeros((k, 3), dtype=cls_color.dtype) + 0.7])

            # GraspNet inference (元コードと同じフォーマット = numpy 入力)
            # objectness_label を全部1として与えると graspable mask が objectness_pred のみで決まる
            ret_dict = {
                "point_clouds": cs.astype(np.float32),
                "coors": cs.astype(np.float32) / self.args.voxel_size,
                "feats": np.ones_like(cs).astype(np.float32),
                "full_point_clouds": cs.astype(np.float32),
                "objectness_label": np.ones(cs.shape[0], dtype=np.int64),
            }
            batch = minkowski_collate_fn([ret_dict])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()

            try:
                end_points = self.gsnet(batch)
            except Exception as e:
                print(f"GSNet failed: {e}")
                return None, None, depth, depth1

            preds = pred_decode(end_points)[0].detach().cpu().numpy()
            gg = GraspGroup(preds)
            if self.args.collision_thresh > 0:
                cd = ModelFreeCollisionDetector(cs.astype(np.float32), voxel_size=self.args.voxel_size_cd)
                cmask = cd.detect(gg, approach_dist=0.05, collision_thresh=self.args.collision_thresh)
                gg = gg[~cmask]
            gg = gg.nms().sort_by_score()
            if len(gg) == 0:
                print("ERROR: No grasps after NMS")
                return None, None, depth, depth1
            gg = gg[:min(50, len(gg))]
            return gg, (cs, cls_color), depth, depth1


def visualize_grasps(rgb_np, gg, K, out_path, top_n=20):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vis = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR).copy()
    def proj(P):
        if P[2] <= 0: return None
        return (int(P[0]*fx/P[2] + cx), int(P[1]*fy/P[2] + cy))
    n = min(top_n, len(gg))
    for i in range(n):
        g = gg[i]; t = g.translation; R = g.rotation_matrix; w = g.width
        c = proj(t)
        if c is None: continue
        end = proj(t + R[:, 0] * 0.05)
        fa = proj(t + R[:, 1] * (w/2))
        fb = proj(t - R[:, 1] * (w/2))
        col = (0, int(255 * g.score), int(255 * (1 - g.score)))
        if fa and fb: cv2.line(vis, fa, fb, col, 2)
        if end: cv2.arrowedLine(vis, end, c, col, 2, tipLength=0.3)
        cv2.circle(vis, c, 4, (0, 255, 255), -1)
        cv2.putText(vis, f"{i}:{g.score:.2f}", (c[0]+5, c[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(vis, f"ASGrasp D435i: top {n} grasps", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(out_path, vis)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="3glass_*/ dir with rgb.png, left_ir.png, right_ir.png, intrinsics.json")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--valid_iters", type=int, default=16)
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[128]*3)
    p.add_argument("--corr_implementation", choices=["reg","alt","reg_cuda","alt_cuda"], default="reg")
    p.add_argument("--shared_backbone", action="store_true")
    p.add_argument("--corr_levels", type=int, default=4)
    p.add_argument("--corr_radius", type=int, default=4)
    p.add_argument("--n_downsample", type=int, default=2)
    p.add_argument("--context_norm", default="batch", choices=["group","batch","instance","none"])
    p.add_argument("--slow_fast_gru", action="store_true")
    p.add_argument("--n_gru_layers", type=int, default=3)
    p.add_argument("--num_sample", type=int, default=96)
    p.add_argument("--depth_min", type=float, default=0.15)
    p.add_argument("--depth_max", type=float, default=1.5)
    p.add_argument("--train_2layer", default=True)
    p.add_argument("--restore_ckpt", default="/asgrasp/checkpoints/raftmvs_2layer.pth")
    p.add_argument("--checkpoint_path", default="/asgrasp/checkpoints/minkuresunet_epoch10.tar")
    p.add_argument("--seed_feat_dim", default=512, type=int)
    p.add_argument("--num_point", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--voxel_size", type=float, default=0.005)
    p.add_argument("--collision_thresh", type=float, default=0.01)
    p.add_argument("--voxel_size_cd", type=float, default=0.01)
    p.add_argument("--graspness_threshold", type=float, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "intrinsics.json")) as f:
        intr = json.load(f)
    baseline = intr["baseline_m"]

    eval = ASGraspD435i(args, intr, baseline)
    rgb_p = os.path.join(args.data_dir, "rgb.png")
    ir1_p = os.path.join(args.data_dir, "left_ir.png")
    ir2_p = os.path.join(args.data_dir, "right_ir.png")
    gg, cloud, depth, depth1 = eval.infer(rgb_p, ir1_p, ir2_p)

    if gg is None:
        print("No grasps generated")
        sys.exit(0)

    # Save outputs
    print(f"  Generated {len(gg)} grasps, saving to {args.out_dir}")

    # 1. grasp poses txt
    with open(os.path.join(args.out_dir, "grasp_poses.txt"), "w") as f:
        f.write("# rank, score, width, depth, height, tx, ty, tz, R(3x3 row-major)\n")
        for i, g in enumerate(gg):
            R = g.rotation_matrix.flatten(); t = g.translation
            f.write(f"{i}, {g.score:.4f}, {g.width:.4f}, {g.depth:.4f}, {g.height:.4f}, "
                    f"{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}, "
                    + ",".join(f"{v:.4f}" for v in R) + "\n")

    # 2. point cloud PLY (combined and per-layer with color coding)
    if cloud is not None:
        cs, color = cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cs)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(color, 0, 1))
        o3d.io.write_point_cloud(os.path.join(args.out_dir, "scene_point_cloud.ply"), pcd)

        # layer0 (front) = 赤系、layer1 (back) = 青系で色分け保存
        l0 = eval._layer0_cloud
        l0_pcd = o3d.geometry.PointCloud()
        l0_pcd.points = o3d.utility.Vector3dVector(l0)
        l0_pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.2, 0.2], (len(l0), 1)))  # red
        o3d.io.write_point_cloud(os.path.join(args.out_dir, "layer0_front_red.ply"), l0_pcd)

        if eval._layer1_cloud is not None:
            l1 = eval._layer1_cloud
            l1_pcd = o3d.geometry.PointCloud()
            l1_pcd.points = o3d.utility.Vector3dVector(l1)
            l1_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.2, 0.4, 1.0], (len(l1), 1)))  # blue
            o3d.io.write_point_cloud(os.path.join(args.out_dir, "layer1_back_blue.ply"), l1_pcd)

            # 2層をマージしたカラー版
            merged = o3d.geometry.PointCloud()
            merged.points = o3d.utility.Vector3dVector(np.vstack([l0, l1]))
            merged.colors = o3d.utility.Vector3dVector(np.vstack([
                np.tile([1.0, 0.2, 0.2], (len(l0), 1)),
                np.tile([0.2, 0.4, 1.0], (len(l1), 1)),
            ]))
            o3d.io.write_point_cloud(os.path.join(args.out_dir, "layers_colored.ply"), merged)

    # 3. depth maps (visualization)
    for name, d in [("layer0_front", depth), ("layer1_back", depth1)]:
        valid = (d > 0.1) & (d < 2.0)
        v = np.zeros_like(d, dtype=np.uint8)
        if valid.sum() > 0:
            v[valid] = np.clip((d[valid] - 0.2) / 0.8 * 255, 0, 255).astype(np.uint8)
        cmap = cv2.applyColorMap(v, cv2.COLORMAP_TURBO); cmap[~valid] = 0
        cv2.imwrite(os.path.join(args.out_dir, f"depth_{name}.jpg"), cmap)

    # 4. RGB overlay with grasps
    _, rgb_np = load_image_crop(rgb_p)
    visualize_grasps(rgb_np, gg, eval.K_rgb, os.path.join(args.out_dir, "grasps_overlay.jpg"))
    print(f"  Done.")
