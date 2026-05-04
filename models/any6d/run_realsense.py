"""
Any6D RealSense用ポーズ推定スクリプト

使い方 (Dockerコンテナ内):
  python run_realsense.py \
    --rgb /data/rgb.png \
    --depth /data/depth.png \
    --mask /data/mask.png \
    --mesh /data/mesh.obj \
    --intrinsics /data/intrinsics.json \
    --output /results_out
"""

import argparse
import json
import os

import cv2
import numpy as np
import trimesh
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from estimater import Any6D
import nvdiffrast.torch as dr

glctx = dr.RasterizeCudaContext()

if __name__ == '__main__':
    seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb",        required=True, help="RGB画像パス")
    parser.add_argument("--depth",      required=True, help="深度画像パス (uint16 PNG)")
    parser.add_argument("--mask",       required=True, help="マスク画像パス (バイナリ PNG)")
    parser.add_argument("--mesh",       required=True, help="OBJメッシュパス")
    parser.add_argument("--intrinsics", required=True, help="カメラ内部パラメータ JSON")
    parser.add_argument("--output",     default="/results_out", help="出力ディレクトリ")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 画像読み込み
    color = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
    mask  = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE).astype(bool)

    # 深度読み込み (intrinsicsからdepth_scale取得)
    with open(args.intrinsics) as f:
        intr = json.load(f)
    depth_scale = intr.get("depth_scale", 0.001)
    depth = cv2.imread(args.depth, cv2.IMREAD_ANYDEPTH).astype(np.float32) * depth_scale

    # カメラ行列
    K = np.array([
        [intr["fx"], 0.0,         intr["cx"]],
        [0.0,        intr["fy"],  intr["cy"]],
        [0.0,        0.0,         1.0],
    ])

    # メッシュ読み込み
    mesh = trimesh.load(args.mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    print(f"メッシュ: 頂点{len(mesh.vertices)} / 面{len(mesh.faces)}")
    print(f"深度スケール: {depth_scale}")
    print(f"カメラ内部パラメータ:\n{K}")

    # ポーズ推定
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=args.output, debug=2)
    pred_pose = est.register_any6d(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name='realsense')

    # 結果保存
    np.savetxt(os.path.join(args.output, 'pred_pose.txt'), pred_pose)
    est.mesh.export(os.path.join(args.output, 'final_mesh.obj'))
    Image.fromarray(color).save(os.path.join(args.output, 'color.png'))
    np.savetxt(os.path.join(args.output, 'K.txt'), K)

    print(f"\n推定ポーズ:\n{pred_pose}")
    print(f"\n結果保存先: {args.output}")
