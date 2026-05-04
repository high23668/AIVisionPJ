"""
Any6D用メッシュ生成スクリプト (ホスト側 / .venv_3d で実行)

アンカー画像からSF3Dで3Dメッシュを生成し、Any6Dに渡せるOBJ形式で保存する。

使い方:
  source .venv_3d/bin/activate
  python scripts/any6d_generate_mesh.py --image data/cup.png --output data/any6d_mesh/cup.obj

  # マスクファイルがある場合（オブジェクト領域だけクロップ）:
  python scripts/any6d_generate_mesh.py --image data/color.png --mask data/mask.png --output data/any6d_mesh/obj.obj
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image

# SF3D パス
SF3D_DIR = Path(__file__).parent.parent / "models" / "image_to_3d" / "sf3d"
sys.path.insert(0, str(SF3D_DIR))
sys.path.insert(0, str(SF3D_DIR / "texture_baker"))
sys.path.insert(0, str(SF3D_DIR / "uv_unwrapper"))


def crop_to_square(image: Image.Image) -> Image.Image:
    """アルファチャンネルのbboxを中心に10%余白付き正方形クロップ"""
    bbox = image.getbbox()
    if bbox is None:
        w, h = image.size
        s = min(w, h)
        return image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))

    x0, y0, x1, y1 = bbox
    s = max(x1 - x0, y1 - y0)
    pad = int(s * 0.1)
    s += pad * 2
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    left = max(0, cx - s // 2)
    top = max(0, cy - s // 2)
    right, bottom = left + s, top + s
    img_w, img_h = image.size
    if right > img_w:
        left, right = max(0, img_w - s), img_w
    if bottom > img_h:
        top, bottom = max(0, img_h - s), img_h

    return image.crop((left, top, right, bottom))


def align_mesh(mesh: trimesh.Trimesh, scale: float = 0.1) -> trimesh.Trimesh:
    """Any6Dと同じ座標系整列処理 (scale + OBB align)"""
    mesh.vertices *= scale

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

    obb = mesh_o3d.get_oriented_bounding_box()
    center = mesh_o3d.get_center()
    mesh_o3d.rotate(np.linalg.inv(obb.R), center=center)

    mesh.vertices = np.asarray(mesh_o3d.vertices)
    return mesh


def generate_mesh(image_path: str, output_path: str, mask_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 画像読み込み
    image = Image.open(image_path).convert("RGB")
    print(f"入力画像: {image_path} ({image.size})")

    # マスク適用（指定時）
    if mask_path:
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask) > 127
        img_arr = np.array(image)
        # RGBAに変換してアルファをマスクから設定（rembgをスキップ）
        rgba_arr = np.zeros((img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8)
        rgba_arr[:, :, :3] = img_arr
        rgba_arr[:, :, 3] = (mask_arr * 255).astype(np.uint8)
        image_rgba = Image.fromarray(rgba_arr, "RGBA")
        print(f"マスク適用: {mask_path}")
    else:
        # マスクなし: rembgで背景除去
        from rembg import remove
        print("背景除去中...")
        image_rgba = remove(image)

    # 正方形クロップ
    image_rgba = crop_to_square(image_rgba)
    print(f"クロップ後: {image_rgba.size}")

    # SF3D ロード & 推論
    from sf3d.system import SF3D
    print("SF3D モデルロード中...")
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval().to(device)

    print("3Dメッシュ生成中...")
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            mesh, _ = model.run_image([image_rgba], bake_resolution=0, remesh="none")

    del model
    torch.cuda.empty_cache()

    # GLB → trimesh
    glb_path = output_path.with_suffix(".glb")
    mesh.export(str(glb_path))
    scene_or_mesh = trimesh.load(str(glb_path))
    if isinstance(scene_or_mesh, trimesh.Scene):
        tri_mesh = trimesh.util.concatenate(scene_or_mesh.dump())
    else:
        tri_mesh = scene_or_mesh

    # 座標系整列 (Any6Dと同じ処理)
    tri_mesh = align_mesh(tri_mesh)

    # OBJ保存
    tri_mesh.export(str(output_path))
    print(f"\n完了: {output_path}")
    print(f"  頂点数: {len(tri_mesh.vertices)} / 面数: {len(tri_mesh.faces)}")
    glb_path.unlink()  # 中間GLB削除


def main():
    parser = argparse.ArgumentParser(description="Any6D用メッシュ生成 (SF3D)")
    parser.add_argument("--image", required=True, help="入力RGBまたはRGBA画像")
    parser.add_argument("--output", required=True, help="出力OBJパス")
    parser.add_argument("--mask", default=None, help="バイナリマスク画像（オプション）")
    args = parser.parse_args()

    generate_mesh(args.image, args.output, args.mask)


if __name__ == "__main__":
    main()
