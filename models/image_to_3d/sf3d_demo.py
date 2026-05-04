"""
SF3D (Stable Fast 3D) デモ
単一RGB画像から3Dメッシュを生成する

使い方:
  python sf3d_demo.py --image data/realsense/rgb.png
  python sf3d_demo.py --image data/realsense/rgb.png --output results/image_to_3d/cup.glb

実行環境:
  source .venv_3d/bin/activate
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# SF3D のソースパスを追加
SF3D_DIR = Path(__file__).parent / "sf3d"
sys.path.insert(0, str(SF3D_DIR))
sys.path.insert(0, str(SF3D_DIR / "texture_baker"))
sys.path.insert(0, str(SF3D_DIR / "uv_unwrapper"))

import numpy as np
import torch
from PIL import Image


def crop_to_square(image: Image.Image) -> Image.Image:
    """オブジェクトのバウンディングボックスを中心に正方形クロップ"""
    # アルファチャンネルでオブジェクト領域を検出
    bbox = image.getbbox()
    if bbox is None:
        # フォールバック: 中央クロップ
        w, h = image.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return image.crop((left, top, left + s, top + s))

    x0, y0, x1, y1 = bbox
    obj_w = x1 - x0
    obj_h = y1 - y0
    s = max(obj_w, obj_h)

    # 余白を追加して正方形に
    pad = int(s * 0.1)
    s = s + pad * 2
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    left = max(0, cx - s // 2)
    top = max(0, cy - s // 2)
    right = left + s
    bottom = top + s

    # 画像範囲を超えないよう調整
    img_w, img_h = image.size
    if right > img_w:
        left = max(0, img_w - s)
        right = img_w
    if bottom > img_h:
        top = max(0, img_h - s)
        bottom = img_h

    return image.crop((left, top, right, bottom))


def remove_background(image: Image.Image) -> Image.Image:
    """rembg で背景除去して RGBA 画像を返す"""
    from rembg import remove
    print("  背景除去中...")
    return remove(image)


def load_model():
    """SF3D モデルをロード"""
    from sf3d.system import SF3D
    print("  SF3D モデルをロード中 (初回はHFからダウンロード)...")
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()
    return model


def run_sf3d(image_path: str, output_path: str = None) -> dict:
    """
    SF3D で単一画像から3Dメッシュを生成

    Args:
        image_path: 入力画像パス
        output_path: 出力 GLB パス (省略時は results/image_to_3d/ に保存)

    Returns:
        dict: {"output": 出力パス, "vram_mb": VRAM使用量, "time_s": 推論時間}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  デバイス: {device}")

    # 出力パス決定
    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"results/image_to_3d/sf3d_{stem}.glb"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 画像読み込み
    print(f"  画像読み込み: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"  画像サイズ: {image.size}")

    # 背景除去
    image_rgba = remove_background(image)

    # 正方形クロップ (SF3Dは512x512想定。縦横比が違うと歪む)
    image_rgba = crop_to_square(image_rgba)
    print(f"  クロップ後サイズ: {image_rgba.size}")

    # VRAM リセット
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # モデルロード (float32のまま: autocast が整合を取る)
    t_load = time.time()
    model = load_model()
    model = model.to(device)

    # 推論 (float16 autocast: SM 7.5 Turing で Flash Attention が有効)
    print("  3Dメッシュ生成中...")
    t_infer = time.time()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            mesh, glob_dict = model.run_image(
                [image_rgba],
                bake_resolution=0,   # テクスチャベイクをスキップ (nvcc不要、ポーズ推定には不要)
                remesh="none",
            )
    infer_time = time.time() - t_infer
    total_time = time.time() - t_load

    # VRAM 計測
    vram_mb = 0
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # メッシュ保存
    print(f"  メッシュ保存: {output_path}")
    mesh.export(output_path)

    # モデル解放
    del model
    torch.cuda.empty_cache()

    return {
        "output": output_path,
        "vram_mb": vram_mb,
        "infer_time_s": infer_time,
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="SF3D: 単一画像から3Dメッシュ生成")
    parser.add_argument("--image", required=True, help="入力画像パス")
    parser.add_argument("--output", default=None, help="出力GLBパス")
    args = parser.parse_args()

    print("=" * 50)
    print("SF3D (Stable Fast 3D) デモ")
    print("=" * 50)

    result = run_sf3d(args.image, args.output)

    print()
    print("=" * 50)
    print("結果")
    print("=" * 50)
    print(f"  出力ファイル : {result['output']}")
    print(f"  推論時間     : {result['infer_time_s']:.2f} 秒")
    print(f"  合計時間     : {result['total_time_s']:.2f} 秒 (モデルロード含む)")
    print(f"  VRAM使用量   : {result['vram_mb']:.0f} MB")
    print()
    print(f"3Dビューアで確認:")
    print(f"  xdg-open {result['output']}")
    subprocess.Popen(["xdg-open", result["output"]])


if __name__ == "__main__":
    main()
