"""
第2段実験 Boxer 一括実行スクリプト

実行環境: models/boxer/.venv
  cd /home/vr01/AIVisionPJ
  models/boxer/.venv/bin/python scripts/exp2/run_boxer.py
  models/boxer/.venv/bin/python scripts/exp2/run_boxer.py --scenes glass_front
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJ  = Path(__file__).resolve().parent.parent.parent
BOXER = PROJ / "models/boxer"
DATA  = PROJ / "data/exp2"
OUT   = PROJ / "results/exp2/rgb_models"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BOXER))

CKPT   = BOXER / "ckpts/boxernet_hw960in4x6d768-wssxpf9p.ckpt"
SCENES = ["glass_front", "glass_oblique", "resin_front", "resin_oblique"]
LABELS = ["glass plate", "transparent glass", "glass", "plastic sheet",
          "transparent object", "acrylic", "window glass"]


def prepare_boxer_dir(scene_name: str) -> Path:
    """Boxer が期待するフラット形式の intrinsics.json を持つ一時ディレクトリを作る。"""
    import shutil
    scene_dir = DATA / scene_name
    tmp_dir = PROJ / f"data/exp2/boxer_tmp/{scene_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # rgb.png / depth.npy をリンク
    for fname in ["rgb.png", "depth.npy"]:
        src = scene_dir / fname
        dst = tmp_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # intrinsics: nested → flat
    intr_nested = json.loads((scene_dir / "intrinsics.json").read_text())
    K = intr_nested["rgb"]
    flat = {
        "fx": K["fx"], "fy": K["fy"],
        "cx": K["cx"], "cy": K["cy"],
        "width": K["width"], "height": K["height"],
        "depth_scale": intr_nested.get("depth_scale", 0.001),
    }
    (tmp_dir / "intrinsics.json").write_text(json.dumps(flat, indent=2))
    return tmp_dir


def run_scene(boxernet, labels, scene_name, thresh2d=0.1, thresh3d=0.2):
    from loaders.realsense_loader import RealSenseLoader
    from utils.image import torch2cv2

    scene_dir = DATA / scene_name
    if not (scene_dir / "rgb.png").exists():
        print(f"  SKIP: {scene_name}")
        return None

    tmp_dir = prepare_boxer_dir(scene_name)
    loader = RealSenseLoader(str(tmp_dir), max_frames=1)
    loader.resize = boxernet.hw

    datum = next(iter(loader))
    device = next(boxernet.parameters()).device
    img_t  = datum["img0"].to(device)
    cam    = datum["cam0"].to(device)
    T_wr   = datum["T_world_rig0"].to(device)
    sdp_w  = datum["sdp_w"].to(device)
    gravity= datum["gravity"].to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = boxernet.forward_single(
            img=img_t, cam=cam, T_world_rig=T_wr,
            sdp_w=sdp_w, gravity=gravity,
            thresh2d=thresh2d, thresh3d=thresh3d,
            labels=labels,
        )
    elapsed = (time.perf_counter() - t0) * 1000

    bb2d = result.get("bb2d", []) or []
    bb3d = result.get("bb3d", []) or []

    # 可視化
    img_cv = torch2cv2(img_t[0])
    for det in bb2d:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        score = det.get("score", 0.0)
        name  = det.get("name", "?")
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{name} {score:.2f}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if not bb2d:
        cv2.putText(img_cv, "No detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imwrite(str(OUT / f"boxer_{scene_name}.jpg"), img_cv)

    dets_out = []
    for obb in bb3d:
        pos = obb.get("pos", [0, 0, 0])
        sz  = obb.get("scale", [0, 0, 0])
        dets_out.append({
            "name": obb.get("name", "?"), "score": round(float(obb.get("prob", 0)), 3),
            "pos_m": [round(float(v), 3) for v in pos],
            "size_m": [round(float(v), 3) for v in sz],
        })

    print(f"  [{scene_name}] 2D={len(bb2d)} 3D={len(bb3d)}  ({elapsed:.0f}ms)")
    for d in dets_out:
        print(f"    {d['name']:25s} score={d['score']:.3f}  "
              f"pos=({d['pos_m'][0]:.3f},{d['pos_m'][1]:.3f},{d['pos_m'][2]:.3f})m  "
              f"size=({d['size_m'][0]:.3f}x{d['size_m'][1]:.3f}x{d['size_m'][2]:.3f})m")
    if not dets_out:
        print(f"    (未検出)")

    return {"scene": scene_name, "infer_ms": round(elapsed),
            "n_2d": len(bb2d), "n_3d": len(bb3d), "detections_3d": dets_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--thresh2d", type=float, default=0.1)
    ap.add_argument("--thresh3d", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Boxer] ロード中... ckpt={CKPT.name}")
    from boxernet.boxernet import BoxerNet
    boxernet = BoxerNet.load_from_checkpoint(str(CKPT), device=device)
    boxernet.eval()
    print(f"  OK (hw={boxernet.hw})")

    results = []
    for scene in args.scenes:
        r = run_scene(boxernet, LABELS, scene, args.thresh2d, args.thresh3d)
        if r:
            results.append(r)

    (OUT / "boxer_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n完了 → {OUT}")


if __name__ == "__main__":
    main()
