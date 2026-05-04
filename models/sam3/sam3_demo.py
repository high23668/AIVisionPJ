"""
SAM3 Demo Script
- Segment Anything with Concepts (Meta AI)
- テキストプロンプトで物体をセグメント
- RealSenseカメラ対応

使い方:
  # 仮想環境: source /home/vr01/AIVisionPJ/.venv_sam3/bin/activate

  # 静止画でテキストセグメント
  python models/sam3/sam3_demo.py --task image --image <path> --prompt "cup"

  # カメラからキャプチャしてセグメント
  python models/sam3/sam3_demo.py --task camera --source 4 --prompt "cup"

  # ライブカメラ (Enterで撮影→セグメント)
  python models/sam3/sam3_demo.py --task live --source 4

  # MJPEGストリーミング (~0.6FPS) → ブラウザで http://localhost:8080 を開く
  python models/sam3/sam3_demo.py --task stream --source 4 --prompt "cup"
"""
import argparse
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def load_model(device="cuda"):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 model (初回はHFからダウンロードします)...")
    t0 = time.perf_counter()
    model = build_sam3_image_model(load_from_HF=True, device=device)
    processor = Sam3Processor(model)
    elapsed = (time.perf_counter() - t0)
    print(f"  OK - SAM3 loaded in {elapsed:.1f}s | CUDA: {torch.cuda.is_available()}")
    return model, processor


def segment_image(processor, pil_img, prompt):
    """Run text-prompted segmentation on a PIL image."""
    t0 = time.perf_counter()
    state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=state, prompt=prompt)
    elapsed = (time.perf_counter() - t0) * 1000

    masks = output["masks"]    # (N, H, W) bool tensors
    boxes = output["boxes"]    # (N, 4) xyxy
    scores = output["scores"]  # (N,)
    print(f"  '{prompt}': {len(masks)} segments in {elapsed:.0f}ms")
    return masks, boxes, scores, elapsed


def draw_masks(frame_bgr, masks, boxes, scores, prompt, alpha=0.5):
    """Overlay segmentation masks on a BGR frame."""
    display = frame_bgr.copy()
    colors = [
        (0, 255, 0), (0, 100, 255), (255, 0, 100),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().squeeze().astype(bool)
        else:
            mask_np = np.array(mask).squeeze().astype(bool)

        color = colors[i % len(colors)]
        overlay = display.copy()
        overlay[mask_np] = color
        display = cv2.addWeighted(display, 1 - alpha, overlay, alpha, 0)

        # BBox
        if isinstance(box, torch.Tensor):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        else:
            x1, y1, x2, y2 = np.array(box).astype(int)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{prompt} {float(score):.2f}"
        cv2.putText(display, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return display


def capture_frame(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source={source}")
        return None, None
    for _ in range(30):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Failed to capture frame")
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), frame


def task_image(processor, image_path, prompt, save_dir="results"):
    """Segment a static image with text prompt."""
    pil_img = Image.open(image_path).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    masks, boxes, scores, ms = segment_image(processor, pil_img, prompt)

    display = draw_masks(frame_bgr, masks, boxes, scores, prompt)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{save_dir}/sam3_result.jpg"
    cv2.imwrite(out_path, display)
    print(f"  Saved: {out_path}")

    import subprocess
    subprocess.Popen(["xdg-open", out_path])
    # cv2.imshow crashes after CUDA inference due to Qt/CUDA conflict
    # cv2.imshow(f"SAM3: '{prompt}'  (any key to close)", np.ascontiguousarray(display))
    input("  確認したらEnterを押してください...")
    cv2.destroyAllWindows()


def task_camera(processor, source, prompt, save_dir="results"):
    """Capture one frame from camera and segment with text prompt."""
    pil_img, frame_bgr = capture_frame(source)
    if pil_img is None:
        return

    masks, boxes, scores, ms = segment_image(processor, pil_img, prompt)

    display = draw_masks(frame_bgr, masks, boxes, scores, prompt)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{save_dir}/sam3_camera_result.jpg"
    cv2.imwrite(out_path, display)
    print(f"  Saved: {out_path}")

    import subprocess
    subprocess.Popen(["xdg-open", out_path])
    # cv2.imshow crashes after CUDA inference due to Qt/CUDA conflict
    # cv2.imshow(f"SAM3: '{prompt}'  (any key to close)", np.ascontiguousarray(display))
    input("  確認したらEnterを押してください...")
    cv2.destroyAllWindows()


def task_live(processor, source):
    """Live camera: type prompt, capture & segment, repeat. 'q' to quit."""
    import subprocess
    print("SAM3 ライブモード")
    print("  プロンプトを入力 → カメラ撮影 → セグメント結果を自動表示")
    print("  'q' + Enter で終了\n")

    while True:
        prompt = input("  プロンプト (例: cup, bottle, person) / q: ").strip()
        if prompt.lower() == "q":
            break
        if not prompt:
            continue

        pil_img, bgr_frame = capture_frame(source)
        if pil_img is None:
            continue

        masks, boxes, scores, ms = segment_image(processor, pil_img, prompt)
        display = draw_masks(bgr_frame, masks, boxes, scores, prompt)

        out_path = f"results/sam3_live_{prompt.replace(' ', '_')}.jpg"
        Path("results").mkdir(exist_ok=True)
        cv2.imwrite(out_path, display)
        print(f"  Saved: {out_path}")
        subprocess.Popen(["xdg-open", out_path])


def task_stream(processor, source, prompt, port=8080):
    """MJPEGストリーミングサーバー。ブラウザで http://localhost:<port> を開く。"""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    latest_frame = {"jpg": None}
    lock = threading.Lock()

    def inference_loop():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera source={source}")
            return
        # warmup
        for _ in range(10):
            cap.read()
        print(f"  推論ループ開始 (prompt='{prompt}', ~0.6FPS)")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            masks, boxes, scores, ms = segment_image(processor, pil_img, prompt)
            display = draw_masks(frame, masks, boxes, scores, prompt)
            # FPS表示
            cv2.putText(display, f"{1000/ms:.1f}FPS  prompt:'{prompt}'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with lock:
                latest_frame["jpg"] = jpg.tobytes()

    class MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # アクセスログを抑制

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""<html><body style="margin:0;background:#000">
<img src="/stream" style="max-width:100%">
</body></html>""")
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with lock:
                            jpg = latest_frame["jpg"]
                        if jpg is None:
                            time.sleep(0.1)
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpg + b"\r\n"
                        )
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print(f"  ブラウザで http://localhost:{port} を開いてください")
    print(f"  Ctrl+C で終了")
    try:
        HTTPServer(("0.0.0.0", port), MJPEGHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n  停止しました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Demo")
    parser.add_argument("--task", default="camera",
                        choices=["image", "camera", "live", "stream"],
                        help="image: 静止画, camera: 1枚撮影, live: ライブ, stream: MJPEGストリーム")
    parser.add_argument("--image", default=None, help="Input image path (--task image用)")
    parser.add_argument("--prompt", default="object", help="テキストプロンプト (例: cup)")
    parser.add_argument("--source", type=int, default=4, help="カメラデバイス番号")
    parser.add_argument("--port", type=int, default=8080, help="ストリームサーバーポート")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, processor = load_model(args.device)

    if args.task == "image":
        if args.image is None:
            print("ERROR: --image が必要です")
            sys.exit(1)
        task_image(processor, args.image, args.prompt)
    elif args.task == "camera":
        task_camera(processor, args.source, args.prompt)
    elif args.task == "live":
        task_live(processor, args.source)
    elif args.task == "stream":
        task_stream(processor, args.source, args.prompt, args.port)
