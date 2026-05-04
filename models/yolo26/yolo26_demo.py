"""
YOLO26 Demo Script
- Object Detection, Segmentation, Pose Estimation, OBB
- Benchmarks: FPS, VRAM usage
"""
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0


def load_model(variant="yolo26n", task="detect"):
    """Load YOLO26 model. variant: yolo26n/s/m/l/x"""
    model_name = f"{variant}.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    return model


def run_detection(model, source, conf=0.5, show_result=False):
    """Run object detection and return results with timing."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    results = model(source, conf=conf, verbose=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000

    detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                })
    return detections, elapsed_ms


def benchmark_variants(source, variants=None):
    """Benchmark multiple YOLO26 variants."""
    if variants is None:
        variants = ["yolo26n", "yolo26s", "yolo26m"]

    results_table = []
    for variant in variants:
        print(f"\n--- Benchmarking {variant} ---")
        torch.cuda.empty_cache()
        vram_before = get_vram_usage()

        model = load_model(variant)
        # Warmup
        model(source, verbose=False)
        vram_after = get_vram_usage()

        # Timed runs
        times = []
        for _ in range(10):
            _, ms = run_detection(model, source)
            times.append(ms)

        avg_ms = np.mean(times)
        fps = 1000 / avg_ms
        vram_used = vram_after - vram_before

        print(f"  Avg inference: {avg_ms:.1f}ms | FPS: {fps:.1f} | VRAM: {vram_used:.0f}MB")
        results_table.append({
            "model": variant,
            "avg_ms": avg_ms,
            "fps": fps,
            "vram_mb": vram_used,
        })
        del model
        torch.cuda.empty_cache()

    return results_table


def demo_segmentation(source, variant="yolo26n"):
    """Demo instance segmentation."""
    print(f"\n--- Segmentation Demo ({variant}-seg) ---")
    model = YOLO(f"{variant}-seg.pt")
    results = model(source, verbose=False)
    for r in results:
        if r.masks is not None:
            print(f"  Detected {len(r.masks)} objects with masks")
    return results


def demo_pose(source, variant="yolo26n"):
    """Demo pose estimation (keypoints)."""
    print(f"\n--- Pose Estimation Demo ({variant}-pose) ---")
    model = YOLO(f"{variant}-pose.pt")
    results = model(source, verbose=False)
    for r in results:
        if r.keypoints is not None:
            print(f"  Detected {len(r.keypoints)} persons with keypoints")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 Demo")
    parser.add_argument("--source", default="https://ultralytics.com/images/bus.jpg",
                        help="Image source (file, URL, or 0 for webcam)")
    parser.add_argument("--variant", default="yolo26n", choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
    parser.add_argument("--task", default="detect", choices=["detect", "segment", "pose", "benchmark"])
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    if args.task == "benchmark":
        results = benchmark_variants(args.source)
        print("\n=== Benchmark Summary ===")
        for r in results:
            print(f"  {r['model']:10s} | {r['avg_ms']:6.1f}ms | {r['fps']:5.1f}FPS | VRAM: {r['vram_mb']:4.0f}MB")
    elif args.task == "detect":
        model = load_model(args.variant)
        detections, ms = run_detection(model, args.source, args.conf)
        print(f"\nDetected {len(detections)} objects in {ms:.1f}ms:")
        for d in detections[:10]:
            print(f"  {d['class']:20s} conf={d['confidence']:.2f}  bbox={[f'{v:.0f}' for v in d['bbox']]}")
    elif args.task == "segment":
        demo_segmentation(args.source, args.variant)
    elif args.task == "pose":
        demo_pose(args.source, args.variant)
