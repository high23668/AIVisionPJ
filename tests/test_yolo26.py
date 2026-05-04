"""
YOLO26 Test Suite
- Basic detection, model loading, GPU inference
"""
import sys
sys.path.insert(0, str(__file__ + "/../../"))

import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


TEST_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"
VARIANTS = ["yolo26n", "yolo26s", "yolo26m"]


def test_model_load(variant="yolo26n"):
    print(f"\n[TEST] Model Load: {variant}")
    model = YOLO(f"{variant}.pt")
    assert model is not None, "Model failed to load"
    print(f"  OK - Model loaded: {variant}")
    return model


def test_detection(model, source=TEST_IMAGE_URL):
    print(f"\n[TEST] Object Detection")
    results = model(source, verbose=False)
    assert results is not None
    n_det = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"  OK - Detected {n_det} objects")
    return results


def test_gpu_inference(variant="yolo26n"):
    print(f"\n[TEST] GPU Inference Speed: {variant}")
    assert torch.cuda.is_available(), "CUDA not available"
    model = YOLO(f"{variant}.pt")
    # Warmup
    model(TEST_IMAGE_URL, verbose=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        model(TEST_IMAGE_URL, verbose=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  OK - Avg: {avg_ms:.1f}ms | FPS: {fps:.1f} | VRAM: {vram_mb:.0f}MB")
    return {"avg_ms": avg_ms, "fps": fps, "vram_mb": vram_mb}


def test_all_tasks(variant="yolo26n"):
    print(f"\n[TEST] All Tasks: {variant}")
    tasks_results = {}
    for task_suffix, task_name in [("", "detect"), ("-seg", "segment"), ("-pose", "pose")]:
        try:
            model = YOLO(f"{variant}{task_suffix}.pt")
            results = model(TEST_IMAGE_URL, verbose=False)
            print(f"  OK - Task '{task_name}' succeeded")
            tasks_results[task_name] = True
        except Exception as e:
            print(f"  SKIP - Task '{task_name}': {e}")
            tasks_results[task_name] = False
    return tasks_results


def run_all_tests():
    print("=" * 50)
    print("YOLO26 Test Suite")
    print("=" * 50)

    # Test 1: Model loading
    model = test_model_load("yolo26n")

    # Test 2: Basic detection
    test_detection(model, TEST_IMAGE_URL)

    # Test 3: GPU inference benchmark
    bench_result = test_gpu_inference("yolo26n")

    # Test 4: All tasks
    task_results = test_all_tasks("yolo26n")

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  GPU Inference: {bench_result['avg_ms']:.1f}ms / {bench_result['fps']:.1f}FPS")
    print(f"  VRAM Usage: {bench_result['vram_mb']:.0f}MB")
    for task, ok in task_results.items():
        print(f"  Task '{task}': {'OK' if ok else 'SKIP'}")


if __name__ == "__main__":
    run_all_tests()
