"""
Qwen3-VL Test Suite
- Model loading, VQA, 2D grounding, robot task grounding
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from models.qwen3vl.qwen3vl_demo import (
    load_model, run_inference, grounding_2d,
    scene_description, robot_task_grounding, vqa
)


TEST_IMAGE = "bus.jpg"


def test_model_load(variant="8b"):
    print(f"\n[TEST] Qwen3-VL Model Load: {variant}")
    model, processor = load_model(variant)
    assert model is not None
    assert processor is not None
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  OK - Model loaded | VRAM: {vram_mb:.0f}MB")
    return model, processor


def test_vqa(model, processor):
    print(f"\n[TEST] Visual Question Answering")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (640, 480))
    response, ms = vqa(model, processor, image, "What is the main subject in this image?")
    assert len(response) > 0, "Empty response"
    print(f"  OK - Response ({ms:.0f}ms): {response[:100]}...")
    return response, ms


def test_grounding_2d(model, processor):
    print(f"\n[TEST] 2D Visual Grounding")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (640, 480))
    response, ms = grounding_2d(model, processor, image, "bus")
    assert len(response) > 0
    print(f"  OK - Grounding ({ms:.0f}ms): {response[:100]}")
    return response


def test_scene_description(model, processor):
    print(f"\n[TEST] Scene Description")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (640, 480))
    response, ms = scene_description(model, processor, image)
    assert len(response) > 10
    print(f"  OK - Description ({ms:.0f}ms): {response[:150]}...")
    return response


def test_robot_task(model, processor):
    print(f"\n[TEST] Robot Task Grounding")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (640, 480))
    response, ms = robot_task_grounding(model, processor, image, "Pick up the bus")
    assert len(response) > 0
    print(f"  OK - Task grounding ({ms:.0f}ms): {response[:150]}...")
    return response


def test_benchmark(model, processor):
    print(f"\n[TEST] Inference Benchmark")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (640, 480))
    times = []
    for _ in range(3):
        _, ms = vqa(model, processor, image, "How many people are in the image?")
        times.append(ms)
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    vram_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  OK - Avg: {avg_ms:.0f}ms | FPS: {fps:.2f} | VRAM: {vram_mb:.0f}MB")
    return {"avg_ms": avg_ms, "fps": fps, "vram_mb": vram_mb}


def run_all_tests(variant="8b"):
    print("=" * 50)
    print(f"Qwen3-VL Test Suite (variant={variant})")
    print("=" * 50)

    model, processor = test_model_load(variant)
    test_vqa(model, processor)
    test_grounding_2d(model, processor)
    test_scene_description(model, processor)
    test_robot_task(model, processor)
    bench = test_benchmark(model, processor)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Avg inference: {bench['avg_ms']:.0f}ms / {bench['fps']:.2f}FPS")
    print(f"  VRAM Usage: {bench['vram_mb']:.0f}MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="8b", choices=["2b", "4b", "8b", "32b"])
    args = parser.parse_args()
    run_all_tests(args.variant)
