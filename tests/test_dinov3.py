"""
DINOv3 Test Suite
- Model loading, feature extraction, k-NN recognition, visualization
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
from models.dinov3.dinov3_demo import (
    load_model, extract_features, knn_recognition,
    benchmark_feature_extraction, visualize_patch_features
)


TEST_IMAGE = "bus.jpg"  # Downloaded by YOLO26 test


def test_model_load(variant="vitb"):
    print(f"\n[TEST] DINOv3 Model Load: {variant}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(variant, device)
    assert model is not None
    assert processor is not None
    print(f"  OK - DINOv3 {variant} loaded on {device}")
    return model, processor, device


def test_feature_extraction(model, processor, device):
    print(f"\n[TEST] Feature Extraction")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (224, 224))
    cls_feat, patch_feat = extract_features(model, processor, image, device)

    assert cls_feat.shape[-1] > 0, "CLS feature empty"
    assert len(patch_feat.shape) == 3, "Patch features should be 3D (1, n_patches, dim)"
    print(f"  OK - CLS: {cls_feat.shape}, Patches: {patch_feat.shape}")
    return cls_feat, patch_feat


def test_knn_recognition(model, processor, device):
    print(f"\n[TEST] k-NN Zero-shot Recognition")
    # Simulate gallery with same image (should get similarity=1.0)
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (224, 224))
    feat_q, _ = extract_features(model, processor, image, device)
    feat_g, _ = extract_features(model, processor, image, device)

    gallery_labels = ["bus"]
    predictions = knn_recognition(feat_q, feat_g, gallery_labels, k=1)
    assert predictions[0]["prediction"] == "bus"
    print(f"  OK - k-NN prediction: {predictions[0]['prediction']} (score: {predictions[0]['top_scores'][0]:.3f})")


def test_benchmark(model, processor, device):
    print(f"\n[TEST] Feature Extraction Benchmark")
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (224, 224))
    avg_ms, fps, vram_mb = benchmark_feature_extraction(model, processor, image, device, n_runs=5)
    assert avg_ms > 0
    print(f"  OK - Avg: {avg_ms:.1f}ms | FPS: {fps:.1f} | VRAM: {vram_mb:.0f}MB")
    return {"avg_ms": avg_ms, "fps": fps, "vram_mb": vram_mb}


def test_visualization(model, processor, device):
    print(f"\n[TEST] Patch Feature Visualization")
    import os
    os.makedirs("results", exist_ok=True)
    image = Image.open(TEST_IMAGE).convert("RGB") if Path(TEST_IMAGE).exists() else Image.new("RGB", (224, 224))
    visualize_patch_features(model, processor, image,
                              save_path="results/dinov3_vitb_features.png",
                              device=device)
    assert Path("results/dinov3_vitb_features.png").exists()
    print(f"  OK - Visualization saved")


def run_all_tests():
    print("=" * 50)
    print("DINOv3 Test Suite")
    print("=" * 50)

    model, processor, device = test_model_load("vitb")
    cls_feat, patch_feat = test_feature_extraction(model, processor, device)
    test_knn_recognition(model, processor, device)
    bench = test_benchmark(model, processor, device)
    test_visualization(model, processor, device)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  Feature dim: {cls_feat.shape[-1]}")
    print(f"  Patch count: {patch_feat.shape[1]}")
    print(f"  Inference: {bench['avg_ms']:.1f}ms / {bench['fps']:.1f}FPS")
    print(f"  VRAM: {bench['vram_mb']:.0f}MB")


if __name__ == "__main__":
    run_all_tests()
