"""
FoundationPose Test Suite
- Runs inside the FoundationPose Docker container
- Tests: installation verification, model loading, pose estimation

Usage:
  docker exec -it foundationpose bash -c "
    source activate my &&
    export LD_LIBRARY_PATH=/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:\$LD_LIBRARY_PATH &&
    cd /home/vr01/AIVisionPJ &&
    python tests/test_foundationpose.py
  "
"""
import sys
import os
from pathlib import Path

# FoundationPose repo path (inside container)
FP_REPO = Path(__file__).parent.parent / "models" / "foundationpose" / "FoundationPose"
sys.path.insert(0, str(FP_REPO))
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all required package imports."""
    print("\n[TEST] FoundationPose Package Imports")
    results = {}

    test_modules = [
        ("mycpp", "C++ pose utils"),
        ("trimesh", "3D mesh library"),
        ("open3d", "Point cloud library"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("torch", "PyTorch"),
        ("nvdiffrast.torch", "nvdiffrast (differentiable rasterizer)"),
    ]

    for mod, desc in test_modules:
        try:
            __import__(mod)
            print(f"  OK - {desc} ({mod})")
            results[mod] = True
        except ImportError as e:
            print(f"  FAIL - {desc}: {e}")
            results[mod] = False

    # Optional but important
    try:
        import kaolin
        print(f"  OK - kaolin {kaolin.__version__} (3D deep learning)")
        results["kaolin"] = True
    except Exception as e:
        print(f"  WARN - kaolin: {e}")
        results["kaolin"] = False

    return results


def test_cuda():
    """Test GPU availability."""
    print("\n[TEST] GPU / CUDA")
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  OK - {device_name} | {vram_gb:.1f}GB VRAM")
    return device_name, vram_gb


def test_estimater_import():
    """Test FoundationPose estimater module."""
    print("\n[TEST] FoundationPose Estimater Import")
    try:
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        print("  OK - FoundationPose, ScorePredictor, PoseRefinePredictor")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_dummy_pose_estimation():
    """Test pose estimation with synthetic data (no real camera/object needed)."""
    print("\n[TEST] Dummy Pose Estimation (Synthetic Data)")
    import torch
    import numpy as np
    import trimesh

    try:
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr

        # Create a simple synthetic cube mesh
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])  # 10cm cube

        # Synthetic inputs
        H, W = 480, 640
        K = np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]])
        color = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        depth = np.ones((H, W), dtype=np.float64) * 0.5  # 0.5m depth
        mask = np.zeros((H, W), dtype=bool)
        mask[180:300, 270:370] = True  # center rectangle

        import time
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        import tempfile
        debug_dir = tempfile.mkdtemp(prefix="fp_test_")
        est = FoundationPose(
            model_pts=mesh.vertices.copy(),
            model_normals=mesh.vertex_normals.copy(),
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug=0,
            debug_dir=debug_dir,
            glctx=glctx,
        )

        t0 = time.perf_counter()
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=4)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert pose.shape == (4, 4), f"Expected 4x4 pose, got {pose.shape}"
        vram_mb = torch.cuda.memory_allocated() / 1024**2

        print(f"  OK - Pose estimated in {elapsed_ms:.0f}ms | VRAM: {vram_mb:.0f}MB")
        print(f"  Translation: {pose[:3, 3]}")
        return {"elapsed_ms": elapsed_ms, "vram_mb": vram_mb, "pose": pose}

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tests():
    print("=" * 50)
    print("FoundationPose Test Suite")
    print("=" * 50)

    import_results = test_imports()
    device_name, vram_gb = test_cuda()
    estimater_ok = test_estimater_import()

    pose_result = None
    if estimater_ok:
        pose_result = test_dummy_pose_estimation()

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    core_ok = all(import_results.get(m, False) for m in ["torch", "trimesh", "open3d", "cv2"])
    print(f"  Core deps: {'OK' if core_ok else 'FAIL'}")
    print(f"  nvdiffrast: {'OK' if import_results.get('nvdiffrast.torch') else 'FAIL'}")
    print(f"  kaolin: {'OK' if import_results.get('kaolin') else 'WARN (optional)'}")
    print(f"  Estimater: {'OK' if estimater_ok else 'FAIL'}")
    if pose_result:
        print(f"  Pose estimation: {pose_result['elapsed_ms']:.0f}ms | VRAM: {pose_result['vram_mb']:.0f}MB")
    print(f"  GPU: {device_name} ({vram_gb:.1f}GB)")


if __name__ == "__main__":
    run_all_tests()
