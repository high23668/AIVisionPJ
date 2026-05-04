"""
FoundationPose Demo Script
- 6DoF pose estimation (model-based and model-free)
- Runs inside the FoundationPose Docker container

Usage (inside container):
  python fp_demo.py --mode model_based --mesh path/to/mesh.obj --image path/to/image.png --depth path/to/depth.png
  python fp_demo.py --mode model_free --ref_dir path/to/ref_images/ --image path/to/image.png

Environment:
  docker exec -it foundationpose bash
  source activate my
  cd /path/to/foundationpose/repo
  python fp_demo.py
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add FoundationPose repo to path
FP_REPO = Path(__file__).parent / "FoundationPose"
sys.path.insert(0, str(FP_REPO))

# Set torch lib path for CUDA extensions
os.environ.setdefault("LD_LIBRARY_PATH",
    "/opt/conda/envs/my/lib/python3.8/site-packages/torch/lib:" +
    os.environ.get("LD_LIBRARY_PATH", ""))


def load_foundationpose(mesh_path=None, ref_dir=None, debug=0):
    """Load FoundationPose estimator.

    Args:
        mesh_path: Path to CAD mesh (.obj) for model-based mode
        ref_dir: Path to reference images directory for model-free mode
        debug: Debug level (0=off, 1=basic, 2=verbose)
    """
    import torch
    from estimater import FoundationPose
    from datareader import YcbineoatReader, LinemodReader

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()

    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=None,
        model_normals=None,
        symmetry_tfs=None,
        mesh=mesh_path,
        scorer=scorer,
        refiner=refiner,
        debug_dir=None,
        debug=debug,
        glctx=glctx,
    )
    return est


def run_pose_estimation_demo(image_path, depth_path, mask_path, mesh_path,
                              K=None, debug=0):
    """Run FoundationPose on a single frame.

    Args:
        image_path: RGB image
        depth_path: Depth image (meters, float32 or uint16 in mm)
        mask_path: Object mask (binary)
        mesh_path: CAD model (.obj)
        K: Camera intrinsics 3x3 (uses default if None)
        debug: Debug level
    Returns:
        pose: 4x4 transformation matrix (object in camera frame)
        elapsed_ms: inference time in milliseconds
    """
    import torch
    import trimesh
    from estimater import FoundationPose
    from Utils import get_mask, depth2xyzmap, toOpen3dCloud

    # Load inputs
    color = cv2.imread(str(image_path))
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float64)
    if depth.max() > 100:  # likely uint16 in mm
        depth = depth / 1000.0
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0

    # Default RealSense D435i intrinsics (640x480)
    if K is None:
        K = np.array([[615.0, 0, 320.0],
                      [0, 615.0, 240.0],
                      [0, 0, 1.0]])

    # Load mesh
    mesh = trimesh.load(str(mesh_path))

    # Initialize estimator
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import nvdiffrast.torch as dr

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug=debug,
        debug_dir=None,
        glctx=glctx,
    )

    # Estimate pose
    t0 = time.perf_counter()
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=4)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Pose estimated in {elapsed_ms:.1f}ms")
    print(f"Translation: {pose[:3, 3]}")
    print(f"Rotation:\n{pose[:3, :3]}")

    return pose, elapsed_ms


def verify_installation():
    """Quick verification that all FoundationPose components are importable."""
    print("Verifying FoundationPose installation...")
    errors = []

    modules = [
        ("mycpp", "C++ extensions"),
        ("trimesh", "trimesh"),
        ("open3d", "open3d"),
        ("cv2", "opencv-python"),
        ("nvdiffrast.torch", "nvdiffrast"),
        ("kaolin", "kaolin"),
    ]

    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"  OK - {display_name}")
        except ImportError as e:
            print(f"  FAIL - {display_name}: {e}")
            errors.append(display_name)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  OK - CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARN - CUDA not available")
    except Exception as e:
        errors.append(f"CUDA: {e}")

    # Check estimator import
    try:
        sys.path.insert(0, str(FP_REPO))
        from estimater import FoundationPose
        print("  OK - FoundationPose estimater")
    except Exception as e:
        print(f"  FAIL - estimater: {e}")
        errors.append("estimater")

    if errors:
        print(f"\nFailed: {errors}")
        return False
    print("\nAll components verified OK")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoundationPose Demo")
    parser.add_argument("--mode", default="verify",
                        choices=["verify", "model_based", "model_free"])
    parser.add_argument("--mesh", help="Path to CAD mesh .obj")
    parser.add_argument("--image", help="RGB image path")
    parser.add_argument("--depth", help="Depth image path")
    parser.add_argument("--mask", help="Object mask path")
    parser.add_argument("--ref_dir", help="Reference images directory (model-free mode)")
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "verify":
        ok = verify_installation()
        sys.exit(0 if ok else 1)
    elif args.mode == "model_based":
        assert args.mesh and args.image and args.depth and args.mask, \
            "model_based mode requires --mesh, --image, --depth, --mask"
        pose, ms = run_pose_estimation_demo(
            args.image, args.depth, args.mask, args.mesh, debug=args.debug
        )
        print(f"\nFinal pose (4x4):\n{pose}")
