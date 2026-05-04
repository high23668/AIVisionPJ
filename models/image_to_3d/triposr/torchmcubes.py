# torchmcubes compatibility shim using scikit-image
# nvcc not available in this environment, so we provide a CPU fallback
import numpy as np
import torch
from skimage.measure import marching_cubes as sk_marching_cubes


def marching_cubes(volume: torch.Tensor, level: float):
    vol_np = volume.detach().cpu().numpy().astype(np.float32)
    try:
        verts, faces, _, _ = sk_marching_cubes(vol_np, level=level, method='lewiner')
    except Exception:
        verts, faces, _, _ = sk_marching_cubes(vol_np, level=level)
    verts_t = torch.from_numpy(verts.astype(np.float32))
    faces_t = torch.from_numpy(faces.astype(np.int64))
    return verts_t, faces_t
