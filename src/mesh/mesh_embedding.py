"""Convert 3D face mesh landmarks into a normalized embedding vector."""

import numpy as np


def mesh_to_embedding(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert 478x3 face landmarks into a normalized geometric embedding.

    Steps:
        1. Center the mesh (translate centroid to origin)
        2. Scale to unit norm (size-invariant)
        3. Flatten to 1D vector (1434-dim)
        4. L2-normalize the final vector

    Args:
        landmarks: (478, 3) float32 array of (x, y, z) landmarks.

    Returns:
        (1434,) float64 L2-normalized embedding vector.
    """
    pts = landmarks.astype(np.float64)
    pts -= pts.mean(axis=0)

    norm = np.linalg.norm(pts)
    if norm > 1e-8:
        pts /= norm

    vec = pts.flatten()
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 1e-8:
        vec /= vec_norm

    return vec
