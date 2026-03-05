"""Anti-spoofing liveness detection using 3D mesh z-depth analysis.

A real face has significant depth variation across 478 landmarks:
    - Nose tip protrudes (high z)
    - Eyes/temples are recessed (low z)
    - Overall z-range ~ 0.04–0.15+

A flat photo or screen attack has nearly co-planar landmarks:
    - Minimal z variation
    - z-range ~ 0.00–0.02

This module checks the z-depth statistics to reject flat/spoofed faces
before they enter the recognition pipeline.
"""

from __future__ import annotations

import numpy as np

from config import SPOOF_Z_RANGE_THRESHOLD, SPOOF_Z_STD_THRESHOLD
from src.logger import get_logger

log = get_logger(__name__)


def check_liveness(landmarks: np.ndarray) -> dict:
    """
    Analyse z-depth of 478 landmarks to detect photo/screen spoofing.

    Args:
        landmarks: (478, 3) float32 array from FaceMesh.extract().

    Returns:
        dict with keys:
            - is_live (bool):   True if face appears 3D (real), False if flat
            - z_range (float):  max(z) − min(z)
            - z_std (float):    standard deviation of z values
            - reason (str):     human-readable explanation
    """
    z = landmarks[:, 2].astype(np.float64)

    z_range = float(z.max() - z.min())
    z_std = float(z.std())

    # Both metrics must pass — a real face has substantial depth spread
    range_ok = z_range >= SPOOF_Z_RANGE_THRESHOLD
    std_ok = z_std >= SPOOF_Z_STD_THRESHOLD
    is_live = range_ok and std_ok

    if is_live:
        reason = "3D depth OK"
        log.debug(
            "Liveness PASS: z_range=%.4f (≥%.4f)  z_std=%.4f (≥%.4f)",
            z_range, SPOOF_Z_RANGE_THRESHOLD, z_std, SPOOF_Z_STD_THRESHOLD,
        )
    else:
        parts = []
        if not range_ok:
            parts.append(
                f"z_range={z_range:.4f} < {SPOOF_Z_RANGE_THRESHOLD:.4f}"
            )
        if not std_ok:
            parts.append(
                f"z_std={z_std:.4f} < {SPOOF_Z_STD_THRESHOLD:.4f}"
            )
        reason = "Flat face detected — " + ", ".join(parts)
        log.warning("Liveness FAIL: %s", reason)

    return {
        "is_live": is_live,
        "z_range": round(z_range, 6),
        "z_std": round(z_std, 6),
        "reason": reason,
    }
