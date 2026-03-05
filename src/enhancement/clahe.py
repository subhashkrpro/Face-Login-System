"""CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessor."""

import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID


def apply_clahe(
    img: np.ndarray,
    clip_limit: float | None = None,
    tile_grid_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Apply CLAHE to a BGR image for contrast enhancement.

    Operates on LAB L-channel only (preserves colour).
    """
    clip_limit = clip_limit if clip_limit is not None else CLAHE_CLIP_LIMIT
    tile_grid_size = tile_grid_size or CLAHE_TILE_GRID

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
