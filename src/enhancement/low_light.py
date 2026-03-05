"""Adaptive gamma correction for low-light face images.

When a face crop is very dark (mean brightness below a threshold),
a gamma < 1 curve brightens the image before CLAHE runs.  This ensures
CLAHE has enough intensity variation to work with.

Pipeline order:  crop → **gamma** → CLAHE → super-resolution
"""

import cv2
import numpy as np

from config import (
    LOW_LIGHT_BRIGHTNESS_THRESHOLD,
    LOW_LIGHT_TARGET_BRIGHTNESS,
    LOW_LIGHT_MAX_GAMMA,
)
from src.logger import get_logger

log = get_logger(__name__)


def _estimate_brightness(img: np.ndarray) -> float:
    """Return mean brightness (L channel, 0-255) of a BGR image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean())


def _compute_gamma(mean_brightness: float) -> float:
    """
    Compute gamma value to lift *mean_brightness* toward the target.

    gamma < 1 → brightens darks (used for low light)
    gamma = 1 → no change
    gamma > 1 → darkens (not used here)

    Formula:  gamma = log(target/255) / log(current/255)
    Clamped to [LOW_LIGHT_MAX_GAMMA .. 1.0] to avoid extreme blow-out.
    """
    if mean_brightness < 1:
        mean_brightness = 1.0  # avoid log(0)

    target = LOW_LIGHT_TARGET_BRIGHTNESS
    gamma = np.log(target / 255.0) / np.log(mean_brightness / 255.0)
    gamma = float(np.clip(gamma, LOW_LIGHT_MAX_GAMMA, 1.0))
    return gamma


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to a BGR image using a LUT (fast)."""
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lut)


def auto_brighten(img: np.ndarray) -> np.ndarray:
    """
    Brighten the image if it's too dark, otherwise return as-is.

    This is the main entry point — call before CLAHE.

    Returns:
        Brightened BGR image (or original if already bright enough).
    """
    brightness = _estimate_brightness(img)

    if brightness >= LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        log.debug("Brightness %.1f ≥ %d — no gamma needed",
                  brightness, LOW_LIGHT_BRIGHTNESS_THRESHOLD)
        return img

    gamma = _compute_gamma(brightness)
    result = apply_gamma(img, gamma)
    new_brightness = _estimate_brightness(result)

    log.info(
        "Low-light correction: brightness %.1f → %.1f (gamma=%.2f)",
        brightness, new_brightness, gamma,
    )
    return result
