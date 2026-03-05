"""Anti-spoofing via texture, noise, and gradient analysis.

Multi-feature approach to detect photo-of-face attacks (screen replay,
printed photos).  Combines four independent signals:

    1. **LBP histogram variance** — real skin has diverse Local Binary
       Pattern codes; flat recaptures are more uniform.
    2. **High-frequency energy ratio** — real faces retain fine detail;
       recaptured images lose high-frequency components.
    3. **Noise kurtosis** — camera sensor noise on a real 3D face creates
       a heavy-tailed (high kurtosis) distribution due to varying surface
       angles and skin micro-texture.  A recaptured photo has more
       Gaussian (low kurtosis) noise because the photo-to-camera capture
       smooths the noise distribution.
       (Real ≈ 25–35, Photo ≈ 5–10)
    4. **Block gradient variance** — a real face has varied edge intensity
       across regions (eyes, nose, lips, hair create gradient diversity).
       A flat photo loses some of this 3D gradient variation.
       (Real ≈ 1000+, Photo ≈ 400–500)
    5. **Local noise level variance** — a real 3D face has spatially
       *non-uniform* sensor noise (smooth forehead = low noise, textured
       eyebrows/hair = high noise).  A flat photo recaptured through a
       camera has *uniform* noise across all regions.
       (Real ≈ 3–8, Photo ≈ 0.5–1.2)

All checks are cheap to compute (pure NumPy/OpenCV, no model needed).
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    SPOOF_LBP_HIST_VARIANCE_THRESHOLD,
    SPOOF_LBP_HF_ENERGY_THRESHOLD,
    SPOOF_NOISE_KURTOSIS_THRESHOLD,
    SPOOF_BLOCK_GRAD_VAR_THRESHOLD,
    SPOOF_LOCAL_NOISE_VAR_THRESHOLD,
)
from src.logger import get_logger

log = get_logger(__name__)

# ── LBP neighbour offsets (8-connected, clockwise from top-left) ──────
_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0,  1), ( 1, 1), ( 1, 0),
    ( 1, -1), ( 0, -1),
]

# Block size for gradient variance analysis
_BLOCK_SIZE = 16


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    """
    Compute uniform LBP (8,1) for a single-channel image.

    Returns a uint8 image of the same size (border pixels = 0).
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    center = gray[1:-1, 1:-1].astype(np.int16)

    for bit, (dy, dx) in enumerate(_OFFSETS):
        neighbour = gray[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx].astype(np.int16)
        lbp[1:-1, 1:-1] |= ((neighbour >= center).astype(np.uint8) << bit)

    return lbp


def _lbp_histogram(lbp: np.ndarray) -> np.ndarray:
    """Normalised 256-bin histogram of LBP codes."""
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _high_freq_energy(gray: np.ndarray) -> float:
    """
    Ratio of high-frequency energy to total energy (Laplacian).

    Real faces have more fine detail → higher ratio.
    Recaptured images lose high-frequency components.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    hf_energy = float(np.mean(lap ** 2))
    total_energy = float(np.mean(gray.astype(np.float64) ** 2))
    if total_energy < 1e-8:
        return 0.0
    return hf_energy / total_energy


def _noise_kurtosis(gray: np.ndarray) -> float:
    """
    Kurtosis of high-frequency noise residual.

    Extract noise by subtracting a Gaussian-blurred version of the image.
    Real 3D faces produce heavy-tailed noise (kurtosis ≈ 25–35) due to
    varying surface angles and skin micro-texture.  Recaptured photos
    have near-Gaussian noise (kurtosis ≈ 5–10) because the double
    digitisation (original capture → screen/print → camera) smooths
    the noise distribution.
    """
    gray_f = gray.astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float64)
    noise = gray_f - blurred

    std = float(np.std(noise))
    if std < 1e-8:
        return 0.0
    # Excess kurtosis: E[(x-µ)^4] / σ^4
    kurt = float(np.mean((noise - noise.mean()) ** 4) / (std ** 4))
    return kurt


def _block_gradient_variance(gray: np.ndarray) -> float:
    """
    Variance of mean gradient magnitude across non-overlapping blocks.

    A real 3D face has highly variable gradient intensity: sharp edges
    around eyes/eyebrows, smooth forehead, textured hair.  A flat photo
    loses some of this 3D gradient diversity.

    Returns the variance of per-block mean gradient magnitudes.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    h, w = gray.shape
    block_means: list[float] = []
    for by in range(0, h - _BLOCK_SIZE + 1, _BLOCK_SIZE):
        for bx in range(0, w - _BLOCK_SIZE + 1, _BLOCK_SIZE):
            block = grad_mag[by : by + _BLOCK_SIZE, bx : bx + _BLOCK_SIZE]
            block_means.append(float(np.mean(block)))

    if len(block_means) < 2:
        return 0.0
    return float(np.var(block_means))


def _local_noise_variance(gray: np.ndarray) -> float:
    """
    Variance of per-block noise standard deviations.

    A real 3D face has spatially *non-uniform* noise: smooth forehead
    has low noise-std, textured eyebrows/hair have high noise-std.
    A flat photo recaptured through a camera has more *uniform* noise
    across all regions (the sensor sees a flat radiating surface).

    Method: subtract a Gaussian-blurred version to isolate noise,
    then compute std-dev of that noise in each 16×16 block.  Finally
    return the variance of those per-block std-devs.

    High value → non-uniform noise → likely real face.
    Low value → uniform noise → likely photo.
    """
    gray_f = gray.astype(np.float64)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float64)
    noise = gray_f - blurred

    h, w = gray.shape
    block_stds: list[float] = []
    for by in range(0, h - _BLOCK_SIZE + 1, _BLOCK_SIZE):
        for bx in range(0, w - _BLOCK_SIZE + 1, _BLOCK_SIZE):
            block = noise[by : by + _BLOCK_SIZE, bx : bx + _BLOCK_SIZE]
            block_stds.append(float(np.std(block)))

    if len(block_stds) < 2:
        return 0.0
    return float(np.var(block_stds))


def check_texture(crop: np.ndarray) -> dict:
    """
    Multi-feature texture analysis to detect photo/screen spoofing.

    Combines five signals — all must pass for the face to be considered
    live.  This catches both solid-colour/flat-image attacks (via LBP +
    HF energy) and realistic photo-of-face attacks (via noise kurtosis,
    block gradient variance, and local noise variance).

    Args:
        crop: BGR face crop (any size, will be resized internally).

    Returns:
        dict with keys:
            - is_live (bool):            True if texture looks like real skin
            - hist_var (float):          LBP histogram variance
            - hf_energy (float):         High-frequency energy ratio
            - noise_kurtosis (float):    Noise residual kurtosis
            - block_grad_var (float):    Block gradient variance
            - local_noise_var (float):   Local noise level variance
            - reason (str):              Human-readable explanation
    """
    # Normalise to consistent size for comparable metrics
    resized = cv2.resize(crop, (128, 128))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 1. LBP histogram variance (catches flat/uniform images)
    lbp = _compute_lbp(gray)
    hist = _lbp_histogram(lbp)
    hist_var = float(np.var(hist))

    # 2. High-frequency energy (catches very smooth images)
    hf = _high_freq_energy(gray)

    # 3. Noise kurtosis (catches screen/print recapture)
    n_kurt = _noise_kurtosis(gray)

    # 4. Block gradient variance (catches flat gradient photos)
    bg_var = _block_gradient_variance(gray)

    # 5. Local noise level variance (catches uniform-noise recaptures)
    ln_var = _local_noise_variance(gray)

    # All five must pass
    var_ok = hist_var >= SPOOF_LBP_HIST_VARIANCE_THRESHOLD
    hf_ok = hf >= SPOOF_LBP_HF_ENERGY_THRESHOLD
    kurt_ok = n_kurt >= SPOOF_NOISE_KURTOSIS_THRESHOLD
    grad_ok = bg_var >= SPOOF_BLOCK_GRAD_VAR_THRESHOLD
    noise_ok = ln_var >= SPOOF_LOCAL_NOISE_VAR_THRESHOLD
    is_live = var_ok and hf_ok and kurt_ok and grad_ok and noise_ok

    # Always log all values for debugging (even on pass)
    log.info(
        "Texture features: hist_var=%.6f  hf=%.4f  kurtosis=%.1f"
        "  block_grad=%.0f  local_noise_var=%.2f",
        hist_var, hf, n_kurt, bg_var, ln_var,
    )

    if is_live:
        reason = "Texture OK"
    else:
        parts = []
        if not var_ok:
            parts.append(
                f"hist_var={hist_var:.6f} < {SPOOF_LBP_HIST_VARIANCE_THRESHOLD:.6f}"
            )
        if not hf_ok:
            parts.append(
                f"hf_energy={hf:.4f} < {SPOOF_LBP_HF_ENERGY_THRESHOLD:.4f}"
            )
        if not kurt_ok:
            parts.append(
                f"noise_kurtosis={n_kurt:.1f} < {SPOOF_NOISE_KURTOSIS_THRESHOLD:.1f}"
            )
        if not grad_ok:
            parts.append(
                f"block_grad_var={bg_var:.0f} < {SPOOF_BLOCK_GRAD_VAR_THRESHOLD:.0f}"
            )
        if not noise_ok:
            parts.append(
                f"local_noise_var={ln_var:.2f} < {SPOOF_LOCAL_NOISE_VAR_THRESHOLD:.1f}"
            )
        reason = "Texture spoof — " + ", ".join(parts)
        log.warning("Texture FAIL: %s", reason)

    return {
        "is_live": is_live,
        "hist_var": round(hist_var, 8),
        "hf_energy": round(hf, 6),
        "noise_kurtosis": round(n_kurt, 2),
        "block_grad_var": round(bg_var, 1),
        "local_noise_var": round(ln_var, 3),
        "reason": reason,
    }
