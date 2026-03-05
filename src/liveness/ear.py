"""Eye Aspect Ratio (EAR) calculation for blink detection."""

import numpy as np

# MediaPipe 478-landmark indices for EAR calculation
# Right eye: outer corner, upper-outer, upper-inner, inner corner, lower-inner, lower-outer
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
# Left eye (mirrored)
LEFT_EYE = (362, 385, 387, 263, 380, 373)


def compute_ear(landmarks: np.ndarray, indices: tuple[int, ...]) -> float:
    """
    Compute Eye Aspect Ratio for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-8:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))
