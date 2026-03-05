"""Face alignment using similarity transform for ArcFace input."""

import cv2
import numpy as np

from config import ARCFACE_INPUT_SIZE

# Standard ArcFace alignment target for 112x112 crop
_ALIGN_DST = np.array(
    [
        [38.2946, 51.6963],   # left eye
        [73.5318, 51.5014],   # right eye
        [56.0252, 71.7366],   # nose tip
    ],
    dtype=np.float32,
)


def align_face(
    frame: np.ndarray,
    keypoints: list[tuple[float, float]],
    output_size: int | None = None,
) -> np.ndarray:
    """
    Align a face using 3 keypoints via similarity transform.

    Uses estimateAffinePartial2D (rotation + uniform scale + translation)
    which is more robust than a full affine with only 3 points.
    """
    output_size = output_size or ARCFACE_INPUT_SIZE
    src_pts = np.array(keypoints[:3], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2),
        _ALIGN_DST.reshape(-1, 1, 2),
    )
    if M is None:
        return cv2.resize(frame, (output_size, output_size))
    return cv2.warpAffine(frame, M, (output_size, output_size))
