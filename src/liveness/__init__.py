"""Anti-spoofing / liveness detection module."""

from .ear import compute_ear
from .blink_detector import BlinkDetector
from .liveness_check import check_liveness

__all__ = [
    "compute_ear",
    "BlinkDetector",
    "check_liveness",
]
