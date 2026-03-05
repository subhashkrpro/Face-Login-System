"""Convenience function for liveness check via blink detection."""

import time

import numpy as np

from config import LIVENESS_TIMEOUT_SEC
from src.logger import get_logger
from .blink_detector import BlinkDetector

log = get_logger(__name__)


def check_liveness(
    stream,
    timeout: float | None = None,
) -> tuple[bool, np.ndarray | None]:
    """
    Read frames from a FastStream and wait for a blink.

    Args:
        stream: A started FastStream instance.
        timeout: Max seconds to wait (default from config).

    Returns:
        (is_live, frame) — True + the frame captured at blink time,
        or (False, None) if no blink detected within timeout.
    """
    timeout = timeout if timeout is not None else LIVENESS_TIMEOUT_SEC
    detector = BlinkDetector()
    deadline = time.time() + timeout
    last_frame = None

    log.info("Please blink to verify you are a live person ... (timeout %.0fs)", timeout)

    while time.time() < deadline:
        frame = stream.read()
        if frame is None:
            continue
        last_frame = frame

        ear, blinked = detector.update(frame)
        if ear is not None:
            status = "CLOSED" if ear < detector.ear_threshold else "open"
            log.debug("EAR=%.3f [%s]  blinks=%d", ear, status, detector.blink_count)

        if blinked:
            log.info("Blink detected! Liveness confirmed.")
            detector.close()
            return True, last_frame

    log.warning("No blink detected within %.0fs — possible spoof.", timeout)
    detector.close()
    return False, None
