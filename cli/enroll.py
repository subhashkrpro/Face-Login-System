"""CLI command: enroll — guided face registration."""

import time
from concurrent.futures import ThreadPoolExecutor

from config import MAX_FACES
from src.logger import get_logger

log = get_logger(__name__)


def cmd_enroll(args):
    """Enroll a face using guided capture with pose diversity."""
    from src.recognition import FaceRecognizer
    from src.recognition.enrollment import GuidedEnrollment
    from debug import DebugSaver

    debug = DebugSaver(enabled=getattr(args, "debug", False))

    # Create recognizer first — its detector's landmarker is shared with
    # GuidedEnrollment so TFLite only loads once (~9s saved).
    recognizer = FaceRecognizer(max_faces=MAX_FACES, debug_saver=debug)

    # Share the detector's FaceLandmarker if available (MediaPipeDetector)
    shared_lm = getattr(recognizer.detector, 'landmarker', None)
    guide = GuidedEnrollment(num_frames=args.frames, landmarker=shared_lm)

    try:
        frames = guide.capture(args.name)
    finally:
        guide.close()

    if not frames:
        log.warning("✗ Enrollment cancelled or no frames captured")
        recognizer.close()
        return

    log.info("Processing %d captured frames ...", len(frames))
    ok = recognizer.enroll_multi(args.name, frames)

    if ok:
        log.info("✓ '%s' enrolled successfully (%d diverse frames)", args.name, len(frames))
    else:
        log.error("✗ Enrollment failed — no face detected or duplicate")

    recognizer.close()
