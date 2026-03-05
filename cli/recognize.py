"""CLI command: recognize — face recognition from camera."""

import time

from config import MAX_FACES, CAMERA_WARMUP_SEC, LIVENESS_ENABLED
from src.logger import get_logger

log = get_logger(__name__)


def cmd_recognize(args):
    """Recognize face from live camera with optional liveness check."""
    from src.capture import FastStream
    from src.recognition import FaceRecognizer
    from src.liveness import check_liveness
    from debug import DebugSaver
    from debug.recognition_debug import RecognitionDebugSaver

    is_debug = getattr(args, "debug", False)
    debug = DebugSaver(enabled=is_debug)
    rec_debug = RecognitionDebugSaver(enabled=is_debug)

    # Start camera FIRST — it warms up while models load in parallel.
    stream = FastStream(buffer_size=1).start()
    t_cam = time.perf_counter()

    recognizer = FaceRecognizer(threshold=args.threshold, max_faces=MAX_FACES,
                                debug_saver=debug, recognition_debug=rec_debug)

    if not recognizer.list_enrolled():
        log.warning("No faces enrolled. Use 'enroll' first.")
        stream.stop()
        recognizer.close()
        return

    try:
        # Only sleep remaining warmup time (model loading already consumed some)
        elapsed = time.perf_counter() - t_cam
        remaining = CAMERA_WARMUP_SEC - elapsed
        if remaining > 0:
            time.sleep(remaining)

        use_liveness = args.liveness if args.liveness is not None else LIVENESS_ENABLED

        if use_liveness:
            is_live, frame = check_liveness(stream)

            if not is_live or frame is None:
                log.warning("Recognition aborted — liveness check failed.")
                return
        else:
            log.warning("Liveness check disabled — photo spoofing is possible.")
            frame = stream.read()
    finally:
        stream.stop()

    if frame is None:
        log.error("Failed to capture frame.")
        recognizer.close()
        return

    results = recognizer.recognize(frame)

    if not results:
        log.warning("No face detected in frame.")
    else:
        for r in results:
            if r["name"] == "unknown":
                log.info("→ Unknown face (best similarity: %.3f)", r["similarity"])
            else:
                log.info("→ Matched: %s (similarity: %.3f)", r["name"], r["similarity"])

    recognizer.close()
