"""CLI command: enhance — capture + detect + CLAHE + SR."""

import time

from config import CAMERA_WARMUP_SEC, CAPTURE_TIMEOUT_SEC
from src.logger import get_logger

log = get_logger(__name__)


def cmd_enhance(args):
    """Capture + enhance faces."""
    from src.capture import FastStream
    from src.enhancement import FaceEnhancer
    from debug import DebugSaver

    debug = DebugSaver(enabled=getattr(args, "debug", False))

    sr_kwargs = {}
    if args.backend == "openvino" and args.device:
        sr_kwargs["device"] = args.device

    enhancer = FaceEnhancer(
        min_confidence=args.confidence,
        sr_backend=args.backend,
        debug_saver=debug,
        **sr_kwargs,
    )

    stream = FastStream(buffer_size=args.frames).start()
    try:
        time.sleep(CAMERA_WARMUP_SEC)
        frames = stream.capture_frames(timeout=CAPTURE_TIMEOUT_SEC)
    finally:
        stream.stop()

    log.info("Captured %d frames, processing ...", len(frames))
    for idx, frame in enumerate(frames):
        enhancer.process_frame(frame, frame_id=idx)

    enhancer.close()
