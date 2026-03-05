
"""
LOGIN — Face Capture, Enhancement & Recognition
=================================================
Subcommands:
    enhance   — Capture frames, detect + CLAHE + SR, save enhanced faces
    enroll    — Enroll a face (ArcFace embedding) under a name
    recognize — Recognize face from live camera against enrolled DB
    list      — List all enrolled faces
    delete    — Delete an enrolled face
    audit     — Scan database for duplicate faces

Usage:
    uv run main.py enhance
    uv run main.py enroll --name Alice
    uv run main.py recognize
    uv run main.py list
    uv run main.py delete --name Alice
    uv run main.py audit
"""

import argparse
import sys

from config import (
    DEFAULT_SR_BACKEND,
    MIN_DETECTION_CONFIDENCE,
    DEFAULT_CAPTURE_FRAMES,
    RECOGNITION_THRESHOLD,
    DUPLICATE_THRESHOLD,
    ENROLL_FRAMES,
    LIVENESS_ENABLED,
)

from cli.enhance import cmd_enhance
from cli.enroll import cmd_enroll
from cli.recognize import cmd_recognize
from cli.list_faces import cmd_list
from cli.delete import cmd_delete
from cli.audit import cmd_audit


def main():
    parser = argparse.ArgumentParser(description="LOGIN — Face Capture, Enhancement & Recognition")

    # Global flags
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Show debug messages")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Show warnings/errors only")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate pipeline images to debug/ folder")

    sub = parser.add_subparsers(dest="command", required=True)

    # enhance
    p_enh = sub.add_parser("enhance", help="Capture + enhance faces")
    p_enh.add_argument("--backend", default=DEFAULT_SR_BACKEND, choices=["realesrgan", "openvino", "gfpgan"])
    p_enh.add_argument("--device", default=None, help="OpenVINO device: NPU, GPU, CPU")
    p_enh.add_argument("--confidence", type=float, default=MIN_DETECTION_CONFIDENCE)
    p_enh.add_argument("--frames", type=int, default=DEFAULT_CAPTURE_FRAMES)

    # enroll
    p_enr = sub.add_parser("enroll", help="Enroll a face with guided capture")
    p_enr.add_argument("--name", required=True, help="Name to register")
    p_enr.add_argument("--frames", type=int, default=ENROLL_FRAMES,
                       help=f"Number of frames to capture (default: {ENROLL_FRAMES})")

    # recognize
    p_rec = sub.add_parser("recognize", help="Recognize face from camera")
    p_rec.add_argument("--threshold", type=float, default=RECOGNITION_THRESHOLD, help="Match threshold (0.0-1.0)")
    liveness_grp = p_rec.add_mutually_exclusive_group()
    liveness_grp.add_argument("--liveness", dest="liveness", action="store_true", default=None,
                              help="Force liveness check on")
    liveness_grp.add_argument("--no-liveness", dest="liveness", action="store_false",
                              help="Skip liveness check")

    # list
    sub.add_parser("list", help="List enrolled faces")

    # delete
    p_del = sub.add_parser("delete", help="Delete an enrolled face")
    p_del.add_argument("--name", required=True)

    # audit
    p_aud = sub.add_parser("audit", help="Scan database for duplicate faces")
    p_aud.add_argument("--threshold", type=float, default=DUPLICATE_THRESHOLD,
                       help=f"Similarity threshold to flag as duplicate (default: {DUPLICATE_THRESHOLD})")

    args = parser.parse_args()

    # ── Setup logging & validate config ───────────────────────────────
    from src.logger import setup_logging, get_logger
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    log = get_logger(__name__)

    from config.validation import validate_config
    from src.exceptions import (
        ConfigValidationError, CameraError, ModelError, FaceError,
    )
    try:
        validate_config()
    except ConfigValidationError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    cmds = {
        "enhance": cmd_enhance,
        "enroll": cmd_enroll,
        "recognize": cmd_recognize,
        "list": cmd_list,
        "delete": cmd_delete,
        "audit": cmd_audit,
    }

    try:
        cmds[args.command](args)
    except CameraError as e:
        log.error("Camera error: %s", e)
        sys.exit(2)
    except ModelError as e:
        log.error("Model error: %s", e)
        sys.exit(3)
    except FaceError as e:
        log.error("Face error: %s", e)
        sys.exit(4)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        log.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
