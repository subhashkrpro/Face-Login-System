"""CLI command: delete — remove an enrolled face."""

from config import MAX_FACES
from src.logger import get_logger

log = get_logger(__name__)


def cmd_delete(args):
    """Delete an enrolled face."""
    from src.recognition import FaceRecognizer

    recognizer = FaceRecognizer(max_faces=MAX_FACES)
    if recognizer.delete(args.name):
        log.info("✓ '%s' deleted", args.name)
    else:
        log.warning("✗ '%s' not found in database", args.name)
    recognizer.close()
