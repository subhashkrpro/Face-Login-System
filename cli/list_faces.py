"""CLI command: list — show all enrolled faces."""

from config import MAX_FACES
from src.logger import get_logger

log = get_logger(__name__)


def cmd_list(args):
    """List enrolled faces."""
    from src.recognition import FaceRecognizer

    recognizer = FaceRecognizer(max_faces=MAX_FACES)
    names = recognizer.list_enrolled()
    recognizer.close()

    if names:
        log.info("%d enrolled face(s):", len(names))
        for n in names:
            log.info("  • %s", n)
    else:
        log.info("No faces enrolled yet.")
