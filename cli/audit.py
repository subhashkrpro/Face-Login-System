"""CLI command: audit — scan database for duplicate faces."""

from config import MAX_FACES
from src.logger import get_logger

log = get_logger(__name__)


def cmd_audit(args):
    """Scan the enrolled database for duplicate faces."""
    from src.recognition import FaceRecognizer, DuplicateChecker

    recognizer = FaceRecognizer(max_faces=MAX_FACES)
    checker = DuplicateChecker(recognizer.index, threshold=args.threshold)

    names = recognizer.list_enrolled()
    log.info("Auditing %d enrolled face(s) (threshold: %.2f)...", len(names), args.threshold)

    duplicates = checker.audit()

    if duplicates:
        log.warning("Found %d duplicate pair(s):", len(duplicates))
        for d in duplicates:
            log.warning("  ⚠ '%s' ↔ '%s'   similarity: %.4f",
                        d["name_a"], d["name_b"], d["similarity"])
        log.info("Use 'delete --name <NAME>' to remove unwanted entries.")
    else:
        log.info("✓ No duplicates found — all enrolled faces are unique.")

    recognizer.close()
