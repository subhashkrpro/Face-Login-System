"""Logger configuration and factory.

Provides a single ``get_logger(name)`` function that all modules use.
Respects ``--verbose`` / ``--quiet`` CLI flags via ``setup_logging()``.

Log levels:
    quiet   → WARNING only (errors + warnings)
    default → INFO  (normal operation messages)
    verbose → DEBUG (detailed timing, internal state)
"""

import logging
import sys

# ── Formatter ─────────────────────────────────────────────────────────────

_FORMAT = "[%(levelname).1s] %(message)s"
_FORMAT_DEBUG = "[%(levelname).1s] %(name)s: %(message)s"

_configured = False


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure the root LOGIN logger. Call once at startup (from main.py).

    Args:
        verbose: Show DEBUG-level messages.
        quiet:   Suppress INFO, show only WARNING+.
    """
    global _configured

    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = _FORMAT_DEBUG if verbose else _FORMAT

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("login")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the ``login`` namespace.

    Usage::

        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("Model loaded")
        log.debug("Embedding shape: %s", emb.shape)
    """
    global _configured
    if not _configured:
        setup_logging()  # sensible defaults if main.py hasn't called it yet

    return logging.getLogger(f"login.{name}")
