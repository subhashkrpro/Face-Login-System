"""Centralized logging for the LOGIN pipeline."""

from .setup import get_logger, setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
]
