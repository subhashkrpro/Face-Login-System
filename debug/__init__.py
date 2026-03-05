"""Debug module — saves intermediate pipeline images for inspection."""

from .debug_saver import DebugSaver
from .recognition_debug import RecognitionDebugSaver

__all__ = ["DebugSaver", "RecognitionDebugSaver"]
