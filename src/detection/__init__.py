"""Face detection module."""

from .detector import BlazeFaceDetector
from .mediapipe_detector import MediaPipeDetector


def create_detector(
    backend: str | None = None,
    min_confidence: float | None = None,
    max_faces: int | None = None,
):
    """Factory: create a face detector by backend name."""
    from config import DETECTOR_BACKEND
    backend = backend or DETECTOR_BACKEND

    if backend == "mediapipe":
        return MediaPipeDetector(
            min_confidence=min_confidence,
            max_faces=max_faces,
        )
    elif backend == "blazeface":
        return BlazeFaceDetector(min_confidence=min_confidence)
    else:
        raise ValueError(f"Unknown detector backend: {backend!r}")


__all__ = ["BlazeFaceDetector", "MediaPipeDetector", "create_detector"]
