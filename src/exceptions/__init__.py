"""Custom exceptions for the LOGIN pipeline."""

from .camera import CameraError, CameraOpenError, CameraTimeoutError
from .model import ModelError, ModelDownloadError, ModelNotFoundError, ModelExtractionError
from .face import FaceError, NoFaceDetectedError, DuplicateFaceError
from .config import ConfigValidationError

__all__ = [
    "CameraError",
    "CameraOpenError",
    "CameraTimeoutError",
    "ModelError",
    "ModelDownloadError",
    "ModelNotFoundError",
    "ModelExtractionError",
    "FaceError",
    "NoFaceDetectedError",
    "DuplicateFaceError",
    "ConfigValidationError",
]
