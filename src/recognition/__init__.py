"""Face recognition module (ArcFace embeddings + recognizer)."""

from .embedding import ArcFaceEmbedder, align_face
from .enrollment import DuplicateChecker
from .recognizer import FaceRecognizer

__all__ = [
    "ArcFaceEmbedder",
    "align_face",
    "DuplicateChecker",
    "FaceRecognizer",
]
