"""Face embedding extraction (ArcFace + alignment + model loading)."""

from .arcface import ArcFaceEmbedder
from .align import align_face
from .model_loader import ensure_arcface_model

__all__ = [
    "ArcFaceEmbedder",
    "align_face",
    "ensure_arcface_model",
]
