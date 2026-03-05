"""Shared utilities."""

from .downloader import download_model
from .similarity import cosine_similarity

__all__ = ["download_model", "cosine_similarity"]
