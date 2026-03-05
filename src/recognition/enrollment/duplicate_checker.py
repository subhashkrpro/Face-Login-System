"""Duplicate face detection — prevents enrolling already-registered faces."""

import numpy as np

from config import DUPLICATE_THRESHOLD
from src.indexing import HNSWIndex
from src.utils import cosine_similarity


class DuplicateChecker:
    """
    Check whether a face embedding already exists in the HNSW index.

    Uses cosine similarity against all enrolled embeddings.
    A match above DUPLICATE_THRESHOLD is considered a duplicate.
    """

    def __init__(self, index: HNSWIndex, threshold: float | None = None):
        self.index = index
        self.threshold = threshold if threshold is not None else DUPLICATE_THRESHOLD

    def is_duplicate(
        self,
        embedding: np.ndarray,
        exclude_name: str | None = None,
    ) -> str | None:
        """
        Check if *embedding* matches an already-enrolled face.

        Args:
            embedding:    512-dim ArcFace embedding to check.
            exclude_name: Skip this name (allows re-enrollment under same name).

        Returns:
            The name of the matching person if duplicate, else ``None``.
        """
        if self.index.count == 0:
            return None

        matches = self.index.search(embedding, k=1)
        if matches:
            best = matches[0]
            if best["similarity"] >= self.threshold:
                if exclude_name and best["name"] == exclude_name:
                    return None  # same person re-enrolling — allowed
                return best["name"]
        return None

    def audit(self) -> list[dict]:
        """
        Compare every enrolled face against every other.

        Returns a list of duplicate pairs:
            [{"name_a": str, "name_b": str, "similarity": float}, ...]
        sorted by similarity descending.
        """
        names = self.index.list_names()
        if len(names) < 2:
            return []

        # Collect all embeddings
        embeddings: dict[str, np.ndarray] = {}
        for name in names:
            vec = self.index.get_embedding(name)
            if vec is not None:
                embeddings[name] = np.array(vec, dtype=np.float32)

        ordered = list(embeddings.keys())
        duplicates = []

        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                name_a, name_b = ordered[i], ordered[j]
                sim = cosine_similarity(embeddings[name_a], embeddings[name_b])
                if sim >= self.threshold:
                    duplicates.append({
                        "name_a": name_a,
                        "name_b": name_b,
                        "similarity": round(float(sim), 4),
                    })

        duplicates.sort(key=lambda d: d["similarity"], reverse=True)
        return duplicates
