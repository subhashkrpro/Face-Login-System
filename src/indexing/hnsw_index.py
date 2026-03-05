"""HNSW-based vector index for fast face embedding search.

Uses hnswlib for O(log n) approximate nearest neighbor search,
supporting incremental inserts without full rebuild.

The index is persisted to disk alongside a JSON label map
so that numeric IDs can be resolved back to enrolled names.
"""

import json
import os

import numpy as np
import hnswlib

from config import (
    HNSW_INDEX_PATH, HNSW_LABELS_PATH,
    HNSW_SPACE, HNSW_EF_CONSTRUCTION, HNSW_M,
    HNSW_EF_SEARCH, HNSW_MAX_ELEMENTS,
    ARCFACE_EMBEDDING_DIM,
)
from src.logger import get_logger

log = get_logger(__name__)


class HNSWIndex:
    """
    HNSW approximate nearest-neighbor index for face embeddings.

    Supports:
        - add(name, embedding)  — insert / update a face
        - search(embedding, k)  — find top-k nearest faces
        - delete(name)          — remove a face (mark deleted)
        - Automatic persistence to disk (index.bin + labels.json)

    All config (space, ef, M, capacity) comes from config/defaults.py.
    """

    def __init__(
        self,
        index_path: str | None = None,
        labels_path: str | None = None,
    ):
        self._index_path = index_path or HNSW_INDEX_PATH
        self._labels_path = labels_path or HNSW_LABELS_PATH
        self._dim = ARCFACE_EMBEDDING_DIM

        # label_id <-> name mapping
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._next_id: int = 0

        # Build or load index
        self._index: hnswlib.Index = hnswlib.Index(space=HNSW_SPACE, dim=self._dim)

        if os.path.isfile(self._index_path) and os.path.isfile(self._labels_path):
            self._load()
        else:
            self._init_empty()

    # ── Init / Load / Save ────────────────────────────────────────────────

    def _init_empty(self):
        """Create a fresh empty index."""
        self._index.init_index(
            max_elements=HNSW_MAX_ELEMENTS,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        self._index.set_ef(HNSW_EF_SEARCH)
        self._name_to_id = {}
        self._id_to_name = {}
        self._next_id = 0
        log.info("New HNSW index (dim=%d, space=%s, M=%d)", self._dim, HNSW_SPACE, HNSW_M)

    def _load(self):
        """Load index + labels from disk."""
        try:
            self._index.load_index(self._index_path, max_elements=HNSW_MAX_ELEMENTS)
            self._index.set_ef(HNSW_EF_SEARCH)

            with open(self._labels_path, "r") as f:
                data = json.load(f)
            self._name_to_id = data["name_to_id"]
            self._id_to_name = {int(k): v for k, v in data["id_to_name"].items()}
            self._next_id = data["next_id"]

            log.info("Loaded HNSW index with %d face(s)", len(self._name_to_id))
        except (json.JSONDecodeError, KeyError, OSError, RuntimeError) as e:
            log.warning("Failed to load HNSW index: %s — creating fresh index", e)
            self._init_empty()

    def _save(self):
        """Persist index + labels to disk."""
        try:
            self._index.save_index(self._index_path)
            with open(self._labels_path, "w") as f:
                json.dump({
                    "name_to_id": self._name_to_id,
                    "id_to_name": {str(k): v for k, v in self._id_to_name.items()},
                    "next_id": self._next_id,
                }, f, indent=2)
        except OSError as e:
            log.error("Failed to save HNSW index: %s", e)

    # ── Public API ────────────────────────────────────────────────────────

    def add(self, name: str, embedding: np.ndarray):
        """
        Add or update a face embedding in the index.

        If the name already exists, the old entry is marked deleted
        and a new one is inserted (hnswlib supports mark_deleted).
        """
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        # Remove old entry if updating
        if name in self._name_to_id:
            old_id = self._name_to_id[name]
            self._index.mark_deleted(old_id)
            del self._id_to_name[old_id]

        new_id = self._next_id
        self._next_id += 1

        # Resize if needed
        if new_id >= self._index.get_max_elements():
            new_max = self._index.get_max_elements() * 2
            self._index.resize_index(new_max)
            log.debug("HNSW index resized to %d", new_max)

        self._index.add_items(emb, np.array([new_id]))
        self._name_to_id[name] = new_id
        self._id_to_name[new_id] = name
        self._save()

    def search(self, embedding: np.ndarray, k: int = 1) -> list[dict]:
        """
        Find the k nearest faces to the query embedding.

        Returns list of {"name": str, "distance": float, "similarity": float}.
        For cosine space, distance in [0, 2] and similarity = 1 - distance.
        """
        if self.count == 0:
            return []

        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        actual_k = min(k, self.count)

        ids, distances = self._index.knn_query(emb, k=actual_k)

        results = []
        for idx, dist in zip(ids[0], distances[0]):
            name = self._id_to_name.get(int(idx), "unknown")
            # hnswlib cosine distance = 1 - cosine_similarity
            similarity = 1.0 - float(dist)
            results.append({
                "name": name,
                "distance": float(dist),
                "similarity": similarity,
            })
        return results

    def delete(self, name: str) -> bool:
        """Remove a face from the index."""
        if name not in self._name_to_id:
            return False

        old_id = self._name_to_id[name]
        self._index.mark_deleted(old_id)
        del self._name_to_id[name]
        del self._id_to_name[old_id]
        self._save()
        log.info("Deleted '%s' from HNSW", name)
        return True

    def list_names(self) -> list[str]:
        """Return all enrolled names."""
        return list(self._name_to_id.keys())

    @property
    def count(self) -> int:
        """Number of active (non-deleted) faces in the index."""
        return len(self._name_to_id)

    def get_embedding(self, name: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a name."""
        if name not in self._name_to_id:
            return None
        idx = self._name_to_id[name]
        return self._index.get_items([idx])[0]
