"""Face recognition via ArcFace embeddings + 3D mesh geometry + HNSW index.

Three-stage recognition:
    Stage 1 — HNSW top-K candidates (fast, uses averaged ArcFace embedding)
    Stage 2 — Verify each candidate against individual frame embeddings
    Stage 3 — Combine ArcFace score with 3D mesh geometric similarity
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from config import (
    FACE_DB_PATH, FACE_FRAMES_DB_PATH, MESH_DB_PATH,
    RECOGNITION_THRESHOLD, MESH_WEIGHT, SPOOF_ENABLED,
    VERIFY_TOP_K, MAX_FACES, MIN_DETECTION_CONFIDENCE,
)
from src.detection import create_detector
from src.indexing import HNSWIndex
from config import SPOOF_LBP_ENABLED
from src.mesh import FaceMesh, mesh_to_embedding, check_liveness, check_texture
from src.utils import cosine_similarity
from src.logger import get_logger
from .embedding import ArcFaceEmbedder
from .enrollment import DuplicateChecker

if TYPE_CHECKING:
    from debug import DebugSaver
    from debug.recognition_debug import RecognitionDebugSaver

log = get_logger(__name__)


class FaceRecognizer:
    """
    Enroll and recognize faces using ArcFace 512-dim + 3D mesh 1434-dim embeddings.

    Uses HNSW index (hnswlib) for O(log n) nearest-neighbor search.
    Three databases are kept on disk:
        - face_db.json     — averaged ArcFace embeddings (synced with HNSW)
        - face_frames.json — individual frame ArcFace embeddings (stage-2 verify)
        - mesh_db.json     — averaged mesh embeddings (stage-3 geometric verify)

    Combined score = (1 - MESH_WEIGHT) × ArcFace_sim + MESH_WEIGHT × mesh_sim

    Workflow:
        1. enroll(name, frame)   — detect → align → ArcFace + mesh → store
        2. recognize(frame)      — HNSW top-K → verify vs frames → combine mesh
        3. Threshold-based matching via combined similarity
    """

    def __init__(
        self,
        threshold: float | None = None,
        max_faces: int | None = None,
        min_detection_confidence: float | None = None,
        debug_saver: DebugSaver | None = None,
        recognition_debug: RecognitionDebugSaver | None = None,
    ):
        t_init = time.perf_counter()
        self.threshold = threshold if threshold is not None else RECOGNITION_THRESHOLD
        self.mesh_weight = MESH_WEIGHT
        self.spoof_enabled = SPOOF_ENABLED
        self.lbp_enabled = SPOOF_LBP_ENABLED
        self._dbg = debug_saver
        self._rdbg = recognition_debug
        _max_faces = max_faces if max_faces is not None else MAX_FACES
        _min_conf = (
            min_detection_confidence
            if min_detection_confidence is not None
            else MIN_DETECTION_CONFIDENCE
        )

        # ── Parallel model loading ────────────────────────────────────
        # Detector (~9s) and ArcFace (~200ms) are independent — load
        # them concurrently so ArcFace init is hidden behind detector.
        with ThreadPoolExecutor(max_workers=2) as pool:
            det_future = pool.submit(
                create_detector, min_confidence=_min_conf, max_faces=_max_faces,
            )
            emb_future = pool.submit(ArcFaceEmbedder)

            # While models load in background, do fast sequential work:
            self.index = HNSWIndex()
            self.dup_checker = DuplicateChecker(self.index)
            self.db: dict[str, list[float]] = {}
            self.frames_db: dict[str, list[list[float]]] = {}
            self.mesh_db: dict[str, list[float]] = {}
            self._load_db()
            self._sync_json_to_hnsw()

            # Collect model results (blocks until both ready)
            self.detector = det_future.result()
            self.embedder = emb_future.result()

        # When using MediaPipeDetector, landmarks come free with detection,
        # so FaceMesh is only needed as fallback (e.g. BlazeFace backend).
        self._detector_provides_landmarks = hasattr(self.detector, 'landmarker')
        if self._detector_provides_landmarks:
            self.mesh_extractor = None
            log.info("Detector provides landmarks — skipping FaceMesh init")
        else:
            self.mesh_extractor = FaceMesh(max_faces=_max_faces)

        ms = (time.perf_counter() - t_init) * 1000
        log.info("FaceRecognizer ready (%.0fms)", ms)

    # ── Database I/O ──────────────────────────────────────────────────────

    def _load_db(self):
        """Load stored embeddings from disk."""
        if os.path.isfile(FACE_DB_PATH):
            try:
                with open(FACE_DB_PATH, "r") as f:
                    self.db = json.load(f)
                log.info("Loaded %d enrolled face(s)", len(self.db))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Corrupted face DB '%s': %s — starting fresh", FACE_DB_PATH, e)
                self.db = {}
        else:
            self.db = {}

        if os.path.isfile(FACE_FRAMES_DB_PATH):
            try:
                with open(FACE_FRAMES_DB_PATH, "r") as f:
                    self.frames_db = json.load(f)
                total = sum(len(v) for v in self.frames_db.values())
                log.info("Loaded %d individual frame embedding(s)", total)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Corrupted frames DB '%s': %s — starting fresh", FACE_FRAMES_DB_PATH, e)
                self.frames_db = {}
        else:
            self.frames_db = {}

        if os.path.isfile(MESH_DB_PATH):
            try:
                with open(MESH_DB_PATH, "r") as f:
                    self.mesh_db = json.load(f)
                log.info("Loaded %d mesh embedding(s)", len(self.mesh_db))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Corrupted mesh DB '%s': %s — starting fresh", MESH_DB_PATH, e)
                self.mesh_db = {}
        else:
            self.mesh_db = {}

    def _save_db(self):
        """Persist all three databases to JSON."""
        try:
            with open(FACE_DB_PATH, "w") as f:
                json.dump(self.db, f, indent=2)
            with open(FACE_FRAMES_DB_PATH, "w") as f:
                json.dump(self.frames_db, f, indent=2)
            with open(MESH_DB_PATH, "w") as f:
                json.dump(self.mesh_db, f, indent=2)
        except OSError as e:
            log.error("Failed to save face database: %s", e)

    def _sync_json_to_hnsw(self):
        """Ensure any JSON-only entries are also in the HNSW index."""
        synced = 0
        for name, vec in self.db.items():
            if name not in self.index.list_names():
                self.index.add(name, np.array(vec, dtype=np.float32))
                synced += 1
        if synced:
            log.info("Synced %d face(s) from JSON -> HNSW index", synced)

    def list_enrolled(self) -> list[str]:
        """Return names of all enrolled faces."""
        return self.index.list_names()

    def delete(self, name: str) -> bool:
        """Remove an enrolled face from HNSW index and all JSON dbs."""
        deleted = self.index.delete(name)
        removed_json = False
        if name in self.db:
            del self.db[name]
            removed_json = True
        if name in self.frames_db:
            del self.frames_db[name]
            removed_json = True
        if name in self.mesh_db:
            del self.mesh_db[name]
            removed_json = True
        if deleted or removed_json:
            self._save_db()
        if deleted:
            log.info("Deleted '%s' from database", name)
        return deleted

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_embedding(self, frame: np.ndarray, face: dict,
                        frame_id: int = 0, face_id: int = 0) -> np.ndarray:
        """Get ArcFace embedding — aligned if keypoints available, else crop."""
        if face["keypoints"]:
            from .embedding import align_face
            aligned = align_face(frame, face["keypoints"])
            # Debug: aligned face (112×112)
            if self._dbg:
                self._dbg.save_aligned(aligned, frame_id=frame_id, face_id=face_id)
            return self.embedder.get_embedding_aligned(frame, face["keypoints"])
        crop = face["crop"]
        # Debug: save crop as "aligned" fallback
        if self._dbg:
            self._dbg.save_aligned(crop, frame_id=frame_id, face_id=face_id)
        return self.embedder.get_embedding(crop)

    def _get_mesh_embedding(self, frame: np.ndarray, face: dict) -> np.ndarray | None:
        """
        Extract 3D mesh embedding from the face crop.

        If the detector already provided landmarks (MediaPipeDetector),
        those are used directly.  Otherwise falls back to FaceMesh.

        Returns a 1434-dim L2-normalized vector, or None if extraction fails.
        """
        # Prefer pre-extracted landmarks from MediaPipeDetector
        lms = face.get("landmarks")
        if lms is not None:
            return mesh_to_embedding(lms)

        crop = face.get("crop")
        if crop is None or crop.size == 0:
            return None
        if self.mesh_extractor is None:
            return None
        try:
            meshes = self.mesh_extractor.extract(crop)
            if not meshes:
                log.debug("Mesh extraction found no face in crop")
                return None
            return mesh_to_embedding(meshes[0])
        except Exception as e:
            log.debug("Mesh extraction failed: %s", e)
            return None

    def _check_spoof(self, frame: np.ndarray, face: dict) -> dict | None:
        """
        Run anti-spoofing: z-depth + LBP texture on the face.

        Uses pre-extracted landmarks from MediaPipeDetector if available,
        otherwise falls back to FaceMesh.  LBP texture runs on crop if
        z-depth passes.

        Returns:
            liveness dict if spoofing is enabled, None if disabled.
        """
        if not self.spoof_enabled:
            return None

        crop = face.get("crop")

        # Prefer pre-extracted landmarks from MediaPipeDetector
        lms = face.get("landmarks")
        if lms is not None:
            result = check_liveness(lms)
            if not result["is_live"]:
                return result
            # LBP texture check
            if self.lbp_enabled and crop is not None and crop.size > 0:
                tex = check_texture(crop)
                if not tex["is_live"]:
                    return {
                        "is_live": False,
                        "z_range": result["z_range"], "z_std": result["z_std"],
                        **{k: tex[k] for k in tex if k not in ("is_live",)},
                    }
                result.update({k: tex[k] for k in tex if k not in ("is_live", "reason")})
            return result

        if crop is None or crop.size == 0:
            return {"is_live": False, "z_range": 0.0, "z_std": 0.0,
                    "reason": "No crop available"}
        if self.mesh_extractor is None:
            return {"is_live": False, "z_range": 0.0, "z_std": 0.0,
                    "reason": "No mesh extractor available"}
        try:
            meshes = self.mesh_extractor.extract(crop)
            if not meshes:
                return {"is_live": False, "z_range": 0.0, "z_std": 0.0,
                        "reason": "No mesh landmarks detected"}
            result = check_liveness(meshes[0])
            if not result["is_live"]:
                return result
            # LBP texture check
            if self.lbp_enabled:
                tex = check_texture(crop)
                if not tex["is_live"]:
                    return {
                        "is_live": False,
                        "z_range": result["z_range"], "z_std": result["z_std"],
                        **{k: tex[k] for k in tex if k not in ("is_live",)},
                    }
                result.update({k: tex[k] for k in tex if k not in ("is_live", "reason")})
            return result
        except Exception as e:
            log.debug("Spoof check failed: %s", e)
            return {"is_live": False, "z_range": 0.0, "z_std": 0.0,
                    "reason": f"Error: {e}"}

    def _check_spoof_from_landmarks(
        self, landmarks: np.ndarray | None,
        crop: np.ndarray | None = None,
    ) -> dict | None:
        """
        Anti-spoofing: z-depth + LBP texture on pre-extracted data.

        Runs z-depth check on landmarks, then (if enabled) LBP texture
        analysis on the face crop. Both must pass for "is_live" = True.
        """
        if not self.spoof_enabled:
            return None
        if landmarks is None:
            return {"is_live": False, "z_range": 0.0, "z_std": 0.0,
                    "reason": "No mesh landmarks available"}

        result = check_liveness(landmarks)
        if not result["is_live"]:
            return result

        # LBP texture check (only if z-depth passed)
        if self.lbp_enabled and crop is not None and crop.size > 0:
            tex = check_texture(crop)
            if not tex["is_live"]:
                return {
                    "is_live": False,
                    "z_range": result["z_range"],
                    "z_std": result["z_std"],
                    **{k: tex[k] for k in tex if k not in ("is_live",)},
                }
            # Merge texture metrics into result
            result.update({k: tex[k] for k in tex if k not in ("is_live", "reason")})

        return result

    def _verify_against_frames(
        self,
        embedding: np.ndarray,
        name: str,
        mesh_emb: np.ndarray | None = None,
    ) -> float:
        """
        Stage 2+3: combine ArcFace frame similarity with mesh geometric similarity.

        Stage 2 — compare ArcFace *embedding* against all individual frame
                   embeddings stored for *name* → best cosine similarity.
        Stage 3 — compare *mesh_emb* against stored average mesh embedding.

        Combined score = (1 - mesh_weight) × arcface_sim + mesh_weight × mesh_sim

        Falls back to ArcFace-only if mesh data is unavailable.
        """
        # ── Stage 2: ArcFace frame verification ──
        arcface_sim = 0.0
        frame_vecs = self.frames_db.get(name)
        if frame_vecs:
            sims = [
                cosine_similarity(embedding, np.array(fv, dtype=np.float32))
                for fv in frame_vecs
            ]
            arcface_sim = float(max(sims))
        else:
            # Fallback: compare against averaged embedding
            avg = self.db.get(name)
            if avg is not None:
                arcface_sim = float(
                    cosine_similarity(embedding, np.array(avg, dtype=np.float32))
                )

        # ── Stage 3: Mesh geometric verification ──
        stored_mesh = self.mesh_db.get(name)
        if mesh_emb is not None and stored_mesh is not None:
            mesh_sim = float(
                cosine_similarity(mesh_emb, np.array(stored_mesh, dtype=np.float64))
            )
            combined = (1 - self.mesh_weight) * arcface_sim + self.mesh_weight * mesh_sim
            log.debug(
                "Verify '%s': arcface=%.3f  mesh=%.3f  combined=%.3f",
                name, arcface_sim, mesh_sim, combined,
            )
            return combined

        # No mesh data — ArcFace only
        return arcface_sim

    # ── Enroll ────────────────────────────────────────────────────────────

    def enroll(self, name: str, frame: np.ndarray) -> bool:
        """
        Enroll a face from a single BGR frame.

        Anti-spoofing check runs first, then ArcFace + mesh embeddings are stored.
        """
        t0 = time.perf_counter()
        faces = self.detector.detect(frame)

        if not faces:
            log.warning("No face detected for '%s'", name)
            return False

        face = faces[0]

        # Anti-spoofing: reject flat/photo faces
        spoof = self._check_spoof(frame, face)
        if spoof and not spoof["is_live"]:
            ms = (time.perf_counter() - t0) * 1000
            log.warning(
                "Spoof rejected for '%s': %s (%.1fms)",
                name, spoof["reason"], ms,
            )
            return False

        embedding = self._get_embedding(frame, face, frame_id=0, face_id=0)

        # Reject if this face already belongs to someone else
        dup = self.dup_checker.is_duplicate(embedding, exclude_name=name)
        if dup:
            ms = (time.perf_counter() - t0) * 1000
            log.warning("Rejected — face already registered as '%s' (%.1fms)", dup, ms)
            return False

        # Extract mesh embedding (geometry)
        mesh_emb = self._get_mesh_embedding(frame, face)

        self.db[name] = embedding.tolist()
        self.frames_db[name] = [embedding.tolist()]
        if mesh_emb is not None:
            self.mesh_db[name] = mesh_emb.tolist()
            log.debug("Stored mesh embedding for '%s' (1434-dim)", name)
        self._save_db()
        self.index.add(name, embedding)

        ms = (time.perf_counter() - t0) * 1000
        log.info("'%s' enrolled (%.1fms, 1 frame, ArcFace+Mesh)", name, ms)
        return True

    def enroll_multi(self, name: str, frames: list[np.ndarray]) -> bool:
        """
        Enroll using multiple frames.

        Anti-spoofing runs per-frame — frames that fail are silently skipped.
        At least one live frame is required.
        """
        embeddings = []
        mesh_embeddings = []
        for idx, frame in enumerate(frames):
            faces = self.detector.detect(frame)
            if not faces:
                continue
            face = faces[0]

            # Per-frame spoof check
            spoof = self._check_spoof(frame, face)
            if spoof and not spoof["is_live"]:
                log.debug(
                    "Skipping frame %d for '%s': %s", idx, name, spoof["reason"]
                )
                continue

            embeddings.append(
                self._get_embedding(frame, face, frame_id=idx, face_id=0)
            )
            m = self._get_mesh_embedding(frame, face)
            if m is not None:
                mesh_embeddings.append(m)

        if not embeddings:
            log.warning("No live faces detected in any frame for '%s'", name)
            return False

        avg = np.mean(embeddings, axis=0)
        avg /= np.linalg.norm(avg)  # re-normalize

        # Reject if this face already belongs to someone else
        dup = self.dup_checker.is_duplicate(avg, exclude_name=name)
        if dup:
            log.warning("Rejected — face already registered as '%s'", dup)
            return False

        # Store ArcFace embeddings
        self.db[name] = avg.tolist()
        self.frames_db[name] = [e.tolist() for e in embeddings]

        # Store averaged mesh embedding
        if mesh_embeddings:
            mesh_avg = np.mean(mesh_embeddings, axis=0)
            mesh_avg /= np.linalg.norm(mesh_avg)  # re-normalize
            self.mesh_db[name] = mesh_avg.tolist()
            log.debug(
                "Stored averaged mesh embedding for '%s' (%d frames)",
                name, len(mesh_embeddings),
            )

        self._save_db()
        self.index.add(name, avg)

        log.info(
            "'%s' enrolled (%d frames, ArcFace+Mesh)", name, len(embeddings)
        )
        return True

    # ── Recognize ─────────────────────────────────────────────────────────

    def recognize(self, frame: np.ndarray) -> list[dict]:
        """
        Three-stage face recognition with anti-spoofing.

        Anti-spoof: Check 3D depth → reject flat/photo faces
        Stage 1:    HNSW search → top-K candidates (fast, averaged ArcFace)
        Stage 2:    Verify each candidate against individual frame ArcFace
        Stage 3:    Combine with mesh geometric similarity for final score

        Combined = (1 - MESH_WEIGHT) × ArcFace + MESH_WEIGHT × Mesh

        Returns list of dicts:
            {"name": str | "unknown" | "spoof",
             "similarity": float, "bbox": (x, y, w, h)}
        """
        t0 = time.perf_counter()
        faces = self.detector.detect(frame)

        if not faces:
            return []

        # Debug: raw frame + detection overlay
        if self._dbg:
            self._dbg.save_raw(frame, frame_id=0)
            self._dbg.save_detected(frame, faces, frame_id=0)
        if self._rdbg:
            self._rdbg.save_raw(frame, frame_id=0)
            self._rdbg.save_detected(frame, faces, frame_id=0)

        results = []
        for face_idx, face in enumerate(faces):
            crop = face.get("crop")

            # ── Extract mesh landmarks ONCE — reuse everywhere ──
            # MediaPipeDetector already provides landmarks per face.
            landmarks = face.get("landmarks")
            if landmarks is None and crop is not None and crop.size > 0:
                # Fallback: extract via separate FaceMesh (BlazeFace backend)
                if self.mesh_extractor is not None:
                    try:
                        meshes = self.mesh_extractor.extract(crop)
                        if meshes:
                            landmarks = meshes[0]
                    except Exception as e:
                        log.debug("Mesh extraction failed: %s", e)

            # Recognition debug: raw crop
            if self._rdbg and crop is not None:
                self._rdbg.save_crop(crop, frame_id=0, face_id=face_idx)

            # Anti-spoofing gate: z-depth + LBP texture
            spoof = self._check_spoof_from_landmarks(landmarks, crop=crop)

            # Recognition debug: spoof analysis with z-depth heatmap
            if self._rdbg and crop is not None:
                self._rdbg.save_spoof(
                    crop, spoof, landmarks=landmarks,
                    frame_id=0, face_id=face_idx,
                )

            if spoof and not spoof["is_live"]:
                log.warning(
                    "Spoof detected (face %d): %s", face_idx, spoof["reason"]
                )
                # Recognition debug: spoof result
                if self._rdbg:
                    self._rdbg.save_result(
                        frame, "spoof", 0.0, face["bbox"],
                        spoof_info=spoof, frame_id=0, face_id=face_idx,
                    )
                results.append({
                    "name": "spoof",
                    "similarity": 0.0,
                    "bbox": face["bbox"],
                    "spoof_info": spoof,
                })
                continue

            # Recognition debug: gamma + CLAHE (show what recognition sees)
            if self._rdbg and crop is not None:
                from src.enhancement.low_light import auto_brighten
                from src.enhancement.clahe import apply_clahe
                gamma_img = auto_brighten(crop)
                self._rdbg.save_gamma(gamma_img, frame_id=0, face_id=face_idx)
                clahe_img = apply_clahe(gamma_img)
                self._rdbg.save_clahe(clahe_img, frame_id=0, face_id=face_idx)

            embedding = self._get_embedding(frame, face,
                                             frame_id=0, face_id=face_idx)

            # Mesh embedding from already-extracted landmarks (no re-extract)
            mesh_emb = None
            if landmarks is not None:
                mesh_emb = mesh_to_embedding(landmarks)

            # Debug: 3D mesh on face crop (pass landmarks — no re-extract)
            if self._dbg:
                self._dbg.save_mesh(
                    face["crop"], landmarks=landmarks,
                    frame_id=0, face_id=face_idx,
                )
            if self._rdbg:
                self._rdbg.save_mesh(
                    face["crop"], landmarks=landmarks,
                    frame_id=0, face_id=face_idx,
                )

            # Stage 1: HNSW fast search → top-K candidates
            candidates = self.index.search(embedding, k=VERIFY_TOP_K)

            # Recognition debug: HNSW candidates
            if self._rdbg and crop is not None:
                self._rdbg.save_candidates(
                    crop, candidates, frame_id=0, face_id=face_idx,
                )

            # Stage 2+3: Verify each candidate (ArcFace frames + mesh geometry)
            best_name = "unknown"
            best_sim = 0.0

            for cand in candidates:
                verified_sim = self._verify_against_frames(
                    embedding, cand["name"], mesh_emb=mesh_emb
                )
                if verified_sim > best_sim:
                    best_sim = verified_sim
                    best_name = cand["name"]

            if best_sim < self.threshold:
                best_name = "unknown"

            results.append({
                "name": best_name,
                "similarity": round(best_sim, 4),
                "bbox": face["bbox"],
            })

            # Debug: recognition result overlay
            if self._dbg:
                self._dbg.save_result(
                    frame, best_name, best_sim, face["bbox"],
                    frame_id=0, face_id=face_idx,
                )
            if self._rdbg:
                self._rdbg.save_result(
                    frame, best_name, best_sim, face["bbox"],
                    frame_id=0, face_id=face_idx,
                )

        ms = (time.perf_counter() - t0) * 1000
        for r in results:
            tag = f"{r['name']} ({r['similarity']:.3f})"
            log.debug("Recognize: %s in %.1fms", tag, ms)

        return results

    def close(self):
        self.detector.close()
        if self.mesh_extractor is not None:
            self.mesh_extractor.close()
