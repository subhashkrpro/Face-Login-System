"""
DebugSaver — saves every intermediate image in the pipeline.

Creates a timestamped session folder under ``debug/`` each run so that
successive runs never overwrite each other.  Images are numbered by
pipeline stage so they appear in order when sorted:

    debug/session_20260301_153012/
        frame0_1_raw.png
        frame0_2_detected.png          # full frame with bbox rectangles
        frame0_face0_3_crop.png        # raw face crop
        frame0_face0_4_clahe.png       # after CLAHE
        frame0_face0_5_sr.png          # after super-resolution
        frame0_face0_6_mesh.png        # 3D mesh overlay
        frame0_face0_7_aligned.png     # 112×112 ArcFace-aligned
        frame0_face0_8_result.png      # annotated with name + similarity

Usage:
    from debug import DebugSaver

    dbg = DebugSaver()                 # creates session folder
    dbg.save_raw(frame, frame_id=0)
    dbg.save_detected(frame, faces, frame_id=0)
    dbg.save_crop(crop, frame_id=0, face_id=0)
    dbg.save_clahe(img, frame_id=0, face_id=0)
    dbg.save_sr(img, frame_id=0, face_id=0)
    dbg.save_mesh(img, landmarks, frame_id=0, face_id=0)
    dbg.save_aligned(img, frame_id=0, face_id=0)
    dbg.save_result(img, name, similarity, bbox, frame_id=0, face_id=0)
"""

import os
from datetime import datetime

import cv2
import numpy as np

from src.logger import get_logger

log = get_logger(__name__)

# MediaPipe FaceMesh connectivity for drawing the 478-landmark mesh
# (subset of FACEMESH_TESSELATION – we only draw the tesselation edges)
_MESH_CONNECTIONS: list[tuple[int, int]] | None = None


def _get_mesh_connections() -> list[tuple[int, int]]:
    """Lazy-load MediaPipe face-mesh tesselation edges."""
    global _MESH_CONNECTIONS
    if _MESH_CONNECTIONS is None:
        try:
            from mediapipe.python.solutions.face_mesh_connections import (
                FACEMESH_TESSELATION,
            )
            _MESH_CONNECTIONS = list(FACEMESH_TESSELATION)
        except ImportError:
            _MESH_CONNECTIONS = []
    return _MESH_CONNECTIONS


class DebugSaver:
    """
    Saves pipeline-stage images into a timestamped session folder.

    Parameters
    ----------
    enabled : bool
        When ``False`` every save method is a no-op (zero overhead).
    base_dir : str | None
        Root directory for debug output.  Defaults to ``<project>/debug``.
    """

    def __init__(self, enabled: bool = True, base_dir: str | None = None):
        self.enabled = enabled
        if not enabled:
            return

        if base_dir is None:
            base_dir = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                "debug",
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        log.info("Debug images → %s", self.session_dir)

    # ── helpers ───────────────────────────────────────────────────────────

    def _write(self, filename: str, img: np.ndarray) -> str | None:
        """Write *img* (BGR uint8) to session folder.  Returns path."""
        if not self.enabled:
            return None
        path = os.path.join(self.session_dir, filename)
        try:
            cv2.imwrite(path, img)
            log.debug("Debug saved: %s", filename)
        except Exception as exc:
            log.warning("Debug save failed (%s): %s", filename, exc)
            return None
        return path

    # ── stage 1: raw frame ────────────────────────────────────────────────

    def save_raw(self, frame: np.ndarray, frame_id: int = 0) -> str | None:
        """Save the original camera frame (unchanged)."""
        return self._write(f"frame{frame_id}_1_raw.png", frame)

    # ── stage 2: detection overlay ────────────────────────────────────────

    def save_detected(
        self,
        frame: np.ndarray,
        faces: list[dict],
        frame_id: int = 0,
    ) -> str | None:
        """Draw bounding boxes + confidence on the full frame and save."""
        if not self.enabled:
            return None

        vis = frame.copy()
        for face in faces:
            x, y, w, h = face["bbox"]
            conf = face.get("confidence", 0.0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis, f"{conf:.2f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            # Draw keypoints if available
            kps = face.get("keypoints")
            if kps:
                for px, py in kps:
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)

        return self._write(f"frame{frame_id}_2_detected.png", vis)

    # ── stage 3: raw face crop ────────────────────────────────────────────

    def save_crop(
        self, crop: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the raw face crop (before any enhancement)."""
        return self._write(
            f"frame{frame_id}_face{face_id}_3_crop.png", crop,
        )

    # ── stage 4: CLAHE enhanced ───────────────────────────────────────────

    def save_clahe(
        self, img: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the face after CLAHE contrast enhancement."""
        return self._write(
            f"frame{frame_id}_face{face_id}_4_clahe.png", img,
        )

    # ── stage 5: super-resolved ───────────────────────────────────────────

    def save_sr(
        self, img: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the face after super-resolution upscaling."""
        return self._write(
            f"frame{frame_id}_face{face_id}_5_sr.png", img,
        )

    # ── stage 6: 3D face mesh ─────────────────────────────────────────────

    def save_mesh(
        self,
        img: np.ndarray,
        landmarks: np.ndarray | None = None,
        frame_id: int = 0,
        face_id: int = 0,
    ) -> str | None:
        """
        Draw 478-point 3D face mesh on *img* and save.

        Parameters
        ----------
        img : np.ndarray
            BGR image (typically the raw crop or enhanced face).
        landmarks : np.ndarray | None
            (478, 3) normalised landmark array from ``FaceMesh.extract()``.
            If ``None`` mesh extraction is attempted automatically.
        """
        if not self.enabled:
            return None

        vis = img.copy()
        h, w = vis.shape[:2]

        if landmarks is None:
            # Try extracting mesh on the fly
            try:
                from src.mesh import FaceMesh
                fm = FaceMesh()
                meshes = fm.extract(vis)
                fm.close()
                if meshes:
                    landmarks = meshes[0]
            except Exception as exc:
                log.debug("Mesh extraction skipped: %s", exc)

        if landmarks is not None and len(landmarks) > 0:
            # Draw tesselation edges
            connections = _get_mesh_connections()
            pts = landmarks[:, :2]  # (478, 2) normalised x, y
            px_pts = (pts * np.array([w, h])).astype(np.int32)

            for i, j in connections:
                if i < len(px_pts) and j < len(px_pts):
                    cv2.line(vis, tuple(px_pts[i]), tuple(px_pts[j]),
                             (200, 200, 200), 1, cv2.LINE_AA)

            # Draw landmark points on top
            for pt in px_pts:
                cv2.circle(vis, tuple(pt), 1, (0, 255, 0), -1)

        return self._write(
            f"frame{frame_id}_face{face_id}_6_mesh.png", vis,
        )

    # ── stage 7: aligned face ─────────────────────────────────────────────

    def save_aligned(
        self, img: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the 112×112 ArcFace-aligned face."""
        return self._write(
            f"frame{frame_id}_face{face_id}_7_aligned.png", img,
        )

    # ── stage 8: recognition result ───────────────────────────────────────

    def save_result(
        self,
        frame: np.ndarray,
        name: str,
        similarity: float,
        bbox: tuple[int, int, int, int],
        frame_id: int = 0,
        face_id: int = 0,
    ) -> str | None:
        """
        Draw recognition result (name + similarity) on *frame* and save.

        Parameters
        ----------
        frame : np.ndarray
            The full camera frame (or a crop).
        name : str
            Matched identity name (or ``"unknown"``).
        similarity : float
            Cosine similarity score.
        bbox : tuple
            (x, y, w, h) bounding box in pixel coords.
        """
        if not self.enabled:
            return None

        vis = frame.copy()
        x, y, w, h = bbox
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{name} ({similarity:.3f})"
        cv2.putText(
            vis, label, (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        return self._write(
            f"frame{frame_id}_face{face_id}_8_result.png", vis,
        )
