"""
RecognitionDebugSaver — saves every intermediate image during recognition.

Creates a timestamped session folder under ``debug/recognition/`` so that
enhancement and recognition debug outputs stay separate.

    debug/recognition/session_20260301_160000/
        frame0_01_raw.png               # original camera frame
        frame0_02_detected.png          # full frame with all bboxes
        frame0_face0_03_crop.png        # raw face crop
        frame0_face0_04_spoof.png       # anti-spoof z-depth heatmap
        frame0_face0_05_gamma.png       # after low-light gamma correction
        frame0_face0_06_clahe.png       # after CLAHE contrast enhancement
        frame0_face0_07_mesh.png        # 3D mesh overlay
        frame0_face0_08_aligned.png     # 112×112 ArcFace-aligned
        frame0_face0_09_candidates.png  # HNSW top-K candidates listed
        frame0_face0_10_result.png      # final annotated result

Usage:
    from debug.recognition_debug import RecognitionDebugSaver

    dbg = RecognitionDebugSaver(enabled=True)
    dbg.save_raw(frame)
    dbg.save_detected(frame, faces)
    dbg.save_crop(crop, face_id=0)
    dbg.save_spoof(crop, spoof_info, face_id=0)
    dbg.save_gamma(img, face_id=0)
    dbg.save_clahe(img, face_id=0)
    dbg.save_mesh(img, landmarks, face_id=0)
    dbg.save_aligned(img, face_id=0)
    dbg.save_candidates(frame, candidates, face_id=0)
    dbg.save_result(frame, name, similarity, bbox, face_id=0)
"""

import os
from datetime import datetime

import cv2
import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


class RecognitionDebugSaver:
    """
    Saves every intermediate recognition image into a session folder.

    Parameters
    ----------
    enabled : bool
        When ``False`` every save method is a no-op (zero overhead).
    base_dir : str | None
        Root directory.  Defaults to ``<project>/debug/recognition``.
    """

    def __init__(self, enabled: bool = True, base_dir: str | None = None):
        self.enabled = enabled
        if not enabled:
            return

        if base_dir is None:
            base_dir = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                "debug", "recognition",
            )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{stamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        log.info("Recognition debug → %s", self.session_dir)

    # ── helpers ───────────────────────────────────────────────────────────

    def _write(self, filename: str, img: np.ndarray) -> str | None:
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

    # ── 01: raw frame ────────────────────────────────────────────────────

    def save_raw(
        self, frame: np.ndarray, frame_id: int = 0,
    ) -> str | None:
        """Save the original camera frame."""
        return self._write(f"frame{frame_id}_01_raw.png", frame)

    # ── 02: detection overlay ────────────────────────────────────────────

    def save_detected(
        self,
        frame: np.ndarray,
        faces: list[dict],
        frame_id: int = 0,
    ) -> str | None:
        """Draw bounding boxes + keypoints on the full frame."""
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
            kps = face.get("keypoints")
            if kps:
                for px, py in kps:
                    cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)

        return self._write(f"frame{frame_id}_02_detected.png", vis)

    # ── 03: raw face crop ────────────────────────────────────────────────

    def save_crop(
        self, crop: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the raw face crop (before any processing)."""
        return self._write(
            f"frame{frame_id}_face{face_id}_03_crop.png", crop,
        )

    # ── 04: anti-spoof z-depth heatmap ───────────────────────────────────

    def save_spoof(
        self,
        crop: np.ndarray,
        spoof_info: dict | None,
        landmarks: np.ndarray | None = None,
        frame_id: int = 0,
        face_id: int = 0,
    ) -> str | None:
        """
        Save anti-spoof analysis — z-depth heatmap overlay on crop.

        If landmarks are available, draws a colour-coded depth map:
            blue = close (high z), red = far (low z).
        Annotates with is_live, z_range, z_std.
        """
        if not self.enabled:
            return None

        vis = crop.copy()
        h, w = vis.shape[:2]

        if landmarks is not None and len(landmarks) > 0:
            z = landmarks[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)  # 0-1

            pts_2d = landmarks[:, :2]
            px_pts = (pts_2d * np.array([w, h])).astype(np.int32)

            for idx, pt in enumerate(px_pts):
                # Blue (close) → Red (far) colourmap
                depth_val = z_norm[idx]
                b = int((1 - depth_val) * 255)
                r = int(depth_val * 255)
                g = 50
                cv2.circle(vis, tuple(pt), 2, (b, g, r), -1)

        # Annotate spoof info
        if spoof_info:
            status = "LIVE" if spoof_info["is_live"] else "SPOOF"
            color = (0, 255, 0) if spoof_info["is_live"] else (0, 0, 255)
            lines = [
                f"{status}",
                f"z_range: {spoof_info['z_range']:.4f}",
                f"z_std: {spoof_info['z_std']:.4f}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(
                    vis, line, (5, 15 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                )

        return self._write(
            f"frame{frame_id}_face{face_id}_04_spoof.png", vis,
        )

    # ── 05: gamma-corrected ──────────────────────────────────────────────

    def save_gamma(
        self, img: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the face after low-light gamma correction."""
        return self._write(
            f"frame{frame_id}_face{face_id}_05_gamma.png", img,
        )

    # ── 06: CLAHE enhanced ───────────────────────────────────────────────

    def save_clahe(
        self, img: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the face after CLAHE contrast enhancement."""
        return self._write(
            f"frame{frame_id}_face{face_id}_06_clahe.png", img,
        )

    # ── 07: 3D mesh overlay ──────────────────────────────────────────────

    def save_mesh(
        self,
        img: np.ndarray,
        landmarks: np.ndarray | None = None,
        frame_id: int = 0,
        face_id: int = 0,
    ) -> str | None:
        """Draw 478-point 3D face mesh on *img* and save."""
        if not self.enabled:
            return None

        vis = img.copy()
        h, w = vis.shape[:2]

        if landmarks is None:
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
            from debug.debug_saver import _get_mesh_connections
            connections = _get_mesh_connections()
            pts = landmarks[:, :2]
            px_pts = (pts * np.array([w, h])).astype(np.int32)

            for i, j in connections:
                if i < len(px_pts) and j < len(px_pts):
                    cv2.line(vis, tuple(px_pts[i]), tuple(px_pts[j]),
                             (200, 200, 200), 1, cv2.LINE_AA)

            for pt in px_pts:
                cv2.circle(vis, tuple(pt), 1, (0, 255, 0), -1)

        return self._write(
            f"frame{frame_id}_face{face_id}_07_mesh.png", vis,
        )

    # ── 08: ArcFace-aligned ──────────────────────────────────────────────

    def save_aligned(
        self, img: np.ndarray, frame_id: int = 0, face_id: int = 0,
    ) -> str | None:
        """Save the 112×112 ArcFace-aligned face."""
        return self._write(
            f"frame{frame_id}_face{face_id}_08_aligned.png", img,
        )

    # ── 09: HNSW candidates ─────────────────────────────────────────────

    def save_candidates(
        self,
        crop: np.ndarray,
        candidates: list[dict],
        frame_id: int = 0,
        face_id: int = 0,
    ) -> str | None:
        """
        Annotate the face crop with HNSW top-K candidate names + scores.

        Each candidate dict has ``{"name": str, "distance": float}``.
        """
        if not self.enabled:
            return None

        # Scale up small crops for readability
        h, w = crop.shape[:2]
        scale = max(1, 200 // max(h, 1))
        vis = cv2.resize(crop, (w * scale, h * scale),
                         interpolation=cv2.INTER_NEAREST)

        cv2.putText(
            vis, "HNSW Top-K:", (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        for idx, cand in enumerate(candidates):
            name = cand.get("name", "?")
            dist = cand.get("distance", 0.0)
            sim = 1.0 - dist if dist <= 1.0 else dist
            line = f"  {idx+1}. {name} (sim={sim:.3f})"
            y_pos = 38 + idx * 18
            cv2.putText(
                vis, line, (5, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
            )

        return self._write(
            f"frame{frame_id}_face{face_id}_09_candidates.png", vis,
        )

    # ── 10: final recognition result ─────────────────────────────────────

    def save_result(
        self,
        frame: np.ndarray,
        name: str,
        similarity: float,
        bbox: tuple[int, int, int, int],
        spoof_info: dict | None = None,
        frame_id: int = 0,
        face_id: int = 0,
    ) -> str | None:
        """
        Draw final recognition result on the full frame.

        Annotates with name, similarity, and optionally spoof status.
        Colour: green = matched, red = unknown, magenta = spoof.
        """
        if not self.enabled:
            return None

        vis = frame.copy()
        x, y, w, h = bbox

        if name == "spoof":
            color = (255, 0, 255)  # magenta
            label = f"SPOOF"
        elif name == "unknown":
            color = (0, 0, 255)  # red
            label = f"unknown ({similarity:.3f})"
        else:
            color = (0, 255, 0)  # green
            label = f"{name} ({similarity:.3f})"

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis, label, (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        # Add spoof details below bbox
        if spoof_info and not spoof_info["is_live"]:
            reason = spoof_info.get("reason", "")
            cv2.putText(
                vis, reason, (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1,
            )

        return self._write(
            f"frame{frame_id}_face{face_id}_10_result.png", vis,
        )
