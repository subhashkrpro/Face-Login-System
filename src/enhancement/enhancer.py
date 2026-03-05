"""Face detection + low-light gamma + CLAHE + super resolution enhancement pipeline."""

from __future__ import annotations

import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from config import ENHANCED_DIR, CLAHE_CLIP_LIMIT
from src.detection import create_detector
from src.logger import get_logger
from .sr import create_sr_backend
from .clahe import apply_clahe
from .low_light import auto_brighten

if TYPE_CHECKING:
    from debug import DebugSaver

log = get_logger(__name__)


class FaceEnhancer:
    """
    Full pipeline: frame -> detect -> crop -> gamma -> CLAHE -> SR 4x -> mesh -> save.

    When a ``DebugSaver`` is provided every intermediate image is persisted
    to a timestamped debug session folder.
    """

    def __init__(self, min_confidence: float | None = None,
                 sr_backend: str | None = None,
                 clip_limit: float | None = None,
                 debug_saver: DebugSaver | None = None,
                 **sr_kwargs):
        # Load detector + SR backend in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            det_future = pool.submit(create_detector, min_confidence=min_confidence)
            sr_future = pool.submit(create_sr_backend, sr_backend, **sr_kwargs)
            self.detector = det_future.result()
            self.sr = sr_future.result()
        self.clip_limit = clip_limit if clip_limit is not None else CLAHE_CLIP_LIMIT
        self._dbg = debug_saver

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> list[str]:
        """Detect faces, enhance each crop, save. Returns saved file paths."""
        t0 = time.perf_counter()

        # Debug: raw camera frame
        if self._dbg:
            self._dbg.save_raw(frame, frame_id=frame_id)

        faces = self.detector.detect(frame)
        t_detect = (time.perf_counter() - t0) * 1000

        if not faces:
            log.debug("Frame %d: no face detected (%.1fms)", frame_id, t_detect)
            return []

        # Debug: detection overlay (bbox + keypoints on full frame)
        if self._dbg:
            self._dbg.save_detected(frame, faces, frame_id=frame_id)

        saved_paths = []
        for i, face in enumerate(faces):
            t1 = time.perf_counter()

            raw_crop = face["crop"]

            # Debug: raw crop (before enhancement)
            if self._dbg:
                self._dbg.save_crop(raw_crop, frame_id=frame_id, face_id=i)

            # Low-light adaptive gamma correction (before CLAHE)
            brightened = auto_brighten(raw_crop)

            clahe_img = apply_clahe(brightened, clip_limit=self.clip_limit)

            # Debug: after CLAHE
            if self._dbg:
                self._dbg.save_clahe(clahe_img, frame_id=frame_id, face_id=i)

            enhanced = self.sr.upscale(clahe_img)
            t_sr = (time.perf_counter() - t1) * 1000

            # Debug: after super-resolution
            if self._dbg:
                self._dbg.save_sr(enhanced, frame_id=frame_id, face_id=i)

            # Debug: 3D face mesh overlay
            if self._dbg:
                self._dbg.save_mesh(enhanced, frame_id=frame_id, face_id=i)

            filename = f"enhanced_frame{frame_id}_face{i}.png"
            path = os.path.join(ENHANCED_DIR, filename)
            if not cv2.imwrite(path, enhanced):
                log.warning("Failed to save enhanced image: %s", path)

            x, y, w, h = face["bbox"]
            eh, ew = enhanced.shape[:2]
            log.info(
                "Frame %d Face %d: bbox=(%d,%d,%d,%d) conf=%.2f | "
                "SR %dx%d -> %dx%d in %.1fms | saved -> %s",
                frame_id, i, x, y, w, h, face["confidence"],
                w, h, ew, eh, t_sr, filename,
            )
            saved_paths.append(path)

        total_ms = (time.perf_counter() - t0) * 1000
        log.info("Frame %d: %d face(s) in %.1fms (detect=%.1fms)",
                 frame_id, len(faces), total_ms, t_detect)
        return saved_paths

    def close(self):
        self.detector.close()
