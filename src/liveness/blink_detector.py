"""Blink detection using MediaPipe FaceLandmarker and EAR."""

import cv2
import numpy as np
import mediapipe as mp

from config import (
    FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH,
    MESH_MIN_DETECTION_CONFIDENCE, MESH_MIN_TRACKING_CONFIDENCE,
    EAR_BLINK_THRESHOLD, BLINK_CONSEC_FRAMES,
)
from src.utils import download_model
from src.exceptions import ModelError
from src.logger import get_logger
from .ear import compute_ear, RIGHT_EYE, LEFT_EYE

log = get_logger(__name__)


class BlinkDetector:
    """
    Detects eye blinks from a sequence of BGR frames.

    Uses MediaPipe FaceLandmarker to extract 478 landmarks, then
    computes the Eye Aspect Ratio (EAR) to detect blink events.

    A blink = EAR drops below threshold for N consecutive frames,
    then recovers above threshold.
    """

    def __init__(
        self,
        ear_threshold: float | None = None,
        consec_frames: int | None = None,
    ):
        self.ear_threshold = ear_threshold or EAR_BLINK_THRESHOLD
        self.consec_frames = consec_frames or BLINK_CONSEC_FRAMES
        self._closed_count = 0
        self._blink_count = 0

        download_model(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=MESH_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MESH_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        try:
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise ModelError(f"Failed to initialize FaceLandmarker for blink detection: {e}") from e
        log.info("BlinkDetector (EAR-based liveness) ready")

    def _get_ear(self, frame: np.ndarray) -> float | None:
        """Extract average EAR from both eyes. Returns None if no face."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        lm = results.face_landmarks[0]
        pts = np.array([(l.x, l.y) for l in lm], dtype=np.float64)

        ear_r = compute_ear(pts, RIGHT_EYE)
        ear_l = compute_ear(pts, LEFT_EYE)
        return (ear_r + ear_l) / 2.0

    def update(self, frame: np.ndarray) -> tuple[float | None, bool]:
        """
        Feed a new frame and check for blink.

        Returns:
            (ear, blinked) — current EAR value (None if no face),
            and whether a blink was just completed.
        """
        ear = self._get_ear(frame)
        if ear is None:
            return None, False

        blinked = False
        if ear < self.ear_threshold:
            self._closed_count += 1
        else:
            if self._closed_count >= self.consec_frames:
                self._blink_count += 1
                blinked = True
            self._closed_count = 0

        return ear, blinked

    @property
    def blink_count(self) -> int:
        return self._blink_count

    def reset(self):
        self._closed_count = 0
        self._blink_count = 0

    def close(self):
        self._landmarker.close()
