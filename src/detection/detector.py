"""BlazeFace face detector using MediaPipe Tasks API."""

import cv2
import numpy as np
import mediapipe as mp

from config import BLAZEFACE_MODEL_URL, BLAZEFACE_MODEL_PATH, MIN_FACE_BBOX_PX, FACE_BBOX_PADDING
from src.utils import download_model
from src.exceptions import ModelError


class BlazeFaceDetector:
    """Lightweight face detector using MediaPipe's BlazeFace model."""

    def __init__(self, min_confidence: float | None = None):
        from config import MIN_DETECTION_CONFIDENCE
        min_confidence = min_confidence if min_confidence is not None else MIN_DETECTION_CONFIDENCE

        download_model(BLAZEFACE_MODEL_URL, BLAZEFACE_MODEL_PATH)

        base_options = mp.tasks.BaseOptions(model_asset_path=BLAZEFACE_MODEL_PATH)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_confidence,
        )
        try:
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        except Exception as e:
            raise ModelError(f"Failed to initialize BlazeFace detector: {e}") from e

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR frame.

        Returns list of dicts with keys:
            bbox: (x, y, w, h) in pixels
            confidence: float
            crop: np.ndarray (BGR face crop)
            keypoints: [(left_eye), (right_eye), (nose)] pixel coords (or None)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        faces = []
        for det in results.detections:
            bb = det.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            bw = min(bb.width, w - x)
            bh = min(bb.height, h - y)

            if bw < MIN_FACE_BBOX_PX or bh < MIN_FACE_BBOX_PX:
                continue

            # ── Expand bbox to include hair, ears, forehead, chin ─────
            pad_x = int(bw * FACE_BBOX_PADDING)
            pad_y = int(bh * FACE_BBOX_PADDING)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            bw_padded = x2 - x1
            bh_padded = y2 - y1

            crop = frame[y1 : y2, x1 : x2].copy()
            conf = det.categories[0].score if det.categories else 0.0

            kps = None
            if det.keypoints and len(det.keypoints) >= 3:
                kps = [
                    (det.keypoints[1].x * w, det.keypoints[1].y * h),
                    (det.keypoints[0].x * w, det.keypoints[0].y * h),
                    (det.keypoints[2].x * w, det.keypoints[2].y * h),
                ]

            faces.append({
                "bbox": (x1, y1, bw_padded, bh_padded),
                "confidence": conf,
                "crop": crop,
                "keypoints": kps,
            })
        return faces

    def close(self):
        self.detector.close()
