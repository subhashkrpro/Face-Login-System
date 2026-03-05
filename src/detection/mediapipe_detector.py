"""MediaPipe-based face detector — reuses FaceLandmarker runtime.

Instead of loading a separate BlazeFace TFLite model (~8s init), this
detector uses MediaPipe's FaceLandmarker which already bundles a face
detector internally.  Since FaceMesh is loaded anyway for mesh/liveness,
this avoids double-loading and cuts init time by ~8 seconds.

The FaceLandmarker detects faces + extracts 478 landmarks in one pass.
We derive bounding boxes and eye/nose keypoints from the landmarks.
Eye keypoints use the centroid of eye-contour landmarks (matching
BlazeFace's eye-center output for consistent ArcFace alignment).
"""

import cv2
import numpy as np
import mediapipe as mp

from config import (
    FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH,
    MIN_FACE_BBOX_PX, FACE_BBOX_PADDING,
    MAX_FACES, MESH_MIN_DETECTION_CONFIDENCE, MESH_MIN_TRACKING_CONFIDENCE,
)
from src.utils import download_model
from src.exceptions import ModelError
from src.logger import get_logger

log = get_logger(__name__)

# ── Eye-contour landmark indices (for computing eye centres) ──────────
# Using 4 corner/midpoints per eye gives a reliable centroid that matches
# the eye-center keypoints BlazeFace returns.
#   outer corner, inner corner, upper lid centre, lower lid centre
_LEFT_EYE_IDXS = (33, 133, 159, 145)
_RIGHT_EYE_IDXS = (263, 362, 386, 374)

_NOSE_TIP_IDX = 1     # nose tip


class MediaPipeDetector:
    """
    Face detector using MediaPipe FaceLandmarker.

    Returns the same dict format as BlazeFaceDetector, so it's a drop-in
    replacement.  Additionally stores raw landmarks in each face dict
    under the key ``"landmarks"`` (478×3 float32) for downstream reuse
    (mesh embedding, liveness) — avoiding redundant extraction.
    """

    def __init__(
        self,
        min_confidence: float | None = None,
        max_faces: int | None = None,
    ):
        min_confidence = (
            min_confidence if min_confidence is not None
            else MESH_MIN_DETECTION_CONFIDENCE
        )
        max_faces = max_faces if max_faces is not None else MAX_FACES

        download_model(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=FACE_LANDMARKER_PATH,
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=MESH_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        try:
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        except Exception as e:
            raise ModelError(
                f"Failed to initialize MediaPipe FaceLandmarker: {e}"
            ) from e

        log.info("MediaPipe FaceLandmarker detector ready")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR frame.

        Returns list of dicts with keys:
            bbox:       (x, y, w, h) in pixels (padded)
            confidence: float (face presence confidence)
            crop:       np.ndarray (BGR face crop)
            keypoints:  [(left_eye), (right_eye), (nose)] pixel coords
            landmarks:  (478, 3) float32 normalised landmarks (for mesh reuse)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)

        faces = []
        for face_landmarks in results.face_landmarks:
            pts = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks],
                dtype=np.float32,
            )

            # Derive bounding box from landmark extents
            xs = pts[:, 0] * w
            ys = pts[:, 1] * h
            x_min = int(max(0, xs.min()))
            y_min = int(max(0, ys.min()))
            x_max = int(min(w, xs.max()))
            y_max = int(min(h, ys.max()))
            bw = x_max - x_min
            bh = y_max - y_min

            if bw < MIN_FACE_BBOX_PX or bh < MIN_FACE_BBOX_PX:
                continue

            # ── Expand bbox to include hair, ears, forehead, chin ─────
            pad_x = int(bw * FACE_BBOX_PADDING)
            pad_y = int(bh * FACE_BBOX_PADDING)
            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_y)
            x2 = min(w, x_max + pad_x)
            y2 = min(h, y_max + pad_y)
            bw_padded = x2 - x1
            bh_padded = y2 - y1

            crop = frame[y1:y2, x1:x2].copy()

            # Keypoints for ArcFace alignment — eye CENTRES (not corners)
            # Averaging 4 contour points (outer, inner, upper, lower) per eye
            # gives a centroid that closely matches BlazeFace's eye-center output.
            left_eye = pts[list(_LEFT_EYE_IDXS), :2].mean(axis=0) * [w, h]
            right_eye = pts[list(_RIGHT_EYE_IDXS), :2].mean(axis=0) * [w, h]
            nose = pts[_NOSE_TIP_IDX, :2] * [w, h]
            kps = [
                (float(left_eye[0]), float(left_eye[1])),
                (float(right_eye[0]), float(right_eye[1])),
                (float(nose[0]), float(nose[1])),
            ]

            # Confidence: average visibility (fallback to 0.99 if unavailable)
            vis = [lm.visibility for lm in face_landmarks
                   if hasattr(lm, 'visibility') and lm.visibility is not None]
            conf = float(np.mean(vis)) if vis else 0.99

            faces.append({
                "bbox": (x1, y1, bw_padded, bh_padded),
                "confidence": conf,
                "crop": crop,
                "keypoints": kps,
                "landmarks": pts,  # 478×3 for mesh/liveness reuse
            })

        return faces

    def close(self):
        self.landmarker.close()
