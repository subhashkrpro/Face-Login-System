"""3D Face Mesh extraction using MediaPipe FaceLandmarker (478 landmarks)."""

import cv2
import numpy as np
import mediapipe as mp

from config import (
    FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH,
    MAX_FACES, MESH_MIN_DETECTION_CONFIDENCE, MESH_MIN_TRACKING_CONFIDENCE,
)
from src.utils import download_model
from src.exceptions import ModelError


class FaceMesh:
    """
    Extracts 478 3D face landmarks from a BGR image.

    Each landmark has (x, y, z) in normalized [0,1] coordinates,
    where z represents depth relative to the face center.
    """

    NUM_LANDMARKS = 478

    def __init__(
        self,
        max_faces: int | None = None,
        min_detection_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
    ):
        max_faces = max_faces if max_faces is not None else MAX_FACES
        min_detection_confidence = (
            min_detection_confidence
            if min_detection_confidence is not None
            else MESH_MIN_DETECTION_CONFIDENCE
        )
        min_tracking_confidence = (
            min_tracking_confidence
            if min_tracking_confidence is not None
            else MESH_MIN_TRACKING_CONFIDENCE
        )

        download_model(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)

        base_options = mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        try:
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise ModelError(f"Failed to initialize FaceLandmarker: {e}") from e

    def extract(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Extract 3D face landmarks from a BGR frame.

        Returns:
            List of (478, 3) float32 arrays — one per detected face.
            Coordinates are normalized [0, 1] with z as relative depth.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(mp_image)

        meshes = []
        for face_landmarks in results.face_landmarks:
            pts = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks],
                dtype=np.float32,
            )
            meshes.append(pts)
        return meshes

    def close(self):
        self.landmarker.close()
