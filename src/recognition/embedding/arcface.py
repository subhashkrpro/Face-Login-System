"""ArcFace face embedding via ONNX Runtime (InsightFace w600k_mbf)."""

import cv2
import numpy as np
import onnxruntime as ort

from config import ARCFACE_INPUT_SIZE, ARCFACE_EMBEDDING_DIM, ARCFACE_ONNX_THREADS
from src.exceptions import ModelError
from src.logger import get_logger
from .model_loader import ensure_arcface_model
from .align import align_face

log = get_logger(__name__)

class ArcFaceEmbedder:
    """
    ArcFace 512-dim face embedding via InsightFace MobileFaceNet (ONNX).

    Auto-downloads the model (~12 MB) on first use.
    Input: aligned 112x112 BGR face.
    Output: L2-normalized 512-dim embedding.
    """

    EMBEDDING_DIM = ARCFACE_EMBEDDING_DIM

    def __init__(self, model_path: str | None = None):
        path = model_path or ensure_arcface_model()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = ARCFACE_ONNX_THREADS

        try:
            self.session = ort.InferenceSession(path, sess_options=opts)
        except Exception as e:
            raise ModelError(f"Failed to load ArcFace ONNX model '{path}': {e}") from e
        self.input_name = self.session.get_inputs()[0].name
        log.info("ArcFace (MobileFaceNet %d-dim) loaded via ONNX Runtime", ARCFACE_EMBEDDING_DIM)

    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """BGR 112x112 -> NCHW float32 normalized to [-1, 1]."""
        img = cv2.resize(face, (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
        return img[np.newaxis, ...]           # add batch dim

    def get_embedding(self, face: np.ndarray) -> np.ndarray:
        """
        Compute 512-dim L2-normalized embedding from a face crop.

        Args:
            face: BGR face image (any size, will be resized to 112x112).

        Returns:
            (512,) float32 L2-normalized embedding.
        """
        blob = self._preprocess(face)
        out = self.session.run(None, {self.input_name: blob})[0]
        emb = out.flatten().astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb /= norm
        return emb

    def get_embedding_aligned(
        self,
        frame: np.ndarray,
        keypoints: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Align face using 3 keypoints, then compute embedding.

        Args:
            frame: Full BGR image.
            keypoints: [(left_eye), (right_eye), (nose)] in pixel coords.

        Returns:
            (512,) L2-normalized embedding.
        """
        aligned = align_face(frame, keypoints)
        return self.get_embedding(aligned)
