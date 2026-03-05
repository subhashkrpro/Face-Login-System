"""GFPGAN face restoration + super resolution backend.

Uses a local reimplementation of the GFPGANv1Clean architecture so that
the heavy ``gfpgan`` / ``basicsr`` PyPI packages are **not** required.
Weights are loaded directly from the official GFPGANv1.4.pth checkpoint.
"""

import cv2
import numpy as np
import torch

from config import SR_UPSCALE
from src.exceptions import ModelError
from src.logger import get_logger
from src.utils import download_model
from .arch_gfpgan import GFPGANv1Clean

log = get_logger(__name__)


class GFPGAN_SR:
    """
    GFPGAN face restoration with built-in super resolution.

    Auto-downloads the model (~350 MB) on first use.
    Runs on CUDA if available, otherwise CPU.

    GFPGAN is specifically designed for face restoration — it produces
    sharper, more natural-looking faces than generic SR models.
    The model operates at 512x512 internally and the output is resized
    to match the requested upscale factor.
    """

    # Official model URL (v1.4 — best quality)
    _MODEL_URL = (
        "https://github.com/TencentARC/GFPGAN/releases/download/"
        "v1.3.0/GFPGANv1.4.pth"
    )
    _FACE_SIZE = 512  # GFPGAN native resolution

    def __init__(self, model_path: str | None = None, upscale: int | None = None):
        from config import GFPGAN_MODEL_PATH
        path = model_path or GFPGAN_MODEL_PATH
        self.scale = upscale or SR_UPSCALE

        # Auto-download
        download_model(self._MODEL_URL, path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.debug("GFPGAN device: %s", self.device)

        # Build architecture and load weights
        self.model = GFPGANv1Clean(
            out_size=self._FACE_SIZE,
            num_style_feat=512,
            channel_multiplier=2,
            narrow=0.5,
            sft_half=True,
        )

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except RuntimeError:
            # weights_only=True may fail on older checkpoints; retry
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # The checkpoint may store weights under different keys
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise ModelError(f"Failed to load GFPGAN weights from '{path}': {e}") from e

        self.model.eval().to(self.device)
        log.info("GFPGAN v1.4 (face restoration, %dx%d) loaded",
                 self._FACE_SIZE, self._FACE_SIZE)

    @torch.no_grad()
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """
        Restore and upscale a face crop.

        Args:
            img: BGR uint8 face crop (any size).

        Returns:
            BGR uint8 restored + upscaled face.
        """
        h, w = img.shape[:2]
        target_h, target_w = h * self.scale, w * self.scale

        # Pre-process: BGR -> RGB, resize to 512x512, normalize to [-1, 1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = cv2.resize(rgb, (self._FACE_SIZE, self._FACE_SIZE),
                          interpolation=cv2.INTER_LANCZOS4)
        tensor = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]

        # Inference
        try:
            output, _ = self.model(tensor)
        except Exception as e:
            log.warning("GFPGAN inference failed, returning resized input: %s", e)
            return cv2.resize(img, (target_w, target_h),
                              interpolation=cv2.INTER_LANCZOS4)

        # Post-process: [-1, 1] -> [0, 255], RGB -> BGR
        output = output.squeeze(0).clamp(-1, 1)
        output = ((output + 1) / 2 * 255.0).cpu().numpy().astype(np.uint8)
        output = output.transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Resize to target dimensions if different from native
        if target_h != self._FACE_SIZE or target_w != self._FACE_SIZE:
            output = cv2.resize(output, (target_w, target_h),
                                interpolation=cv2.INTER_LANCZOS4)

        return output
