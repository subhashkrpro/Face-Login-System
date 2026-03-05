"""Real-ESRGAN 4x Super Resolution via PyTorch (.pth model)."""

import cv2
import numpy as np
import torch

from config import (
    REALESRGAN_MODEL_URL, REALESRGAN_MODEL_PATH,
    SR_UPSCALE, SR_TILE_SIZE, SR_TILE_PAD,
)
from src.utils import download_model
from src.exceptions import ModelError
from src.logger import get_logger
from .sr_model import SRVGGNetCompact

log = get_logger(__name__)


class RealESRGAN_SR:
    """
    Real-ESRGAN 4x Super Resolution.

    Auto-downloads .pth model on first use (~5 MB).
    Runs on CUDA if available, otherwise CPU.
    """

    def __init__(self, model_path: str | None = None):
        path = model_path or REALESRGAN_MODEL_PATH
        download_model(REALESRGAN_MODEL_URL, path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = SR_UPSCALE
        self.tile_size = SR_TILE_SIZE
        self.tile_pad = SR_TILE_PAD
        log.debug("PyTorch device: %s", self.device)

        self.model = SRVGGNetCompact()
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            if "params_ema" in state_dict:
                state_dict = state_dict["params_ema"]
            elif "params" in state_dict:
                state_dict = state_dict["params"]
            self.model.load_state_dict(state_dict, strict=True)
        except (RuntimeError, FileNotFoundError, KeyError) as e:
            raise ModelError(f"Failed to load Real-ESRGAN model '{path}': {e}") from e
        self.model.eval().to(self.device)
        log.info("Real-ESRGAN (SRVGGNetCompact x4) loaded")

    @torch.no_grad()
    def _infer(self, img: np.ndarray) -> np.ndarray:
        """Run inference on a single BGR uint8 image."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        out = out.squeeze(0).clamp(0, 1).cpu().numpy()
        out = (out.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale image 4x. Tiles large images automatically."""
        h, w = img.shape[:2]
        if h <= self.tile_size and w <= self.tile_size:
            return self._infer(img)
        return self._upscale_tiled(img)

    def _upscale_tiled(self, img: np.ndarray) -> np.ndarray:
        """Process large images in overlapping tiles."""
        h, w = img.shape[:2]
        out_h, out_w = h * self.scale, w * self.scale
        output = np.zeros((out_h, out_w, 3), dtype=np.float64)
        weight = np.zeros((out_h, out_w, 3), dtype=np.float64)

        tile = self.tile_size
        pad = self.tile_pad

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y1, x1 = max(0, y - pad), max(0, x - pad)
                y2, x2 = min(h, y + tile + pad), min(w, x + tile + pad)

                sr_tile = self._infer(img[y1:y2, x1:x2])

                oy1 = (y - y1) * self.scale
                ox1 = (x - x1) * self.scale
                oy2 = oy1 + min(tile, h - y) * self.scale
                ox2 = ox1 + min(tile, w - x) * self.scale

                dy1, dx1 = y * self.scale, x * self.scale
                dy2, dx2 = dy1 + (oy2 - oy1), dx1 + (ox2 - ox1)

                output[dy1:dy2, dx1:dx2] += sr_tile[oy1:oy2, ox1:ox2].astype(np.float64)
                weight[dy1:dy2, dx1:dx2] += 1.0

        return (output / np.maximum(weight, 1e-8)).astype(np.uint8)
