"""OpenVINO Super Resolution backend (NPU / GPU / CPU)."""

import os
import cv2
import numpy as np

from config import OPENVINO_SR_MODEL_XML, OPENVINO_SR_MODEL_BIN, SR_UPSCALE
from src.exceptions import ModelError, ModelNotFoundError
from src.logger import get_logger

log = get_logger(__name__)


class OpenVINO_SR:
    """
    Super Resolution via OpenVINO IR model (.xml + .bin).

    Auto-selects best device: NPU -> GPU -> CPU.
    """

    def __init__(self, model_xml: str | None = None, model_bin: str | None = None,
                 device: str | None = None):
        try:
            from openvino import Core
        except ImportError as e:
            raise ModelError("OpenVINO not installed. Install with: uv add openvino") from e

        xml_path = model_xml or OPENVINO_SR_MODEL_XML
        bin_path = model_bin or OPENVINO_SR_MODEL_BIN
        self.scale = SR_UPSCALE

        if not os.path.isfile(xml_path):
            raise ModelNotFoundError(
                xml_path,
                hint=(
                    "Download from Intel Open Model Zoo:\n"
                    "  omz_downloader --name single-image-super-resolution-1032 --output_dir models\n"
                    "  omz_converter  --name single-image-super-resolution-1032 --output_dir models"
                ),
            )

        core = Core()
        available = core.available_devices
        log.debug("OpenVINO devices: %s", available)

        if device is None:
            device = "CPU"
            for preferred in ("NPU", "GPU"):
                if preferred in available:
                    device = preferred
                    break

        log.debug("Loading OpenVINO SR on: %s", device)
        try:
            model = core.read_model(model=xml_path, weights=bin_path)
            self._compiled = core.compile_model(model, device)
        except Exception as e:
            raise ModelError(f"Failed to load OpenVINO SR model: {e}") from e
        self._input = self._compiled.input(0)
        self._output = self._compiled.output(0)
        log.info("OpenVINO SR loaded on %s", device)

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale image using the OpenVINO IR model."""
        input_shape = self._input.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_shape[3], input_shape[2]))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        result = self._compiled([blob])[self._output]

        out = np.squeeze(result, axis=0)
        out = np.transpose(out, (1, 2, 0))
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
