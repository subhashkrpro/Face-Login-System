"""Super Resolution backends (Real-ESRGAN, OpenVINO)."""

from .sr_factory import create_sr_backend
from .sr_model import SRVGGNetCompact
from .sr_realesrgan import RealESRGAN_SR
from .sr_openvino import OpenVINO_SR

__all__ = [
    "create_sr_backend",
    "SRVGGNetCompact",
    "RealESRGAN_SR",
    "OpenVINO_SR",
]
