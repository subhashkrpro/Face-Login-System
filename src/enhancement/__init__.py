"""Image enhancement module (low-light gamma + CLAHE + Super Resolution)."""

from .clahe import apply_clahe
from .low_light import auto_brighten
from .sr import create_sr_backend, SRVGGNetCompact, RealESRGAN_SR, OpenVINO_SR
from .enhancer import FaceEnhancer

__all__ = [
    "apply_clahe",
    "auto_brighten",
    "create_sr_backend",
    "SRVGGNetCompact",
    "RealESRGAN_SR",
    "OpenVINO_SR",
    "FaceEnhancer",
]
