"""Factory function to create a Super Resolution backend."""

from src.exceptions import ConfigValidationError


def create_sr_backend(backend: str | None = None, **kwargs):
    """
    Create a Super Resolution backend.

    Args:
        backend: "realesrgan" (default), "openvino", or "gfpgan"
        **kwargs: Passed to the backend constructor.
    """
    from config import DEFAULT_SR_BACKEND
    backend = (backend or DEFAULT_SR_BACKEND).lower().strip()

    if backend in ("realesrgan", "real-esrgan", "pytorch", "pth"):
        from .sr_realesrgan import RealESRGAN_SR
        return RealESRGAN_SR(**kwargs)
    elif backend in ("openvino", "ov", "npu"):
        from .sr_openvino import OpenVINO_SR
        return OpenVINO_SR(**kwargs)
    elif backend in ("gfpgan", "gfp"):
        from .sr_gfpgan import GFPGAN_SR
        return GFPGAN_SR(**kwargs)
    else:
        raise ConfigValidationError(
            "DEFAULT_SR_BACKEND", backend,
            "must be 'realesrgan', 'openvino', or 'gfpgan'",
        )
