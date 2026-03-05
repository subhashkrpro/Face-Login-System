"""Configuration validation — runs on import to catch bad values early."""

from src.exceptions import ConfigValidationError


def _check_range(name: str, value, lo, hi, typ=None):
    """Validate that value is of type `typ` and within [lo, hi]."""
    if typ and not isinstance(value, typ):
        raise ConfigValidationError(name, value, f"expected {typ.__name__}, got {type(value).__name__}")
    if value < lo or value > hi:
        raise ConfigValidationError(name, value, f"must be between {lo} and {hi}")


def _check_positive(name: str, value, typ=None):
    """Validate that value is positive (> 0)."""
    if typ and not isinstance(value, typ):
        raise ConfigValidationError(name, value, f"expected {typ.__name__}, got {type(value).__name__}")
    if value <= 0:
        raise ConfigValidationError(name, value, "must be positive (> 0)")


def _check_non_negative(name: str, value, typ=None):
    """Validate that value is non-negative (>= 0)."""
    if typ and not isinstance(value, typ):
        raise ConfigValidationError(name, value, f"expected {typ.__name__}, got {type(value).__name__}")
    if value < 0:
        raise ConfigValidationError(name, value, "must be non-negative (>= 0)")


def _check_choice(name: str, value, choices):
    """Validate that value is one of the allowed choices."""
    if value not in choices:
        raise ConfigValidationError(name, value, f"must be one of {choices}")


def validate_config() -> None:
    """
    Validate all config values. Raises ConfigValidationError on first failure.

    Called from main.py at startup, before any models are loaded.
    """
    from config import (
        # Camera
        CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_BUFFER_SIZE,
        DEFAULT_CAPTURE_FRAMES, CAMERA_WARMUP_SEC, CAPTURE_TIMEOUT_SEC,
        ENROLL_FRAMES, ENROLL_BLUR_THRESH,
        # Thresholds
        MIN_DETECTION_CONFIDENCE, RECOGNITION_THRESHOLD, DUPLICATE_THRESHOLD,
        CLAHE_CLIP_LIMIT, EAR_BLINK_THRESHOLD, FACE_BBOX_PADDING, MESH_WEIGHT,
        # Integers
        VERIFY_TOP_K, MAX_FACES, BLINK_CONSEC_FRAMES,
        # SR
        SR_UPSCALE, SR_TILE_SIZE, SR_NUM_FEAT, SR_NUM_CONV,
        DEFAULT_SR_BACKEND,
        # HNSW
        HNSW_SPACE, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_MAX_ELEMENTS,
        # ArcFace
        ARCFACE_INPUT_SIZE, ARCFACE_EMBEDDING_DIM, ARCFACE_ONNX_THREADS,
        # Liveness/download
        LIVENESS_TIMEOUT_SEC, DOWNLOAD_TIMEOUT_SEC,
        # Anti-spoofing
        SPOOF_Z_RANGE_THRESHOLD, SPOOF_Z_STD_THRESHOLD,
    )

    # ── Camera ────────────────────────────────────────────────────────
    _check_positive("CAMERA_WIDTH", CAMERA_WIDTH, int)
    _check_positive("CAMERA_HEIGHT", CAMERA_HEIGHT, int)
    _check_positive("CAMERA_FPS", CAMERA_FPS, int)
    _check_positive("CAMERA_BUFFER_SIZE", CAMERA_BUFFER_SIZE, int)
    _check_positive("DEFAULT_CAPTURE_FRAMES", DEFAULT_CAPTURE_FRAMES, int)
    _check_non_negative("CAMERA_WARMUP_SEC", CAMERA_WARMUP_SEC, (int, float))
    _check_positive("CAPTURE_TIMEOUT_SEC", CAPTURE_TIMEOUT_SEC, (int, float))
    _check_positive("ENROLL_FRAMES", ENROLL_FRAMES, int)
    _check_non_negative("ENROLL_BLUR_THRESH", ENROLL_BLUR_THRESH, (int, float))

    # ── Thresholds (0.0 – 1.0) ────────────────────────────────────────
    _check_range("MIN_DETECTION_CONFIDENCE", MIN_DETECTION_CONFIDENCE, 0.0, 1.0, (int, float))
    _check_range("RECOGNITION_THRESHOLD", RECOGNITION_THRESHOLD, 0.0, 1.0, (int, float))
    _check_range("DUPLICATE_THRESHOLD", DUPLICATE_THRESHOLD, 0.0, 1.0, (int, float))
    _check_range("EAR_BLINK_THRESHOLD", EAR_BLINK_THRESHOLD, 0.0, 1.0, (int, float))

    # ── Detector backend ──────────────────────────────────────────────
    from config import DETECTOR_BACKEND
    _check_choice("DETECTOR_BACKEND", DETECTOR_BACKEND, ("mediapipe", "blazeface"))
    _check_range("FACE_BBOX_PADDING", FACE_BBOX_PADDING, 0.0, 2.0, (int, float))
    _check_range("MESH_WEIGHT", MESH_WEIGHT, 0.0, 1.0, (int, float))

    # ── CLAHE ─────────────────────────────────────────────────────────
    _check_positive("CLAHE_CLIP_LIMIT", CLAHE_CLIP_LIMIT, (int, float))

    # ── Low-light ─────────────────────────────────────────────────────
    from config import (
        LOW_LIGHT_BRIGHTNESS_THRESHOLD, LOW_LIGHT_TARGET_BRIGHTNESS,
        LOW_LIGHT_MAX_GAMMA,
    )
    _check_range("LOW_LIGHT_BRIGHTNESS_THRESHOLD", LOW_LIGHT_BRIGHTNESS_THRESHOLD, 1, 255, (int, float))
    _check_range("LOW_LIGHT_TARGET_BRIGHTNESS", LOW_LIGHT_TARGET_BRIGHTNESS, 1, 255, (int, float))
    _check_range("LOW_LIGHT_MAX_GAMMA", LOW_LIGHT_MAX_GAMMA, 0.05, 1.0, (int, float))

    # ── Integers > 0 ──────────────────────────────────────────────────
    _check_positive("VERIFY_TOP_K", VERIFY_TOP_K, int)
    _check_positive("MAX_FACES", MAX_FACES, int)
    _check_positive("BLINK_CONSEC_FRAMES", BLINK_CONSEC_FRAMES, int)

    # ── Super Resolution ──────────────────────────────────────────────
    _check_positive("SR_UPSCALE", SR_UPSCALE, int)
    _check_positive("SR_TILE_SIZE", SR_TILE_SIZE, int)
    _check_positive("SR_NUM_FEAT", SR_NUM_FEAT, int)
    _check_positive("SR_NUM_CONV", SR_NUM_CONV, int)
    _check_choice("DEFAULT_SR_BACKEND", DEFAULT_SR_BACKEND, ("realesrgan", "openvino", "gfpgan"))

    # ── HNSW ──────────────────────────────────────────────────────────
    _check_choice("HNSW_SPACE", HNSW_SPACE, ("cosine", "l2", "ip"))
    _check_positive("HNSW_M", HNSW_M, int)
    _check_positive("HNSW_EF_CONSTRUCTION", HNSW_EF_CONSTRUCTION, int)
    _check_positive("HNSW_EF_SEARCH", HNSW_EF_SEARCH, int)
    _check_positive("HNSW_MAX_ELEMENTS", HNSW_MAX_ELEMENTS, int)

    # ── ArcFace ───────────────────────────────────────────────────────
    _check_positive("ARCFACE_INPUT_SIZE", ARCFACE_INPUT_SIZE, int)
    _check_positive("ARCFACE_EMBEDDING_DIM", ARCFACE_EMBEDDING_DIM, int)
    _check_positive("ARCFACE_ONNX_THREADS", ARCFACE_ONNX_THREADS, int)

    # ── Anti-spoofing ─────────────────────────────────────────────────
    _check_range("SPOOF_Z_RANGE_THRESHOLD", SPOOF_Z_RANGE_THRESHOLD, 0.0, 1.0, (int, float))
    _check_range("SPOOF_Z_STD_THRESHOLD", SPOOF_Z_STD_THRESHOLD, 0.0, 1.0, (int, float))

    from config import (
        SPOOF_LBP_HIST_VARIANCE_THRESHOLD, SPOOF_LBP_HF_ENERGY_THRESHOLD,
    )
    _check_non_negative("SPOOF_LBP_HIST_VARIANCE_THRESHOLD", SPOOF_LBP_HIST_VARIANCE_THRESHOLD, (int, float))
    _check_non_negative("SPOOF_LBP_HF_ENERGY_THRESHOLD", SPOOF_LBP_HF_ENERGY_THRESHOLD, (int, float))

    from config import (
        SPOOF_NOISE_KURTOSIS_THRESHOLD, SPOOF_BLOCK_GRAD_VAR_THRESHOLD,
        SPOOF_LOCAL_NOISE_VAR_THRESHOLD,
    )
    _check_non_negative("SPOOF_NOISE_KURTOSIS_THRESHOLD", SPOOF_NOISE_KURTOSIS_THRESHOLD, (int, float))
    _check_non_negative("SPOOF_BLOCK_GRAD_VAR_THRESHOLD", SPOOF_BLOCK_GRAD_VAR_THRESHOLD, (int, float))
    _check_non_negative("SPOOF_LOCAL_NOISE_VAR_THRESHOLD", SPOOF_LOCAL_NOISE_VAR_THRESHOLD, (int, float))

    # ── Timeouts ──────────────────────────────────────────────────────
    _check_positive("LIVENESS_TIMEOUT_SEC", LIVENESS_TIMEOUT_SEC, (int, float))
    _check_positive("DOWNLOAD_TIMEOUT_SEC", DOWNLOAD_TIMEOUT_SEC, (int, float))
