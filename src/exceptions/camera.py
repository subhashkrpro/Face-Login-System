"""Camera-related exceptions."""


class CameraError(RuntimeError):
    """Base exception for camera errors."""


class CameraOpenError(CameraError):
    """Raised when the camera cannot be opened."""

    def __init__(self, source: int | str = 0):
        super().__init__(f"Failed to open camera source: {source}")
        self.source = source


class CameraTimeoutError(CameraError):
    """Raised when the camera does not deliver frames in time."""

    def __init__(self, expected_frames: int, timeout: float):
        super().__init__(
            f"Camera did not deliver {expected_frames} frames "
            f"within {timeout:.1f}s timeout"
        )
        self.expected_frames = expected_frames
        self.timeout = timeout
