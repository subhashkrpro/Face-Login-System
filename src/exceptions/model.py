"""Model-related exceptions."""


class ModelError(Exception):
    """Base exception for model errors."""


class ModelDownloadError(ModelError):
    """Raised when a model file cannot be downloaded."""

    def __init__(self, url: str, reason: str = ""):
        msg = f"Failed to download model from {url}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.url = url


class ModelNotFoundError(ModelError, FileNotFoundError):
    """Raised when a required model file is missing on disk."""

    def __init__(self, path: str, hint: str = ""):
        msg = f"Model file not found: {path}"
        if hint:
            msg += f"\n{hint}"
        super().__init__(msg)
        self.path = path


class ModelExtractionError(ModelError):
    """Raised when an archive does not contain the expected model."""

    def __init__(self, archive: str, expected: str):
        super().__init__(f"'{expected}' not found inside '{archive}'")
        self.archive = archive
        self.expected = expected
