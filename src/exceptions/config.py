"""Configuration validation exceptions."""


class ConfigValidationError(ValueError):
    """Raised when a config value fails validation."""

    def __init__(self, param: str, value, reason: str):
        super().__init__(f"Invalid config '{param}' = {value!r}: {reason}")
        self.param = param
        self.value = value
        self.reason = reason
