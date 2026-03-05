"""Face processing exceptions."""


class FaceError(Exception):
    """Base exception for face-related errors."""


class NoFaceDetectedError(FaceError):
    """Raised when no face is detected in a frame."""

    def __init__(self, context: str = "frame"):
        super().__init__(f"No face detected in {context}")
        self.context = context


class DuplicateFaceError(FaceError):
    """Raised when enrolling a face that already exists in the database."""

    def __init__(self, name: str, existing_name: str, similarity: float = 0.0):
        super().__init__(
            f"Face for '{name}' matches already-enrolled '{existing_name}' "
            f"(similarity: {similarity:.3f})"
        )
        self.name = name
        self.existing_name = existing_name
        self.similarity = similarity
