"""Camera capture configuration."""

import cv2

# ── Camera hardware ──────────────────────────────────────────────────────
CAMERA_SOURCE = 0
CAMERA_BACKEND = cv2.CAP_DSHOW       # Windows; use cv2.CAP_V4L2 on Linux
CAMERA_CODEC = "MJPG"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60
CAMERA_BUFFER_SIZE = 1                # Internal OpenCV buffer (keep at 1)

# ── Capture behaviour ────────────────────────────────────────────────────
DEFAULT_CAPTURE_FRAMES = 3
DEFAULT_BUFFER_SIZE = 3               # Pre-buffer deque size
CAMERA_WARMUP_SEC = 1.0               # Auto-exposure settling time
CAPTURE_TIMEOUT_SEC = 5.0             # Max wait for buffered frames
# ── Guided enrollment ───────────────────────────────────────────────
ENROLL_FRAMES = 5                     # Total frames to capture during guided enrollment
ENROLL_POSE_HOLD_SEC = 0.8            # Hold pose for this long before capture
ENROLL_POSE_YAW_THRESH = 12.0         # Degrees yaw to qualify as left/right
ENROLL_POSE_PITCH_THRESH = 10.0       # Degrees pitch to qualify as up/down
ENROLL_BLUR_THRESH = 30.0             # Laplacian blur threshold (reject blurry frames)