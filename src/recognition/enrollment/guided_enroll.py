"""Guided enrollment — phone-style face registration.

Clean oval guide with smooth circular progress, just like Face ID.
User slowly moves their head in a circle. The oval border fills green
progressively. Frames are captured silently at diverse head angles.
"""

import math
import time
import cv2
import numpy as np
import mediapipe as mp

from config import (
    CAMERA_SOURCE, CAMERA_BACKEND, CAMERA_CODEC,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_BUFFER_SIZE,
    FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH,
    ENROLL_FRAMES,
)
from src.utils import download_model
from src.logger import get_logger
from src.exceptions import CameraOpenError

log = get_logger(__name__)

# ── UI constants ──────────────────────────────────────────────────────────
_OVAL_W = 140                # Oval semi-axis width
_OVAL_H = 185               # Oval semi-axis height
_OVAL_THICK = 5              # Base oval thickness
_PROGRESS_THICK = 7          # Filled arc thickness
_CAPTURE_ANGLE_GAP = 25      # Min degrees between captures on the circle
_BLUR_THRESH = 12.0          # Laplacian blur threshold (webcam-friendly)


class GuidedEnrollment:
    """
    Phone-style guided enrollment with oval progress.

    Uses nose-tip offset from face center for head pose —
    simple, fast, and reliable (no solvePnP wrapping issues).

    Pass a pre-loaded ``landmarker`` to avoid duplicate TFLite init
    (saves ~9 seconds when the caller already has a FaceLandmarker).
    """

    def __init__(self, num_frames: int | None = None,
                 landmarker=None):
        self.num_frames = num_frames if num_frames is not None else ENROLL_FRAMES
        if landmarker is not None:
            self._landmarker = landmarker
            self._owns_landmarker = False
            log.info("GuidedEnrollment: reusing shared FaceLandmarker")
        else:
            self._landmarker = self._init_landmarker()
            self._owns_landmarker = True

    def _init_landmarker(self):
        download_model(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ── Pose estimation (nose offset) ─────────────────────────────────────

    @staticmethod
    def _estimate_head_offset(landmarks):
        """
        Estimate head direction from nose-tip offset vs face center.

        Uses landmarks:
          - nose tip (1)
          - left eye outer (33), right eye outer (263)
          - chin (152), forehead approx (10)

        Returns (nx, ny) normalized offset in range ~[-1, 1].
          nx > 0 = head turned right, nx < 0 = left
          ny > 0 = head tilted down, ny < 0 = up
        """
        nose = landmarks[1]              # Nose tip
        left_eye = landmarks[33]         # Left eye outer corner
        right_eye = landmarks[263]       # Right eye outer corner
        chin = landmarks[152]            # Chin
        forehead = landmarks[10]         # Forehead center

        # Face center = midpoint between eyes (x) and between forehead & chin (y)
        face_cx = (left_eye.x + right_eye.x) / 2
        face_cy = (forehead.y + chin.y) / 2

        # Face dimensions for normalization
        face_w = abs(right_eye.x - left_eye.x)
        face_h = abs(chin.y - forehead.y)

        if face_w < 0.01 or face_h < 0.01:
            return 0.0, 0.0

        # Nose offset relative to face center, normalized by face size
        nx = (nose.x - face_cx) / face_w   # Positive = right
        ny = (nose.y - face_cy) / face_h   # Positive = down

        # Scale to make full-range more reachable
        nx = float(np.clip(nx * 2.5, -1.0, 1.0))
        ny = float(np.clip(ny * 2.5, -1.0, 1.0))

        return nx, ny

    # ── Quality check ─────────────────────────────────────────────────────

    @staticmethod
    def _is_sharp(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= _BLUR_THRESH

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw_ui(self, frame, cx, cy, progress, face_found,
                 phase, num_captured, total):
        """Draw the clean oval UI overlay."""
        h, w = frame.shape[:2]

        # ── Dim background outside oval ───────────────────────────────
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (_OVAL_W, _OVAL_H), 0, 0, 360, 255, -1)
        dark = (frame * 0.3).astype(np.uint8)
        frame_out = np.where(mask[:, :, None] == 255, frame, dark)
        np.copyto(frame, frame_out)

        # ── Base oval (subtle gray) ───────────────────────────────────
        cv2.ellipse(frame, (cx, cy), (_OVAL_W, _OVAL_H), 0, 0, 360,
                    (70, 70, 70), _OVAL_THICK)

        # ── Progress arc (green, fills clockwise from top) ────────────
        if progress > 0:
            arc_end = int(progress * 360)
            # Draw from -90 (top) clockwise
            cv2.ellipse(frame, (cx, cy), (_OVAL_W, _OVAL_H), -90,
                        0, arc_end, (0, 210, 80), _PROGRESS_THICK)

        # ── Top text area ─────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Phase instruction
        if phase == "center":
            text = "Look straight at the camera"
            color = (255, 255, 255)
        elif phase == "circle":
            text = "Slowly move your head in a circle"
            color = (255, 255, 255)
        elif phase == "done":
            text = "Done!"
            color = (0, 230, 80)
        else:
            text = "Position your face in the oval"
            color = (180, 180, 180)

        # Center the text
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        tx = (w - tw) // 2
        cv2.putText(frame, text, (tx, 33),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                     cv2.LINE_AA)

        # ── Counter (bottom-right, subtle) ────────────────────────────
        counter = f"{num_captured}/{total}"
        cv2.putText(frame, counter, (w - 55, h - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                     cv2.LINE_AA)

        # ── No face warning ───────────────────────────────────────────
        if not face_found:
            warn = "No face detected"
            (ww, _), _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            wx = (w - ww) // 2
            cv2.putText(frame, warn, (wx, h - 15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1,
                         cv2.LINE_AA)

    # ── Main capture loop ─────────────────────────────────────────────────

    def capture(self, name: str) -> list[np.ndarray]:
        """
        Phone-style enrollment: oval guide + smooth circular progress.

        Phase 1: Center capture (look straight)
        Phase 2: Circle movement (frames captured at diverse angles)

        Returns list of BGR frames. Empty if cancelled.
        """
        cap = cv2.VideoCapture(CAMERA_SOURCE, CAMERA_BACKEND)
        if not cap.isOpened():
            raise CameraOpenError(CAMERA_SOURCE)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_CODEC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        window = f"Enroll - {name}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        captured_frames: list[np.ndarray] = []
        captured_angles: list[float] = []  # Angles at which captures happened
        phase = "center"  # "center" → "circle" → "done"
        last_capture_time = 0.0
        center_hold_start = None

        # Screen center (oval stays here always)
        scx, scy = CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2

        log.info("Face registration for '%s'", name)
        log.info("Follow on-screen instructions. Press ESC to cancel.")

        while phase != "done":
            ret, frame = cap.read()
            if not ret:
                continue

            # ── Detect face ───────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            face_found = bool(result.face_landmarks)
            nx, ny = 0.0, 0.0

            if face_found:
                lm = result.face_landmarks[0]
                nx, ny = self._estimate_head_offset(lm)

            now = time.time()

            num_captured = len(captured_frames)
            progress = num_captured / self.num_frames

            # ── Phase: CENTER ─────────────────────────────────────────
            if phase == "center":
                head_dist = math.hypot(nx, ny)
                if face_found and head_dist < 0.3:
                    # Face is roughly centered / looking straight
                    if center_hold_start is None:
                        center_hold_start = now
                    elif (now - center_hold_start) > 0.5:
                        if self._is_sharp(frame):
                            captured_frames.append(frame.copy())
                            log.info("[1/%d] Center captured", self.num_frames)
                            phase = "circle"
                            last_capture_time = now
                        else:
                            center_hold_start = now
                else:
                    center_hold_start = None

            # ── Phase: CIRCLE ─────────────────────────────────────────
            elif phase == "circle":
                if face_found and (now - last_capture_time) > 0.35:
                    head_dist = math.hypot(nx, ny)

                    # Capture when head is turned enough (not center)
                    if head_dist > 0.20:
                        angle = math.degrees(math.atan2(ny, nx)) % 360

                        # Check if this angle is far enough from all previous
                        too_close = any(
                            min(abs(angle - a), 360 - abs(angle - a)) < _CAPTURE_ANGLE_GAP
                            for a in captured_angles
                        )

                        if not too_close and self._is_sharp(frame):
                            captured_frames.append(frame.copy())
                            captured_angles.append(angle)
                            last_capture_time = now
                            n = len(captured_frames)
                            log.info("[%d/%d] Captured at %.0f°", n, self.num_frames, angle)

                            if n >= self.num_frames:
                                phase = "done"

            # ── Draw UI ───────────────────────────────────────────────
            display = frame.copy()
            num_captured = len(captured_frames)
            progress = num_captured / self.num_frames

            self._draw_ui(display, scx, scy, progress, face_found,
                          phase, num_captured, self.num_frames)

            cv2.imshow(window, display)

            if phase == "done":
                cv2.waitKey(800)  # Show "Done!" briefly
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                log.info("Enrollment cancelled by user.")
                captured_frames = []
                break

        cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

        return captured_frames

    def close(self):
        if self._owns_landmarker:
            self._landmarker.close()
