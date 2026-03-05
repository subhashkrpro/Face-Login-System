"""Threaded webcam capture with pre-buffering."""

import cv2
import threading
import time
from collections import deque

from config import (
    CAMERA_SOURCE, CAMERA_BACKEND, CAMERA_CODEC,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_BUFFER_SIZE,
    DEFAULT_BUFFER_SIZE, CAPTURE_TIMEOUT_SEC,
)
from src.exceptions import CameraOpenError, CameraTimeoutError


class FastStream:
    """Threaded webcam capture with configurable pre-buffering."""

    def __init__(self, src=None, buffer_size=None):
        src = src if src is not None else CAMERA_SOURCE
        buffer_size = buffer_size or DEFAULT_BUFFER_SIZE

        self.cap = cv2.VideoCapture(src, CAMERA_BACKEND)

        if not self.cap.isOpened():
            raise CameraOpenError(src)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_CODEC))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        self.grabbed, self.frame = self.cap.read()
        self.running = True
        self._lock = threading.Lock()

        self._buffer_size = buffer_size
        self._frame_buffer = deque(maxlen=buffer_size)
        self._buffer_ready = threading.Event()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.001)
                continue
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame

            if not self._buffer_ready.is_set():
                self._frame_buffer.append(frame)
                if len(self._frame_buffer) >= self._buffer_size:
                    self._buffer_ready.set()

    def read(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def capture_frames(self, timeout=None) -> list:
        """Wait for buffer_size frames, return copies."""
        timeout = timeout or CAPTURE_TIMEOUT_SEC
        if not self._buffer_ready.wait(timeout=timeout):
            raise CameraTimeoutError(self._buffer_size, timeout)
        return list(self._frame_buffer)

    def stop(self):
        self.running = False
        self.cap.release()
