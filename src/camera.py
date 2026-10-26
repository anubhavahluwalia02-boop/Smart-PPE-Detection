"""
Thread-safe OpenCV Camera Manager for video streaming and capture.
"""
import time
import threading
import logging
from typing import Optional, Generator
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages webcam capture in a thread-safe manner,
    handling camera release, reconnection, and fallback test frames.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.last_frame: Optional[np.ndarray] = None
        self.error_message: Optional[str] = None

    def start(self) -> bool:
        """Initializes the webcam device."""
        with self.lock:
            if self.is_running and self.cap and self.cap.isOpened():
                return True

            self.error_message = None
            try:
                # On Windows, cv2.CAP_DSHOW can prevent delays
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap or not self.cap.isOpened():
                    # Retry with directshow on Windows
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

                if not self.cap or not self.cap.isOpened():
                    self.error_message = f"Webcam at index {self.camera_index} could not be opened. Check permissions or camera connection."
                    self.is_running = False
                    return False

                # Optimize stream parameters
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                self.is_running = True
                return True
            except Exception as e:
                self.error_message = f"Camera initialization exception: {str(e)}"
                self.is_running = False
                return False

    def stop(self):
        """Releases the camera device cleanly."""
        with self.lock:
            self.is_running = False
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a single frame from the camera."""
        with self.lock:
            if not self.is_running or not self.cap or not self.cap.isOpened():
                return None

            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None

            self.last_frame = frame
            return frame.copy()

    def generate_placeholder_frame(self, message: str = "Camera Standby") -> np.ndarray:
        """Generates a synthetic camera screen when the camera is offline."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background dark blue-grey
        img[:] = (35, 30, 30)

        # Crosshatch pattern
        for y in range(0, 480, 40):
            cv2.line(img, (0, y), (640, y), (45, 40, 40), 1)
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 480), (45, 40, 40), 1)

        cv2.putText(
            img,
            "SMART PPE DETECTION SYSTEM",
            (120, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (46, 204, 113),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            message,
            (170, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            "Click 'Start Camera' button to stream",
            (160, 295),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
            cv2.LINE_AA
        )
        return img
