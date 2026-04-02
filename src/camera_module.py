"""
camera_module.py — Threaded Video Frame Capture
================================================
Runs a dedicated capture thread that continuously reads frames from
the camera and pushes them into a bounded deque (ring buffer).

Design decisions for Pi Zero 2W:
  • Threaded capture decouples I/O-bound camera reads from the
    CPU-bound inference pipeline, preventing frame drops.
  • A small deque (default 4 frames) caps memory usage at
    ~4 × 640×480×3 ≈ 3.5 MB — negligible on 512 MB RAM.
  • JPEG quality is kept at 80 to reduce SD card write size.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.config import (
    CAMERA_INDEX,
    CAPTURE_FPS,
    FRAME_BUFFER_SIZE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)

logger = logging.getLogger(__name__)


class CameraModule:
    """Manages threaded video capture from an OpenCV-compatible camera."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = CAPTURE_FPS,
        buffer_size: int = FRAME_BUFFER_SIZE,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps

        # Thread-safe ring buffer: newest frames push oldest out
        self._buffer: deque[Tuple[float, np.ndarray]] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # Control flags
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Stats
        self._frames_captured = 0
        self._frames_dropped = 0

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Open camera and start the capture thread."""
        if self._running.is_set():
            logger.warning("Camera is already running.")
            return

        self._cap = self._open_camera()
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Camera started — index=%d  resolution=%dx%d  target_fps=%d",
            self._camera_index, self._width, self._height, self._fps,
        )

    def stop(self) -> None:
        """Signal the capture thread to stop and release the camera."""
        if not self._running.is_set():
            return

        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info(
            "Camera stopped — captured=%d  dropped=%d",
            self._frames_captured, self._frames_dropped,
        )

    # ── Public Interface ─────────────────────────────────────

    def get_latest_frame(self) -> Optional[Tuple[float, np.ndarray]]:
        """
        Return the most recent (timestamp, frame) or None if buffer is empty.
        Non-blocking: the caller never waits for a new frame.
        """
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def get_all_frames(self) -> list:
        """Drain the buffer and return all available frames."""
        with self._lock:
            frames = list(self._buffer)
            self._buffer.clear()
        return frames

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def stats(self) -> dict:
        return {
            "captured": self._frames_captured,
            "dropped": self._frames_dropped,
            "buffer_size": len(self._buffer),
        }

    # ── Internals ────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        """Try to open the camera with configured resolution & FPS."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._camera_index}. "
                "Check connection and /dev/video* permissions."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Minimize internal buffering to get fresh frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def _capture_loop(self) -> None:
        """
        Main capture loop running on a background thread.
        Respects target FPS and handles transient camera errors.
        """
        frame_interval = 1.0 / self._fps
        consecutive_errors = 0
        max_consecutive_errors = 30  # ~2 seconds at 15 FPS

        while self._running.is_set():
            loop_start = time.monotonic()

            try:
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_errors += 1
                    self._frames_dropped += 1

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            "Camera returned %d consecutive bad frames — "
                            "attempting reconnect...", consecutive_errors,
                        )
                        self._attempt_reconnect()
                        consecutive_errors = 0

                    time.sleep(frame_interval)
                    continue

                # Successful read — reset error counter
                consecutive_errors = 0
                timestamp = time.time()

                with self._lock:
                    self._buffer.append((timestamp, frame))

                self._frames_captured += 1

            except Exception as exc:
                logger.error("Frame capture error: %s", exc, exc_info=True)
                self._frames_dropped += 1
                time.sleep(frame_interval)
                continue

            # Maintain target FPS (sleep for remaining interval)
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _attempt_reconnect(self) -> None:
        """Release and re-open the camera after persistent failures."""
        logger.warning("Reconnecting camera...")
        try:
            if self._cap is not None:
                self._cap.release()
            time.sleep(1.0)  # Brief cooldown before retry
            self._cap = self._open_camera()
            logger.info("Camera reconnected successfully.")
        except RuntimeError as exc:
            logger.error("Reconnect failed: %s", exc)

    # ── Context Manager ──────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
