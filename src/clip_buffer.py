"""
clip_buffer.py — Rolling Frame Buffer for Spatio-Temporal Clip Capture
=======================================================================
Maintains a fixed-size rolling window of recent camera frames. When an
anomaly is flagged by the ActiveLearner, the ClipBuffer can instantly
produce a short video clip spanning frames BEFORE and AFTER the event.

This temporal context is critical for the cloud-side ConvLSTM model,
which needs sequential frame data to detect motion-based anomalies
(e.g., a cow suddenly crossing a highway, a vehicle swerving).

Memory budget on Pi Zero 2W:
  30 frames × 640×480×3 bytes ≈ 27 MB  (within the 512 MB budget)

This module is fully self-contained and does NOT modify any existing code.

Usage:
    buffer = ClipBuffer()
    buffer.push(timestamp, frame)       # called every tick
    ...
    clip = buffer.capture_clip()        # called when anomaly detected
"""

import logging
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Self-contained configuration ─────────────────────────────
# These defaults mirror the existing ADAS config but are defined
# here to avoid modifying utils/config.py.

_DEFAULT_FPS = 15
_DEFAULT_BUFFER_SECONDS = 2
_DEFAULT_PRE_EVENT_FRAMES = 15
_DEFAULT_POST_EVENT_FRAMES = 15


class ClipBuffer:
    """
    Thread-safe rolling buffer that stores recent frames for clip extraction.

    The buffer is a bounded deque that automatically discards the oldest
    frames when full. When an anomaly is detected, `capture_clip()` grabs
    the surrounding frames to create a temporal context clip.
    """

    def __init__(
        self,
        buffer_seconds: float = _DEFAULT_BUFFER_SECONDS,
        fps: int = _DEFAULT_FPS,
        pre_event_frames: int = _DEFAULT_PRE_EVENT_FRAMES,
        post_event_frames: int = _DEFAULT_POST_EVENT_FRAMES,
    ) -> None:
        # Total buffer capacity = enough frames to cover the rolling window
        self._buffer_capacity = int(buffer_seconds * fps) + post_event_frames
        self._pre_event_frames = pre_event_frames
        self._post_event_frames = post_event_frames
        self._fps = fps

        # Thread-safe ring buffer: (timestamp, frame) tuples
        self._buffer: deque[Tuple[float, np.ndarray]] = deque(
            maxlen=self._buffer_capacity
        )
        self._lock = threading.Lock()

        # Cooldown: prevent capturing overlapping clips for rapid-fire anomalies
        self._last_clip_time: float = 0.0
        self._cooldown_seconds: float = 1.0  # Minimum gap between clip captures

        # Stats
        self._frames_buffered: int = 0
        self._clips_captured: int = 0

    # ── Public Interface ─────────────────────────────────────

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        """
        Add a frame to the rolling buffer. Called every tick from main loop.

        Parameters
        ----------
        timestamp : float
            Unix timestamp of frame capture.
        frame : np.ndarray
            BGR image from OpenCV.
        """
        with self._lock:
            self._buffer.append((timestamp, frame.copy()))
        self._frames_buffered += 1

    def capture_clip(
        self,
        pre_frames: Optional[int] = None,
    ) -> Optional[List[Tuple[float, np.ndarray]]]:
        """
        Extract a clip of frames surrounding the current moment (anomaly event).

        Returns the last `pre_frames` frames from the buffer immediately.

        Parameters
        ----------
        pre_frames : int, optional
            Number of frames before the event (default: _DEFAULT_PRE_EVENT_FRAMES).

        Returns
        -------
        list of (timestamp, frame) or None
            The pre-event clip, or None if on cooldown / buffer too small.
        """
        pre = pre_frames or self._pre_event_frames
        now = time.monotonic()

        # Cooldown check: don't spam clips for rapid anomalies
        if now - self._last_clip_time < self._cooldown_seconds:
            logger.debug("Clip capture skipped — cooldown active.")
            return None

        with self._lock:
            available = len(self._buffer)
            if available == 0:
                logger.warning("Clip capture failed — buffer is empty.")
                return None

            # Take up to `pre` most recent frames
            take = min(pre, available)
            clip_frames = list(self._buffer)[-take:]

        self._last_clip_time = now
        self._clips_captured += 1

        logger.info(
            "Clip captured — %d pre-event frames (buffer had %d).",
            len(clip_frames), available,
        )
        return clip_frames

    def capture_full_clip(self) -> Optional[List[Tuple[float, np.ndarray]]]:
        """
        Capture a full clip using the entire buffer contents.
        Useful for grabbing maximum temporal context.

        Returns
        -------
        list of (timestamp, frame) or None
        """
        now = time.monotonic()
        if now - self._last_clip_time < self._cooldown_seconds:
            return None

        with self._lock:
            if not self._buffer:
                return None
            clip_frames = list(self._buffer)

        self._last_clip_time = now
        self._clips_captured += 1

        logger.info("Full clip captured — %d frames.", len(clip_frames))
        return clip_frames

    @property
    def buffer_size(self) -> int:
        """Current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> dict:
        return {
            "frames_buffered": self._frames_buffered,
            "clips_captured": self._clips_captured,
            "buffer_capacity": self._buffer_capacity,
            "current_size": self.buffer_size,
        }

    def clear(self) -> None:
        """Flush the buffer (e.g., on memory pressure)."""
        with self._lock:
            self._buffer.clear()
        logger.debug("Clip buffer cleared.")
