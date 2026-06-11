"""
data_logger.py — Synchronized Frame + Telemetry Writer
=======================================================
Packages a flagged video frame with its correlated JSON telemetry
data into a timestamped "data packet" and writes it to local storage.

File Layout on SD Card:
    data/
    ├── frames/
    │   ├── 20260403_021500_123.jpg
    │   ├── 20260403_021502_456.jpg
    │   └── ...
    └── telemetry/
        ├── 20260403_021500_123.json
        ├── 20260403_021502_456.json
        └── ...

The frame and JSON share the same timestamp-based filename so they
can be trivially correlated during post-processing.

Performance notes:
  • JPEG encoding is done via OpenCV (cv2.imencode) which is faster
    than Pillow on ARM for this use case.
  • JSON writes are tiny (~500 bytes) so overhead is negligible.
  • Writes are synchronous-on-thread but called from the main loop,
    not from the camera thread, to avoid I/O contention.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.config import CLIPS_DIR, FRAMES_DIR, JPEG_QUALITY, TELEMETRY_DIR

logger = logging.getLogger(__name__)


class DataLogger:
    """
    Writes anomaly frames and their telemetry to local storage.

    Each save produces two files with matching timestamps:
      - A JPEG image (compressed frame)
      - A JSON document (telemetry + inference metadata)
    """

    def __init__(
        self,
        frames_dir: Path = FRAMES_DIR,
        telemetry_dir: Path = TELEMETRY_DIR,
        clips_dir: Path = CLIPS_DIR,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        self._frames_dir = frames_dir
        self._telemetry_dir = telemetry_dir
        self._clips_dir = clips_dir
        self._jpeg_quality = jpeg_quality

        # Ensure output directories exist
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry_dir.mkdir(parents=True, exist_ok=True)
        self._clips_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self._packets_saved = 0
        self._total_bytes_written = 0

    # ── Public Interface ─────────────────────────────────────

    def save_packet(
        self,
        frame: np.ndarray,
        timestamp: float,
        telemetry: Optional[dict] = None,
        inference: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Save a synchronized data packet (frame + telemetry JSON).

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.
        timestamp : float
            Unix timestamp of the frame capture.
        telemetry : dict, optional
            OBD-II sensor readings (from TelemetryPacket.to_dict()).
        inference : dict, optional
            YOLO inference results (from InferenceResult.to_dict()).

        Returns
        -------
        str or None
            The base filename (without extension) on success, None on failure.
        """
        try:
            # Generate a unique, sortable filename from the timestamp
            base_name = self._timestamp_to_filename(timestamp)

            # ── Save JPEG frame ──────────────────────────────
            frame_path = self._frames_dir / f"{base_name}.jpg"
            success = self._save_frame(frame, frame_path)
            if not success:
                return None

            # ── Save JSON telemetry ──────────────────────────
            json_path = self._telemetry_dir / f"{base_name}.json"
            json_data = self._build_json_packet(
                timestamp=timestamp,
                frame_filename=f"{base_name}.jpg",
                telemetry=telemetry,
                inference=inference,
            )
            self._save_json(json_data, json_path)

            self._packets_saved += 1

            logger.info(
                "Packet saved: %s  (frame=%.1f KB)",
                base_name,
                os.path.getsize(frame_path) / 1024,
            )
            return base_name

        except Exception as exc:
            logger.error("Failed to save data packet: %s", exc, exc_info=True)
            return None

    def save_clip(
        self,
        clip_frames: list[tuple[float, np.ndarray]],
        event_timestamp: float,
        telemetry: Optional[dict] = None,
        inference: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Save a temporal anomaly clip as an .npz file for cloud scoring.

        The cloud scorer expects `frames` and `timestamps` arrays, so this
        method is the bridge between edge capture and cloud-side ConvLSTM
        scoring.
        """
        if not clip_frames:
            return None

        try:
            base_name = self._timestamp_to_filename(event_timestamp)
            clip_path = self._clips_dir / f"{base_name}.npz"

            timestamps = np.array([ts for ts, _ in clip_frames], dtype=np.float64)
            frames = np.stack([frame for _, frame in clip_frames], axis=0)
            metadata = {
                "event_timestamp": event_timestamp,
                "telemetry": telemetry or {},
                "inference": inference or {},
            }

            np.savez_compressed(
                str(clip_path),
                frames=frames,
                timestamps=timestamps,
                metadata=np.array(metadata, dtype=object),
            )
            self._total_bytes_written += os.path.getsize(clip_path)

            logger.info(
                "Clip saved: %s  (frames=%d, size=%.1f KB)",
                clip_path.name,
                len(clip_frames),
                os.path.getsize(clip_path) / 1024,
            )
            return base_name

        except Exception as exc:
            logger.error("Failed to save clip: %s", exc, exc_info=True)
            return None

    @property
    def stats(self) -> dict:
        return {
            "packets_saved": self._packets_saved,
            "total_bytes_written": self._total_bytes_written,
            "total_mb_written": round(self._total_bytes_written / (1024 * 1024), 2),
        }

    # ── Internals ────────────────────────────────────────────

    def _save_frame(self, frame: np.ndarray, path: Path) -> bool:
        """Encode and write a frame as JPEG."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        success, encoded = cv2.imencode(".jpg", frame, encode_params)

        if not success:
            logger.error("JPEG encoding failed for %s", path.name)
            return False

        encoded.tofile(str(path))
        self._total_bytes_written += os.path.getsize(path)
        return True

    def _save_json(self, data: dict, path: Path) -> None:
        """Write JSON telemetry to disk."""
        raw = json.dumps(data, indent=2, ensure_ascii=False)
        path.write_text(raw, encoding="utf-8")
        self._total_bytes_written += len(raw.encode("utf-8"))

    @staticmethod
    def _build_json_packet(
        timestamp: float,
        frame_filename: str,
        telemetry: Optional[dict],
        inference: Optional[dict],
    ) -> dict:
        """Assemble the complete JSON document for one data packet."""
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        packet = {
            "metadata": {
                "timestamp_unix": timestamp,
                "timestamp_utc": dt.isoformat(),
                "frame_file": frame_filename,
                "node_id": "adas-pi-001",  # Unique per device in fleet
            },
            "telemetry": telemetry or {},
            "inference": inference or {},
        }
        return packet

    @staticmethod
    def _timestamp_to_filename(ts: float) -> str:
        """
        Convert a Unix timestamp into a filename-safe string.
        Format: YYYYMMDD_HHMMSS_mmm  (mmm = milliseconds)
        Example: 20260403_021500_123
        """
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        millis = int((ts % 1) * 1000)
        return dt.strftime("%Y%m%d_%H%M%S") + f"_{millis:03d}"
