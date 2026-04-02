"""
main.py — ADAS Pipeline Orchestrator
=========================================
The main entry point that initializes all subsystems, manages the
processing loops, and ensures the Pi Zero 2W's memory stays within
safe bounds.

Pipeline Flow (per tick):
  1. Grab the latest frame from the camera ring buffer.
  2. Read the latest OBD-II telemetry snapshot.
  3. Run YOLOv8n inference → get confidence scores.
  4. If anomaly detected (low confidence) → save frame + telemetry.
  5. Force garbage collection periodically to reclaim memory.

The main loop runs on the primary thread while camera capture and
OBD-II simulation each run on dedicated daemon threads.
"""

import gc
import logging
import signal
import sys
import time
from typing import Optional

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from utils.config import (
    CONFIDENCE_THRESHOLD,
    GC_INTERVAL_SECONDS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_MEMORY_PERCENT,
)

from src.camera_module import CameraModule
from src.obd_simulator import OBDSimulator
from src.active_learner import ActiveLearner
from src.data_logger import DataLogger

logger = logging.getLogger("adas")


# ── Logging Setup ────────────────────────────────────────────

def setup_logging() -> None:
    """Configure root logger with console output."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Memory Monitor ───────────────────────────────────────────

def check_memory_pressure() -> bool:
    """
    Return True if system memory usage exceeds the safe threshold.
    Falls back to False if psutil is not installed.
    """
    if not HAS_PSUTIL:
        return False

    mem = psutil.virtual_memory()
    if mem.percent >= MAX_MEMORY_PERCENT:
        logger.warning(
            "⚠ Memory pressure: %.1f%% used (threshold: %d%%)",
            mem.percent, MAX_MEMORY_PERCENT,
        )
        return True
    return False


def log_system_stats() -> None:
    """Log current CPU and memory usage."""
    if not HAS_PSUTIL:
        return

    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    logger.info(
        "System — CPU: %.1f%%  RAM: %.1f%% (%d MB / %d MB)",
        cpu,
        mem.percent,
        mem.used // (1024 * 1024),
        mem.total // (1024 * 1024),
    )


# ── Main Pipeline ────────────────────────────────────────────

class EdgeDashPipeline:
    """
    Orchestrates the full dashcam data collection pipeline.

    Manages lifecycle of:
      - CameraModule      (threaded frame capture)
      - OBDSimulator       (threaded telemetry generation)
      - ActiveLearner      (YOLOv8n inference)
      - DataLogger         (frame + JSON writer)
    """

    def __init__(self) -> None:
        self.camera = CameraModule()
        self.obd = OBDSimulator()
        self.learner = ActiveLearner()
        self.logger = DataLogger()

        self._running = False
        self._tick_count = 0
        self._last_gc = time.monotonic()

    def start(self) -> None:
        """Initialize all subsystems."""
        logger.info("=" * 60)
        logger.info("  ADAS — Advanced Driver Assistance System")
        logger.info("  Confidence threshold: %.2f", CONFIDENCE_THRESHOLD)
        logger.info("=" * 60)

        # Load YOLO model (with graceful fallback)
        self.learner.load_model()

        # Start background threads
        self.camera.start()
        self.obd.start()

        self._running = True
        logger.info("All subsystems initialized. Pipeline is RUNNING.")

    def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        self._running = False
        self.camera.stop()
        self.obd.stop()

        # Final stats
        logger.info("─" * 40)
        logger.info("Pipeline shut down. Final statistics:")
        logger.info("  Camera:    %s", self.camera.stats)
        logger.info("  OBD:       packets_generated=%d", self.obd._packets_generated)
        logger.info("  Learner:   %s", self.learner.stats)
        logger.info("  Logger:    %s", self.logger.stats)
        logger.info("─" * 40)

    def run_forever(self) -> None:
        """
        Main processing loop. Runs until interrupted (Ctrl-C / SIGTERM).

        Each iteration:
          1. Check memory pressure → pause if overloaded.
          2. Grab latest frame and telemetry.
          3. Run inference → decide if anomaly.
          4. Save data if anomaly detected.
          5. Periodic garbage collection.
        """
        self.start()

        try:
            while self._running:
                self._tick()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt.")
        except Exception as exc:
            logger.critical("Unhandled exception in main loop: %s", exc, exc_info=True)
        finally:
            self.stop()

    def _tick(self) -> None:
        """Single iteration of the main pipeline loop."""
        self._tick_count += 1

        # ── 1. Memory guard ──────────────────────────────────
        if check_memory_pressure():
            logger.warning("Pausing pipeline for 2s to relieve memory...")
            gc.collect()
            time.sleep(2.0)
            return

        # ── 2. Get latest frame ──────────────────────────────
        frame_data = self.camera.get_latest_frame()
        if frame_data is None:
            # Camera hasn't produced a frame yet — wait briefly
            time.sleep(0.05)
            return

        timestamp, frame = frame_data

        # ── 3. Get latest telemetry ──────────────────────────
        telemetry_packet = self.obd.get_latest()
        telemetry_dict = telemetry_packet.to_dict() if telemetry_packet else {}

        # ── 4. Run inference ─────────────────────────────────
        result = self.learner.process_frame(frame)

        # ── 5. Save if anomaly ───────────────────────────────
        if result.is_anomaly:
            self.logger.save_packet(
                frame=frame,
                timestamp=timestamp,
                telemetry=telemetry_dict,
                inference=result.to_dict(),
            )

        # ── 6. Periodic housekeeping ─────────────────────────
        now = time.monotonic()
        if now - self._last_gc >= GC_INTERVAL_SECONDS:
            gc.collect()
            self._last_gc = now

            # Log stats every GC cycle
            if self._tick_count % 100 == 0:
                log_system_stats()
                logger.info(
                    "Pipeline tick #%d — anomalies=%d/%d",
                    self._tick_count,
                    self.learner.stats["anomalies_flagged"],
                    self.learner.stats["frames_processed"],
                )

        # Small yield to prevent CPU spin on Pi Zero
        time.sleep(0.01)


# ── Entry Point ──────────────────────────────────────────────

def main() -> None:
    """Entry point for `python -m src.main` (run from ADAS directory)."""
    setup_logging()

    pipeline = EdgeDashPipeline()

    # Handle SIGTERM gracefully (systemd, Docker stop)
    def _signal_handler(signum, frame):
        logger.info("Received signal %d — shutting down...", signum)
        pipeline._running = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    pipeline.run_forever()


if __name__ == "__main__":
    main()
