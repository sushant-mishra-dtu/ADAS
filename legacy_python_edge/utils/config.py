"""
config.py — Centralized Configuration for ADAS
===================================================
All tunable constants live here. Optimized defaults for
Raspberry Pi Zero 2W (512 MB RAM, quad-core 1 GHz).
"""

import os
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
TELEMETRY_DIR = DATA_DIR / "telemetry"
CLIPS_DIR = DATA_DIR / "clips"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories on import (idempotent)
for _dir in (FRAMES_DIR, TELEMETRY_DIR, CLIPS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Camera Settings ──────────────────────────────────────────
CAMERA_INDEX = 0               # /dev/video0  (or 0 for default)
CAPTURE_FPS = 15               # Target FPS (Pi Zero can sustain ~15 @ 640x480)
FRAME_WIDTH = 640              # Capture width  (keep low for Pi Zero)
FRAME_HEIGHT = 480             # Capture height
JPEG_QUALITY = 80              # JPEG encode quality (0-100, lower = smaller file)

# ── Frame Buffer ─────────────────────────────────────────────
FRAME_BUFFER_SIZE = 4          # Max frames held in the ring buffer
                               # Small value prevents RAM overload on Pi Zero

# ── OBD-II / CAN Bus Simulator ──────────────────────────────
OBD_POLL_INTERVAL = 0.1        # Seconds between telemetry samples (10 Hz)

# Realistic value ranges for simulated data
OBD_RPM_RANGE = (700, 6500)            # Engine RPM
OBD_SPEED_RANGE = (0, 120)             # km/h
OBD_STEERING_RANGE = (-540, 540)       # Degrees (-left, +right)
OBD_BRAKE_PRESSURE_RANGE = (0.0, 1.0)  # Normalized 0..1

# ── Active Learning / YOLO ───────────────────────────────────
YOLO_MODEL_PATH = str(MODELS_DIR / "yolov8n.pt")  # YOLOv8 Nano weights
YOLO_IMGSZ = 320              # Inference resolution (smaller = faster on Pi)
YOLO_CONF = 0.25              # YOLO's own internal confidence filter

# Anomaly / uncertainty threshold:
# If the HIGHEST detection confidence in a frame is BELOW this value,
# we consider the frame an "edge case" worth saving.
CONFIDENCE_THRESHOLD = 0.35

# ── System Resource Limits ───────────────────────────────────
MAX_MEMORY_PERCENT = 80        # Pause capture if RAM usage exceeds this %
GC_INTERVAL_SECONDS = 30       # Force garbage collection every N seconds

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("ADAS_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(name)-16s] %(levelname)-7s %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
