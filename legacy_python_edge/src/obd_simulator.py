"""
obd_simulator.py — Mock OBD-II / CAN Bus Telemetry Generator
=============================================================
Simulates realistic vehicle telematics data on a background thread,
allowing the rest of the pipeline to be developed without real
OBD-II hardware.

Generated signals:
  • Engine RPM       (700–6 500)
  • Vehicle Speed    (0–120 km/h)
  • Steering Angle   (−540° to +540°)
  • Braking Pressure (0.0–1.0 normalized)

The simulator uses smooth random walks so values change gradually,
mimicking real driving patterns (acceleration → cruise → braking).
"""

import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from utils.config import (
    OBD_BRAKE_PRESSURE_RANGE,
    OBD_POLL_INTERVAL,
    OBD_RPM_RANGE,
    OBD_SPEED_RANGE,
    OBD_STEERING_RANGE,
)

logger = logging.getLogger(__name__)


@dataclass
class TelemetryPacket:
    """Immutable snapshot of vehicle telematics at a single timestamp."""

    timestamp: float
    engine_rpm: float
    vehicle_speed_kmh: float
    steering_angle_deg: float
    brake_pressure: float

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON encoding)."""
        return asdict(self)


class OBDSimulator:
    """
    Background thread that generates mock CAN bus telemetry.

    Usage:
        sim = OBDSimulator()
        sim.start()
        packet = sim.get_latest()   # non-blocking
        sim.stop()
    """

    def __init__(self, poll_interval: float = OBD_POLL_INTERVAL) -> None:
        self._poll_interval = poll_interval

        # Current simulated state (smooth random walk)
        self._rpm = float(random.randint(*OBD_RPM_RANGE[:1] * 2))
        self._speed = 0.0
        self._steering = 0.0
        self._brake = 0.0

        # Thread-safe latest reading
        self._latest: Optional[TelemetryPacket] = None
        self._lock = threading.Lock()

        # Control
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._packets_generated = 0

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Begin generating mock telemetry on a background thread."""
        if self._running.is_set():
            logger.warning("OBD simulator is already running.")
            return

        self._running.set()
        self._thread = threading.Thread(
            target=self._generate_loop,
            name="OBDSimThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("OBD-II simulator started — poll_interval=%.2fs", self._poll_interval)

    def stop(self) -> None:
        """Stop the telemetry generator."""
        if not self._running.is_set():
            return

        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("OBD-II simulator stopped — packets_generated=%d", self._packets_generated)

    # ── Public Interface ─────────────────────────────────────

    def get_latest(self) -> Optional[TelemetryPacket]:
        """Return the most recent telemetry reading (non-blocking)."""
        with self._lock:
            return self._latest

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ── Internals ────────────────────────────────────────────

    def _generate_loop(self) -> None:
        """Main generation loop — smooth random walk for each signal."""
        while self._running.is_set():
            try:
                self._step_simulation()

                packet = TelemetryPacket(
                    timestamp=time.time(),
                    engine_rpm=round(self._rpm, 1),
                    vehicle_speed_kmh=round(self._speed, 1),
                    steering_angle_deg=round(self._steering, 1),
                    brake_pressure=round(self._brake, 3),
                )

                with self._lock:
                    self._latest = packet

                self._packets_generated += 1

            except Exception as exc:
                logger.error("OBD simulation error: %s", exc, exc_info=True)

            time.sleep(self._poll_interval)

    def _step_simulation(self) -> None:
        """
        Advance each signal by a small random delta, clamped to safe ranges.
        The step sizes are tuned so values change realistically over time.
        """
        # Engine RPM — drifts ±200 per tick
        self._rpm += random.uniform(-200, 200)
        self._rpm = self._clamp(self._rpm, *OBD_RPM_RANGE)

        # Vehicle Speed — drifts ±5 km/h per tick
        self._speed += random.uniform(-5, 5)
        self._speed = self._clamp(self._speed, *OBD_SPEED_RANGE)

        # Steering Angle — drifts ±30 deg per tick
        self._steering += random.uniform(-30, 30)
        self._steering = self._clamp(self._steering, *OBD_STEERING_RANGE)

        # Braking Pressure — drifts ±0.1 per tick
        self._brake += random.uniform(-0.1, 0.1)
        self._brake = self._clamp(self._brake, *OBD_BRAKE_PRESSURE_RANGE)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp a value to [lo, hi]."""
        return max(lo, min(hi, value))

    # ── Context Manager ──────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
