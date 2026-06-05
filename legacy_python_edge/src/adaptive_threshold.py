"""
adaptive_threshold.py — Environment-Aware Dynamic Anomaly Threshold
=====================================================================
Replaces the fixed CONFIDENCE_THRESHOLD=0.35 with an intelligent,
self-adjusting threshold that adapts to current driving conditions.

This module is fully self-contained and does NOT modify any existing code.

Components:
  1. EnvironmentClassifier — categorizes driving conditions from frame stats
  2. AdaptiveThreshold     — rolling-window dynamic threshold engine
  3. ThresholdResult       — dataclass holding the decision + diagnostics

How it works:
  - Maintains a rolling window of recent YOLOv8 confidence scores
  - Computes: rolling_mean, rolling_std
  - Dynamic threshold = rolling_mean - k × rolling_std
  - k (sensitivity) varies by environment:
      Clear day → k=1.0   (standard sensitivity)
      Night     → k=1.5   (more selective — model is always confused)
      Rain/fog  → k=1.5   (same — don't save every foggy frame)
      Dense     → k=1.2   (slightly more selective in crowds)
      Open road → k=0.8   (more sensitive — anomalies are rare here)
  - Hard bounds prevent extreme values: [min_threshold, max_threshold]

Usage:
    threshold = AdaptiveThreshold()

    # Each frame: feed the max YOLO confidence + frame metadata
    result = threshold.update(
        max_confidence=0.42,
        frame_brightness=0.65,
        num_detections=3,
    )
    print(result.is_anomaly)         # True/False
    print(result.dynamic_threshold)  # 0.28  (adapted)
    print(result.environment)        # "CLEAR_DAY"
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 200       # Rolling window of recent confidence scores
DEFAULT_BASELINE = 0.35         # Fallback threshold (matches existing config)
DEFAULT_MIN_THRESHOLD = 0.10    # Never go below this (too permissive)
DEFAULT_MAX_THRESHOLD = 0.60    # Never go above this (too restrictive)
DEFAULT_WARMUP_FRAMES = 50      # Use baseline until this many scores collected
DEFAULT_K = 1.0                 # Default sensitivity multiplier


# ══════════════════════════════════════════════════════════════
#  Environment Classification
# ══════════════════════════════════════════════════════════════

class Environment(Enum):
    """Driving environment categories with sensitivity multipliers."""

    CLEAR_DAY = ("clear_day", 1.0)
    NIGHT = ("night", 1.5)
    RAIN_FOG = ("rain_fog", 1.5)
    DENSE_TRAFFIC = ("dense_traffic", 1.2)
    OPEN_ROAD = ("open_road", 0.8)
    UNKNOWN = ("unknown", 1.0)

    def __init__(self, label: str, sensitivity: float) -> None:
        self.label = label
        self.sensitivity = sensitivity


class EnvironmentClassifier:
    """
    Classifies driving environment using frame statistics.

    Uses:
      - Frame brightness (mean pixel value / 255)
      - Detection density (number of YOLO detections)
      - Brightness variance (low variance + low brightness = fog/rain)

    This is a lightweight heuristic — not a learned classifier.
    """

    def __init__(
        self,
        night_brightness: float = 0.25,
        fog_brightness_range: tuple = (0.25, 0.55),
        fog_variance_threshold: float = 0.02,
        dense_detection_count: int = 5,
        open_detection_count: int = 1,
    ) -> None:
        self._night_brightness = night_brightness
        self._fog_bright_lo = fog_brightness_range[0]
        self._fog_bright_hi = fog_brightness_range[1]
        self._fog_var_threshold = fog_variance_threshold
        self._dense_count = dense_detection_count
        self._open_count = open_detection_count

        # Smoothing: track recent classifications to prevent flickering
        self._history: deque = deque(maxlen=10)

    def classify(
        self,
        brightness: float,
        num_detections: int,
        brightness_variance: float = 0.1,
    ) -> Environment:
        """
        Classify the current driving environment.

        Parameters
        ----------
        brightness : float
            Mean frame brightness ∈ [0, 1].
        num_detections : int
            Number of YOLO detections in the current frame.
        brightness_variance : float
            Variance of pixel brightness (low = uniform = fog/rain).

        Returns
        -------
        Environment
        """
        # Night: very dark frame
        if brightness < self._night_brightness:
            env = Environment.NIGHT

        # Rain/fog: medium brightness but very low variance (uniform gray)
        elif (
            self._fog_bright_lo <= brightness <= self._fog_bright_hi
            and brightness_variance < self._fog_var_threshold
        ):
            env = Environment.RAIN_FOG

        # Dense traffic: many detections
        elif num_detections >= self._dense_count:
            env = Environment.DENSE_TRAFFIC

        # Open road: very few or no detections
        elif num_detections <= self._open_count:
            env = Environment.OPEN_ROAD

        # Default: clear day
        else:
            env = Environment.CLEAR_DAY

        self._history.append(env)
        return env

    def get_smoothed(self) -> Environment:
        """Return the most common environment over the recent window."""
        if not self._history:
            return Environment.UNKNOWN

        from collections import Counter
        counts = Counter(self._history)
        return counts.most_common(1)[0][0]


# ══════════════════════════════════════════════════════════════
#  Threshold Result
# ══════════════════════════════════════════════════════════════

@dataclass
class ThresholdResult:
    """Result of the adaptive threshold decision for one frame."""

    is_anomaly: bool                # The final decision
    dynamic_threshold: float        # The threshold that was used
    max_confidence: float           # The YOLO confidence input
    environment: str                # Classified driving environment
    sensitivity: float              # k multiplier used
    rolling_mean: float             # Mean of recent confidences
    rolling_std: float              # Std dev of recent confidences
    is_warmup: bool                 # True if still in warmup period
    reason: str                     # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "dynamic_threshold": round(self.dynamic_threshold, 4),
            "max_confidence": round(self.max_confidence, 4),
            "environment": self.environment,
            "sensitivity": round(self.sensitivity, 2),
            "rolling_mean": round(self.rolling_mean, 4),
            "rolling_std": round(self.rolling_std, 4),
            "is_warmup": self.is_warmup,
            "reason": self.reason,
        }


# ══════════════════════════════════════════════════════════════
#  Adaptive Threshold Engine
# ══════════════════════════════════════════════════════════════

class AdaptiveThreshold:
    """
    Self-adjusting anomaly threshold based on rolling statistics.

    Instead of a fixed 0.35 cutoff, the threshold adapts:
      threshold = clamp(rolling_mean - k * rolling_std, min, max)

    Where k varies by environment (night/rain → more selective).

    Parameters
    ----------
    window_size : int
        Number of recent scores to track (default: 200).
    baseline : float
        Fixed threshold used during warmup (default: 0.35).
    min_threshold : float
        Minimum allowed threshold (default: 0.10).
    max_threshold : float
        Maximum allowed threshold (default: 0.60).
    warmup_frames : int
        Number of frames before adaptive mode kicks in (default: 50).
    base_k : float
        Base sensitivity multiplier (default: 1.0).
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        baseline: float = DEFAULT_BASELINE,
        min_threshold: float = DEFAULT_MIN_THRESHOLD,
        max_threshold: float = DEFAULT_MAX_THRESHOLD,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
        base_k: float = DEFAULT_K,
    ) -> None:
        self._window_size = window_size
        self._baseline = baseline
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._warmup_frames = warmup_frames
        self._base_k = base_k

        # Rolling window of max confidence scores
        self._scores: deque = deque(maxlen=window_size)

        # Environment classifier
        self._env_classifier = EnvironmentClassifier()

        # Stats
        self._frame_count: int = 0
        self._anomaly_count: int = 0

    def update(
        self,
        max_confidence: float,
        frame_brightness: float = 0.5,
        num_detections: int = 0,
        brightness_variance: float = 0.1,
    ) -> ThresholdResult:
        """
        Feed one frame's data and get the adaptive anomaly decision.

        Parameters
        ----------
        max_confidence : float
            Maximum YOLO detection confidence for this frame.
        frame_brightness : float
            Mean frame brightness ∈ [0, 1] (default: 0.5).
        num_detections : int
            Number of detections in this frame (default: 0).
        brightness_variance : float
            Pixel brightness variance (default: 0.1).

        Returns
        -------
        ThresholdResult
        """
        self._frame_count += 1
        self._scores.append(max_confidence)

        # ── Classify environment ─────────────────────────────
        env = self._env_classifier.classify(
            brightness=frame_brightness,
            num_detections=num_detections,
            brightness_variance=brightness_variance,
        )

        # ── Check warmup ─────────────────────────────────────
        is_warmup = len(self._scores) < self._warmup_frames

        if is_warmup:
            # Use fixed baseline during warmup
            threshold = self._baseline
            rolling_mean = self._baseline
            rolling_std = 0.0
            k = self._base_k
            reason = f"Warmup ({len(self._scores)}/{self._warmup_frames})"
        else:
            # ── Compute rolling statistics ───────────────────
            scores_arr = np.array(self._scores, dtype=np.float64)
            rolling_mean = float(np.mean(scores_arr))
            rolling_std = float(np.std(scores_arr))

            # ── Environment-aware sensitivity ────────────────
            k = self._base_k * env.sensitivity

            # ── Dynamic threshold ────────────────────────────
            threshold = rolling_mean - k * rolling_std

            # ── Clamp to safe bounds ─────────────────────────
            threshold = max(self._min_threshold, min(self._max_threshold, threshold))

            reason = (
                f"Adaptive: mean={rolling_mean:.3f} - "
                f"{k:.1f}×std={rolling_std:.3f} → {threshold:.3f} "
                f"[{env.label}]"
            )

        # ── Make decision ────────────────────────────────────
        is_anomaly = (
            max_confidence < threshold
            or num_detections == 0
        )

        if is_anomaly:
            self._anomaly_count += 1

        return ThresholdResult(
            is_anomaly=is_anomaly,
            dynamic_threshold=threshold,
            max_confidence=max_confidence,
            environment=env.label,
            sensitivity=k if not is_warmup else self._base_k,
            rolling_mean=rolling_mean if not is_warmup else self._baseline,
            rolling_std=rolling_std if not is_warmup else 0.0,
            is_warmup=is_warmup,
            reason=reason,
        )

    def get_current_threshold(self) -> float:
        """Return the current dynamic threshold without updating."""
        if len(self._scores) < self._warmup_frames:
            return self._baseline

        scores_arr = np.array(self._scores, dtype=np.float64)
        mean = float(np.mean(scores_arr))
        std = float(np.std(scores_arr))

        env = self._env_classifier.get_smoothed()
        k = self._base_k * env.sensitivity
        threshold = mean - k * std

        return max(self._min_threshold, min(self._max_threshold, threshold))

    def reset(self) -> None:
        """Reset all state (e.g. on route change)."""
        self._scores.clear()
        self._frame_count = 0
        self._anomaly_count = 0

    @property
    def stats(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": (
                round(self._anomaly_count / max(self._frame_count, 1), 4)
            ),
            "window_fill": f"{len(self._scores)}/{self._window_size}",
            "current_threshold": round(self.get_current_threshold(), 4),
            "is_warmup": len(self._scores) < self._warmup_frames,
        }
