"""
test_adaptive_threshold.py — Tests for Adaptive Threshold + Environment Classifier
=====================================================================================
Fully self-contained. Does NOT modify any existing project files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════
#  Environment Classifier Tests
# ══════════════════════════════════════════════════════════════

class TestEnvironmentClassifier:
    """Tests for driving environment classification."""

    def _make_classifier(self):
        from src.adaptive_threshold import EnvironmentClassifier
        return EnvironmentClassifier()

    def test_night_classification(self):
        from src.adaptive_threshold import Environment
        clf = self._make_classifier()
        env = clf.classify(brightness=0.10, num_detections=2)
        assert env == Environment.NIGHT

    def test_clear_day_classification(self):
        from src.adaptive_threshold import Environment
        clf = self._make_classifier()
        env = clf.classify(brightness=0.65, num_detections=3, brightness_variance=0.1)
        assert env == Environment.CLEAR_DAY

    def test_dense_traffic_classification(self):
        from src.adaptive_threshold import Environment
        clf = self._make_classifier()
        env = clf.classify(brightness=0.60, num_detections=8)
        assert env == Environment.DENSE_TRAFFIC

    def test_open_road_classification(self):
        from src.adaptive_threshold import Environment
        clf = self._make_classifier()
        env = clf.classify(brightness=0.70, num_detections=0)
        assert env == Environment.OPEN_ROAD

    def test_rain_fog_classification(self):
        from src.adaptive_threshold import Environment
        clf = self._make_classifier()
        # Medium brightness + very low variance = fog
        env = clf.classify(
            brightness=0.40, num_detections=2, brightness_variance=0.01
        )
        assert env == Environment.RAIN_FOG

    def test_sensitivity_values(self):
        """Each environment should have a defined sensitivity."""
        from src.adaptive_threshold import Environment
        assert Environment.CLEAR_DAY.sensitivity == 1.0
        assert Environment.NIGHT.sensitivity == 1.5
        assert Environment.OPEN_ROAD.sensitivity == 0.8

    def test_smoothed_classification(self):
        """get_smoothed should return the most common recent environment."""
        from src.adaptive_threshold import Environment
        clf = self._make_classifier()
        # Feed mostly night
        for _ in range(7):
            clf.classify(brightness=0.10, num_detections=1)
        for _ in range(3):
            clf.classify(brightness=0.70, num_detections=2)

        smoothed = clf.get_smoothed()
        assert smoothed == Environment.NIGHT


# ══════════════════════════════════════════════════════════════
#  Adaptive Threshold Tests
# ══════════════════════════════════════════════════════════════

class TestAdaptiveThreshold:
    """Tests for the adaptive threshold engine."""

    def _make_threshold(self, **kwargs):
        from src.adaptive_threshold import AdaptiveThreshold
        return AdaptiveThreshold(**kwargs)

    def test_warmup_uses_baseline(self):
        """During warmup, should use the fixed baseline."""
        at = self._make_threshold(baseline=0.35, warmup_frames=50)
        result = at.update(max_confidence=0.20)
        assert result.is_warmup
        assert result.dynamic_threshold == pytest.approx(0.35)
        assert result.is_anomaly  # 0.20 < 0.35

    def test_warmup_ends(self):
        """After enough frames, warmup should end."""
        at = self._make_threshold(warmup_frames=10)
        for _ in range(15):
            result = at.update(max_confidence=0.50)
        assert not result.is_warmup

    def test_threshold_adapts_to_high_confidence(self):
        """If recent frames are all high-confidence, threshold should rise."""
        at = self._make_threshold(warmup_frames=10, window_size=50)

        # Feed high-confidence frames
        for _ in range(60):
            result = at.update(
                max_confidence=0.80,
                frame_brightness=0.60,
                num_detections=3,
            )

        # Threshold should be above baseline 0.35
        assert result.dynamic_threshold > 0.35, (
            f"Threshold {result.dynamic_threshold} should be > 0.35 "
            f"after consistently high confidence scores"
        )

    def test_threshold_adapts_to_low_confidence(self):
        """If recent frames are all low-confidence, threshold should drop."""
        at = self._make_threshold(warmup_frames=10, window_size=50)

        # Feed low-confidence frames (model always confused)
        for _ in range(60):
            result = at.update(
                max_confidence=0.20,
                frame_brightness=0.60,
                num_detections=2,
            )

        # Threshold should drop below baseline
        assert result.dynamic_threshold < 0.35, (
            f"Threshold {result.dynamic_threshold} should be < 0.35 "
            f"after consistently low confidence scores"
        )

    def test_threshold_clamped_min(self):
        """Threshold should never go below min_threshold."""
        at = self._make_threshold(
            warmup_frames=5, min_threshold=0.10, window_size=20
        )

        # Feed very low and variable confidences
        for i in range(30):
            at.update(max_confidence=0.05 + (i % 3) * 0.02)

        threshold = at.get_current_threshold()
        assert threshold >= 0.10

    def test_threshold_clamped_max(self):
        """Threshold should never go above max_threshold."""
        at = self._make_threshold(
            warmup_frames=5, max_threshold=0.60, window_size=20
        )

        # Feed very high confidences
        for _ in range(30):
            at.update(max_confidence=0.95)

        threshold = at.get_current_threshold()
        assert threshold <= 0.60

    def test_zero_detections_is_anomaly(self):
        """Frame with zero detections should always be anomaly."""
        at = self._make_threshold(warmup_frames=5)

        # Even during non-warmup with high confidence
        for _ in range(10):
            at.update(max_confidence=0.90, num_detections=5)

        # Zero detections should be anomaly regardless of confidence
        result = at.update(max_confidence=0.90, num_detections=0)
        assert result.is_anomaly

    def test_night_sensitivity(self):
        """Night environment should use higher k (more selective)."""
        at = self._make_threshold(warmup_frames=5)

        # Feed frames classified as night
        for _ in range(20):
            result = at.update(
                max_confidence=0.30,
                frame_brightness=0.10,  # dark → NIGHT
                num_detections=1,
            )

        # Sensitivity should be > 1.0 (night multiplier = 1.5)
        assert result.sensitivity > 1.0

    def test_open_road_sensitivity(self):
        """Open road should use lower k (more sensitive to anomalies)."""
        at = self._make_threshold(warmup_frames=5)

        for _ in range(20):
            result = at.update(
                max_confidence=0.70,
                frame_brightness=0.65,
                num_detections=0,  # empty road → OPEN_ROAD
            )

        assert result.sensitivity < 1.0

    def test_reset(self):
        """Reset should clear all state."""
        at = self._make_threshold()
        for _ in range(20):
            at.update(max_confidence=0.50)

        at.reset()
        assert at.stats["frame_count"] == 0
        assert at.stats["anomaly_count"] == 0
        assert at.stats["is_warmup"]

    def test_stats(self):
        at = self._make_threshold(warmup_frames=5)
        for _ in range(10):
            at.update(max_confidence=0.50, num_detections=3)

        stats = at.stats
        assert stats["frame_count"] == 10
        assert "current_threshold" in stats
        assert "anomaly_rate" in stats

    def test_threshold_result_to_dict(self):
        at = self._make_threshold()
        result = at.update(max_confidence=0.40)
        d = result.to_dict()
        assert "is_anomaly" in d
        assert "dynamic_threshold" in d
        assert "environment" in d
        assert "reason" in d


# ══════════════════════════════════════════════════════════════
#  Integration: Adaptive Threshold Over Time
# ══════════════════════════════════════════════════════════════

class TestAdaptiveThresholdIntegration:
    """End-to-end scenario tests."""

    def test_transition_day_to_night(self):
        """Threshold should adapt when driving from day into night."""
        from src.adaptive_threshold import AdaptiveThreshold

        at = AdaptiveThreshold(warmup_frames=10, window_size=30)

        # Phase 1: Daytime driving (high confidence)
        for _ in range(30):
            at.update(
                max_confidence=0.75, frame_brightness=0.65, num_detections=3
            )
        day_threshold = at.get_current_threshold()

        # Phase 2: Enter night (low confidence, dark frames)
        for _ in range(30):
            at.update(
                max_confidence=0.25, frame_brightness=0.10, num_detections=1
            )
        night_threshold = at.get_current_threshold()

        # Night threshold should be lower (model is always confused)
        assert night_threshold < day_threshold, (
            f"Night threshold ({night_threshold:.3f}) should be lower than "
            f"day threshold ({day_threshold:.3f})"
        )

    def test_anomaly_rate_stays_reasonable(self):
        """Adaptive threshold should keep anomaly rate around 10-20%."""
        from src.adaptive_threshold import AdaptiveThreshold

        at = AdaptiveThreshold(warmup_frames=20, window_size=100)

        # Simulate 200 frames of mixed driving
        np.random.seed(42)
        for _ in range(200):
            conf = np.random.uniform(0.15, 0.85)
            at.update(
                max_confidence=conf,
                frame_brightness=0.55,
                num_detections=np.random.randint(0, 6),
            )

        rate = at.stats["anomaly_rate"]
        # Should save between 5% and 50% (not 0% or 100%)
        assert 0.05 < rate < 0.50, f"Anomaly rate {rate} is out of expected range"


# ── Run directly ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
