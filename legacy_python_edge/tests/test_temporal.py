"""
test_temporal.py — Tests for Spatio-Temporal Anomaly Scoring Pipeline
======================================================================
Fully self-contained test suite. Does NOT modify any existing project files.

Tests covering:
  1. ClipBuffer — push, capture, cooldown, edge cases
  2. MockTemporalScorer — fallback scoring
  3. ConvLSTMCell — output shape verification (requires PyTorch)
  4. SpatioTemporalScorer — full forward pass (requires PyTorch)
  5. CloudAnomalyScorer — end-to-end clip scoring
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))


# ══════════════════════════════════════════════════════════════
#  1. ClipBuffer Tests
# ══════════════════════════════════════════════════════════════

class TestClipBuffer:
    """Tests for src/clip_buffer.py — rolling frame buffer."""

    def _make_buffer(self, **kwargs):
        from src.clip_buffer import ClipBuffer
        return ClipBuffer(**kwargs)

    def _make_frame(self, value: int = 128) -> np.ndarray:
        """Create a mock 640×480×3 BGR frame."""
        return np.full((480, 640, 3), value, dtype=np.uint8)

    def test_push_and_size(self):
        """Buffer size should grow as frames are pushed."""
        buf = self._make_buffer(buffer_seconds=1, fps=10)
        assert buf.buffer_size == 0

        for i in range(5):
            buf.push(time.time(), self._make_frame(i))

        assert buf.buffer_size == 5

    def test_buffer_capacity_limit(self):
        """Buffer should not exceed its configured capacity."""
        buf = self._make_buffer(
            buffer_seconds=1, fps=10,
            pre_event_frames=5, post_event_frames=5,
        )
        capacity = buf._buffer_capacity  # 10 + 5 = 15

        # Push more than capacity
        for i in range(30):
            buf.push(time.time(), self._make_frame(i))

        assert buf.buffer_size == capacity

    def test_capture_clip_returns_frames(self):
        """capture_clip should return the requested number of pre-event frames."""
        buf = self._make_buffer(pre_event_frames=5)

        for i in range(10):
            buf.push(float(i), self._make_frame(i * 10))

        clip = buf.capture_clip(pre_frames=5)
        assert clip is not None
        assert len(clip) == 5

        # Should be the 5 most recent frames
        timestamps = [ts for ts, _ in clip]
        assert timestamps == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_capture_clip_partial_buffer(self):
        """If buffer has fewer frames than requested, return what's available."""
        buf = self._make_buffer(pre_event_frames=20)

        for i in range(3):
            buf.push(float(i), self._make_frame())

        clip = buf.capture_clip(pre_frames=20)
        assert clip is not None
        assert len(clip) == 3

    def test_capture_clip_cooldown(self):
        """Second capture within cooldown period should return None."""
        buf = self._make_buffer(pre_event_frames=5)
        buf._cooldown_seconds = 2.0

        for i in range(10):
            buf.push(float(i), self._make_frame())

        first = buf.capture_clip()
        assert first is not None

        # Immediate second call should be blocked by cooldown
        second = buf.capture_clip()
        assert second is None

    def test_capture_clip_empty_buffer(self):
        """capture_clip on empty buffer should return None."""
        buf = self._make_buffer()
        assert buf.capture_clip() is None

    def test_stats(self):
        """Stats should track pushes and captures correctly."""
        buf = self._make_buffer()
        for i in range(5):
            buf.push(float(i), self._make_frame())

        buf.capture_clip()

        stats = buf.stats
        assert stats["frames_buffered"] == 5
        assert stats["clips_captured"] == 1

    def test_clear(self):
        """clear() should empty the buffer."""
        buf = self._make_buffer()
        for i in range(5):
            buf.push(float(i), self._make_frame())

        buf.clear()
        assert buf.buffer_size == 0


# ══════════════════════════════════════════════════════════════
#  2. MockTemporalScorer Tests
# ══════════════════════════════════════════════════════════════

class TestMockTemporalScorer:
    """Tests for the fallback frame-differencing scorer."""

    def test_score_range(self):
        """Score should be in [0, 1]."""
        from src.temporal_model import MockTemporalScorer
        scorer = MockTemporalScorer()

        frames = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(10)]
        score = scorer.score_clip(frames)

        assert 0.0 <= score <= 1.0

    def test_identical_frames_low_score(self):
        """Identical frames should produce a low motion score."""
        from src.temporal_model import MockTemporalScorer
        scorer = MockTemporalScorer()

        frame = np.full((128, 128, 3), 128, dtype=np.uint8)
        frames = [frame.copy() for _ in range(10)]
        score = scorer.score_clip(frames)

        assert score < 0.5, f"Identical frames got score {score}, expected < 0.5"

    def test_high_motion_high_score(self):
        """Rapidly changing frames should produce a higher score."""
        from src.temporal_model import MockTemporalScorer
        scorer = MockTemporalScorer()

        frames = []
        for i in range(10):
            val = 0 if i % 2 == 0 else 255
            frames.append(np.full((128, 128, 3), val, dtype=np.uint8))

        score = scorer.score_clip(frames)
        assert score > 0.5, f"High motion frames got score {score}, expected > 0.5"

    def test_single_frame(self):
        """Single frame should return 0.5 (not enough context)."""
        from src.temporal_model import MockTemporalScorer
        scorer = MockTemporalScorer()

        frames = [np.zeros((128, 128, 3), dtype=np.uint8)]
        assert scorer.score_clip(frames) == 0.5

    def test_callable(self):
        """MockTemporalScorer should be callable."""
        from src.temporal_model import MockTemporalScorer
        scorer = MockTemporalScorer()

        frames = [np.zeros((128, 128, 3), dtype=np.uint8)] * 5
        score = scorer(frames)
        assert isinstance(score, float)


# ══════════════════════════════════════════════════════════════
#  3. ConvLSTM & SpatioTemporalScorer Tests (require PyTorch)
# ══════════════════════════════════════════════════════════════

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestConvLSTMCell:
    """Tests for the ConvLSTM cell."""

    def test_output_shape(self):
        """ConvLSTMCell output should match expected dimensions."""
        from src.temporal_model import ConvLSTMCell

        cell = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=3)
        x = torch.randn(2, 64, 16, 16)

        h, c = cell(x)

        assert h.shape == (2, 32, 16, 16), f"h shape: {h.shape}"
        assert c.shape == (2, 32, 16, 16), f"c shape: {c.shape}"

    def test_output_shape_with_state(self):
        """ConvLSTMCell should accept prior state correctly."""
        from src.temporal_model import ConvLSTMCell

        cell = ConvLSTMCell(input_dim=64, hidden_dim=32)
        x = torch.randn(1, 64, 8, 8)
        h0 = torch.zeros(1, 32, 8, 8)
        c0 = torch.zeros(1, 32, 8, 8)

        h1, c1 = cell(x, (h0, c0))

        assert h1.shape == h0.shape
        assert c1.shape == c0.shape

    def test_sequential_updates(self):
        """Running multiple timesteps should update the state."""
        from src.temporal_model import ConvLSTMCell

        cell = ConvLSTMCell(input_dim=32, hidden_dim=32)
        state = None

        for _ in range(5):
            x = torch.randn(1, 32, 8, 8)
            state = cell(x, state)

        h, c = state
        assert h.shape == (1, 32, 8, 8)
        assert torch.any(h != 0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestSpatioTemporalScorer:
    """Tests for the full ConvLSTM-based anomaly scorer model."""

    def test_forward_pass_shape(self):
        """Forward pass should produce (batch, 1) output."""
        from src.temporal_model import SpatioTemporalScorer

        model = SpatioTemporalScorer(hidden_dim=32, num_layers=1)
        clip = torch.randn(1, 10, 3, 128, 128)

        with torch.no_grad():
            score = model(clip)

        assert score.shape == (1, 1), f"Output shape: {score.shape}"

    def test_score_range(self):
        """Output score should be in [0, 1] due to sigmoid."""
        from src.temporal_model import SpatioTemporalScorer

        model = SpatioTemporalScorer(hidden_dim=32)
        clip = torch.randn(2, 5, 3, 128, 128)

        with torch.no_grad():
            scores = model(clip)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_batch_processing(self):
        """Model should handle batch inputs correctly."""
        from src.temporal_model import SpatioTemporalScorer

        model = SpatioTemporalScorer(hidden_dim=32)
        clip = torch.randn(4, 8, 3, 128, 128)

        with torch.no_grad():
            scores = model(clip)

        assert scores.shape == (4, 1)

    def test_param_count(self):
        """Model should have a reasonable parameter count."""
        from src.temporal_model import SpatioTemporalScorer

        model = SpatioTemporalScorer(hidden_dim=64)
        count = model.param_count

        assert 100_000 < count < 2_000_000, f"Param count: {count}"

    def test_model_summary(self):
        """get_summary() should return a non-empty string."""
        from src.temporal_model import SpatioTemporalScorer

        model = SpatioTemporalScorer(hidden_dim=64)
        summary = model.get_summary()
        assert "SpatioTemporalScorer" in summary
        assert "ConvLSTM" in summary

    def test_multi_layer(self):
        """Model should work with multiple stacked ConvLSTM layers."""
        from src.temporal_model import SpatioTemporalScorer

        model = SpatioTemporalScorer(hidden_dim=32, num_layers=3)
        clip = torch.randn(1, 5, 3, 128, 128)

        with torch.no_grad():
            score = model(clip)

        assert score.shape == (1, 1)
        assert 0.0 <= score.item() <= 1.0


# ══════════════════════════════════════════════════════════════
#  4. CloudAnomalyScorer Tests
# ══════════════════════════════════════════════════════════════

class TestCloudAnomalyScorer:
    """End-to-end tests for the cloud-side scorer."""

    def _create_mock_clip(self, tmp_path, num_frames=10) -> Path:
        """Create a mock .npz clip file for testing."""
        frames = np.random.randint(0, 255, (num_frames, 128, 128, 3), dtype=np.uint8)
        timestamps = np.arange(num_frames, dtype=np.float64) * 0.067

        clip_path = tmp_path / "test_clip.npz"
        np.savez_compressed(str(clip_path), frames=frames, timestamps=timestamps)
        return clip_path

    def test_score_clip_with_mock(self, tmp_path):
        """CloudAnomalyScorer should score a clip using mock scorer."""
        from cloud_backend.cloud_scorer import CloudAnomalyScorer

        scorer = CloudAnomalyScorer()
        scorer.load_model()

        clip_path = self._create_mock_clip(tmp_path)
        result = scorer.score_clip(str(clip_path))

        assert result is not None
        assert 0.0 <= result.temporal_score <= 1.0
        assert result.num_frames == 10
        assert isinstance(result.description, str)
        assert len(result.per_frame_motion) == 9

    def test_scoring_result_to_dict(self, tmp_path):
        """ScoringResult.to_dict() should produce a valid dict."""
        from cloud_backend.cloud_scorer import CloudAnomalyScorer

        scorer = CloudAnomalyScorer()
        scorer.load_model()

        clip_path = self._create_mock_clip(tmp_path)
        result = scorer.score_clip(str(clip_path))

        d = result.to_dict()
        assert "temporal_score" in d
        assert "is_critical_anomaly" in d
        assert "description" in d
        assert isinstance(d["per_frame_motion"], list)

    def test_score_all_clips(self, tmp_path):
        """score_all_clips should process all .npz files in a directory."""
        from cloud_backend.cloud_scorer import CloudAnomalyScorer

        scorer = CloudAnomalyScorer()
        scorer.load_model()

        for i in range(3):
            frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
            timestamps = np.arange(5, dtype=np.float64)
            path = tmp_path / f"clip_{i}.npz"
            np.savez_compressed(str(path), frames=frames, timestamps=timestamps)

        results = scorer.score_all_clips(str(tmp_path))
        assert len(results) == 3

    def test_score_clip_too_short(self, tmp_path):
        """Clip with < 2 frames should return None."""
        from cloud_backend.cloud_scorer import CloudAnomalyScorer

        scorer = CloudAnomalyScorer()
        scorer.load_model()

        frames = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        timestamps = np.array([0.0])
        path = tmp_path / "short.npz"
        np.savez_compressed(str(path), frames=frames, timestamps=timestamps)

        result = scorer.score_clip(str(path))
        assert result is None

    def test_stats_tracking(self, tmp_path):
        """Scorer stats should track processed clips."""
        from cloud_backend.cloud_scorer import CloudAnomalyScorer

        scorer = CloudAnomalyScorer()
        scorer.load_model()

        clip_path = self._create_mock_clip(tmp_path)
        scorer.score_clip(str(clip_path))

        assert scorer.stats["clips_scored"] == 1


# ── Run directly ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
