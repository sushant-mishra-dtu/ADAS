"""
test_kalman_tracker.py — Tests for Kalman Tracker + Trajectory Analyzer
=========================================================================
Fully self-contained. Does NOT modify any existing project files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════
#  IoU Tests
# ══════════════════════════════════════════════════════════════

class TestIoU:
    """Tests for IoU computation."""

    def test_perfect_overlap(self):
        from src.kalman_tracker import compute_iou
        box = np.array([10, 10, 50, 50])
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        from src.kalman_tracker import compute_iou
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        from src.kalman_tracker import compute_iou
        a = np.array([0, 0, 20, 20])
        b = np.array([10, 10, 30, 30])
        # Intersection: 10×10=100, Union: 400+400-100=700
        assert compute_iou(a, b) == pytest.approx(100 / 700, abs=0.01)

    def test_iou_matrix(self):
        from src.kalman_tracker import compute_iou_matrix
        a = np.array([[0, 0, 10, 10], [20, 20, 40, 40]])
        b = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        matrix = compute_iou_matrix(a, b)
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == pytest.approx(1.0)
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        assert matrix[1, 1] == 0.0


# ══════════════════════════════════════════════════════════════
#  Greedy Matching Tests
# ══════════════════════════════════════════════════════════════

class TestGreedyMatching:
    """Tests for greedy IoU matching."""

    def test_perfect_match(self):
        from src.kalman_tracker import greedy_iou_match
        iou = np.array([[1.0, 0.0], [0.0, 1.0]])
        matches, unmatched_t, unmatched_d = greedy_iou_match(iou, 0.3)
        assert len(matches) == 2
        assert len(unmatched_t) == 0
        assert len(unmatched_d) == 0

    def test_no_match_below_threshold(self):
        from src.kalman_tracker import greedy_iou_match
        iou = np.array([[0.1, 0.05], [0.02, 0.15]])
        matches, unmatched_t, unmatched_d = greedy_iou_match(iou, 0.3)
        assert len(matches) == 0
        assert len(unmatched_t) == 2
        assert len(unmatched_d) == 2

    def test_more_detections_than_tracks(self):
        from src.kalman_tracker import greedy_iou_match
        iou = np.array([[0.8, 0.1, 0.0]])  # 1 track, 3 detections
        matches, unmatched_t, unmatched_d = greedy_iou_match(iou, 0.3)
        assert len(matches) == 1
        assert matches[0] == (0, 0)
        assert len(unmatched_d) == 2

    def test_empty_inputs(self):
        from src.kalman_tracker import greedy_iou_match
        iou = np.empty((0, 0))
        matches, unmatched_t, unmatched_d = greedy_iou_match(iou, 0.3)
        assert len(matches) == 0


# ══════════════════════════════════════════════════════════════
#  Box Conversion Tests
# ══════════════════════════════════════════════════════════════

class TestBoxConversion:
    """Tests for xyxy ↔ z conversion."""

    def test_roundtrip(self):
        from src.kalman_tracker import xyxy_to_z, z_to_xyxy
        original = np.array([100.0, 200.0, 160.0, 280.0])
        z = xyxy_to_z(original)
        recovered = z_to_xyxy(z)
        np.testing.assert_array_almost_equal(recovered, original, decimal=1)

    def test_center_computation(self):
        from src.kalman_tracker import xyxy_to_z
        box = np.array([0.0, 0.0, 100.0, 50.0])
        z = xyxy_to_z(box)
        assert z[0] == pytest.approx(50.0)  # cx
        assert z[1] == pytest.approx(25.0)  # cy


# ══════════════════════════════════════════════════════════════
#  KalmanBoxTracker Tests
# ══════════════════════════════════════════════════════════════

class TestKalmanBoxTracker:
    """Tests for single-object Kalman tracker."""

    def _make_tracker(self, bbox=None):
        from src.kalman_tracker import KalmanBoxTracker
        if bbox is None:
            bbox = np.array([100, 200, 150, 280])
        return KalmanBoxTracker(bbox=bbox, track_id=1, class_name="car")

    def test_initial_state(self):
        t = self._make_tracker()
        assert t.id == 1
        assert t.class_name == "car"
        assert t.age == 0
        assert t.hits == 1

    def test_predict_advances_age(self):
        t = self._make_tracker()
        t.predict()
        assert t.age == 1
        assert t.time_since_update == 1

    def test_update_resets_time_since_update(self):
        t = self._make_tracker()
        t.predict()
        assert t.time_since_update == 1
        t.update(np.array([102, 202, 152, 282]))
        assert t.time_since_update == 0
        assert t.hits == 2

    def test_predict_moves_towards_velocity(self):
        """After seeing a moving object, predict should extrapolate."""
        t = self._make_tracker(np.array([100, 100, 120, 120]))

        # Feed a series of detections moving right
        for x in range(1, 6):
            t.predict()
            t.update(np.array([100 + x * 10, 100, 120 + x * 10, 120]))

        # Now predict without update — should continue rightward
        t.predict()
        bbox = t.get_bbox()
        cx = (bbox[0] + bbox[2]) / 2.0
        # Center should be beyond 152 (the last update was at ~150 center)
        assert cx > 150, f"Predicted center {cx} should be > 150"

    def test_velocity_extraction(self):
        t = self._make_tracker()
        vx, vy = t.get_velocity()
        # Initially zero velocity
        assert vx == pytest.approx(0.0, abs=1.0)
        assert vy == pytest.approx(0.0, abs=1.0)

    def test_trajectory_recorded(self):
        t = self._make_tracker()
        t.predict()
        t.update(np.array([110, 210, 160, 290]))
        assert len(t.trajectory) == 2  # initial + 1 update

    def test_to_dict(self):
        t = self._make_tracker()
        d = t.to_dict()
        assert "id" in d
        assert "cx" in d
        assert "vx" in d
        assert "bbox" in d
        assert isinstance(d["bbox"], list)
        assert len(d["bbox"]) == 4

    def test_speed(self):
        t = self._make_tracker()
        speed = t.get_speed()
        assert isinstance(speed, float)
        assert speed >= 0.0


# ══════════════════════════════════════════════════════════════
#  MultiObjectTracker Tests
# ══════════════════════════════════════════════════════════════

class TestMultiObjectTracker:
    """Tests for multi-object tracking."""

    def _make_tracker(self, **kwargs):
        from src.kalman_tracker import MultiObjectTracker
        return MultiObjectTracker(**kwargs)

    def test_creates_tracks_for_new_detections(self):
        mot = self._make_tracker()
        detections = [
            [100, 200, 150, 280, 0.9, "car"],
            [300, 100, 350, 160, 0.8, "person"],
        ]
        mot.update(detections)
        assert mot.active_count == 2

    def test_assigns_unique_ids(self):
        mot = self._make_tracker()
        det = [[100, 200, 150, 280, 0.9, "car"]]
        # Feed same detection for enough frames to confirm (hits >= 3)
        for _ in range(4):
            result = mot.update(det)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_persistent_id_across_frames(self):
        """Same object detected in same position should keep its ID."""
        mot = self._make_tracker()

        # Feed enough frames to confirm the track
        det = [[100, 200, 150, 280, 0.9, "car"]]
        for _ in range(3):
            mot.update(det)

        # Frame 4 — confirmed, should return the track
        r1 = mot.update([[100, 200, 150, 280, 0.9, "car"]])
        # Frame 5 — slightly moved
        r2 = mot.update([[102, 202, 152, 282, 0.85, "car"]])

        assert len(r1) == 1
        assert len(r2) == 1
        assert r1[0]["id"] == r2[0]["id"]

    def test_removes_stale_tracks(self):
        mot = self._make_tracker(max_age=3)

        # Create a track
        mot.update([[100, 200, 150, 280, 0.9, "car"]])

        # Run updates with no detections — track should die after max_age
        for _ in range(5):
            mot.update([])

        assert mot.active_count == 0

    def test_multiple_objects_tracked(self):
        mot = self._make_tracker(min_hits=1)

        detections = [
            [10, 10, 30, 30, 0.9, "car"],
            [200, 200, 250, 250, 0.8, "truck"],
        ]

        # Feed same detections for 3 frames
        for _ in range(3):
            results = mot.update(detections)

        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert len(ids) == 2  # Two distinct IDs

    def test_stats(self):
        mot = self._make_tracker()
        mot.update([[100, 200, 150, 280, 0.9, "car"]])
        stats = mot.stats
        assert stats["frame_count"] == 1
        assert stats["total_created"] == 1

    def test_empty_detections(self):
        """Tracker should handle empty detection lists."""
        mot = self._make_tracker()
        results = mot.update([])
        assert results == []
        assert mot.active_count == 0


# ══════════════════════════════════════════════════════════════
#  Trajectory Analyzer Tests
# ══════════════════════════════════════════════════════════════

class TestTrajectoryAnalyzer:
    """Tests for trajectory anomaly detection."""

    def _make_straight_track(self):
        """Create a track moving in a straight line (low anomaly)."""
        from src.kalman_tracker import KalmanBoxTracker
        track = KalmanBoxTracker(
            bbox=np.array([100, 100, 120, 120]),
            track_id=1, class_name="car",
        )
        # Simulate straight-line motion (moving right)
        for i in range(1, 15):
            track.predict()
            track.update(np.array([100 + i * 5, 100, 120 + i * 5, 120]))
        return track

    def _make_erratic_track(self):
        """Create a track with erratic direction changes (high anomaly)."""
        from src.kalman_tracker import KalmanBoxTracker
        track = KalmanBoxTracker(
            bbox=np.array([100, 100, 120, 120]),
            track_id=2, class_name="person",
        )
        # Zigzag motion: right, up, left, down, right...
        positions = [
            (100, 100), (130, 100), (130, 70), (100, 70),
            (100, 100), (130, 130), (100, 130), (70, 100),
            (100, 70), (130, 100), (100, 130), (70, 70),
            (130, 70), (70, 130), (130, 130),
        ]
        for x, y in positions[1:]:
            track.predict()
            track.update(np.array([x, y, x + 20, y + 20]))
        return track

    def test_straight_line_low_score(self):
        from src.kalman_tracker import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer()
        track = self._make_straight_track()
        score = analyzer.analyze_single(track)
        assert score.score < 0.3, f"Straight line got score {score.score}"

    def test_erratic_motion_high_score(self):
        from src.kalman_tracker import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer()
        track = self._make_erratic_track()
        score = analyzer.analyze_single(track)
        assert score.score > 0.3, f"Erratic motion got score {score.score}"
        assert score.direction_changes > 0

    def test_analyze_multiple(self):
        from src.kalman_tracker import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer()
        tracks = [self._make_straight_track(), self._make_erratic_track()]
        results = analyzer.analyze(tracks)
        assert len(results) == 2

    def test_short_trajectory_zero_score(self):
        """Track with too few points should get score 0."""
        from src.kalman_tracker import KalmanBoxTracker, TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer(min_trajectory_length=5)
        track = KalmanBoxTracker(
            bbox=np.array([10, 10, 30, 30]),
            track_id=3, class_name="car",
        )
        # Only 2 updates (below min_trajectory_length)
        track.predict()
        track.update(np.array([15, 15, 35, 35]))
        # Force confirmed
        track.hits = 5

        score = analyzer.analyze_single(track)
        assert score.score == 0.0
        assert "Insufficient" in score.reason

    def test_trajectory_score_to_dict(self):
        from src.kalman_tracker import TrajectoryAnalyzer
        analyzer = TrajectoryAnalyzer()
        track = self._make_straight_track()
        score = analyzer.analyze_single(track)
        d = score.to_dict()
        assert "track_id" in d
        assert "score" in d
        assert "reason" in d
        assert 0.0 <= d["score"] <= 1.0


# ── Run directly ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
