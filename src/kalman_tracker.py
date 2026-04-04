"""
kalman_tracker.py — Multi-Object Kalman Tracker with Trajectory Analysis
==========================================================================
Assigns persistent IDs to YOLO detections across frames using a Kalman
filter for state prediction and greedy IoU matching for data association.

This module is fully self-contained and does NOT modify any existing code.
It uses only numpy — no scipy or external tracking libraries required.

Components:
  1. KalmanBoxTracker   — single-object Kalman filter tracker
  2. MultiObjectTracker — manages all active trackers with ID assignment
  3. TrajectoryAnalyzer — detects anomalous movement patterns

State Vector (7D):
  [x_center, y_center, area, aspect_ratio, dx, dy, d_area]

Usage:
    tracker = MultiObjectTracker()
    analyzer = TrajectoryAnalyzer()

    # Each frame: feed YOLO detections
    detections = [[x1, y1, x2, y2, conf, cls], ...]
    tracked = tracker.update(detections)
    for obj in tracked:
        print(f"ID={obj['id']}  pos=({obj['cx']:.0f}, {obj['cy']:.0f})  "
              f"vel=({obj['vx']:.1f}, {obj['vy']:.1f})")

    # Analyze trajectories for anomalies
    scores = analyzer.analyze(tracker.get_all_tracks())
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
DEFAULT_MAX_AGE = 7          # Remove track after N frames without a match
DEFAULT_MIN_HITS = 3         # Track must be matched N times before it is confirmed
DEFAULT_IOU_THRESHOLD = 0.3  # Minimum IoU to associate a detection with a track
TRAJECTORY_WINDOW = 15       # Number of past positions to keep for trajectory analysis


# ══════════════════════════════════════════════════════════════
#  IoU Computation (dependency-free)
# ══════════════════════════════════════════════════════════════

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two boxes.

    Parameters
    ----------
    box_a, box_b : np.ndarray
        Bounding boxes as [x1, y1, x2, y2].

    Returns
    -------
    float
        IoU value ∈ [0, 1].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.

    Parameters
    ----------
    boxes_a : np.ndarray, shape (N, 4)
    boxes_b : np.ndarray, shape (M, 4)

    Returns
    -------
    np.ndarray, shape (N, M)
        IoU values for all pairs.
    """
    n = len(boxes_a)
    m = len(boxes_b)
    iou_matrix = np.zeros((n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes_a[i], boxes_b[j])

    return iou_matrix


# ══════════════════════════════════════════════════════════════
#  Greedy IoU Matching (scipy-free Hungarian alternative)
# ══════════════════════════════════════════════════════════════

def greedy_iou_match(
    iou_matrix: np.ndarray,
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy assignment: repeatedly pick the highest IoU pair above threshold.

    Parameters
    ----------
    iou_matrix : np.ndarray, shape (N, M)
        IoU values between N tracks and M detections.
    threshold : float
        Minimum IoU to consider a valid match.

    Returns
    -------
    matches : list of (track_idx, detection_idx)
    unmatched_tracks : list of track indices
    unmatched_detections : list of detection indices
    """
    n_tracks, n_dets = iou_matrix.shape
    matched_tracks = set()
    matched_dets = set()
    matches = []

    if n_tracks == 0 or n_dets == 0:
        return (
            [],
            list(range(n_tracks)),
            list(range(n_dets)),
        )

    # Flatten and sort by IoU descending
    flat = []
    for i in range(n_tracks):
        for j in range(n_dets):
            if iou_matrix[i, j] >= threshold:
                flat.append((iou_matrix[i, j], i, j))

    flat.sort(key=lambda x: x[0], reverse=True)

    for _, track_idx, det_idx in flat:
        if track_idx not in matched_tracks and det_idx not in matched_dets:
            matches.append((track_idx, det_idx))
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


# ══════════════════════════════════════════════════════════════
#  Box Conversion Utilities
# ══════════════════════════════════════════════════════════════

def xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to Kalman state [cx, cy, area, aspect_ratio].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    area = w * h
    aspect = w / max(h, 1e-6)
    return np.array([cx, cy, area, aspect], dtype=np.float64)


def z_to_xyxy(z: np.ndarray) -> np.ndarray:
    """
    Convert Kalman measurement [cx, cy, area, aspect_ratio] to [x1, y1, x2, y2].
    """
    cx, cy, area, aspect = z[0], z[1], z[2], z[3]
    area = max(area, 1.0)
    aspect = max(aspect, 1e-6)
    w = np.sqrt(area * aspect)
    h = area / max(w, 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════
#  Kalman Box Tracker (single object)
# ══════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Kalman filter tracker for a single bounding box.

    State vector (7D):
        [cx, cy, area, aspect_ratio, d_cx, d_cy, d_area]

    The filter uses a constant-velocity motion model:
        position(t+1) = position(t) + velocity(t)

    Parameters
    ----------
    bbox : np.ndarray
        Initial bounding box [x1, y1, x2, y2].
    track_id : int
        Unique track identifier.
    class_name : str
        YOLO class label for this object.
    confidence : float
        Initial detection confidence.
    """

    def __init__(
        self,
        bbox: np.ndarray,
        track_id: int,
        class_name: str = "unknown",
        confidence: float = 0.0,
    ) -> None:
        self.id = track_id
        self.class_name = class_name
        self.confidence = confidence

        # ── Kalman filter matrices ───────────────────────────
        # State: [cx, cy, area, ar, d_cx, d_cy, d_area]
        self.dim_x = 7  # state dimension
        self.dim_z = 4  # measurement dimension

        # State vector
        self.x = np.zeros(self.dim_x, dtype=np.float64)

        # Initialize from first detection
        z = xyxy_to_z(bbox)
        self.x[:4] = z    # position
        self.x[4:] = 0.0  # zero initial velocity

        # State transition matrix (constant velocity model)
        # x(t+1) = F @ x(t)
        self.F = np.eye(self.dim_x, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += d_cx
        self.F[1, 5] = 1.0  # cy += d_cy
        self.F[2, 6] = 1.0  # area += d_area

        # Measurement matrix (we observe [cx, cy, area, ar])
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance matrices
        self.P = np.eye(self.dim_x, dtype=np.float64) * 10.0  # state covariance
        self.P[4, 4] = 1000.0  # high uncertainty for initial velocities
        self.P[5, 5] = 1000.0
        self.P[6, 6] = 1000.0

        self.Q = np.eye(self.dim_x, dtype=np.float64) * 1.0   # process noise
        self.Q[4, 4] = 0.01
        self.Q[5, 5] = 0.01
        self.Q[6, 6] = 0.0001

        self.R = np.eye(self.dim_z, dtype=np.float64) * 10.0  # measurement noise

        # ── Track lifecycle ──────────────────────────────────
        self.age = 0                # Total frames since creation
        self.hits = 1               # Total successful matches
        self.hit_streak = 1         # Consecutive frames with a match
        self.time_since_update = 0  # Frames since last matched detection

        # ── Trajectory history ───────────────────────────────
        self.trajectory: deque = deque(maxlen=TRAJECTORY_WINDOW)
        self.trajectory.append((self.x[0], self.x[1]))  # initial position

    # ── Kalman Predict ───────────────────────────────────────

    def predict(self) -> np.ndarray:
        """
        Advance state by one timestep using the motion model.

        Returns
        -------
        np.ndarray
            Predicted bounding box [x1, y1, x2, y2].
        """
        # Ensure area doesn't go negative
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0

        # State prediction: x = F @ x
        self.x = self.F @ self.x

        # Covariance prediction: P = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        # Reset hit streak if no update this frame
        if self.time_since_update > 1:
            self.hit_streak = 0

        return self.get_bbox()

    # ── Kalman Update ────────────────────────────────────────

    def update(
        self,
        bbox: np.ndarray,
        class_name: str = "",
        confidence: float = 0.0,
    ) -> None:
        """
        Correct the state using a matched YOLO detection.

        Parameters
        ----------
        bbox : np.ndarray
            Matched detection [x1, y1, x2, y2].
        class_name : str
            Updated class label.
        confidence : float
            Detection confidence.
        """
        z = xyxy_to_z(bbox)

        # Innovation (residual): y = z - H @ x
        y = z - self.H @ self.x

        # Innovation covariance: S = H @ P @ H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P @ H^T @ S^-1
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ self.H.T @ S_inv

        # State update: x = x + K @ y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K @ H) @ P
        I = np.eye(self.dim_x, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        # Update metadata
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        if class_name:
            self.class_name = class_name
        self.confidence = confidence

        # Record position in trajectory
        self.trajectory.append((self.x[0], self.x[1]))

    # ── Accessors ────────────────────────────────────────────

    def get_bbox(self) -> np.ndarray:
        """Return current state as [x1, y1, x2, y2]."""
        return z_to_xyxy(self.x[:4])

    def get_velocity(self) -> Tuple[float, float]:
        """Return current (vx, vy) in pixels/frame."""
        return float(self.x[4]), float(self.x[5])

    def get_center(self) -> Tuple[float, float]:
        """Return current (cx, cy)."""
        return float(self.x[0]), float(self.x[1])

    def get_speed(self) -> float:
        """Return scalar speed in pixels/frame."""
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx**2 + vy**2))

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed after enough consecutive hits."""
        return self.hits >= DEFAULT_MIN_HITS

    def to_dict(self) -> dict:
        """Serialize track state to dict."""
        cx, cy = self.get_center()
        vx, vy = self.get_velocity()
        bbox = self.get_bbox()
        return {
            "id": self.id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "cx": round(cx, 1),
            "cy": round(cy, 1),
            "vx": round(vx, 2),
            "vy": round(vy, 2),
            "speed": round(self.get_speed(), 2),
            "bbox": [round(float(v), 1) for v in bbox],
            "age": self.age,
            "hits": self.hits,
            "hit_streak": self.hit_streak,
            "time_since_update": self.time_since_update,
            "confirmed": self.is_confirmed,
        }


# ══════════════════════════════════════════════════════════════
#  Multi-Object Tracker
# ══════════════════════════════════════════════════════════════

class MultiObjectTracker:
    """
    Manages multiple KalmanBoxTracker instances across frames.

    Each frame:
      1. Predict all existing tracks forward
      2. Compute IoU between predicted boxes and new YOLO detections
      3. Greedy match: assign detections to tracks
      4. Update matched tracks with their assigned detections
      5. Create new tracks for unmatched detections
      6. Remove stale tracks that haven't been matched for too long

    Parameters
    ----------
    max_age : int
        Delete tracks unseen for this many frames (default: 7).
    min_hits : int
        Track must be matched this many times to be "confirmed" (default: 3).
    iou_threshold : float
        Minimum IoU for a valid track–detection match (default: 0.3).
    """

    def __init__(
        self,
        max_age: int = DEFAULT_MAX_AGE,
        min_hits: int = DEFAULT_MIN_HITS,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold

        self._tracks: List[KalmanBoxTracker] = []
        self._next_id: int = 1
        self._frame_count: int = 0

        # Stats
        self._total_tracks_created: int = 0
        self._total_tracks_removed: int = 0

    def update(
        self, detections: List[List[float]]
    ) -> List[dict]:
        """
        Process one frame of YOLO detections.

        Parameters
        ----------
        detections : list of [x1, y1, x2, y2, confidence, class_id]
            Current frame's YOLO detections. class_id can be int or str.

        Returns
        -------
        list of dict
            Active confirmed tracks with id, position, velocity, etc.
        """
        self._frame_count += 1

        # Parse detections
        det_boxes = []
        det_confs = []
        det_classes = []
        for det in detections:
            det_boxes.append(np.array(det[:4], dtype=np.float64))
            det_confs.append(float(det[4]) if len(det) > 4 else 0.0)
            det_classes.append(str(det[5]) if len(det) > 5 else "unknown")

        # ── 1. Predict all existing tracks ───────────────────
        predicted_boxes = []
        for track in self._tracks:
            pred_box = track.predict()
            predicted_boxes.append(pred_box)

        # ── 2. Compute IoU matrix ────────────────────────────
        if predicted_boxes and det_boxes:
            pred_array = np.array(predicted_boxes)
            det_array = np.array(det_boxes)
            iou_matrix = compute_iou_matrix(pred_array, det_array)
        else:
            iou_matrix = np.empty((len(predicted_boxes), len(det_boxes)))

        # ── 3. Greedy matching ───────────────────────────────
        matches, unmatched_trks, unmatched_dets = greedy_iou_match(
            iou_matrix, self._iou_threshold
        )

        # ── 4. Update matched tracks ─────────────────────────
        for track_idx, det_idx in matches:
            self._tracks[track_idx].update(
                bbox=det_boxes[det_idx],
                class_name=det_classes[det_idx],
                confidence=det_confs[det_idx],
            )

        # ── 5. Create new tracks for unmatched detections ────
        for det_idx in unmatched_dets:
            new_track = KalmanBoxTracker(
                bbox=det_boxes[det_idx],
                track_id=self._next_id,
                class_name=det_classes[det_idx],
                confidence=det_confs[det_idx],
            )
            self._tracks.append(new_track)
            self._next_id += 1
            self._total_tracks_created += 1

        # ── 6. Remove dead tracks ────────────────────────────
        alive = []
        for track in self._tracks:
            if track.time_since_update <= self._max_age:
                alive.append(track)
            else:
                self._total_tracks_removed += 1
                logger.debug(
                    "Track #%d (%s) removed — unseen for %d frames.",
                    track.id, track.class_name, track.time_since_update,
                )
        self._tracks = alive

        # ── 7. Return confirmed tracks ───────────────────────
        results = []
        for track in self._tracks:
            if track.is_confirmed and track.time_since_update == 0:
                results.append(track.to_dict())

        return results

    def get_all_tracks(self) -> List[KalmanBoxTracker]:
        """Return all active tracks (including unconfirmed)."""
        return list(self._tracks)

    def get_confirmed_tracks(self) -> List[KalmanBoxTracker]:
        """Return only confirmed tracks."""
        return [t for t in self._tracks if t.is_confirmed]

    @property
    def active_count(self) -> int:
        return len(self._tracks)

    @property
    def stats(self) -> dict:
        return {
            "frame_count": self._frame_count,
            "active_tracks": self.active_count,
            "confirmed_tracks": len(self.get_confirmed_tracks()),
            "total_created": self._total_tracks_created,
            "total_removed": self._total_tracks_removed,
            "next_id": self._next_id,
        }


# ══════════════════════════════════════════════════════════════
#  Trajectory Analyzer
# ══════════════════════════════════════════════════════════════

@dataclass
class TrajectoryScore:
    """Anomaly score for a single tracked object's trajectory."""

    track_id: int
    class_name: str
    score: float                    # ∈ [0, 1], higher = more anomalous
    direction_changes: int          # Number of sharp turns
    avg_speed: float                # Average speed in px/frame
    max_acceleration: float         # Peak acceleration event
    is_erratic: bool                # True if score > erratic_threshold
    reason: str                     # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "score": round(self.score, 4),
            "direction_changes": self.direction_changes,
            "avg_speed": round(self.avg_speed, 2),
            "max_acceleration": round(self.max_acceleration, 2),
            "is_erratic": self.is_erratic,
            "reason": self.reason,
        }


class TrajectoryAnalyzer:
    """
    Analyzes tracked object trajectories for anomalous behavior.

    Flags:
      - Sudden direction changes (>45° within 3 frames)
      - Rapid acceleration or deceleration
      - Erratic lateral movement (high variance in heading)

    Parameters
    ----------
    direction_threshold_deg : float
        Minimum angle change (degrees) to count as a "sharp turn" (default: 45).
    erratic_threshold : float
        Trajectory score above this is flagged as erratic (default: 0.5).
    min_trajectory_length : int
        Minimum positions in trajectory to analyze (default: 5).
    """

    def __init__(
        self,
        direction_threshold_deg: float = 45.0,
        erratic_threshold: float = 0.5,
        min_trajectory_length: int = 5,
    ) -> None:
        self._direction_threshold = np.radians(direction_threshold_deg)
        self._erratic_threshold = erratic_threshold
        self._min_length = min_trajectory_length

    def analyze(self, tracks: List[KalmanBoxTracker]) -> List[TrajectoryScore]:
        """
        Analyze all tracks and return trajectory anomaly scores.

        Parameters
        ----------
        tracks : list of KalmanBoxTracker

        Returns
        -------
        list of TrajectoryScore
        """
        results = []
        for track in tracks:
            if not track.is_confirmed:
                continue
            score = self._score_trajectory(track)
            results.append(score)
        return results

    def analyze_single(self, track: KalmanBoxTracker) -> TrajectoryScore:
        """Analyze a single track."""
        return self._score_trajectory(track)

    def _score_trajectory(self, track: KalmanBoxTracker) -> TrajectoryScore:
        """Compute anomaly score for a single track's trajectory."""
        positions = list(track.trajectory)

        # Not enough data
        if len(positions) < self._min_length:
            return TrajectoryScore(
                track_id=track.id,
                class_name=track.class_name,
                score=0.0,
                direction_changes=0,
                avg_speed=0.0,
                max_acceleration=0.0,
                is_erratic=False,
                reason="Insufficient trajectory data",
            )

        # ── Compute velocities ───────────────────────────────
        positions_arr = np.array(positions, dtype=np.float64)
        velocities = np.diff(positions_arr, axis=0)  # (N-1, 2)
        speeds = np.linalg.norm(velocities, axis=1)   # (N-1,)

        avg_speed = float(np.mean(speeds))

        # ── Compute accelerations ────────────────────────────
        if len(speeds) >= 2:
            accels = np.abs(np.diff(speeds))
            max_accel = float(np.max(accels))
        else:
            accels = np.array([0.0])
            max_accel = 0.0

        # ── Compute direction changes ────────────────────────
        direction_changes = 0
        if len(velocities) >= 2:
            for i in range(1, len(velocities)):
                v_prev = velocities[i - 1]
                v_curr = velocities[i]

                # Skip near-zero velocity (stationary object)
                norm_prev = np.linalg.norm(v_prev)
                norm_curr = np.linalg.norm(v_curr)
                if norm_prev < 1e-3 or norm_curr < 1e-3:
                    continue

                # Angle between consecutive velocity vectors
                cos_angle = np.clip(
                    np.dot(v_prev, v_curr) / (norm_prev * norm_curr),
                    -1.0, 1.0,
                )
                angle = np.arccos(cos_angle)

                if angle > self._direction_threshold:
                    direction_changes += 1

        # ── Heading variance (lateral jitter) ────────────────
        if len(velocities) >= 3:
            headings = np.arctan2(velocities[:, 1], velocities[:, 0])
            # Filter out near-stationary points
            moving = speeds[: len(headings)] > 1e-3
            if np.sum(moving) >= 2:
                heading_var = float(np.var(headings[moving]))
            else:
                heading_var = 0.0
        else:
            heading_var = 0.0

        # ── Combine into a single anomaly score ──────────────
        # Normalize each component to [0, 1] and weight them
        n = len(positions)

        # Direction change rate (0 = none, 1 = change every frame)
        dir_score = min(direction_changes / max(n - 2, 1), 1.0)

        # Acceleration score (normalized by average speed)
        accel_score = min(max_accel / max(avg_speed, 1.0), 1.0)

        # Heading variance score (sigmoid mapping)
        heading_score = min(heading_var / 1.0, 1.0)  # var > 1.0 is very erratic

        # Weighted combination
        score = (
            0.40 * dir_score
            + 0.35 * accel_score
            + 0.25 * heading_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        is_erratic = score >= self._erratic_threshold

        # Generate reason
        reasons = []
        if direction_changes > 0:
            reasons.append(f"{direction_changes} sharp direction change(s)")
        if accel_score > 0.3:
            reasons.append(f"high acceleration (peak={max_accel:.1f} px/f²)")
        if heading_score > 0.3:
            reasons.append(f"heading jitter (var={heading_var:.3f})")
        if not reasons:
            reasons.append("normal trajectory")

        return TrajectoryScore(
            track_id=track.id,
            class_name=track.class_name,
            score=score,
            direction_changes=direction_changes,
            avg_speed=avg_speed,
            max_acceleration=max_accel,
            is_erratic=is_erratic,
            reason="; ".join(reasons),
        )
