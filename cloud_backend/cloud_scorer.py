"""
cloud_scorer.py — Cloud-Side Spatio-Temporal Anomaly Scorer
=============================================================
Loads saved video clips (.npz) from the edge device and runs them
through the ConvLSTM-based SpatioTemporalScorer to produce enriched
anomaly scores with temporal context.

This module is designed to run on a cloud server or development machine
with sufficient compute — NOT on the Raspberry Pi edge device.

This module is fully self-contained and does NOT modify any existing code.

Workflow:
  1. Load a .npz clip file (frames + timestamps + metadata)
  2. Preprocess frames (resize to 128×128, normalize)
  3. Run through the ConvLSTM model
  4. Produce a ScoringResult with temporal anomaly score + description
  5. Save enriched results as JSON alongside the original clip

Usage:
    scorer = CloudAnomalyScorer()
    scorer.load_model()
    result = scorer.score_clip("data/clips/20260403_021500_123.npz")
    print(result)
    # ScoringResult(temporal_score=0.78, is_critical=True, ...)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Ensure legacy_python_edge is in sys.path so that src imports work
_repo_root = Path(__file__).resolve().parent.parent
_legacy_edge = _repo_root / "legacy_python_edge"
if _legacy_edge.exists() and str(_legacy_edge) not in sys.path:
    sys.path.append(str(_legacy_edge))

logger = logging.getLogger(__name__)

# ── Self-contained configuration ─────────────────────────────
DEFAULT_TEMPORAL_MODEL_PATH = "models/temporal_scorer.pt"
DEFAULT_INPUT_SIZE = 128
DEFAULT_ANOMALY_THRESHOLD = 0.6
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_CLIPS_DIR = str(_legacy_edge / "data" / "clips")

# Conditional PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# ── Scoring Result ───────────────────────────────────────────

@dataclass
class ScoringResult:
    """Result of running the spatio-temporal model on a video clip."""

    clip_path: str                          # Path to the source .npz file
    temporal_score: float                   # Anomaly score ∈ [0, 1]
    is_critical_anomaly: bool               # True if score > threshold
    num_frames: int                         # Number of frames in the clip
    clip_duration_seconds: float            # Duration of the clip
    description: str                        # Human-readable anomaly description
    per_frame_motion: List[float] = field(  # Frame-to-frame motion intensity
        default_factory=list
    )

    def to_dict(self) -> dict:
        return {
            "clip_path": self.clip_path,
            "temporal_score": round(self.temporal_score, 4),
            "is_critical_anomaly": self.is_critical_anomaly,
            "num_frames": self.num_frames,
            "clip_duration_seconds": round(self.clip_duration_seconds, 2),
            "description": self.description,
            "per_frame_motion": [round(m, 4) for m in self.per_frame_motion],
        }


# ── Cloud Anomaly Scorer ────────────────────────────────────

class CloudAnomalyScorer:
    """
    Loads saved clips and runs them through the temporal model.

    Supports two backends:
      1. PyTorch ConvLSTM (when torch + model weights are available)
      2. Mock scorer using frame differencing (fallback)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_TEMPORAL_MODEL_PATH,
        input_size: int = DEFAULT_INPUT_SIZE,
        threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    ) -> None:
        self._model_path = model_path
        self._input_size = input_size
        self._threshold = threshold
        self._model = None
        self._mock_scorer = None
        self._use_mock = False

        # Stats
        self._clips_scored: int = 0
        self._critical_count: int = 0

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the ConvLSTM model weights, or fall back to mock scorer."""
        if not HAS_TORCH:
            logger.warning(
                "PyTorch not available — using mock temporal scorer."
            )
            self._setup_mock()
            return

        try:
            from src.temporal_model import SpatioTemporalScorer

            self._model = SpatioTemporalScorer(
                hidden_dim=DEFAULT_HIDDEN_DIM,
                num_layers=DEFAULT_NUM_LAYERS,
            )

            # Try to load pretrained weights
            weights_path = Path(self._model_path)
            if weights_path.exists():
                state_dict = torch.load(
                    str(weights_path), map_location="cpu", weights_only=True
                )
                self._model.load_state_dict(state_dict)
                logger.info("Loaded temporal model weights from: %s", weights_path)
            else:
                logger.info(
                    "No pretrained weights found at %s — "
                    "using randomly initialized model (suitable for testing).",
                    weights_path,
                )

            self._model.eval()
            logger.info("Temporal model ready.\n%s", self._model.get_summary())

        except Exception as exc:
            logger.warning(
                "Failed to load temporal model: %s — using mock scorer.", exc
            )
            self._setup_mock()

    def _setup_mock(self) -> None:
        """Initialize the mock frame-differencing scorer."""
        from src.temporal_model import MockTemporalScorer
        self._mock_scorer = MockTemporalScorer()
        self._use_mock = True

    # ── Public Interface ─────────────────────────────────────

    def score_clip(self, clip_path: str) -> Optional[ScoringResult]:
        """
        Load a clip from disk and score it for spatio-temporal anomalies.

        Parameters
        ----------
        clip_path : str
            Path to a .npz clip file.

        Returns
        -------
        ScoringResult or None
            Scoring result with temporal anomaly score, or None on error.
        """
        try:
            # ── Load clip data ───────────────────────────────
            clip_data = np.load(clip_path, allow_pickle=True)
            frames = clip_data["frames"]        # (N, H, W, 3) — BGR uint8
            timestamps = clip_data["timestamps"]  # (N,) — unix timestamps

            num_frames = len(frames)
            if num_frames < 2:
                logger.warning("Clip %s has < 2 frames, skipping.", clip_path)
                return None

            duration = float(timestamps[-1] - timestamps[0])

            # ── Compute per-frame motion ─────────────────────
            motion_scores = self._compute_motion(frames)

            # ── Run temporal model ───────────────────────────
            if self._use_mock or self._model is None:
                temporal_score = self._mock_scorer.score_clip(list(frames))
            else:
                temporal_score = self._run_model_inference(frames)

            # ── Determine criticality ────────────────────────
            is_critical = temporal_score >= self._threshold

            # ── Generate description ─────────────────────────
            description = self._generate_description(
                temporal_score, motion_scores, num_frames, duration
            )

            self._clips_scored += 1
            if is_critical:
                self._critical_count += 1

            result = ScoringResult(
                clip_path=str(clip_path),
                temporal_score=temporal_score,
                is_critical_anomaly=is_critical,
                num_frames=num_frames,
                clip_duration_seconds=duration,
                description=description,
                per_frame_motion=motion_scores,
            )

            logger.info(
                "Scored clip: %s — score=%.3f  critical=%s  frames=%d",
                Path(clip_path).name, temporal_score, is_critical, num_frames,
            )
            return result

        except Exception as exc:
            logger.error("Failed to score clip %s: %s", clip_path, exc, exc_info=True)
            return None

    def score_all_clips(
        self, clips_dir: Optional[str] = None
    ) -> List[ScoringResult]:
        """
        Batch-score all .npz clips in a directory.

        Parameters
        ----------
        clips_dir : str, optional
            Directory containing .npz clip files (default: data/clips/).

        Returns
        -------
        list of ScoringResult
        """
        target_dir = Path(clips_dir) if clips_dir else Path(DEFAULT_CLIPS_DIR)
        clip_files = sorted(target_dir.glob("*.npz"))

        if not clip_files:
            logger.info("No clips found in %s", target_dir)
            return []

        logger.info("Scoring %d clips in %s ...", len(clip_files), target_dir)

        results = []
        for clip_path in clip_files:
            result = self.score_clip(str(clip_path))
            if result is not None:
                results.append(result)

                # Save enriched result JSON alongside the clip
                self._save_result(result, clip_path)

        logger.info(
            "Batch scoring complete — %d/%d clips scored, %d critical.",
            len(results), len(clip_files), self._critical_count,
        )
        return results

    # ── Internals ────────────────────────────────────────────

    def _run_model_inference(self, frames: np.ndarray) -> float:
        """Run the ConvLSTM model on a clip of frames."""
        # Preprocess: resize, normalize, convert BGR→RGB
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (self._input_size, self._input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            # Convert HWC → CHW
            chw = np.transpose(normalized, (2, 0, 1))
            processed.append(chw)

        # Stack into (1, seq_len, 3, H, W) tensor
        clip_array = np.stack(processed, axis=0)  # (T, 3, H, W)
        clip_tensor = torch.from_numpy(clip_array).unsqueeze(0)  # (1, T, 3, H, W)

        # Inference (no gradients needed)
        with torch.no_grad():
            score = self._model(clip_tensor)  # (1, 1)

        return float(score.squeeze().item())

    @staticmethod
    def _compute_motion(frames: np.ndarray) -> List[float]:
        """
        Compute per-frame motion intensity using absolute frame differences.

        Returns a list of N-1 motion values, one per consecutive frame pair.
        """
        motion = []
        for i in range(1, len(frames)):
            diff = np.abs(
                frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)
            )
            motion.append(float(np.mean(diff) / 255.0))
        return motion

    @staticmethod
    def _generate_description(
        score: float,
        motion_scores: List[float],
        num_frames: int,
        duration: float,
    ) -> str:
        """Generate a human-readable description of the anomaly analysis."""
        avg_motion = np.mean(motion_scores) if motion_scores else 0.0
        max_motion = max(motion_scores) if motion_scores else 0.0

        # Classify severity
        if score >= 0.8:
            severity = "CRITICAL"
            detail = (
                "Extreme temporal anomaly detected. The scene exhibits "
                "highly unusual motion patterns — possible sudden obstacle, "
                "vehicle swerving, or near-miss collision event."
            )
        elif score >= 0.6:
            severity = "HIGH"
            detail = (
                "Significant temporal anomaly. Unusual motion dynamics "
                "detected across the clip — possible erratic driving, "
                "unexpected road user behavior, or abrupt scene change."
            )
        elif score >= 0.4:
            severity = "MODERATE"
            detail = (
                "Moderate temporal variation. Some unusual motion detected "
                "but within semi-normal parameters. Worth reviewing for "
                "potential training value."
            )
        else:
            severity = "LOW"
            detail = (
                "Low temporal anomaly. Motion patterns appear relatively "
                "normal. This clip may have been flagged primarily by the "
                "single-frame YOLO detector."
            )

        return (
            f"[{severity}] Temporal anomaly score: {score:.3f} | "
            f"{num_frames} frames over {duration:.1f}s | "
            f"Avg motion: {avg_motion:.4f}, Peak motion: {max_motion:.4f} | "
            f"{detail}"
        )

    @staticmethod
    def _save_result(result: ScoringResult, clip_path: Path) -> None:
        """Save the enriched scoring result as JSON next to the clip file."""
        json_path = clip_path.with_suffix(".scored.json")
        json_data = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        json_path.write_text(json_data, encoding="utf-8")
        logger.debug("Saved scoring result: %s", json_path.name)

    @property
    def stats(self) -> dict:
        return {
            "clips_scored": self._clips_scored,
            "critical_anomalies": self._critical_count,
            "using_mock": self._use_mock,
        }
