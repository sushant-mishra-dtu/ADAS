"""
active_learner.py — YOLOv8 Nano Inference & Uncertainty Scoring
================================================================
Runs a quantized YOLOv8n model on incoming frames and computes an
"uncertainty score" to decide whether a frame is an edge case worth
saving for active learning.

Active Learning Logic:
  1. Run YOLOv8n inference on a frame (down-scaled to 320px).
  2. Extract the maximum detection confidence across all detections.
  3. If max_confidence < CONFIDENCE_THRESHOLD → frame is uncertain
     and likely an anomaly / novel scenario → SAVE IT.
  4. If no detections at all → also flag it (empty road / rare scene).

This "save on low confidence" approach prioritizes under-represented
samples, which is the core idea behind pool-based active learning.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from utils.config import (
    CONFIDENCE_THRESHOLD,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of running YOLOv8 inference on a single frame."""

    max_confidence: float          # Highest detection confidence (0.0 if none)
    num_detections: int            # Total detection count
    is_anomaly: bool               # True → frame should be saved
    inference_time_ms: float       # Wall-clock inference time
    class_names: List[str]         # Detected class labels
    confidences: List[float]       # Per-detection confidence scores

    def to_dict(self) -> dict:
        return {
            "max_confidence": round(self.max_confidence, 4),
            "num_detections": self.num_detections,
            "is_anomaly": self.is_anomaly,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "class_names": self.class_names,
            "confidences": [round(c, 4) for c in self.confidences],
        }


class ActiveLearner:
    """
    Lightweight active learning filter around YOLOv8 Nano.

    Loads the model once and exposes a `process_frame()` method
    that returns an InferenceResult with the anomaly decision.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        imgsz: int = YOLO_IMGSZ,
        conf: float = YOLO_CONF,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model_path = model_path
        self._imgsz = imgsz
        self._conf = conf
        self._threshold = threshold
        self._model = None

        # Stats
        self._frames_processed = 0
        self._anomalies_flagged = 0

    # ── Lifecycle ────────────────────────────────────────────

    def load_model(self) -> None:
        """
        Load the YOLOv8 Nano model.

        Wrapped in a try/except so the pipeline can still run
        (with a mock fallback) even if the model file is missing —
        useful during development on non-Pi machines.
        """
        try:
            from ultralytics import YOLO

            logger.info("Loading YOLOv8 model from: %s", self._model_path)
            self._model = YOLO(self._model_path)

            # Warm-up inference (allocates ONNX/TFLite buffers)
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                imgsz=self._imgsz,
                conf=self._conf,
                verbose=False,
            )
            logger.info("YOLOv8 model loaded and warmed up.")

        except Exception as exc:
            logger.warning(
                "Could not load YOLO model (%s). "
                "Falling back to mock inference. Error: %s",
                self._model_path, exc,
            )
            self._model = None

    # ── Public Interface ─────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> InferenceResult:
        """
        Run inference on a single frame and return the anomaly decision.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (any resolution — will be resized internally).

        Returns
        -------
        InferenceResult
            Contains confidence scores, detection count, and anomaly flag.
        """
        start = time.monotonic()

        if self._model is not None:
            result = self._run_real_inference(frame)
        else:
            result = self._run_mock_inference(frame)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Determine anomaly: low confidence or zero detections
        is_anomaly = (
            result["max_conf"] < self._threshold
            or result["num_det"] == 0
        )

        self._frames_processed += 1
        if is_anomaly:
            self._anomalies_flagged += 1

        inference_result = InferenceResult(
            max_confidence=result["max_conf"],
            num_detections=result["num_det"],
            is_anomaly=is_anomaly,
            inference_time_ms=elapsed_ms,
            class_names=result["classes"],
            confidences=result["confs"],
        )

        if is_anomaly:
            logger.debug(
                "ANOMALY flagged — max_conf=%.3f  detections=%d  time=%.1fms",
                result["max_conf"], result["num_det"], elapsed_ms,
            )

        return inference_result

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frames_processed,
            "anomalies_flagged": self._anomalies_flagged,
            "model_loaded": self._model is not None,
        }

    # ── Internals ────────────────────────────────────────────

    def _run_real_inference(self, frame: np.ndarray) -> dict:
        """Run actual YOLOv8 inference and extract results."""
        results = self._model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._conf,
            verbose=False,
        )

        # Extract detection data from the first (only) result
        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return {"max_conf": 0.0, "num_det": 0, "classes": [], "confs": []}

        confs = boxes.conf.cpu().numpy().tolist()
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        class_names = [r.names[cid] for cid in class_ids]

        return {
            "max_conf": max(confs),
            "num_det": len(confs),
            "classes": class_names,
            "confs": confs,
        }

    def _run_mock_inference(self, frame: np.ndarray) -> dict:
        """
        Fallback when YOLO model isn't available.

        Uses simple image statistics (mean brightness, edge density)
        to simulate varying confidence levels. This lets the full
        pipeline be tested without the actual model file.
        """
        import random

        # Simulate inference delay (~50-150ms on Pi Zero)
        time.sleep(random.uniform(0.05, 0.15))

        # Use frame variance as a rough "interestingness" proxy
        gray = np.mean(frame) / 255.0  # 0..1 brightness
        noise = random.uniform(-0.2, 0.2)
        fake_conf = max(0.0, min(1.0, gray * 0.6 + 0.2 + noise))

        num_det = random.randint(0, 5) if fake_conf > 0.2 else 0
        mock_classes = ["car", "person", "truck", "bus", "motorcycle"]
        classes = random.sample(mock_classes, min(num_det, len(mock_classes)))
        confs = [round(random.uniform(0.1, fake_conf), 3) for _ in classes]

        return {
            "max_conf": max(confs) if confs else 0.0,
            "num_det": num_det,
            "classes": classes,
            "confs": confs,
        }
