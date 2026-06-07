"""
temporal_model.py — ConvLSTM Spatio-Temporal Anomaly Scorer
=============================================================
A recurrent neural network that processes a sequence of video frames
and outputs a single anomaly score ∈ [0, 1]. Higher scores indicate
more anomalous temporal dynamics (sudden motion changes, unexpected
object trajectories, abrupt scene transitions).

Architecture:
  1. CNN Feature Extractor — reduces each 128×128 frame into a
     compact spatial feature map (3 conv + batchnorm + relu layers)
  2. ConvLSTM Layer — processes the sequence of feature maps to
     capture temporal dependencies across frames
  3. Classification Head — global average pooling → FC → sigmoid

This model is designed for CLOUD-SIDE inference only. The edge device
(Pi Zero 2W) does NOT run this model — it only captures and saves clips.

This module is fully self-contained and does NOT modify any existing code.

Total parameters: ~500K (lightweight enough for CPU inference on cloud)

Usage:
    model = SpatioTemporalScorer()
    clip = torch.randn(1, 15, 3, 128, 128)   # (batch, seq, C, H, W)
    score = model(clip)                        # tensor([0.7231])
"""

import logging
import math
from typing import List, Optional, Tuple

# NOTE: TYPE_CHECKING imports help static analyzers (Pylance) resolve
# torch/nn symbols during type checking without executing heavy imports.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Expose names to static type checkers (no runtime import).
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

import numpy as np

logger = logging.getLogger(__name__)

# ── Self-contained configuration ─────────────────────────────
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_INPUT_SIZE = 128
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info(
        "PyTorch not installed — temporal model will use mock scorer. "
        "Install with: pip install torch"
    )

if TYPE_CHECKING or HAS_TORCH:
    class ConvLSTMCell(nn.Module):
        """
        A single Convolutional LSTM cell.

        Unlike a standard LSTM that uses fully-connected (matrix multiply)
        gates, a ConvLSTM uses 2D convolutions. This preserves the spatial
        structure of feature maps — critical for video understanding.

        Gate equations (all convolutions are 2D):
            i = σ(W_xi * X_t + W_hi * H_{t-1} + b_i)   # input gate
            f = σ(W_xf * X_t + W_hf * H_{t-1} + b_f)   # forget gate
            o = σ(W_xo * X_t + W_ho * H_{t-1} + b_o)   # output gate
            g = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g) # cell candidate
            C_t = f ⊙ C_{t-1} + i ⊙ g                   # new cell state
            H_t = o ⊙ tanh(C_t)                          # new hidden state

        Parameters
        ----------
        input_dim : int
            Number of channels in the input tensor X_t.
        hidden_dim : int
            Number of channels in the hidden state H_t.
        kernel_size : int
            Size of the convolutional kernel (applied with same-padding).
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int = 3,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.kernel_size = kernel_size
            self.padding = kernel_size // 2  # "same" padding

            # Single convolution that computes all 4 gates at once
            # Input: concat(X_t, H_{t-1}) → 4 × hidden_dim channels
            self.conv_gates = nn.Conv2d(
                in_channels=input_dim + hidden_dim,
                out_channels=4 * hidden_dim,
                kernel_size=kernel_size,
                padding=self.padding,
                bias=True,
            )

            self._initialize_weights()

        def _initialize_weights(self) -> None:
            """Xavier uniform initialization for stable training."""
            nn.init.xavier_uniform_(self.conv_gates.weight)
            if self.conv_gates.bias is not None:
                nn.init.zeros_(self.conv_gates.bias)

        def forward(
            self,
            x: "torch.Tensor",
            state: Optional[Tuple["torch.Tensor", "torch.Tensor"]] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Forward pass for one timestep.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch, input_dim, H, W).
            state : tuple of (h, c), optional
                Previous hidden and cell states. If None, initialized to zeros.

            Returns
            -------
            h_next, c_next : tuple of torch.Tensor
                Updated hidden and cell states, each (batch, hidden_dim, H, W).
            """
            batch, _, height, width = x.size()

            if state is None:
                device = x.device
                h = torch.zeros(batch, self.hidden_dim, height, width, device=device)
                c = torch.zeros(batch, self.hidden_dim, height, width, device=device)
            else:
                h, c = state

            # Concatenate input and previous hidden state along channel dim
            combined = torch.cat([x, h], dim=1)  # (B, input+hidden, H, W)

            # Compute all 4 gates in one convolution
            gates = self.conv_gates(combined)  # (B, 4*hidden, H, W)

            # Split into individual gates
            i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

            i = torch.sigmoid(i_gate)   # input gate
            f = torch.sigmoid(f_gate)   # forget gate
            o = torch.sigmoid(o_gate)   # output gate
            g = torch.tanh(g_gate)      # cell candidate

            # Update cell state and hidden state
            c_next = f * c + i * g
            h_next = o * torch.tanh(c_next)

            return h_next, c_next


    # ══════════════════════════════════════════════════════════
    #  CNN Feature Extractor
    # ══════════════════════════════════════════════════════════

    class CNNFeatureExtractor(nn.Module):
        """
        Lightweight CNN that reduces a 128×128×3 input frame into a
        compact spatial feature map suitable for the ConvLSTM.

        Architecture:
            Conv(3→32, 3×3) + BN + ReLU + MaxPool(2)   → 64×64×32
            Conv(32→64, 3×3) + BN + ReLU + MaxPool(2)  → 32×32×64
            Conv(64→out, 3×3) + BN + ReLU + MaxPool(2) → 16×16×out

        Total downsampling: 8× spatial, producing a 16×16 feature map.
        """

        def __init__(self, out_channels: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                # Block 1: 128×128×3 → 64×64×32
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 2: 64×64×32 → 32×32×64
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                # Block 3: 32×32×64 → 16×16×out_channels
                nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : torch.Tensor
                (batch, 3, 128, 128) — a single RGB frame.

            Returns
            -------
            torch.Tensor
                (batch, out_channels, 16, 16) — spatial feature map.
            """
            return self.net(x)


    # ══════════════════════════════════════════════════════════
    #  Spatio-Temporal Scorer (Full Model)
    # ══════════════════════════════════════════════════════════

    class SpatioTemporalScorer(nn.Module):
        """
        Complete spatio-temporal anomaly scoring model.

        Pipeline:
            Input clip: (batch, seq_len, 3, 128, 128)
                           │
                    CNN Feature Extractor (per frame)
                           │
                    (batch, seq_len, hidden_dim, 16, 16)
                           │
                    ConvLSTM (processes sequence)
                           │
                    Final hidden state: (batch, hidden_dim, 16, 16)
                           │
                    Global Average Pooling → (batch, hidden_dim)
                           │
                    FC → Dropout → FC → Sigmoid → (batch, 1)
                           │
                    Anomaly score ∈ [0, 1]

        Parameters
        ----------
        hidden_dim : int
            Channel dimension for ConvLSTM hidden state (default: 64).
        num_layers : int
            Number of stacked ConvLSTM layers (default: 1).
        dropout : float
            Dropout rate in the classification head (default: 0.3).
        """

        def __init__(
            self,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            num_layers: int = DEFAULT_NUM_LAYERS,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            # Per-frame spatial feature extraction
            self.cnn = CNNFeatureExtractor(out_channels=hidden_dim)

            # Stacked ConvLSTM layers
            self.convlstm_layers = nn.ModuleList()
            for i in range(num_layers):
                cell_input_dim = hidden_dim  # All layers use same dim
                self.convlstm_layers.append(
                    ConvLSTMCell(
                        input_dim=cell_input_dim,
                        hidden_dim=hidden_dim,
                        kernel_size=3,
                    )
                )

            # Classification head: spatial features → scalar anomaly score
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

            self._param_count = sum(p.numel() for p in self.parameters())
            logger.info(
                "SpatioTemporalScorer initialized — %d parameters (%.2f MB)",
                self._param_count,
                self._param_count * 4 / (1024 * 1024),  # float32
            )

        def forward(self, clip: "torch.Tensor") -> "torch.Tensor":
            """
            Score a video clip for temporal anomalies.

            Parameters
            ----------
            clip : torch.Tensor
                Shape: (batch, seq_len, 3, 128, 128) — a sequence of RGB frames.

            Returns
            -------
            torch.Tensor
                Shape: (batch, 1) — anomaly scores in [0, 1].
            """
            batch_size, seq_len, C, H, W = clip.size()

            # ── Step 1: Extract CNN features for every frame ──
            # Reshape to process all frames in one batch through CNN
            flat_frames = clip.view(batch_size * seq_len, C, H, W)
            features = self.cnn(flat_frames)  # (B*T, hidden, fH, fW)

            _, feat_c, feat_h, feat_w = features.size()
            features = features.view(batch_size, seq_len, feat_c, feat_h, feat_w)

            # ── Step 2: Process sequence through ConvLSTM ─────
            # Initialize hidden states for each layer
            states: List[Optional[Tuple["torch.Tensor", "torch.Tensor"]]] = [None] * self.num_layers

            for t in range(seq_len):
                x = features[:, t]  # (B, hidden, fH, fW)

                for layer_idx, cell in enumerate(self.convlstm_layers):
                    h, c = cell(x, states[layer_idx])
                    states[layer_idx] = (h, c)
                    x = h  # Output of this layer feeds into the next

            # Final hidden state from the last ConvLSTM layer
            assert states[-1] is not None
            final_h = states[-1][0]  # (B, hidden_dim, fH, fW)

            # ── Step 3: Global Average Pooling ────────────────
            pooled = final_h.mean(dim=[2, 3])  # (B, hidden_dim)

            # ── Step 4: Classification Head ───────────────────
            score = self.classifier(pooled)  # (B, 1)

            return score

        @property
        def param_count(self) -> int:
            return self._param_count

        def get_summary(self) -> str:
            """Human-readable model summary."""
            return (
                f"SpatioTemporalScorer(\n"
                f"  CNN: 3→32→64→{self.hidden_dim} channels, 8× downsample\n"
                f"  ConvLSTM: {self.num_layers} layer(s), "
                f"{self.hidden_dim} hidden channels, 3×3 kernel\n"
                f"  Head: {self.hidden_dim}→128→1 (sigmoid)\n"
                f"  Total params: {self._param_count:,}\n"
                f")"
            )


# ══════════════════════════════════════════════════════════════
#  Mock Scorer (Fallback when PyTorch is unavailable)
# ══════════════════════════════════════════════════════════════

class MockTemporalScorer:
    """
    Lightweight fallback that scores clips using simple frame differencing
    instead of a neural network. Used when PyTorch is not installed.

    Strategy:
      1. Compute pixel-wise absolute difference between consecutive frames.
      2. Average the differences to get a "motion intensity" score.
      3. High motion intensity → higher anomaly score (captures sudden
         changes like swerving, sudden obstacles, etc.)
    """

    def __init__(self) -> None:
        logger.info("MockTemporalScorer initialized (no PyTorch).")

    def score_clip(self, frames: List[np.ndarray]) -> float:
        """
        Score a list of BGR frames using frame differencing.

        Parameters
        ----------
        frames : list of np.ndarray
            Sequence of BGR images.

        Returns
        -------
        float
            Anomaly score ∈ [0, 1]. Higher = more anomalous motion.
        """
        if len(frames) < 2:
            return 0.5  # Not enough temporal context

        diffs = []
        for i in range(1, len(frames)):
            # Convert to grayscale for faster comparison
            prev_gray = np.mean(frames[i - 1], axis=2)
            curr_gray = np.mean(frames[i], axis=2)

            # Absolute pixel difference, normalized to [0, 1]
            diff = np.abs(curr_gray - prev_gray) / 255.0
            diffs.append(np.mean(diff))

        # Average motion intensity across all frame pairs
        mean_motion = np.mean(diffs)

        # Scale to [0, 1] using a sigmoid-like curve
        # Typical motion is ~0.02-0.05; anomalous is >0.1
        score = 1.0 / (1.0 + math.exp(-20 * (mean_motion - 0.08)))

        return float(np.clip(score, 0.0, 1.0))

    def __call__(self, frames: List[np.ndarray]) -> float:
        return self.score_clip(frames)
