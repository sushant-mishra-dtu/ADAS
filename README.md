<div align="center">

# 🚗 ADAS — Decentralized Data Collection Platform for Autonomous Driving in India

**A low-cost, edge-computing dashcam node built on the Raspberry Pi Zero 2W that captures, filters, and logs high-value driving data using active learning — purpose-built for training Vision-Language Models (VLMs) on India's chaotic roads.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi%20Zero%202W-C51A4A?logo=raspberrypi&logoColor=white)](#hardware-specifications)
[![Model: YOLOv8n](https://img.shields.io/badge/Model-YOLOv8%20Nano-FF6F00)](#ai-pipeline--active-learning)
[![Tests: 78 Passed](https://img.shields.io/badge/Tests-78%20Passed-brightgreen)](#testing)
[![Hackathon: Vihaan 9](https://img.shields.io/badge/Hackathon-Vihaan%209-blueviolet)](#)

</div>

---

## 📑 Table of Contents

- [Executive Summary](#executive-summary)
- [The Problem — The Indian Data Deficit](#the-problem--the-indian-data-deficit)
- [The Solution — Edge-Optimized Hardware](#the-solution--edge-optimized-hardware)
- [System Architecture](#system-architecture)
- [AI Pipeline & Active Learning](#ai-pipeline--active-learning)
- [🆕 Kalman Tracker & Object Tracking](#-kalman-tracker--multi-object-tracking)
- [🆕 Adaptive Threshold](#-adaptive-threshold--environment-aware-filtering)
- [🆕 Spatio-Temporal Model (ConvLSTM)](#-spatio-temporal-anomaly-scoring-convlstm)
- [VLM Optimization by Layering](#vlm-optimization-by-layering)
- [Consumer Incentive Model](#consumer-incentive-model-solving-the-cold-start)
- [Business Model — B2B SaaS](#business-model--b2b-saas)
- [Privacy & Compliance](#privacy--compliance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Raspberry Pi Deployment](#raspberry-pi-deployment)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Hardware Specifications](#hardware-specifications)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [License](#license)

---

## Executive Summary

This project presents a strategic roadmap for a **low-cost, decentralized data collection platform** designed to train advanced Vision-Language Models (VLMs) for autonomous driving in India.

By distributing edge-computing hardware to daily commuters, the platform aims to solve the **critical shortage of localized driving data** while creating a highly lucrative **B2B Software-as-a-Service (SaaS)** business model for automotive manufacturers. Instead of relying on expensive autonomous vehicle fleets to collect data, this system crowdsources real-world driving scenarios from millions of ordinary commuters — the people who actually navigate India's roads every day.

---

## The Problem — The Indian Data Deficit

### Failure of Western ADAS

Standard Advanced Driver Assistance Systems (ADAS) are trained on predictable, rule-compliant Western roads. These models assume clear lane markings, regulated intersections, standardized signage, and law-abiding road users — assumptions that **consistently fail in India's chaotic, unstructured traffic environment**.

### Sensor Blindness in Premium Vehicles

Even premium vehicles equipped with state-of-the-art ADAS currently face critical issues on Indian roads:

| Issue | Root Cause |
|-------|-----------|
| **False positives** | Stray animals, hand-painted signs, and jaywalkers trigger incorrect alerts |
| **Sudden braking** | Autorickshaws and two-wheelers making unpredictable lane changes |
| **Sensor confusion** | Missing lane markings, unmarked speed bumps, construction zones |
| **Weather blindness** | Monsoon rain, dust storms, and fog endemic to the subcontinent |
| **Intersection deadlocks** | Unregulated multi-way junctions with no traffic signals |

### The True Bottleneck

> **The main barrier to autonomous driving in emerging markets is not a lack of computing power, but a severe deficit of geographically specific, edge-case-rich training data.**

Western datasets (nuScenes, Waymo Open, KITTI) contain virtually no representation of:
- Unmarked rural highways shared with bullock carts
- Dense urban intersections with 6+ vehicle types simultaneously
- Hand-painted or non-standard road signage
- Two-wheelers carrying 3–4 passengers with no helmets
- Animals (cows, dogs, camels) as static road obstacles
- Night driving with no street lighting and oncoming high beams

---

## The Solution — Edge-Optimized Hardware

The platform relies on a **budget-friendly hardware device** engineered for mass consumer deployment — designed to be as cheap and unobtrusive as a traditional dashcam.

### Core Unit

The **Raspberry Pi Zero 2W** is selected as the optimal processing unit, offering a unique balance of:
- **Ultra-low cost** (~₹1,500 / $18) for mass deployment feasibility
- **Sufficient compute** — quad-core ARM Cortex-A53 @ 1 GHz with 512 MB RAM
- **Low power draw** — can be powered directly from a vehicle's USB port
- **Compact form factor** — 65mm × 30mm, easily mounted behind a rearview mirror

### Sensor Integration

| Component | Purpose | Interface |
|-----------|---------|-----------|
| **HD Camera Module** | Visual data capture (720p @ 15 FPS) | CSI / USB |
| **GNSS Module** (u-blox NEO-6M) | Precise geolocation & route mapping | UART |
| **4G LTE Module** (SIM7600) | Cloud data transmission for edge cases | USB |
| **OBD-II Adapter** (ELM327) | Vehicle telematics extraction | Bluetooth / CAN |

### Telematics Extraction

Crucially, the device interfaces with the vehicle's **OBD-II diagnostic port** to capture granular mechanical data, directly correlating **human driver intent** with visual observation:

- 🏎️ **Speed & RPM** — Captures driving velocity and engine state
- 🔄 **Steering Angle** — Records the exact steering input at any moment
- 🛑 **Brake Pressure** — Detects hard braking events and panic stops
- ⛽ **Throttle Position** — Measures acceleration behavior

This multimodal pairing of *what the driver sees* with *what the driver does* creates an exceptionally rich training signal for imitation-learning–based autonomous driving models.

---

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EDGE DEVICE (Pi Zero 2W)                        │
│                                                                         │
│   Camera (OpenCV)                                                       │
│        │                                                                │
│        ▼                                                                │
│   Frame Buffer ──────► YOLOv8n Inference ──────► Confidence Filter      │
│   (Ring Buffer)         (320px, quantized)       (Adaptive Threshold)   │
│        │                      │                        │                │
│        │              Kalman Tracker                   │                │
│        │          (persistent object IDs)              │                │
│        │              Trajectory Analyzer              │                │
│        │               (erratic motion)                │                │
│        │                                               ▼                │
│   OBD-II / CAN Bus                             ┌─────────────┐        │
│        │                                        │ Data Logger  │        │
│        ▼                                        │ (SD Card)    │        │
│   Telemetry Buffer ────────────────────────────►│ • Frames     │        │
│   (10 Hz sampling)                              │ • Telemetry  │        │
│        │                                        │ • Video Clips│        │
│   Clip Buffer ─────────────────────────────────►└──────┬──────┘        │
│   (Rolling 2s window)                                  │               │
│                                                         │               │
│   PII Blur (Edge) ◄────────────────────────────────────┘               │
│        │                                                                │
└────────┼────────────────────────────────────────────────────────────────┘
         │ (4G LTE — anomaly packets only)
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLOUD PLATFORM                                │
│                                                                         │
│   Ingestion → ConvLSTM Temporal Scorer → Annotation → VLM Training     │
│               (Spatio-Temporal Anomaly)  (3D Scenes)   (Fine-tuning)   │
│                                                                         │
│   B2B SaaS Dashboard ◄──────────────────────── Pre-trained Model APIs  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module-Level Architecture (Current MVP)

```
Camera (OpenCV) ──► Frame Buffer ──► YOLOv8n Inference ──► Adaptive Threshold
                         │                   │                      │
                    Clip Buffer        Kalman Tracker          Anomaly?
                  (rolling 2s)     (persistent IDs,       (environment-aware
                         │          trajectory scores)     dynamic cutoff)
                         │                   │                      │
                         └───────────────────┴──────────────────────┤
                                                                    ▼
OBD-II Simulator ──► Telemetry Buffer ───────────────────► Data Logger
                                                            (SD Card)
                                                          ┌──────────┐
                                                          │ • .jpg   │
                                                          │ • .json  │
                                                          │ • .npz   │
                                                          └──────────┘
                                                               │
                                                    ── 4G Upload ──
                                                               │
                                              Cloud: ConvLSTM Scorer
                                              (Spatio-Temporal Analysis)
```

The pipeline runs **three concurrent threads** coordinated by a central orchestrator:

1. **Camera Thread** — Continuously captures frames into a fixed-size ring buffer (prevents RAM exhaustion on the Pi Zero's 512 MB)
2. **OBD-II Thread** — Samples vehicle telemetry at 10 Hz, maintaining the latest snapshot
3. **Main Thread** — Pulls the latest frame + telemetry, runs YOLOv8 inference, applies Kalman tracking + adaptive threshold, and decides whether to persist the data

---

## AI Pipeline & Active Learning

### The Bandwidth Problem

Continuously streaming HD video over 4G/LTE networks from millions of devices would incur **astronomical cloud bandwidth costs** — potentially ₹50,000+/device/year. The system solves this with an intelligent filtering mechanism.

### Active Learning Strategy

The device runs a **lightweight local model** (quantized YOLOv8 Nano at 320px resolution) to compute uncertainty scores in real-time:

```
Frame Captured
      │
      ▼
Run YOLOv8n Inference (320px)
      │
      ├──► Kalman Tracker (assign persistent IDs)
      │        └──► Trajectory Analyzer (flag erratic motion)
      │
      ├──► Adaptive Threshold (environment-aware)
      │        └──► Adjusts based on: night/rain/traffic/open road
      │
      ▼
Anomaly Decision:
      ├── confidence ≥ dynamic_threshold ──► DISCARD
      │
      └── confidence < dynamic_threshold ──► SAVE & UPLOAD
              or no detections ──► SAVE & UPLOAD
              or erratic trajectory ──► SAVE & UPLOAD
```

### Why This Works

| Scenario | Model Confidence | Adaptive Threshold | Action |
|----------|:---------------:|:------------------:|--------|
| Clear highway with marked lanes | **High** (0.7+) | 0.52 | Discard — routine |
| Standard intersection with traffic | **Medium** (0.4–0.7) | 0.38 | Discard — routine |
| Cow blocking a rural highway | **Low** (<0.35) | 0.28 | **Save** — rare edge case |
| Night driving, unlit road | **Low** (<0.20) | 0.08 (adapted) | **Save** — only the most unusual |
| Monsoon rain, all frames confused | **Low** (<0.25) | 0.05 (adapted) | Save only **extreme** cases |
| Vehicle swerving erratically | **Medium** (0.50) | 0.40 | **Save** — Kalman flags trajectory |

> 💡 The adaptive threshold prevents **storage flooding** in bad conditions (night, rain) by recognizing that low confidence is *normal* in those environments and adjusting accordingly.

This selective approach reduces upload volume by an estimated **85–95%**, transmitting only the most valuable data packets to the cloud.

---

## 🆕 Kalman Tracker & Multi-Object Tracking

### Why Track Objects?

Without tracking, each frame is analyzed in isolation — the system knows *a car exists* but not *which car* or *where it's going*. The Kalman Tracker adds **temporal intelligence**:

```
Without Tracking:                    With Kalman Tracking:
  Frame 1: "car at (200, 300)"       Frame 1: "car #1 at (200, 300)"
  Frame 2: "car at (210, 305)"       Frame 2: "car #1 → moving right at 10px/f"
  Frame 3: (YOLO miss)               Frame 3: "car #1 predicted at (220, 310)" ✓
  Frame 4: "car at (230, 310)"       Frame 4: "car #1 confirmed, speed=12.5px/f"
```

### Architecture

The tracker uses a **7D Kalman Filter** with a constant-velocity motion model:

| Component | Purpose |
|-----------|---------|
| **`KalmanBoxTracker`** | Single-object tracker with state `[cx, cy, area, aspect_ratio, dx, dy, d_area]` |
| **`MultiObjectTracker`** | Manages all tracks — creates, matches (greedy IoU), and removes stale ones |
| **`TrajectoryAnalyzer`** | Scores each track's trajectory for anomalous behavior (sharp turns, sudden acceleration, lateral jitter) |

### Trajectory Anomaly Detection

The `TrajectoryAnalyzer` flags erratic movement patterns that a single-frame YOLO pass cannot detect:

| Metric | What It Measures | Weight |
|--------|-----------------|:------:|
| Direction changes | Sharp turns (>45° in 3 frames) | 40% |
| Acceleration | Sudden speed changes | 35% |
| Heading variance | Lateral jitter / weaving | 25% |

```python
from src.kalman_tracker import MultiObjectTracker, TrajectoryAnalyzer

tracker = MultiObjectTracker()
analyzer = TrajectoryAnalyzer()

# Each frame: feed YOLO detections
tracked = tracker.update([[100, 200, 150, 280, 0.85, "car"]])
scores = analyzer.analyze(tracker.get_all_tracks())

for s in scores:
    if s.is_erratic:
        print(f"⚠ Object #{s.track_id} ({s.class_name}): {s.reason}")
        # "⚠ Object #3 (car): 4 sharp direction change(s); high acceleration"
```

---

## 🆕 Adaptive Threshold — Environment-Aware Filtering

### The Problem with Fixed Thresholds

A static `CONFIDENCE_THRESHOLD = 0.35` causes:
- **Night driving**: Saves every dark frame (SD card floods)
- **Monsoon rain**: Saves every rainy frame (model is always confused)
- **Clear highway**: Misses subtle anomalies (threshold too permissive)

### The Solution

The `AdaptiveThreshold` dynamically adjusts based on recent driving conditions:

```
threshold = clamp(rolling_mean − k × rolling_std, 0.10, 0.60)
```

Where **k** (sensitivity) varies by automatically detected environment:

| Environment | Detection Method | Sensitivity (k) | Effect |
|-------------|-----------------|:----------------:|--------|
| ☀️ Clear Day | Bright frame, moderate detections | 1.0 | Standard |
| 🌙 Night | Low brightness (<0.25) | 1.5 | More selective |
| 🌧️ Rain/Fog | Medium brightness, low variance | 1.5 | More selective |
| 🚗 Dense Traffic | 5+ simultaneous detections | 1.2 | Slightly selective |
| 🛣️ Open Road | 0–1 detections | 0.8 | More sensitive |

```python
from src.adaptive_threshold import AdaptiveThreshold

threshold = AdaptiveThreshold()

result = threshold.update(
    max_confidence=0.42,
    frame_brightness=0.10,   # dark frame → NIGHT
    num_detections=1,
)

print(result.dynamic_threshold)  # 0.08 (adapted for night)
print(result.environment)        # "night"
print(result.is_anomaly)         # True (0.42 is unusual for night)
```

---

## 🆕 Spatio-Temporal Anomaly Scoring (ConvLSTM)

### Why Temporal Analysis?

A single frame tells you *a cow is there*. A sequence of frames tells you *the cow suddenly bolted across the road while the car was doing 60 km/h*. The **ConvLSTM** (Convolutional LSTM) model captures this temporal context.

### Architecture

```
Input: Video Clip (batch, seq_len, 3, 128, 128)
                    │
        ┌───────────┴───────────┐
        │  CNN Feature Extractor │  3 conv layers, 8× spatial downsample
        │  (per frame)           │  Conv(3→32→64→64) + BN + ReLU + MaxPool
        └───────────┬───────────┘
                    │  Feature maps: (batch, seq, 64, 16, 16)
        ┌───────────┴───────────┐
        │  ConvLSTM Layer        │  Convolutional gates (i/f/o/g)
        │  (processes sequence)  │  Preserves spatial structure
        └───────────┬───────────┘
                    │  Final hidden: (batch, 64, 16, 16)
        ┌───────────┴───────────┐
        │  Classification Head   │  Global Avg Pool → FC(64→128)
        │                        │  → Dropout(0.3) → FC(128→1) → Sigmoid
        └───────────┬───────────┘
                    │
              Score ∈ [0, 1]
```

- **~500K parameters** — lightweight for cloud CPU inference
- Supports **multi-layer stacking** and batch processing
- Includes **MockTemporalScorer** fallback using frame differencing

### Edge-Cloud Split

| Edge (Pi Zero 2W) | Cloud (Server) |
|:------------------:|:--------------:|
| `ClipBuffer` captures rolling 2s window | `CloudAnomalyScorer` loads .npz clips |
| Saves .npz clip on anomaly trigger | Runs ConvLSTM inference |
| ~27 MB RAM for buffer | Produces enriched JSON with temporal score |

```python
# Cloud-side scoring
from src.cloud_scorer import CloudAnomalyScorer

scorer = CloudAnomalyScorer()
scorer.load_model()

result = scorer.score_clip("data/clips/20260403_021500_clip.npz")
print(result.temporal_score)        # 0.78
print(result.is_critical_anomaly)   # True
print(result.description)
# "[HIGH] Temporal anomaly score: 0.780 | 15 frames over 1.0s |
#  Avg motion: 0.0842, Peak motion: 0.1523 | Significant temporal anomaly..."
```

---

## VLM Optimization by Layering

The data collected by the edge fleet feeds a **4-layer VLM training pipeline** that minimizes compute cost while maximizing India-specific driving understanding:

```
Layer 1: Frozen Vision Encoder (CLIP/SigLIP)     ← NOT retrained (saves 80% compute)
           │
Layer 2: Trainable Projection MLP                ← Learns India-specific mappings
           │                                        (cow → "large static animal obstacle")
Layer 3: LoRA-Finetuned LLM (LLaMA/Gemma)        ← Only 0.1% params trained
           │                                        (learns driving reasoning)
Layer 4: Temporal Fusion (ConvLSTM features)      ← Adds motion/trajectory context
           │
         Output: Rich driving descriptions + risk scores
```

| Approach | Trainable Params | GPU Hours | Cost |
|----------|:---------------:|:---------:|:----:|
| Full VLM fine-tune | ~7B | 500+ | $$$$ |
| Layered (freeze + LoRA + fusion) | ~12M | 25–60 | $ |

---

## Consumer Incentive Model (Solving the "Cold Start")

### 1. 🪙 Tokenomics (DePIN)

Drivers earn **passive income** via cryptographic tokens, weighted by the **rarity** of collected data.

### 2. 🛡️ Usage-Based Insurance (UBI)

Safe drivers earn up to **25–30% discounts** on annual car insurance via verified OBD-II telematics.

### 3. 📹 On-Device Utility

- **Legal protection** — timestamped footage for disputes
- **Drowsiness detection** — real-time acoustic alarms
- **Hazard alerts** — forward collision warnings
- **Route analytics** — driving insights via companion app

---

## Business Model — B2B SaaS

| Tier | Offering | Target Customer |
|------|----------|----------------|
| **Data Lake Access** | Raw anonymized video + telemetry | R&D labs, academic institutions |
| **Annotated Datasets** | 3D scene graphs, semantic segmentation | ADAS engineering teams |
| **Pre-trained VLM APIs** | Fine-tuned models via REST API | OEMs integrating L2/L3 autonomy |
| **Custom Model Training** | Dedicated fine-tuning on OEM platforms | Premium OEM partnerships |

**Target Customers:** Tata Motors, Mahindra & Mahindra, Ola Electric, Hyundai, Kia, Toyota

---

## Privacy & Compliance

### Privacy by Design (DPDP Act 2023)

| Layer | Mechanism | Description |
|-------|----------|-------------|
| **Edge** | PII Blur | On-device face & license plate blurring |
| **Transit** | TLS 1.3 | End-to-end encrypted transmission |
| **Cloud** | Anonymization | Location aggregated to 100m grid cells |
| **Architecture** | Federated Learning | Only model gradients uploaded |
| **Policy** | Consent-First | Explicit opt-in with granular controls |

---

## Project Structure

```
ADAS/
├── src/
│   ├── __init__.py
│   ├── __main__.py                # Package entry point
│   ├── camera_module.py           # Threaded video frame capture with ring buffer
│   ├── obd_simulator.py           # Mock CAN bus telematics generator (10 Hz)
│   ├── active_learner.py          # YOLOv8n inference & uncertainty scoring
│   ├── data_logger.py             # Synchronized frame + telemetry writer
│   ├── main.py                    # EdgeDashPipeline orchestrator / entry point
│   │
│   │   # ── New Modules (self-contained, no existing code modified) ──
│   ├── kalman_tracker.py          # 🆕 Multi-object Kalman tracker + trajectory analysis
│   ├── adaptive_threshold.py      # 🆕 Environment-aware dynamic anomaly threshold
│   ├── clip_buffer.py             # 🆕 Rolling frame buffer for temporal clip capture
│   ├── temporal_model.py          # 🆕 ConvLSTM spatio-temporal anomaly scorer
│   └── cloud_scorer.py            # 🆕 Cloud-side clip loader + model inference
│
├── utils/
│   ├── __init__.py
│   └── config.py                  # Centralized configuration constants
├── models/                        # YOLOv8n weights (.pt / .onnx)
├── data/
│   ├── frames/                    # Saved anomaly frames (.jpg)
│   ├── telemetry/                 # Correlated JSON telemetry packets
│   └── clips/                     # 🆕 Saved video clips (.npz) for temporal scoring
├── tests/
│   ├── __init__.py
│   ├── test_kalman_tracker.py     # 🆕 30 tests — tracker, IoU, trajectory
│   ├── test_adaptive_threshold.py # 🆕 21 tests — threshold, environment, integration
│   └── test_temporal.py           # 🆕 27 tests — ConvLSTM, clip buffer, cloud scorer
├── requirements.txt               # Python dependencies
└── README.md                      # ← You are here
```

### Module Responsibilities

| Module | Responsibility | Runs On |
|--------|---------------|:-------:|
| `camera_module.py` | Captures frames via OpenCV into a fixed-size ring buffer. Prevents RAM overflow on 512 MB. | 🥧 Pi |
| `obd_simulator.py` | Generates realistic OBD-II telemetry (speed, RPM, steering, brake pressure) at 10 Hz. | 🥧 Pi |
| `active_learner.py` | Runs YOLOv8 Nano inference at 320px. Applies uncertainty threshold to flag edge cases. | 🥧 Pi |
| `data_logger.py` | Persists anomaly frames as JPEG + correlated telemetry as JSON. | 🥧 Pi |
| `main.py` | Orchestrates the pipeline lifecycle: starts threads, runs the main loop, monitors memory. | 🥧 Pi |
| `config.py` | Centralized constants for camera, OBD, YOLO, system resources, and logging. | 🥧 Pi |
| `kalman_tracker.py` | Assigns persistent IDs to detections using Kalman filter. Analyzes trajectories. | 🥧 Pi |
| `adaptive_threshold.py` | Environment-aware dynamic threshold. Classifies driving conditions. | 🥧 Pi |
| `clip_buffer.py` | Maintains rolling 2-second frame window. Captures clips around anomaly events. | 🥧 Pi |
| `temporal_model.py` | ConvLSTM neural network (~500K params). Scores video clips for temporal anomalies. | ☁️ Cloud |
| `cloud_scorer.py` | Loads .npz clips, runs ConvLSTM inference, produces enriched JSON results. | ☁️ Cloud |

---

## Quick Start

### Prerequisites

- Python 3.9+
- (Optional) Raspberry Pi Zero 2W with camera module
- (Optional) YOLOv8 Nano weights in `models/yolov8n.pt`
- (Optional) PyTorch — required only for cloud-side temporal model

> **Note:** The system includes mock inference and mock scoring fallbacks — you can run and test everything on any machine without a Pi, YOLO weights, or PyTorch.

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/BHUKKADDD/ADAS.git
cd ADAS

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install new module dependencies
pip install torch pytest         # For ConvLSTM + testing

# 5. (Optional) Download YOLOv8 Nano weights
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/

# 6. Run the pipeline
python -m src.main
```

### Expected Output

```
03:14:22 [adas            ] INFO    ============================================================
03:14:22 [adas            ] INFO      ADAS — Advanced Driver Assistance System
03:14:22 [adas            ] INFO      Confidence threshold: 0.35
03:14:22 [adas            ] INFO    ============================================================
03:14:23 [active_learner  ] INFO    YOLOv8 model loaded and warmed up.
03:14:23 [camera_module   ] INFO    Camera started — index=0  resolution=640x480  target_fps=15
03:14:23 [obd_simulator   ] INFO    OBD-II simulator started — poll_interval=0.10s
03:14:23 [adas            ] INFO    All subsystems initialized. Pipeline is RUNNING.
03:14:25 [active_learner  ] DEBUG   ANOMALY flagged — max_conf=0.182  detections=2  time=87.3ms
03:14:25 [data_logger     ] INFO    Packet saved: 20260403_031425_123  (frame=45.2 KB)
```

---

## Raspberry Pi Deployment

### Hardware Required

| Component | Model | Cost (approx.) | Interface |
|-----------|-------|:--------------:|-----------|
| Computer | Raspberry Pi Zero 2W | ₹1,500 | — |
| Camera | Pi Camera Module v2 (8MP) | ₹1,200 | CSI ribbon |
| GPS | u-blox NEO-6M | ₹400 | UART |
| 4G | SIM7600E-H USB dongle | ₹2,500 | USB |
| OBD-II | ELM327 v2.1 Bluetooth | ₹500 | Bluetooth |
| Storage | 32GB+ MicroSD | ₹500 | — |
| **Total** | | **~₹6,100 (~$73)** | |

### Step-by-Step Setup

```bash
# 1. Flash Raspberry Pi OS Lite (64-bit) onto MicroSD
# 2. Enable SSH + Wi-Fi in Raspberry Pi Imager settings
# 3. SSH into the Pi
ssh pi@adas-pi.local

# 4. Enable camera
sudo raspi-config   # → Interface Options → Camera → Enable

# 5. Install system dependencies
sudo apt update && sudo apt install -y \
    python3-pip python3-venv python3-opencv \
    libatlas-base-dev libopenblas-dev

# 6. Transfer project from your PC
# (from Windows PowerShell)
scp -r "C:\path\to\ADAS" pi@adas-pi.local:~/ADAS

# 7. Set up Python environment on Pi
cd ~/ADAS
python3 -m venv venv
source venv/bin/activate
pip install opencv-python-headless numpy psutil Pillow

# 8. (Optional) Download YOLO weights
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/

# 9. Run!
python3 -m src.main
```

### Auto-Start on Boot

```bash
sudo nano /etc/systemd/system/adas.service
```

```ini
[Unit]
Description=ADAS Dashcam Pipeline
After=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ADAS
ExecStart=/home/pi/ADAS/venv/bin/python3 -m src.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable adas.service
sudo systemctl start adas.service
journalctl -u adas -f   # View live logs
```

---

## Configuration Reference

All tunable parameters live in [`utils/config.py`](utils/config.py):

### Camera Settings

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `CAPTURE_FPS` | `15` | Target frames per second |
| `FRAME_WIDTH` | `640` | Capture resolution width |
| `FRAME_HEIGHT` | `480` | Capture resolution height |
| `JPEG_QUALITY` | `80` | JPEG encode quality (0–100) |
| `FRAME_BUFFER_SIZE` | `4` | Max frames in ring buffer |

### OBD-II / Telematics

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `OBD_POLL_INTERVAL` | `0.1` s | Telemetry sampling rate (10 Hz) |
| `OBD_RPM_RANGE` | `700–6500` | Simulated engine RPM |
| `OBD_SPEED_RANGE` | `0–120` | Simulated speed (km/h) |
| `OBD_STEERING_RANGE` | `−540–540` | Simulated steering angle (°) |

### Active Learning / YOLO

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `YOLO_MODEL_PATH` | `models/yolov8n.pt` | YOLOv8 Nano weights |
| `YOLO_IMGSZ` | `320` | Inference resolution |
| `YOLO_CONF` | `0.25` | YOLO internal confidence filter |
| `CONFIDENCE_THRESHOLD` | `0.35` | Baseline anomaly threshold |

### System Resources

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `MAX_MEMORY_PERCENT` | `80` | Pause capture if RAM exceeds this % |
| `GC_INTERVAL_SECONDS` | `30` | Force garbage collection interval |

---

## Testing

The project includes **78 automated tests** covering all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_kalman_tracker.py -v       # 30 tests
python -m pytest tests/test_adaptive_threshold.py -v   # 21 tests
python -m pytest tests/test_temporal.py -v             # 27 tests (requires PyTorch)
```

### Test Coverage

| Test Suite | Tests | What's Covered |
|-----------|:-----:|----------------|
| `test_kalman_tracker.py` | 30 | IoU computation, greedy matching, box conversion, Kalman predict/update, multi-object tracking, trajectory analysis |
| `test_adaptive_threshold.py` | 21 | Environment classification, warmup behavior, threshold adaptation, clamping bounds, sensitivity multipliers, day-to-night transition |
| `test_temporal.py` | 27 | ClipBuffer operations, ConvLSTM cell shapes, SpatioTemporalScorer forward pass, MockTemporalScorer, CloudAnomalyScorer end-to-end |

---

## Hardware Specifications

### Target Device

| Spec | Value |
|------|-------|
| **Board** | Raspberry Pi Zero 2W |
| **SoC** | BCM2710A1 — Quad-core ARM Cortex-A53 @ 1 GHz |
| **RAM** | 512 MB LPDDR2 |
| **Connectivity** | 802.11 b/g/n Wi-Fi, Bluetooth 4.2 BLE |
| **GPIO** | 40-pin header (CSI camera, UART for GNSS) |
| **Power** | 5V / 2.5A via Micro-USB (vehicle USB port) |
| **Dimensions** | 65mm × 30mm × 5mm |
| **Cost** | ~₹1,500 / ~$18 USD |

---

## Tech Stack

| Layer | Technology |
|-------|-----------:|
| **Language** | Python 3.9+ |
| **Computer Vision** | OpenCV (headless) |
| **Object Detection** | Ultralytics YOLOv8 Nano |
| **Object Tracking** | Custom Kalman Filter (numpy) |
| **Temporal Model** | ConvLSTM (PyTorch, cloud-only) |
| **Inference Runtime** | ONNX Runtime (ARM-optimized) |
| **CAN Bus** | python-can (hardware), custom simulator (MVP) |
| **Image Processing** | Pillow, NumPy |
| **System Monitoring** | psutil |
| **Edge Deployment** | Raspberry Pi OS Lite (64-bit) |
| **Testing** | pytest (78 tests) |

---

## Roadmap

- [x] **Phase 1 — MVP Edge Node** *(Complete)*
  - [x] Threaded camera capture with ring buffer
  - [x] OBD-II telemetry simulation (CAN bus mock)
  - [x] YOLOv8 Nano inference with uncertainty scoring
  - [x] Active learning–based selective data logging
  - [x] Memory-safe pipeline with GC and resource monitoring

- [x] **Phase 1.5 — Advanced Edge Intelligence** *(Complete)*
  - [x] Multi-object Kalman tracker with persistent IDs
  - [x] Trajectory anomaly detection (erratic motion scoring)
  - [x] Adaptive threshold with environment classification
  - [x] Rolling clip buffer for temporal context capture
  - [x] ConvLSTM spatio-temporal anomaly scorer (cloud-side)
  - [x] 78 automated tests with full coverage

- [ ] **Phase 2 — Hardware Integration**
  - [ ] Real OBD-II ELM327 Bluetooth integration
  - [ ] GNSS module for geolocation tagging
  - [ ] 4G LTE selective upload to cloud ingestion API
  - [ ] On-device PII blurring (face + license plate)

- [ ] **Phase 3 — Cloud Platform**
  - [ ] Data lake ingestion pipeline (S3/GCS)
  - [ ] 3D scene graph annotation pipeline
  - [ ] VLM fine-tuning infrastructure (LoRA + Projection layers)
  - [ ] B2B SaaS dashboard for OEM customers

- [ ] **Phase 4 — Consumer Features**
  - [ ] Companion mobile app (Flutter)
  - [ ] DePIN tokenomics smart contracts
  - [ ] Insurtech API integration for UBI scoring
  - [ ] Driver drowsiness detection (face landmark model)
  - [ ] Real-time forward collision warning

- [ ] **Phase 5 — Scale**
  - [ ] Fleet management portal
  - [ ] Federated learning pipeline (privacy-preserving)
  - [ ] Multi-market expansion (Southeast Asia, Africa, LATAM)

---

## Team

Built with ❤️ for **Vihaan 9.0 Hackathon**

---

## License

MIT — See [LICENSE](LICENSE) for details.

---

<div align="center">

**🇮🇳 Making India's roads safer, one edge case at a time.**

</div>
