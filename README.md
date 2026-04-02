<![CDATA[<div align="center">

# 🚗 ADAS — Decentralized Data Collection Platform for Autonomous Driving in India

**A low-cost, edge-computing dashcam node built on the Raspberry Pi Zero 2W that captures, filters, and logs high-value driving data using active learning — purpose-built for training Vision-Language Models (VLMs) on India's chaotic roads.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi%20Zero%202W-C51A4A?logo=raspberrypi&logoColor=white)](#hardware-specifications)
[![Model: YOLOv8n](https://img.shields.io/badge/Model-YOLOv8%20Nano-FF6F00)](#ai-pipeline--active-learning)
[![Hackathon: Vihaan 9](https://img.shields.io/badge/Hackathon-Vihaan%209-blueviolet)](#)

</div>

---

## 📑 Table of Contents

- [Executive Summary](#executive-summary)
- [The Problem — The Indian Data Deficit](#the-problem--the-indian-data-deficit)
- [The Solution — Edge-Optimized Hardware](#the-solution--edge-optimized-hardware)
- [System Architecture](#system-architecture)
- [AI Pipeline & Active Learning](#ai-pipeline--active-learning)
- [Consumer Incentive Model](#consumer-incentive-model-solving-the-cold-start)
- [Business Model — B2B SaaS](#business-model--b2b-saas)
- [Privacy & Compliance](#privacy--compliance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
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
│   (Ring Buffer)         (320px, quantized)       (threshold < 0.35)     │
│                                                        │                │
│   OBD-II / CAN Bus                                     │                │
│        │                                               ▼                │
│        ▼                                        ┌─────────────┐        │
│   Telemetry Buffer ────────────────────────────►│ Data Logger  │        │
│   (10 Hz sampling)                              │ (SD Card)    │        │
│                                                  └──────┬──────┘        │
│                                                         │               │
│   PII Blur (Edge) ◄────────────────────────────────────┘               │
│        │                                                                │
└────────┼────────────────────────────────────────────────────────────────┘
         │ (4G LTE — anomaly packets only)
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLOUD PLATFORM                                │
│                                                                         │
│   Ingestion API ──► Data Lake ──► Annotation Pipeline ──► VLM Training │
│                      (S3/GCS)     (3D Scene Graphs)       (Fine-tuning) │
│                                                                         │
│   B2B SaaS Dashboard ◄──────────────────────── Pre-trained Model APIs  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module-Level Architecture (Current MVP)

```
Camera (OpenCV) ──► Frame Buffer ──► YOLOv8n Inference ──► Confidence Filter
                                                                │
OBD-II Simulator ──► Telemetry Buffer ─────────────────────────►├──► Data Logger
                                                                │      (SD Card)
                                                         Anomaly Detected?
                                                        (confidence < threshold)
```

The pipeline runs **three concurrent threads** coordinated by a central orchestrator:

1. **Camera Thread** — Continuously captures frames into a fixed-size ring buffer (prevents RAM exhaustion on the Pi Zero's 512 MB)
2. **OBD-II Thread** — Samples vehicle telemetry at 10 Hz, maintaining the latest snapshot
3. **Main Thread** — Pulls the latest frame + telemetry, runs YOLOv8 inference, and decides whether to persist the data

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
      ▼
Extract max detection confidence
      │
      ├── confidence ≥ 0.35 ──► DISCARD (common / well-understood scene)
      │
      └── confidence < 0.35 ──► SAVE & UPLOAD (edge case / novel scenario)
              or
          no detections at all ──► SAVE & UPLOAD (empty road / rare scene)
```

### Why This Works

| Scenario | Model Confidence | Action |
|----------|:---------------:|--------|
| Clear highway with marked lanes | **High** (0.7+) | Discard — already well-represented in training data |
| Standard intersection with traffic lights | **Medium** (0.4–0.7) | Discard — routine scenario |
| Cow blocking a rural highway | **Low** (< 0.35) | **Save** — rare edge case, high training value |
| Unmarked jugaad vehicle | **Very Low** (< 0.15) | **Save** — model has never seen this class |
| Empty unlit road at night | **Zero** (no detections) | **Save** — novel environment condition |

This selective approach reduces upload volume by an estimated **85–95%**, transmitting only the most valuable data packets to the cloud.

### VLM Training (Cloud-Side)

The curated dataset of rare edge cases uploaded from the fleet is processed through:

1. **3D Scene Graph Annotation** — Spatial relationships between all detected objects
2. **Rich Text Captioning** — Natural language descriptions of the driving scenario
3. **Vision-Language Model Fine-tuning** — Training VLMs that use human-like *reasoning* (not just pattern matching) to make driving decisions

---

## Consumer Incentive Model (Solving the "Cold Start")

To convince the masses to install the hardware and continuously share data, the project relies on a robust **incentive triad**:

### 1. 🪙 Tokenomics (DePIN — Decentralized Physical Infrastructure Network)

Drivers act as stakeholders in the network, earning **passive income** via cryptographic tokens:
- Tokens are algorithmically rewarded based on the **volume and rarity** of collected data
- Rare edge-case data (e.g., first capture of a novel road hazard in a new geography) commands higher rewards
- Token value is backed by the commercial B2B data licensing revenue

### 2. 🛡️ Usage-Based Insurance (UBI)

By providing verified OBD-II telematics to **Insurtech partners**:
- Safe drivers earn **substantial discounts** (up to 25–30%) on annual car insurance premiums
- Driving behavior is scored based on real telemetry: acceleration patterns, braking frequency, cornering forces
- Partners include ACKO, Digit Insurance, and similar Indian Insurtech companies

### 3. 📹 On-Device Utility

The device provides immediate, tangible value to the driver as a **smart dashcam**:
- **Legal protection** — Timestamped HD footage for accident disputes or false traffic citations
- **Drowsiness detection** — Real-time acoustic alarms if driver inattention is detected
- **Hazard alerts** — Localized ADAS features including forward collision warnings and pedestrian alerts
- **Route analytics** — Driving pattern insights via a companion mobile app

---

## Business Model — B2B SaaS

### Revenue Engine

The project's commercial engine involves selling **tiered access** to the data lake, annotated datasets, and pre-trained VLM APIs to Indian OEMs (Original Equipment Manufacturers):

| Tier | Offering | Target Customer |
|------|----------|----------------|
| **Data Lake Access** | Raw anonymized video + telemetry streams | R&D labs, academic institutions |
| **Annotated Datasets** | 3D scene graphs, semantic segmentation, text captions | ADAS engineering teams |
| **Pre-trained VLM APIs** | Fine-tuned models via REST API with Indian road understanding | OEMs integrating L2/L3 autonomy |
| **Custom Model Training** | Dedicated model fine-tuning on OEM-specific vehicle platforms | Premium OEM partnerships |

### Target Customers

- 🇮🇳 **Tata Motors** — Localizing ADAS for Nexon, Harrier, and upcoming EV platforms
- 🇮🇳 **Mahindra & Mahindra** — Self-driving capabilities for XUV, Thar, and electric SUVs
- 🇮🇳 **Ola Electric** — Autonomous two-wheeler navigation in dense urban traffic
- 🌏 **Global OEMs** (Hyundai, Kia, Toyota) — Indian market localization of existing ADAS stacks

---

## Privacy & Compliance

### Privacy by Design (DPDP Act 2023)

The architecture complies strictly with India's **Digital Personal Data Protection (DPDP) Act 2023**:

| Layer | Mechanism | Description |
|-------|----------|-------------|
| **Edge** | PII Blur | On-device face & license plate blurring before any data leaves the vehicle |
| **Transit** | TLS 1.3 | End-to-end encrypted data transmission over 4G |
| **Cloud** | Data Anonymization | All location data is aggregated to 100m grid cells; no individual tracking |
| **Architecture** | Federated Learning | Raw data never leaves the vehicle — only model gradients are uploaded |
| **Policy** | Consent-First | Explicit opt-in required; granular controls over what data is shared |

---

## Project Structure

```
ADAS/
├── src/
│   ├── __init__.py
│   ├── __main__.py            # Package entry point
│   ├── camera_module.py       # Threaded video frame capture with ring buffer
│   ├── obd_simulator.py       # Mock CAN bus telematics generator (10 Hz)
│   ├── active_learner.py      # YOLOv8n inference & uncertainty scoring
│   ├── data_logger.py         # Synchronized frame + telemetry writer
│   └── main.py                # EdgeDashPipeline orchestrator / entry point
├── utils/
│   ├── __init__.py
│   └── config.py              # Centralized configuration constants
├── models/                    # YOLOv8n weights (.pt / .onnx)
├── data/
│   ├── frames/                # Saved anomaly frames (.jpg)
│   └── telemetry/             # Correlated JSON telemetry packets
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # ← You are here
```

### Module Responsibilities

| Module | Responsibility | Thread |
|--------|---------------|--------|
| `camera_module.py` | Captures frames via OpenCV into a fixed-size ring buffer. Prevents RAM overflow on the Pi Zero's 512 MB. | Daemon thread |
| `obd_simulator.py` | Generates realistic OBD-II telemetry (speed, RPM, steering, brake pressure) at 10 Hz. Simulates CAN bus for MVP development. | Daemon thread |
| `active_learner.py` | Runs YOLOv8 Nano inference at 320px. Applies uncertainty threshold to flag edge cases. Falls back to mock inference when model is unavailable. | Main thread (called per tick) |
| `data_logger.py` | Persists anomaly frames as JPEG + correlated telemetry as JSON with matching timestamps. | Main thread (called on anomaly) |
| `main.py` | Orchestrates the pipeline lifecycle: starts threads, runs the main loop, monitors memory pressure, performs periodic GC. | Main thread |
| `config.py` | Centralized constants for camera, OBD, YOLO, system resources, and logging. All tunable from one place. | N/A (imported) |

---

## Quick Start

### Prerequisites

- Python 3.9+
- (Optional) Raspberry Pi Zero 2W with camera module
- (Optional) YOLOv8 Nano weights in `models/yolov8n.pt`

> **Note:** The system includes a mock inference fallback — you can run the full pipeline on any machine without a Pi or YOLO weights for development and testing.

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/adas-edge-platform.git
cd adas-edge-platform/ADAS

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download YOLOv8 Nano weights
#    If skipped, the system falls back to mock inference automatically
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/

# 5. Run the pipeline
python -m src.main
```

### Expected Output

```
03:14:22 [adas            ] INFO    ============================================================
03:14:22 [adas            ] INFO      ADAS — Advanced Driver Assistance System
03:14:22 [adas            ] INFO      Confidence threshold: 0.35
03:14:22 [adas            ] INFO    ============================================================
03:14:23 [active_learner  ] INFO    YOLOv8 model loaded and warmed up.
03:14:23 [camera_module   ] INFO    Camera thread started (device 0).
03:14:23 [obd_simulator   ] INFO    OBD-II simulator thread started (10 Hz).
03:14:23 [adas            ] INFO    All subsystems initialized. Pipeline is RUNNING.
03:14:25 [active_learner  ] DEBUG   ANOMALY flagged — max_conf=0.182  detections=2  time=87.3ms
03:14:25 [data_logger     ] INFO    Saved anomaly #1 → frames/20260403_031425.jpg
```

---

## Configuration Reference

All tunable parameters live in [`utils/config.py`](utils/config.py):

### Camera Settings

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `CAMERA_INDEX` | `0` | OpenCV camera device index (`/dev/video0`) |
| `CAPTURE_FPS` | `15` | Target frames per second (Pi Zero sustains ~15 @ 640×480) |
| `FRAME_WIDTH` | `640` | Capture resolution width in pixels |
| `FRAME_HEIGHT` | `480` | Capture resolution height in pixels |
| `JPEG_QUALITY` | `80` | JPEG encode quality (0–100; lower = smaller file) |
| `FRAME_BUFFER_SIZE` | `4` | Max frames held in the ring buffer (prevents RAM overload) |

### OBD-II / Telematics

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `OBD_POLL_INTERVAL` | `0.1` s | Telemetry sampling rate (10 Hz) |
| `OBD_RPM_RANGE` | `700–6500` | Simulated engine RPM range |
| `OBD_SPEED_RANGE` | `0–120` | Simulated speed range (km/h) |
| `OBD_STEERING_RANGE` | `−540–540` | Simulated steering angle (degrees) |
| `OBD_BRAKE_PRESSURE_RANGE` | `0.0–1.0` | Normalized brake pressure |

### Active Learning / YOLO

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `YOLO_MODEL_PATH` | `models/yolov8n.pt` | Path to YOLOv8 Nano weights |
| `YOLO_IMGSZ` | `320` | Inference resolution (smaller = faster on Pi) |
| `YOLO_CONF` | `0.25` | YOLO's internal confidence filter |
| `CONFIDENCE_THRESHOLD` | `0.35` | Save frames with max confidence **below** this value |

### System Resources

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `MAX_MEMORY_PERCENT` | `80` | Pause capture if RAM usage exceeds this % |
| `GC_INTERVAL_SECONDS` | `30` | Force garbage collection every N seconds |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

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

### Sensor Stack

| Component | Model | Cost (approx.) | Interface |
|-----------|-------|:--------------:|-----------|
| Camera | Pi Camera Module v2 (8MP, 1080p) | ₹1,200 | CSI ribbon cable |
| GNSS | u-blox NEO-6M GPS module | ₹400 | UART (3.3V TX/RX) |
| 4G LTE | SIM7600E-H hat / USB dongle | ₹2,500 | USB |
| OBD-II | ELM327 v2.1 Bluetooth adapter | ₹500 | Bluetooth SPP |
| **Total BOM** | | **~₹5,600 (~$67)** | |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.9+ |
| **Computer Vision** | OpenCV (headless) |
| **Object Detection** | Ultralytics YOLOv8 Nano |
| **Inference Runtime** | ONNX Runtime (optimized for ARM) |
| **CAN Bus** | python-can (hardware), custom simulator (MVP) |
| **Image Processing** | Pillow, NumPy |
| **System Monitoring** | psutil |
| **Edge Deployment** | Raspberry Pi OS Lite (64-bit) |

---

## Roadmap

- [x] **Phase 1 — MVP Edge Node** *(Current)*
  - [x] Threaded camera capture with ring buffer
  - [x] OBD-II telemetry simulation (CAN bus mock)
  - [x] YOLOv8 Nano inference with uncertainty scoring
  - [x] Active learning–based selective data logging
  - [x] Memory-safe pipeline with GC and resource monitoring

- [ ] **Phase 2 — Hardware Integration**
  - [ ] Real OBD-II ELM327 Bluetooth integration
  - [ ] GNSS module for geolocation tagging
  - [ ] 4G LTE selective upload to cloud ingestion API
  - [ ] On-device PII blurring (face + license plate)

- [ ] **Phase 3 — Cloud Platform**
  - [ ] Data lake ingestion pipeline (S3/GCS)
  - [ ] 3D scene graph annotation pipeline
  - [ ] VLM fine-tuning infrastructure
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
]]>
