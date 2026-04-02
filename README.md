# 🚗 ADAS — Advanced Driver Assistance System

A low-cost, edge-computing dashcam node built for the **Raspberry Pi Zero 2W** that captures, filters, and logs high-value driving data using active learning.

## Architecture

```
Camera (OpenCV) ──► Frame Buffer ──► YOLOv8n Inference ──► Confidence Filter
                                                                │
OBD-II Simulator ──► Telemetry Buffer ─────────────────────────►├──► Data Logger
                                                                │      (SD Card)
                                                         Anomaly Detected?
                                                        (confidence < threshold)
```

## Project Structure

```
ADAS/
├── src/
│   ├── camera_module.py      # Threaded video frame capture
│   ├── obd_simulator.py      # Mock CAN bus telematics generator
│   ├── active_learner.py     # YOLOv8n inference & uncertainty scoring
│   ├── data_logger.py        # Synchronized frame + telemetry writer
│   └── main.py               # Orchestrator / entry point
├── utils/
│   └── config.py             # Centralized configuration constants
├── models/                   # YOLOv8n weights (.pt / .onnx)
├── data/
│   ├── frames/               # Saved anomaly frames (.jpg)
│   └── telemetry/            # Correlated JSON telemetry
├── tests/                    # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Navigate to the ADAS directory
cd ADAS

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python -m src.main
```

## Configuration

All tunable parameters live in `utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `CAPTURE_FPS` | `15` | Target frames per second |
| `FRAME_WIDTH` | `640` | Capture resolution width |
| `FRAME_HEIGHT` | `480` | Capture resolution height |
| `CONFIDENCE_THRESHOLD` | `0.35` | Save frames with max confidence **below** this |
| `FRAME_BUFFER_SIZE` | `4` | Max frames held in memory |
| `OBD_POLL_INTERVAL` | `0.1` | Telemetry sampling rate (seconds) |

## Hardware

- **Board:** Raspberry Pi Zero 2W (512 MB RAM, quad-core ARM Cortex-A53)
- **Camera:** Raspberry Pi Camera Module v2 / USB webcam
- **OBD-II:** ELM327-compatible CAN bus adapter (simulated for MVP)

## License

MIT — Built for the hackathon 🏁
