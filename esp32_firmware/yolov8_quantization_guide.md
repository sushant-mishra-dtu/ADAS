# YOLOv8 Quantization for ESP32-P4-EYE (ESP-DL)

To run YOLOv8 on the ESP32-P4 using the ESP-DL library, the model must be quantized to INT8 format. This allows the model to run efficiently using the hardware SIMD instructions (PIE) on the RISC-V cores.

## Prerequisites
1. Python 3.8+
2. `ultralytics` package
3. Espressif `esp-dl` repository cloned locally.

## Steps

### 1. Export YOLOv8 to ONNX
First, export your PyTorch `.pt` model to ONNX format.
```bash
yolo export model=models/yolov8n.pt format=onnx imgsz=224,224
```
*Note: Using a smaller image size like 224x224 or 192x192 is highly recommended for ESP32-P4 to achieve usable framerates.*

### 2. Prepare Calibration Dataset
Create a directory with 50-100 sample images from your target environment (e.g., Indian roads) for the INT8 calibration process.

### 3. Use ESP-DL Quantization Tool
Navigate to the `esp-dl/tools/quantization` directory in your ESP-DL clone.

Run the ESP-DL optimization and quantization script (this converts the `.onnx` to an `.espdl` or `.cpp` array).
```python
from optimizer import *
from calibrator import *
from evaluator import *

# 1. Optimize ONNX model
model = Model("yolov8n.onnx")
model.optimize()
model.save("yolov8n_optimized.onnx")

# 2. Quantize (requires writing a small Python script per esp-dl documentation)
# This will output yolov8n.cpp / yolov8n.hpp which you include in your ESP-IDF project.
```

### 4. Integrate into Firmware
Copy the generated `yolov8n.cpp` and `yolov8n.hpp` into the `esp32_firmware/main/` directory.

In your `inference_task.cpp`, initialize the model array:
```cpp
#include "yolov8n.hpp"
// Initialize model with quantized weights...
```
