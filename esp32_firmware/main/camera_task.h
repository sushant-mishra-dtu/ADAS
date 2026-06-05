#pragma once

#include "esp_camera.h"

// Camera pin definitions for ESP32-P4-EYE (MIPI-CSI or parallel, adjust as needed)
// Assuming standard config for ESP32-P4-EYE
#define CAM_PIN_PWDN    -1
#define CAM_PIN_RESET   -1
#define CAM_PIN_XCLK    -1
#define CAM_PIN_SIOD    -1
#define CAM_PIN_SIOC    -1
// ... other pins ...

void camera_task(void *pvParameters);
esp_err_t init_camera();
