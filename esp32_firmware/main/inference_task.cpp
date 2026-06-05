#include "inference_task.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Note: Requires ESP-DL library to be included in CMakeLists.h
// #include "esp_dl.h"

static const char *TAG = "INFERENCE_TASK";

void inference_task(void *pvParameters) {
    ESP_LOGI(TAG, "Inference Task started");
    
    // Initialize ESP-DL Model (e.g., yolov8n_quantized.espdl)
    // Model *model = new Model();
    
    while (1) {
        // Wait for frame from camera task queue (downsampled)
        // Run ESP-DL inference
        // model->run(input_tensor);
        
        // Process bounding boxes
        // If anomaly detected (e.g., person too close), signal H264 Encoder task to save clip
        
        vTaskDelay(pdMS_TO_TICKS(100)); // ~10 fps inference loop
    }
}
