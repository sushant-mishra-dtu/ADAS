#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "camera_task.h"
#include "can_obd_task.h"
#include "h264_encoder_task.h"
#include "inference_task.h"

static const char *TAG = "ADAS_MAIN";

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting ADAS Edge Node (ESP32-P4-EYE)");

    // Initialize NVS (often required for Wi-Fi and other components)
    // esp_err_t ret = nvs_flash_init();
    // ...

    // Spawn FreeRTOS tasks
    xTaskCreatePinnedToCore(camera_task, "camera_task", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(can_obd_task, "can_obd_task", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(inference_task, "inference_task", 8192, NULL, 4, NULL, 1);
    xTaskCreatePinnedToCore(h264_encoder_task, "h264_encoder_task", 8192, NULL, 4, NULL, 0);

    while (1) {
        ESP_LOGI(TAG, "Main loop running...");
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
