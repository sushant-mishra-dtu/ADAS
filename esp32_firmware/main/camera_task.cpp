#include "camera_task.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "CAMERA_TASK";

esp_err_t init_camera() {
    camera_config_t config = {};
    // For ESP32-P4, configuration will differ significantly if using MIPI-CSI.
    // This is a placeholder for standard esp32-camera init structure.
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_PIN_D0;
    config.pin_d1 = CAM_PIN_D1;
    config.pin_d2 = CAM_PIN_D2;
    config.pin_d3 = CAM_PIN_D3;
    config.pin_d4 = CAM_PIN_D4;
    config.pin_d5 = CAM_PIN_D5;
    config.pin_d6 = CAM_PIN_D6;
    config.pin_d7 = CAM_PIN_D7;
    config.pin_xclk = CAM_PIN_XCLK;
    config.pin_pclk = CAM_PIN_PCLK;
    config.pin_vsync = CAM_PIN_VSYNC;
    config.pin_href = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn = CAM_PIN_PWDN;
    config.pin_reset = CAM_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565; // or PIXFORMAT_YUV422 for inference
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2; // Use double buffering in PSRAM

    if (CAM_PIN_XCLK < 0 || CAM_PIN_PCLK < 0 || CAM_PIN_VSYNC < 0 || CAM_PIN_HREF < 0) {
        ESP_LOGE(TAG, "Camera pins are not configured for this board.");
        return ESP_ERR_NOT_SUPPORTED;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
        return err;
    }
    ESP_LOGI(TAG, "Camera initialized");
    return ESP_OK;
}

void camera_task(void *pvParameters) {
    if (init_camera() != ESP_OK) {
        vTaskDelete(NULL);
        return;
    }

    while (1) {
        // camera_fb_t *fb = esp_camera_fb_get();
        // if (!fb) {
        //     ESP_LOGE(TAG, "Camera capture failed");
        //     vTaskDelay(pdMS_TO_TICKS(100));
        //     continue;
        // }
        // 
        // // Push frame pointer to inference queue or h264 encoder queue
        // 
        // esp_camera_fb_return(fb);

        // Dummy delay
        vTaskDelay(pdMS_TO_TICKS(33)); // ~30 fps
    }
}
