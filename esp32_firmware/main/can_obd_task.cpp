#include "can_obd_task.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "CAN_OBD_TASK";

#define TX_GPIO_NUM    21 // Example pins, must be changed for P4
#define RX_GPIO_NUM    22

esp_err_t init_can_obd() {
    twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT((gpio_num_t)TX_GPIO_NUM, (gpio_num_t)RX_GPIO_NUM, TWAI_MODE_NORMAL);
    twai_timing_config_t t_config = TWAI_TIMING_CONFIG_500KBITS();
    twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

    if (twai_driver_install(&g_config, &t_config, &f_config) == ESP_OK) {
        if (twai_start() == ESP_OK) {
            ESP_LOGI(TAG, "TWAI (CAN) driver installed and started.");
            return ESP_OK;
        }
    }
    ESP_LOGE(TAG, "Failed to initialize TWAI (CAN) driver.");
    return ESP_FAIL;
}

void can_obd_task(void *pvParameters) {
    // init_can_obd();
    ESP_LOGI(TAG, "CAN/OBD-II Task started");
    while (1) {
        // twai_message_t rx_msg;
        // if (twai_receive(&rx_msg, pdMS_TO_TICKS(100)) == ESP_OK) {
        //     // Parse OBD-II PID responses here
        // }

        // Periodically send OBD-II requests (e.g., speed, RPM)
        vTaskDelay(pdMS_TO_TICKS(100)); // 10Hz
    }
}
