// AS5600 ESP-IDF example using the pinned vroom AS5600 HAL.
//
// Purpose:
//   - Exercise the real I2C path and the AS5600 HAL on an ESP32.
//   - Print raw angle, degrees, normalized angle, delta, and magnet status.
//
// This example does not program ZPOS/MPOS/MANG or burn OTP. It only calls Init
// and read/status APIs. The AS5600 HAL owns register semantics; this file owns
// ESP-IDF I2C transport and board configuration.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "as5600.hpp"

namespace {

constexpr char kTag[] = "as5600_example";
constexpr i2c_port_t kI2cPort = I2C_NUM_0;
constexpr uint8_t kAs5600Address = 0x36;
constexpr int kI2cTimeoutMs = 1000;

struct I2cContext {
    i2c_port_t port;
    TickType_t timeout_ticks;
};

I2cContext g_i2c_context{
    kI2cPort,
    pdMS_TO_TICKS(kI2cTimeoutMs),
};

AS5600_RET_TYPE esp_i2c_read(void* handle,
                             uint8_t address,
                             uint8_t reg,
                             uint8_t* buffer,
                             uint8_t length) {
    if (handle == nullptr || buffer == nullptr || length == 0) {
        return AS5600_RET_ERR;
    }

    auto* context = static_cast<I2cContext*>(handle);
    i2c_cmd_handle_t command = i2c_cmd_link_create();
    if (command == nullptr) {
        return AS5600_RET_ERR;
    }

    i2c_master_start(command);
    i2c_master_write_byte(command, static_cast<uint8_t>((address << 1) | I2C_MASTER_WRITE), true);
    i2c_master_write_byte(command, reg, true);
    i2c_master_start(command);
    i2c_master_write_byte(command, static_cast<uint8_t>((address << 1) | I2C_MASTER_READ), true);

    if (length > 1) {
        i2c_master_read(command, buffer, length - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(command, buffer + length - 1, I2C_MASTER_NACK);
    i2c_master_stop(command);

    const esp_err_t result = i2c_master_cmd_begin(context->port, command, context->timeout_ticks);
    i2c_cmd_link_delete(command);
    return result == ESP_OK ? AS5600_RET_OK : AS5600_RET_I2C_FAIL;
}

AS5600_RET_TYPE esp_i2c_write(void* handle,
                              uint8_t address,
                              uint8_t reg,
                              uint8_t* buffer,
                              uint8_t length) {
    if (handle == nullptr || buffer == nullptr || length == 0) {
        return AS5600_RET_ERR;
    }

    auto* context = static_cast<I2cContext*>(handle);
    i2c_cmd_handle_t command = i2c_cmd_link_create();
    if (command == nullptr) {
        return AS5600_RET_ERR;
    }

    i2c_master_start(command);
    i2c_master_write_byte(command, static_cast<uint8_t>((address << 1) | I2C_MASTER_WRITE), true);
    i2c_master_write_byte(command, reg, true);
    i2c_master_write(command, buffer, length, true);
    i2c_master_stop(command);

    const esp_err_t result = i2c_master_cmd_begin(context->port, command, context->timeout_ticks);
    i2c_cmd_link_delete(command);
    return result == ESP_OK ? AS5600_RET_OK : AS5600_RET_I2C_FAIL;
}

AS5600_RET_TYPE esp_delay_ms(void* /*handle*/, uint32_t delay_ms) {
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
    return AS5600_RET_OK;
}

void init_i2c() {
    i2c_config_t config{};
    config.mode = I2C_MODE_MASTER;
    config.sda_io_num = static_cast<gpio_num_t>(CONFIG_AS5600_SDA_GPIO);
    config.scl_io_num = static_cast<gpio_num_t>(CONFIG_AS5600_SCL_GPIO);
    config.sda_pullup_en = GPIO_PULLUP_ENABLE;
    config.scl_pullup_en = GPIO_PULLUP_ENABLE;
    config.master.clk_speed = CONFIG_AS5600_I2C_FREQ_HZ;

    ESP_ERROR_CHECK(i2c_param_config(kI2cPort, &config));
    ESP_ERROR_CHECK(i2c_driver_install(kI2cPort, I2C_MODE_MASTER, 0, 0, 0));
}

}  // namespace

extern "C" void app_main(void) {
    ESP_LOGI(kTag, "AS5600 ESP-IDF example");
    ESP_LOGI(kTag,
             "I2C port=%d SDA=%d SCL=%d frequency=%d Hz address=0x%02X",
             static_cast<int>(kI2cPort),
             CONFIG_AS5600_SDA_GPIO,
             CONFIG_AS5600_SCL_GPIO,
             CONFIG_AS5600_I2C_FREQ_HZ,
             kAs5600Address);

    init_i2c();

    AS5600 encoder(&g_i2c_context, esp_i2c_read, esp_i2c_write, esp_delay_ms);
    const AS5600_RET_TYPE init_result = encoder.Init();
    if (init_result != AS5600_RET_OK) {
        ESP_LOGE(kTag, "AS5600 Init failed, return code=%u", init_result);
        ESP_LOGE(kTag, "Check 3.3 V, GND, SDA/SCL, pull-ups, and the magnet");
        return;
    }

    ESP_LOGI(kTag, "AS5600 responded at 0x%02X", kAs5600Address);

    if (!encoder.MagnetDetected()) {
        ESP_LOGW(kTag, "No magnet detected");
    } else if (encoder.MagnetTooWeak()) {
        ESP_LOGW(kTag, "Magnet detected but too weak");
    } else if (encoder.MagnetTooStrong()) {
        ESP_LOGW(kTag, "Magnet detected but too strong");
    } else {
        ESP_LOGI(kTag, "Magnet strength is in range");
    }

    while (true) {
        uint16_t raw_angle = 0;
        const AS5600_RET_TYPE read_result = encoder.ReadRawAngle(&raw_angle);
        if (read_result != AS5600_RET_OK) {
            ESP_LOGE(kTag, "AS5600 angle read failed, return code=%u", read_result);
        } else {
            const float degrees = (static_cast<float>(raw_angle) * 360.0F) / 4096.0F;
            const float normalized = static_cast<float>(raw_angle) / 4096.0F;
            const int16_t delta = encoder.GetDelta();
            ESP_LOGI(kTag,
                     "raw=%4u deg=%7.2f norm=%.5f delta=%+5d magnet=%d weak=%d strong=%d",
                     raw_angle,
                     degrees,
                     normalized,
                     delta,
                     encoder.MagnetDetected(),
                     encoder.MagnetTooWeak(),
                     encoder.MagnetTooStrong());
        }

        vTaskDelay(pdMS_TO_TICKS(CONFIG_AS5600_SAMPLE_PERIOD_MS));
    }
}
