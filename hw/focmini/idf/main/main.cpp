// DFRobot DRI0058 + original ESP32-WROOM native ESP-IDF bring-up application.
//
// Purpose: guarded open-loop first spin and optional AS5600 closed-loop mode.
// Units: menu voltages are mV converted to V; targets and velocity limits are
// mechanical rad/s. The motor amplitude is distinct from the full bus-voltage
// driver range; centered sine modulation requires 2*amplitude <= driver range.
// Assumptions: DRI0058, measured supply and verified pole-pairs in menuconfig,
// original ESP32-WROOM pin map, unloaded motor, and no current sensing. Voltage
// mode cannot enforce the driver's 2.5 A/channel rating. Build only: `pio run
// -d hw/focmini/idf -e esp32dev_idf`; native Kconfig defaults remain disarmed.

#include <cmath>

#include "driver/gpio.h"
#include "esp_attr.h"
#include "esp_err.h"
#include "esp_intr_alloc.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_simplefoc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/gpio_reg.h"
#include "soc/soc.h"

#include "../../shared/focmini_control.h"

#if !defined(CONFIG_IDF_TARGET_ESP32) || !CONFIG_IDF_TARGET_ESP32
#error "FOCMini direct GPIO-bank cutoff is for original ESP32-WROOM only"
#endif

namespace {
constexpr gpio_num_t kPwmA = GPIO_NUM_25;
constexpr gpio_num_t kPwmB = GPIO_NUM_26;
constexpr gpio_num_t kPwmC = GPIO_NUM_27;
constexpr gpio_num_t kEnable = GPIO_NUM_32;
constexpr gpio_num_t kReset = GPIO_NUM_33;
constexpr gpio_num_t kFault = GPIO_NUM_34;
constexpr gpio_num_t kSleep = GPIO_NUM_14;
constexpr gpio_num_t kI2cScl = GPIO_NUM_22;
constexpr gpio_num_t kI2cSda = GPIO_NUM_21;
constexpr float kPwmFrequencyHz = 20000.0f;
constexpr float kAlignmentVoltageV = 0.30f;
constexpr char kTag[] = "focmini";
constexpr int kSerialBytesPerLoop = 64;

constexpr float kBusVoltageV = CONFIG_FOCMINI_BUS_MV / 1000.0f;
constexpr float kDriverVoltageRangeV = kBusVoltageV;
constexpr float kMotorVoltageAmplitudeV =
    CONFIG_FOCMINI_VOLTAGE_LIMIT_MV / 1000.0f;
constexpr float kVelocityLimitRadPerSec =
    CONFIG_FOCMINI_VELOCITY_LIMIT_MRAD_S / 1000.0f;

volatile bool fault_latched = false;
portMUX_TYPE power_mux = portMUX_INITIALIZER_UNLOCKED;
focmini::CommandParser command_parser;
focmini::MotionInterlock motion;
bool driver_awake = false;
bool foc_initialized = false;
bool fault_reported = false;

BLDCMotor motor(CONFIG_FOCMINI_POLE_PAIRS);
BLDCDriver3PWM driver(kPwmA, kPwmB, kPwmC);
#if CONFIG_FOCMINI_USE_AS5600
AS5600 sensor(I2C_NUM_0, kI2cScl, kI2cSda);
#endif

// Original ESP32-WROOM: EN GPIO32 is bank 1; SLEEP GPIO14 is bank 0. Direct
// register writes keep this cutoff callable while flash cache is unavailable.
void IRAM_ATTR cut_external_gate_from_isr() {
    REG_WRITE(GPIO_OUT1_W1TC_REG, 1U << (kEnable - GPIO_NUM_32));
    REG_WRITE(GPIO_OUT_W1TC_REG, 1U << kSleep);
}

void IRAM_ATTR on_fault(void *) {
    portENTER_CRITICAL_ISR(&power_mux);
    fault_latched = true;
    cut_external_gate_from_isr();
    portEXIT_CRITICAL_ISR(&power_mux);
}

// Healthy idle leaves DRV8313 awake so its FAULT output can be sampled; EN
// alone keeps all motor outputs disabled. Only latched shutdown cuts SLEEP.
void disable_motor_output() {
  gpio_set_level(kEnable, 0);
}

void cut_power_stage() {
    gpio_set_level(kEnable, 0);
    gpio_set_level(kSleep, 0);
}

bool enable_power_stage_if_healthy() {
    bool enabled = false;
    portENTER_CRITICAL(&power_mux);
    if (driver_awake && !fault_latched && gpio_get_level(kFault) == 1) {
        gpio_set_level(kEnable, 1);
        enabled = true;
    } else {
        fault_latched = true;
        cut_external_gate_from_isr();
    }
    portEXIT_CRITICAL(&power_mux);
    return enabled;
}

bool safe_configuration() {
    return CONFIG_FOCMINI_POLE_PAIRS > 0 && std::isfinite(kBusVoltageV) &&
           kBusVoltageV >= 8.0f &&
           kBusVoltageV <= 30.0f && kDriverVoltageRangeV > 0.0f &&
           kDriverVoltageRangeV <= kBusVoltageV &&
           std::isfinite(kMotorVoltageAmplitudeV) &&
           kMotorVoltageAmplitudeV > 0.0f &&
           2.0f * kMotorVoltageAmplitudeV <= kDriverVoltageRangeV &&
           kAlignmentVoltageV > 0.0f &&
           kAlignmentVoltageV <= kMotorVoltageAmplitudeV &&
           2.0f * kAlignmentVoltageV <= kDriverVoltageRangeV &&
           std::isfinite(kVelocityLimitRadPerSec) &&
           kVelocityLimitRadPerSec > 0.0f &&
           kVelocityLimitRadPerSec <= 5.0f;
}

void latch_shutdown(const char *message) {
    fault_latched = true;
    motion.latch_fault();
    driver_awake = false;
    cut_power_stage();
    if (!fault_reported) {
        if (message != nullptr) ESP_LOGE(kTag, "%s", message);
        fault_reported = true;
    }
}

void service_fault() {
    if (fault_latched || (driver_awake && gpio_get_level(kFault) == 0)) {
        latch_shutdown("FAULT or startup failure latched; stage disabled until reboot");
    }
}

void stop_motion() {
    motion.stop();
    motor.target = 0.0f;
    disable_motor_output();
}

void apply_command(const focmini::Command &command) {
    using focmini::CommandType;
    switch (command.type) {
        case CommandType::kNone:
            return;
        case CommandType::kInvalid:
            ESP_LOGW(kTag, "Rejected line: use ARM, T<rad/s> (limit %.3f), STOP, or T0",
                     kVelocityLimitRadPerSec);
            return;
        case CommandType::kArm:
            if (!motion.arm()) {
                ESP_LOGW(kTag, "ARM rejected: setup is not ready or fault is latched");
                return;
            }
            // Re-ARM drops a prior target/field and waits for an explicit target.
            motor.target = 0.0f;
            disable_motor_output();
#if CONFIG_FOCMINI_USE_AS5600
            if (!foc_initialized) {
                if (!enable_power_stage_if_healthy()) {
                    latch_shutdown("ARM failed: FAULT active during AS5600 calibration");
                    return;
                }
                // Calibration is energized only after explicit ARM. The ISR
                // cuts both external gates even if initFOC() blocks.
                if (!motor.initFOC() || fault_latched ||
                    gpio_get_level(kFault) == 0) {
                    latch_shutdown("AS5600 alignment failed or FAULT occurred; reboot required");
                    return;
                }
                foc_initialized = true;
                disable_motor_output();
            }
            ESP_LOGI(kTag, "AS5600 calibrated and armed; T limit %.3f rad/s, or STOP/T0",
                     kVelocityLimitRadPerSec);
#else
            ESP_LOGI(kTag, "Open-loop armed with stage off; send T1 (limit %.3f rad/s)",
                     kVelocityLimitRadPerSec);
#endif
            return;
        case CommandType::kStop:
            stop_motion();
            ESP_LOGI(kTag, "Stopped: target zero, EN low; send ARM before another target");
            return;
        case CommandType::kTarget:
            if (command.target_rad_per_sec == 0.0f) {
                stop_motion();
                ESP_LOGI(kTag, "T0 stopped and disarmed the power stage");
                return;
            }
            if (!std::isfinite(command.target_rad_per_sec) ||
                std::fabs(command.target_rad_per_sec) > kVelocityLimitRadPerSec) {
                ESP_LOGW(kTag, "Rejected target: finite and within %.3f rad/s required",
                         kVelocityLimitRadPerSec);
                return;
            }
            if (!motion.set_target(command.target_rad_per_sec,
                                   kVelocityLimitRadPerSec)) {
                ESP_LOGW(kTag, "Target rejected: send ARM first; invalid lines preserve the target");
                return;
            }
            motor.target = motion.target_rad_per_sec();
            if (!enable_power_stage_if_healthy()) {
                latch_shutdown("Target rejected: FAULT active; stage remains off");
                return;
            }
            ESP_LOGI(kTag, "Target accepted in mechanical rad/s");
            return;
    }
}

bool initialize_safe_gpio() {
    gpio_config_t outputs{};
    outputs.pin_bit_mask = (1ULL << kEnable) | (1ULL << kReset) | (1ULL << kSleep);
    outputs.mode = GPIO_MODE_OUTPUT;
    outputs.pull_up_en = GPIO_PULLUP_DISABLE;
    outputs.pull_down_en = GPIO_PULLDOWN_DISABLE;
    outputs.intr_type = GPIO_INTR_DISABLE;
    if (gpio_config(&outputs) != ESP_OK) return false;
    cut_power_stage();
    if (gpio_set_level(kReset, 1) != ESP_OK) return false;  // RESET active low.

    gpio_config_t input{};
    input.pin_bit_mask = 1ULL << kFault;
    input.mode = GPIO_MODE_INPUT;
    input.pull_up_en = GPIO_PULLUP_DISABLE;  // External 10 kOhm pull-up required.
    input.pull_down_en = GPIO_PULLDOWN_DISABLE;
    input.intr_type = GPIO_INTR_DISABLE;
    if (gpio_config(&input) != ESP_OK) return false;
    // TI specifies approximately 1 ms from nSLEEP rising before the DRV8313 is
    // operational. Wait 10 ms with EN low, pulse RESET only during boot, then
    // settle another 10 ms before interpreting the FAULT pin.
    if (gpio_set_level(kSleep, 1) != ESP_OK) return false;
    esp_rom_delay_us(10000);
    if (gpio_set_level(kReset, 0) != ESP_OK) return false;
    esp_rom_delay_us(10000);
    if (gpio_set_level(kReset, 1) != ESP_OK) return false;
    esp_rom_delay_us(10000);

    const bool fault_high_before_isr = gpio_get_level(kFault) == 1;
    if (gpio_set_intr_type(kFault, GPIO_INTR_NEGEDGE) != ESP_OK) return false;
    if (gpio_install_isr_service(ESP_INTR_FLAG_IRAM) != ESP_OK) return false;
    if (gpio_isr_handler_add(kFault, on_fault, nullptr) != ESP_OK) return false;

    bool ready = false;
    portENTER_CRITICAL(&power_mux);
    if (fault_high_before_isr && !fault_latched && gpio_get_level(kFault) == 1) {
        driver_awake = true;
        ready = true;
    } else {
        fault_latched = true;
        cut_external_gate_from_isr();
    }
    portEXIT_CRITICAL(&power_mux);
    return ready && !fault_latched && gpio_get_level(kFault) == 1;
}
}  // namespace

extern "C" void app_main(void) {
    if (!initialize_safe_gpio()) {
        latch_shutdown("GPIO/IRAM FAULT interrupt setup failed; stage left disabled");
        return;
    }
    Serial.begin(115200);

    if (!safe_configuration()) {
        ESP_LOGW(kTag, "DISARMED: configure verified pole pairs and measured bus in menuconfig");
        return;
    }
    driver.voltage_power_supply = kBusVoltageV;
    driver.voltage_limit = kDriverVoltageRangeV;
    driver.pwm_frequency = kPwmFrequencyHz;
#if CONFIG_FOCMINI_USE_AS5600
    sensor.init();
    motor.linkSensor(&sensor);
    motor.controller = MotionControlType::velocity;
    motor.voltage_sensor_align = kAlignmentVoltageV;
    motor.PID_velocity.P = 0.2f;
    motor.PID_velocity.I = 2.0f;
    motor.LPF_velocity.Tf = 0.05f;
#else
    motor.controller = MotionControlType::velocity_openloop;
#endif

#if CONFIG_SOC_MCPWM_SUPPORTED
    const bool driver_ready = driver.init(0);
#else
    const bool driver_ready = driver.init({kPwmA, kPwmB, kPwmC});
#endif
    if (!driver_ready) {
        latch_shutdown("PWM driver initialization failed; stage left disabled");
        return;
    }
    motor.linkDriver(&driver);
    motor.voltage_limit = kMotorVoltageAmplitudeV;
    motor.velocity_limit = kVelocityLimitRadPerSec;
    if (!motor.init()) {
        latch_shutdown("Motor initialization failed; stage left disabled");
        return;
    }
    if (fault_latched || gpio_get_level(kFault) == 0) {
        latch_shutdown("FAULT occurred during motor initialization");
        return;
    }
    motion.mark_ready();
#if CONFIG_FOCMINI_USE_AS5600
    ESP_LOGI(kTag, "READY DISARMED AS5600; ARM for alignment; velocity limit %.3f rad/s",
             kVelocityLimitRadPerSec);
#else
    ESP_LOGI(kTag, "READY DISARMED OPEN LOOP; ARM then T1; limit %.3f rad/s; STOP/T0 cuts power",
             kVelocityLimitRadPerSec);
#endif

    while (true) {
        service_fault();
        if (motion.fault_latched()) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        int serial_bytes = 0;
        while (Serial.available() > 0 && serial_bytes < kSerialBytesPerLoop) {
            const focmini::Command command =
                command_parser.feed(static_cast<char>(Serial.read()));
            apply_command(command);
            ++serial_bytes;
            if (fault_latched) break;
        }
        service_fault();
        if (!motion.fault_latched() && motion.active()) {
            motor.loopFOC();
            if (!fault_latched && gpio_get_level(kFault) == 1) {
                motor.move(motion.target_rad_per_sec());
            } else {
                service_fault();
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}
