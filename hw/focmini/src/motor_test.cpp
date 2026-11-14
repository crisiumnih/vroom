// Standalone DRI0058 sensorless, open-loop diagnostic. Build this source
// instead of main.cpp with FOCMINI_PROGRAM=motor_test. It never runs on boot;
// only an explicit RUN10 serial command requests a 10 s unloaded spin.

#include <Arduino.h>
#include <SimpleFOC.h>

#include "../shared/motor_test_command.h"

#include "driver/gpio.h"
#include "esp_intr_alloc.h"
#include "freertos/FreeRTOS.h"
#include "soc/gpio_reg.h"
#include "soc/soc.h"

#if !defined(CONFIG_IDF_TARGET_ESP32) || !CONFIG_IDF_TARGET_ESP32
#error "This diagnostic's direct GPIO cutoff is for original ESP32-WROOM only"
#endif

namespace {
constexpr int kPwmA = 25, kPwmB = 26, kPwmC = 27;
constexpr int kEnable = 32, kReset = 33, kFault = 34, kSleep = 14;
constexpr int kPolePairs = 7;
constexpr float kBusVoltageV = 9.0f;
constexpr float kDriverVoltageLimitV = 9.0f;
// Prior operator-reported baseline: 1 V phase amplitude, 1 rad/s, unloaded.
// These are not a current limit and are not validated under load.
constexpr float kPhaseVoltageV = 1.0f;
constexpr float kOpenLoopSpeedRadPerSec = 1.0f;
constexpr uint32_t kRunDurationMs = 10000;
constexpr float kPwmFrequencyHz = 20000.0f;

BLDCMotor motor(kPolePairs);
BLDCDriver3PWM driver(kPwmA, kPwmB, kPwmC);
portMUX_TYPE power_mux = portMUX_INITIALIZER_UNLOCKED;
volatile bool fault_latched = false;
bool driver_awake = false;
bool ready = false;
bool running = false;
bool fault_reported = false;
uint32_t run_started_ms = 0;
focmini::MotorTestParser command_parser;

void IRAM_ATTR cut_external_gate_from_isr() {
  REG_WRITE(GPIO_OUT1_W1TC_REG, 1U << (kEnable - 32));
  REG_WRITE(GPIO_OUT_W1TC_REG, 1U << kSleep);
}

void IRAM_ATTR on_fault(void *) {
  portENTER_CRITICAL_ISR(&power_mux);
  fault_latched = true;
  cut_external_gate_from_isr();
  portEXIT_CRITICAL_ISR(&power_mux);
}

void cut_power_stage() {
  digitalWrite(kEnable, LOW);
  digitalWrite(kSleep, LOW);
}

void latch_shutdown(const char *message) {
  fault_latched = true;
  running = false;
  ready = false;
  driver_awake = false;
  cut_power_stage();
  if (!fault_reported) {
    if (message != nullptr) Serial.println(message);
    fault_reported = true;
  }
}

bool enable_stage_if_healthy() {
  bool enabled = false;
  portENTER_CRITICAL(&power_mux);
  if (driver_awake && !fault_latched && digitalRead(kFault) == HIGH) {
    digitalWrite(kEnable, HIGH);
    enabled = true;
  } else {
    fault_latched = true;
    cut_external_gate_from_isr();
  }
  portEXIT_CRITICAL(&power_mux);
  return enabled;
}

bool wake_driver_and_check_fault() {
  // EN remains low through the wake/reset sequence and initial FAULT check.
  digitalWrite(kEnable, LOW);
  digitalWrite(kSleep, HIGH);
  delay(10);
  digitalWrite(kReset, LOW);
  delay(10);
  digitalWrite(kReset, HIGH);
  delay(10);

  const bool fault_high_before_isr = digitalRead(kFault) == HIGH;
  if (gpio_set_intr_type(static_cast<gpio_num_t>(kFault), GPIO_INTR_NEGEDGE) != ESP_OK ||
      gpio_install_isr_service(ESP_INTR_FLAG_IRAM) != ESP_OK ||
      gpio_isr_handler_add(static_cast<gpio_num_t>(kFault), on_fault, nullptr) != ESP_OK) {
    latch_shutdown("DISARMED: FAULT interrupt setup failed; stage remains off.");
    return false;
  }

  bool healthy = false;
  portENTER_CRITICAL(&power_mux);
  if (fault_high_before_isr && !fault_latched && digitalRead(kFault) == HIGH) {
    driver_awake = true;
    healthy = true;
  } else {
    fault_latched = true;
    cut_external_gate_from_isr();
  }
  portEXIT_CRITICAL(&power_mux);

  if (!healthy || fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("DISARMED: DRI0058 FAULT active after wake/reset; stage off.");
    return false;
  }
  return true;
}

void stop_run(const char *message) {
  running = false;
  motor.target = 0.0f;
  digitalWrite(kEnable, LOW);
  if (message != nullptr) Serial.println(message);
}

void start_run10() {
  if (!ready || fault_latched) {
    Serial.println("RUN10 rejected: setup is not READY or FAULT is latched.");
    return;
  }
  if (running) {
    Serial.println("RUN10 ignored: already running; send STOP to stop early.");
    return;
  }
  motor.target = kOpenLoopSpeedRadPerSec;
  if (!enable_stage_if_healthy()) {
    latch_shutdown("RUN10 rejected: DRI FAULT active; stage off.");
    return;
  }
  run_started_ms = millis();
  running = true;
  Serial.println("RUN10 started: 1 rad/s open loop for at most 10 s; STOP to abort.");
}

void read_serial_commands() {
  unsigned int bytes_read = 0;
  while (Serial.available() > 0 && bytes_read < 32) {
    const char ch = static_cast<char>(Serial.read());
    ++bytes_read;
    switch (command_parser.feed(ch)) {
      case focmini::MotorTestCommand::kRun10:
        start_run10();
        break;
      case focmini::MotorTestCommand::kStop:
        stop_run("STOP received: EN low.");
        break;
      case focmini::MotorTestCommand::kInvalid:
        Serial.println("Invalid command; use RUN10 or STOP.");
        break;
      case focmini::MotorTestCommand::kNone:
        break;
    }
  }
}
}  // namespace

void setup() {
  Serial.begin(115200);
  pinMode(kEnable, OUTPUT);
  digitalWrite(kEnable, LOW);
  pinMode(kReset, OUTPUT);
  pinMode(kSleep, OUTPUT);
  pinMode(kFault, INPUT);  // GPIO34 requires an external 10 kΩ pull-up to 3.3 V.
  cut_power_stage();
  digitalWrite(kReset, HIGH);

  if (!wake_driver_and_check_fault()) return;
  driver.voltage_power_supply = kBusVoltageV;
  driver.voltage_limit = kDriverVoltageLimitV;
  driver.pwm_frequency = kPwmFrequencyHz;
  if (!driver.init()) {
    latch_shutdown("DISARMED: PWM driver initialization failed; stage off.");
    return;
  }

  motor.linkDriver(&driver);  // No encoder or current-sense object in this file.
  motor.controller = MotionControlType::velocity_openloop;
  motor.torque_controller = TorqueControlType::voltage;
  motor.voltage_limit = kPhaseVoltageV;
  motor.velocity_limit = kOpenLoopSpeedRadPerSec;
  if (!motor.init()) {
    latch_shutdown("DISARMED: motor initialization failed; stage off.");
    return;
  }
  if (fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("DISARMED: DRI0058 FAULT during initialization; stage off.");
    return;
  }

  ready = true;
  Serial.println("READY DISARMED MOTOR TEST: sensors unused; type RUN10 or STOP.");
}

void loop() {
  if (fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("LATCHED: DRI FAULT; stage off. Inspect cause and reboot.");
    delay(100);
    return;
  }

  read_serial_commands();
  if (fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("LATCHED: DRI FAULT; stage off. Inspect cause and reboot.");
    return;
  }

  if (!running) return;
  if (millis() - run_started_ms >= kRunDurationMs) {
    stop_run("RUN10 complete: 10 s elapsed; EN low.");
    return;
  }

  motor.move(kOpenLoopSpeedRadPerSec);
  if (fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("LATCHED: DRI FAULT during RUN10; stage off.");
  }
}
