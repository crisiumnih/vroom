// DFRobot DRI0058 SimpleFOCmini + original ESP32-WROOM bring-up target.
//
// Purpose: primary AS5600/ACS724 closed-loop FOC firmware for the ESP32-WROOM.
// Select src/motor_test.cpp separately via FOCMINI_PROGRAM=motor_test for an
// unloaded sensorless diagnostic; this is not an RL deployment target.
// Units: targets and limits are mechanical rad/s; voltages are V. The 1.50 V
// motor amplitude is a phase-vector command, while the driver range is the
// full 9 V bus. Centered sine modulation needs 2*amplitude <= driver range.
// Assumptions: DRI0058, measured 9 V supply, verified 7 pole pairs, original
// ESP32-WROOM GPIO map below, unloaded motor, and two ACS724-05AB inline current
// sensors with 10k/18k output dividers. AS5600 mode uses FOC current control;
// current limit is 0.5 A. Repro:
// `pio run -d hw/focmini -e esp32dev_open_loop` (AS5600: esp32dev_as5600).

#include <Arduino.h>
#include <SimpleFOC.h>

#include <cmath>

#include "../shared/focmini_control.h"
#include "driver/gpio.h"
#include "esp_intr_alloc.h"
#include "freertos/FreeRTOS.h"
#include "soc/gpio_reg.h"
#include "soc/soc.h"

#if !defined(CONFIG_IDF_TARGET_ESP32) || !CONFIG_IDF_TARGET_ESP32
#error "FOCMini direct GPIO-bank cutoff is for original ESP32-WROOM only"
#endif

#ifndef FOCMINI_USE_AS5600
#define FOCMINI_USE_AS5600 0
#endif
#ifndef FOCMINI_ACS724_MONITOR_ONLY
#define FOCMINI_ACS724_MONITOR_ONLY 0
#endif
#ifndef FOCMINI_USE_ACS724
#define FOCMINI_USE_ACS724 0
#endif
#if FOCMINI_USE_AS5600 && !FOCMINI_USE_ACS724
#error "AS5600 FOC-current mode requires the two ACS724 phase-current channels"
#endif
#if FOCMINI_USE_AS5600
#include <Wire.h>
#endif

extern BLDCMotor motor;
#if FOCMINI_USE_ACS724
extern InlineCurrentSense current_sense;
#endif
#if FOCMINI_USE_AS5600
extern MagneticSensorI2C sensor;
#endif

namespace focmini {
// DRI0058 IN1/IN2/IN3, external EN, RESET, active-low FAULT, SLEEP.
constexpr int kPwmA = 25, kPwmB = 26, kPwmC = 27, kEnable = 32;
constexpr int kReset = 33, kFault = 34, kSleep = 14;
// ADC1-only inputs on the original ESP32-WROOM. Each sensor OUT needs its own
// 10k (OUT-to-ADC) / 18k (ADC-to-GND) divider; never feed 5 V directly to GPIO.
constexpr int kPhaseCurrentPinA = 35, kPhaseCurrentPinB = 36;
constexpr float kCurrentDividerTopOhms = 10000.0f;
constexpr float kCurrentDividerBottomOhms = 18000.0f;
constexpr float kCurrentDividerRatio = kCurrentDividerBottomOhms /
                                      (kCurrentDividerTopOhms + kCurrentDividerBottomOhms);
constexpr float kAcs72405AbSensitivityMilliVPerA = 400.0f;
constexpr float kAcs72405AbEsp32SensitivityMilliVPerA =
    kAcs72405AbSensitivityMilliVPerA * kCurrentDividerRatio;
constexpr unsigned int kCurrentSampleCount = 32;
constexpr float kMotorCurrentLimitA = 0.5f;
constexpr uint32_t kTelemetryIntervalMs = 250;
constexpr float kPwmFrequencyHz = 20000.0f;
#if FOCMINI_USE_AS5600
constexpr int kI2cSda = 21, kI2cScl = 22;
#endif
constexpr float kAlignmentVoltageV = 0.30f;
constexpr float kBusVoltageV = 9.0f;  // Measured by the operator for this setup.
constexpr float kDriverVoltageRangeV = kBusVoltageV;
constexpr float kMotorVoltageAmplitudeV = 1.50f;
constexpr float kVelocityLimitRadPerSec = 5.0f;
constexpr int kMotorPolePairs = 7;  // Verified; this is pole pairs, not poles.
constexpr int kSerialBytesPerLoop = 64;

volatile bool fault_latched = false;
portMUX_TYPE power_mux = portMUX_INITIALIZER_UNLOCKED;
CommandParser command_parser;
MotionInterlock motion;
bool driver_awake = false;
bool foc_initialized = false;
bool fault_reported = false;
uint32_t last_telemetry_ms = 0;
float current_zero_a_mV = 0.0f;
float current_zero_b_mV = 0.0f;

float average_adc_millivolts(int pin) {
  uint32_t total_mV = 0;
  for (unsigned int i = 0; i < kCurrentSampleCount; ++i) {
    total_mV += analogReadMilliVolts(pin);
    delay(2);
  }
  return static_cast<float>(total_mV) / kCurrentSampleCount;
}

void print_current_sensor_monitor() {
  const float phase_a_mV = average_adc_millivolts(kPhaseCurrentPinA);
  const float phase_b_mV = average_adc_millivolts(kPhaseCurrentPinB);
  const float phase_a_A = (phase_a_mV - current_zero_a_mV) /
                          kAcs72405AbEsp32SensitivityMilliVPerA;
  const float phase_b_A = (phase_b_mV - current_zero_b_mV) /
                          kAcs72405AbEsp32SensitivityMilliVPerA;
  Serial.print("ACS724,phaseA_adc_mV=");
  Serial.print(phase_a_mV, 1);
  Serial.print(",phaseA_current_A=");
  Serial.print(phase_a_A, 3);
  Serial.print(",phaseB_adc_mV=");
  Serial.print(phase_b_mV, 1);
  Serial.print(",phaseB_current_A=");
  Serial.println(phase_b_A, 3);
}

// ESP32-WROOM register writes are IRAM-safe and do not depend on GPIO APIs that
// may be placed in flash. EN is GPIO32 (bank 1); SLEEP is GPIO14 (bank 0).
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

// In healthy idle the DRV8313 stays awake so nFAULT is valid; EN alone holds
// the motor outputs disabled. The two-pin cutoff is reserved for a latched
// fault or failed initialization.
void disable_motor_output() {
  digitalWrite(kEnable, LOW);
}

void cut_power_stage() {
  digitalWrite(kEnable, LOW);
  digitalWrite(kSleep, LOW);
}

bool enable_power_stage_if_healthy() {
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

bool configuration_is_safe() {
  return kMotorPolePairs > 0 && kBusVoltageV >= 8.0f &&
         kBusVoltageV <= 30.0f && kDriverVoltageRangeV > 0.0f &&
         kDriverVoltageRangeV <= kBusVoltageV &&
         kMotorVoltageAmplitudeV > 0.0f &&
         2.0f * kMotorVoltageAmplitudeV <= kDriverVoltageRangeV &&
         kAlignmentVoltageV > 0.0f &&
         kAlignmentVoltageV <= kMotorVoltageAmplitudeV &&
         2.0f * kAlignmentVoltageV <= kDriverVoltageRangeV &&
         kVelocityLimitRadPerSec > 0.0f &&
         kVelocityLimitRadPerSec <= 5.0f;
}

void latch_shutdown(const char *message) {
  fault_latched = true;
  motion.latch_fault();
  driver_awake = false;
  cut_power_stage();
  if (!fault_reported) {
    if (message != nullptr) Serial.println(message);
    fault_reported = true;
  }
}

void service_fault() {
  if (fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("FAULT/STARTUP FAILURE LATCHED: stage off; reboot after inspection.");
  }
}

void stop_motion() {
  motion.stop();
  motor.target = 0.0f;
  disable_motor_output();
}

bool wake_driver_and_check_fault() {
  // DRV8313 nSLEEP needs settling time before the bridge is operational. The
  // 10 ms wait is conservative relative to the ~1 ms datasheet wake interval.
  // Keep external EN low throughout wake/reset and fault validation.
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
    latch_shutdown("DISARMED: IRAM FAULT interrupt setup failed.");
    return false;
  }

  bool ready = false;
  portENTER_CRITICAL(&power_mux);
  if (fault_high_before_isr && !fault_latched && digitalRead(kFault) == HIGH) {
    driver_awake = true;
    ready = true;
  } else {
    fault_latched = true;
    cut_external_gate_from_isr();
  }
  portEXIT_CRITICAL(&power_mux);
  if (!ready || fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("DISARMED: DRI0058 FAULT is active after wake/reset.");
    return false;
  }
  return true;
}

void apply_command(const Command &command);

void print_telemetry() {
  const uint32_t now = millis();
  if (now - last_telemetry_ms < kTelemetryIntervalMs) return;
  last_telemetry_ms = now;

#if FOCMINI_USE_AS5600
  if (!motion.active()) sensor.update();
  Serial.print("TEL,angle_rad=");
  Serial.print(sensor.getMechanicalAngle(), 4);
  Serial.print(",position_rad=");
  Serial.print(sensor.getAngle(), 4);
  Serial.print(",measured_speed_rad_s=");
  Serial.print(sensor.getVelocity(), 3);
#else
  Serial.print("TEL,angle_rad=NA,position_rad=NA,measured_speed_rad_s=NA");
#endif

#if FOCMINI_USE_ACS724
  PhaseCurrent_s phase_current = current_sense.getPhaseCurrents();
  phase_current.c = -(phase_current.a + phase_current.b);
  Serial.print(",phaseA_current_A=");
  Serial.print(phase_current.a, 3);
  Serial.print(",phaseB_current_A=");
  Serial.print(phase_current.b, 3);
  Serial.print(",phaseC_current_A_inferred=");
  Serial.print(phase_current.c, 3);
#else
  Serial.print(",phase_currents_A=NA");
#endif
  Serial.println();
}
}  // namespace focmini

using namespace focmini;
BLDCMotor motor(kMotorPolePairs);
#if FOCMINI_USE_AS5600
MagneticSensorI2C sensor = MagneticSensorI2C(AS5600_I2C);
#endif
// Deliberately omit SimpleFOC's EN pin. Only guarded application code may
// enable the external DRI0058 gate; motor.init() cannot re-enable it.
BLDCDriver3PWM driver(kPwmA, kPwmB, kPwmC);
#if FOCMINI_USE_ACS724
InlineCurrentSense current_sense(
    kAcs72405AbEsp32SensitivityMilliVPerA,
    kPhaseCurrentPinA, kPhaseCurrentPinB);
#endif

namespace focmini {
void apply_command(const Command &command) {
  switch (command.type) {
    case CommandType::kNone:
      return;
    case CommandType::kInvalid:
      Serial.println("Rejected line: use ARM, T<rad/s> (<= 5), STOP, or T0.");
      return;
    case CommandType::kArm:
      if (!motion.arm()) {
        Serial.println("ARM rejected: setup is not ready or fault is latched.");
        return;
      }
      // Re-ARM is a deliberate reset to zero: drop any old open-loop field
      // before returning to the waiting-for-target state.
      motor.target = 0.0f;
      disable_motor_output();
#if FOCMINI_USE_AS5600
      if (!foc_initialized) {
        if (!enable_power_stage_if_healthy()) {
          latch_shutdown("ARM failed: FAULT active during calibration.");
          return;
        }
        // Alignment is intentionally energized only after explicit ARM. The
        // IRAM fault handler cuts EN/SLEEP even while initFOC() blocks.
        if (!motor.initFOC() || fault_latched || digitalRead(kFault) == LOW) {
          latch_shutdown("AS5600 alignment failed or FAULT occurred; reboot required.");
          return;
        }
        foc_initialized = true;
        disable_motor_output();
      }
      Serial.println("ARMED AS5600 mode; send T<rad/s> (<= 5), or STOP/T0.");
#else
      // Open-loop ARM only unlocks commands; it never energizes a zero field.
      Serial.println("ARMED OPEN LOOP; send T1 for the first unloaded spin, or STOP/T0.");
#endif
      return;
    case CommandType::kStop:
      stop_motion();
      Serial.println("STOPPED: target zero, EN low; send ARM before another target.");
      return;
    case CommandType::kTarget:
      if (command.target_rad_per_sec == 0.0f) {
        stop_motion();
        Serial.println("STOPPED: T0 disarmed and set EN low.");
        return;
      }
      if (!std::isfinite(command.target_rad_per_sec) ||
          std::fabs(command.target_rad_per_sec) > kVelocityLimitRadPerSec) {
        Serial.println("Rejected target: finite and within +/-5 rad/s required.");
        return;
      }
      if (!motion.set_target(command.target_rad_per_sec,
                             kVelocityLimitRadPerSec)) {
        Serial.println("Target rejected: send ARM first; invalid lines do not change target.");
        return;
      }
      motor.target = motion.target_rad_per_sec();
      if (!enable_power_stage_if_healthy()) {
        latch_shutdown("Target rejected: FAULT active; stage remains off.");
        return;
      }
      Serial.println("Target accepted in rad/s.");
      return;
  }
}
}  // namespace focmini

void setup() {
  Serial.begin(115200);
#if FOCMINI_ACS724_MONITOR_ONLY
  // Standalone sensor check: do not wake the DRI, initialize PWM, or allow
  // motor output. Calibrate zero-current offsets with phase current absent.
  pinMode(kEnable, OUTPUT);
  digitalWrite(kEnable, LOW);
  pinMode(kSleep, OUTPUT);
  digitalWrite(kSleep, LOW);
  pinMode(kReset, OUTPUT);
  digitalWrite(kReset, LOW);
  analogReadResolution(12);
  analogSetPinAttenuation(kPhaseCurrentPinA, ADC_11db);
  analogSetPinAttenuation(kPhaseCurrentPinB, ADC_11db);
  delay(100);
  current_zero_a_mV = average_adc_millivolts(kPhaseCurrentPinA);
  current_zero_b_mV = average_adc_millivolts(kPhaseCurrentPinB);
  Serial.println("ACS724_MONITOR_ONLY: motor output disabled; zero offsets captured.");
  Serial.print("ZERO,phaseA_adc_mV=");
  Serial.print(current_zero_a_mV, 1);
  Serial.print(",phaseB_adc_mV=");
  Serial.println(current_zero_b_mV, 1);
  return;
#endif
  // Arduino-ESP32 requires pinMode before digitalWrite. Configure EN first
  // and hold the output stage off before waking the driver.
  pinMode(kEnable, OUTPUT);
  digitalWrite(kEnable, LOW);
  pinMode(kReset, OUTPUT);
  pinMode(kSleep, OUTPUT);
  pinMode(kFault, INPUT);  // GPIO34 requires the external FAULT pull-up.
  cut_power_stage();
  digitalWrite(kReset, HIGH);
  if (!wake_driver_and_check_fault()) return;

  if (!configuration_is_safe()) {
    Serial.println("DISARMED: invalid pole-pair, bus, or voltage-range settings.");
    return;
  }
  driver.voltage_power_supply = kBusVoltageV;
  driver.voltage_limit = kDriverVoltageRangeV;
  driver.pwm_frequency = kPwmFrequencyHz;
#if FOCMINI_USE_AS5600
  Wire.begin(kI2cSda, kI2cScl);
  sensor.init(&Wire);
  motor.linkSensor(&sensor);
#endif
  if (!driver.init()) {
    latch_shutdown("DISARMED: PWM driver initialization failed.");
    return;
  }
  motor.linkDriver(&driver);
#if FOCMINI_USE_ACS724
  current_sense.linkDriver(&driver);
  if (!current_sense.init()) {
    latch_shutdown("DISARMED: ACS724 current-sense initialization failed.");
    return;
  }
  Serial.println("CURRENT_SENSE_READY: two ACS724-05AB channels calibrated.");
  motor.linkCurrentSense(&current_sense);
#endif
#if FOCMINI_USE_AS5600
  motor.controller = MotionControlType::velocity;
  motor.torque_controller = TorqueControlType::foc_current;
  motor.current_limit = kMotorCurrentLimitA;
#else
  motor.controller = MotionControlType::velocity_openloop;
  motor.torque_controller = TorqueControlType::voltage;
#endif
  motor.voltage_limit = kMotorVoltageAmplitudeV;
  motor.velocity_limit = kVelocityLimitRadPerSec;
#if FOCMINI_USE_AS5600
  motor.voltage_sensor_align = kAlignmentVoltageV;
#endif
  if (!motor.init()) {
    latch_shutdown("DISARMED: motor initialization failed.");
    return;
  }
  if (fault_latched || digitalRead(kFault) == LOW) {
    latch_shutdown("DISARMED: FAULT occurred during initialization.");
    return;
  }
  motion.mark_ready();
#if FOCMINI_USE_AS5600
  Serial.println("READY DISARMED AS5600: send ARM to calibrate; then T<rad/s>.");
#else
  Serial.println("READY DISARMED OPEN LOOP: send ARM, then T1; STOP or T0 cuts power.");
#endif
}

void loop() {
#if FOCMINI_ACS724_MONITOR_ONLY
  print_current_sensor_monitor();
  delay(250);
#else
  service_fault();
  if (motion.fault_latched()) {
    if (!fault_reported) {
      cut_power_stage();
      Serial.println("LATCHED: power cycle after checking the driver and wiring.");
      fault_reported = true;
    }
    delay(100);
    return;
  }

  int serial_bytes = 0;
  while (Serial.available() > 0 && serial_bytes < kSerialBytesPerLoop) {
    const Command command = command_parser.feed(static_cast<char>(Serial.read()));
    apply_command(command);
    ++serial_bytes;
    if (fault_latched) break;
  }
  service_fault();
  if (motion.fault_latched()) return;

  if (motion.active()) {
    motor.loopFOC();
    if (fault_latched || digitalRead(kFault) == LOW) {
      service_fault();
      return;
    }
    motor.move(motion.target_rad_per_sec());
  }
  print_telemetry();
#endif
}
