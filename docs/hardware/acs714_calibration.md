# ACS714 calibration procedure

Status: firmware and host fitting script implemented; no hardware measurements
have been performed yet.

This procedure calibrates the complete signal chain:

```text
ACS714 output -> 10 kΩ / 10 kΩ divider -> ESP32 ADC1 -> ESP-IDF calibration
```

The nominal ACS714 relationship at 5 V is:

```text
V_sensor = 2.5 V + 0.185 V/A × I
```

With equal divider resistors:

```text
V_adc = V_sensor / 2
V_adc = 1.25 V + 0.0925 V/A × I
```

The nominal value is only a starting point. The firmware measures the actual
ADC voltage and the host script fits the actual intercept and slope.

## Wiring

For the Pololu ACS714 ±5 A carrier:

```text
ACS714 VCC  -> regulated 5 V
ACS714 GND  -> ESP32 GND
ACS714 OUT  -> 10 kΩ -> ADC node
ADC node    -> 10 kΩ -> GND
ADC node    -> ESP32 ADC1 input
```

The sensor logic ground and ESP32 ground must be common. The current path is
the high-current path through the carrier; the OUT pin is the low-voltage
analog signal path. Use a fuse and a load rated for the planned current. Keep
the carrier current-path voltage below the limit specified for that carrier;
the 60 V capability of the bench supply is not a reason to apply 60 V to this
board.

Do not connect the ACS714 OUT pin directly to the ESP32. When the ACS714 is
powered from 5 V, its nominal output can exceed 3.3 V.

## Run the firmware

The project is `hw/esp32/acs714_calibration`. It uses ADC1, 12-bit conversion,
ADC attenuation suitable for the approximately 0.8–1.7 V divider output, and
the ESP-IDF line-fitting calibration scheme. The ADC channel is configurable
because ADC1 channel-to-GPIO mappings differ between ESP32 variants.

Use the ESP-IDF 5.5.x environment and run:

```bash
cd hw/esp32/acs714_calibration
idf.py --version
idf.py set-target <target>
idf.py menuconfig
idf.py build
idf.py flash monitor
```

The firmware prints records in this form:

```text
ZERO,voltage_mv,stddev_mv,min_mv,max_mv,samples
POINT,current_a,voltage_mv,stddev_mv,min_mv,max_mv,samples
```

The `ZERO` and `POINT` records are machine-readable. Other monitor text is
diagnostic and may be ignored by the fitting script.

The console is implemented with the ESP-IDF REPL. Use `help` to list commands;
the previous `fgets()` loop is intentionally not used because some console
configurations return immediately with no input and can starve the idle task.

For wiring debug, enter `m` to start continuous monitoring and `x` to stop it.
The output format is:

```text
ADC,timestamp_us,raw_adc,calibrated_adc_node_mv
```

The final field is the voltage at the ESP32 side of the divider. With no
current and an equal divider, it should be approximately 1250 mV. The raw
capture can be inspected directly or filtered with the host tools.

## Calibration sequence

1. Power the ACS714 and ESP32. Do not pass current through the ACS714.
2. Wait approximately 30 seconds for the sensor and supply to settle.
3. Send `z` in the monitor. Save the emitted `ZERO` record.
4. Connect a controlled DC load through the ACS714 current path.
5. Apply a known positive current, for example 0.5 A, 1 A, 2 A, and 3 A.
6. Verify each current with a calibrated meter or shunt. The power-supply
   current display is not the calibration reference unless independently known
   to be accurate.
7. At each stable current, send `p <current_a>`, for example `p 1.000`.
8. Reverse the current path only if a negative-current calibration is needed;
   do not reverse the bench supply unless the load is designed for it.
9. Repeat the zero measurement after the current tests.
10. Save the complete monitor output as the raw capture.

At least three distinct current values are required. A positive and negative
set of points is preferred for a bidirectional sensor. Use a current span of
at least 1 A for the initial fit, and stay below the ACS714's ±5 A range.

Fit the output with:

```bash
python tools/hw_bench/fit_acs714.py capture.txt \
  --output results/hardware/acs714_calibration/calibration.json
```

The fitted model is:

```text
V_adc_mV = intercept_mV + slope_mV_per_A × current_A
current_A = (V_adc_mV - intercept_mV) / slope_mV_per_A
```

The fitted intercept is the calibrated ADC voltage at zero current. The
startup firmware should recapture that zero offset whenever the motor is
disabled and current is guaranteed to be zero. The fitted slope is normally
stored and reused.

## Acceptance checks

Review the generated JSON and confirm:

- `status` is `complete`.
- The fitted slope is positive and reasonably close to 92.5 mV/A.
- Zero-current voltage is near half of the 5 V sensor output after division.
- The residual error is small compared with the sensor's specified accuracy.
- No ADC sample is clipped.
- Reversing current changes the measured sign.

The initial result is a room-temperature calibration. Repeat the gain
calibration after changing the sensor, divider, ADC configuration, sensor
supply, or wiring. A temperature study is required before claiming accuracy
across the full operating range.

## Evidence record

Keep these files together under an ignored result directory:

```text
raw_monitor.txt
calibration.json
firmware_commit.txt
idf_version.txt
sdkconfig
board_and_wiring.md
```

The current repository contains the implementation and procedure only. No
hardware result is claimed until the firmware has been run on the actual board.
