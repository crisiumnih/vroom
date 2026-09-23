# AS5600 ESP32 bring-up

Status: the ESP-IDF transport adapter and example are implemented and build
tested with ESP-IDF 5.5.4 for the `esp32` target. No board-level readings have
been recorded yet.

The project uses the pinned `third_party/as5600_driver` submodule. Its
hardware-independent callbacks are adapted to the ESP-IDF legacy I2C master
driver in `hw/esp32/as5600_example/main/main.cpp`.

## Signal path and assumptions

The AS5600 uses I2C address `0x36`. The example defaults to:

```text
SDA = GPIO21
SCL = GPIO22
I2C = 400 kHz
sample period = 100 ms
```

The AS5600 raw angle is a 12-bit value from 0 through 4095. The example
converts it as:

```text
angle_deg = raw_angle × 360 / 4096
angle_01  = raw_angle / 4096
```

The HAL supplies a wrap-aware delta. The example also reports magnet detected,
too weak, and too strong status. These values are diagnostic; they do not
establish mechanical zero, direction, or usable control-loop latency.

## Wiring

Use 3.3 V logic and a common ground:

```text
AS5600 VCC -> ESP32 3V3
AS5600 GND -> ESP32 GND
AS5600 SDA -> ESP32 GPIO21 by default
AS5600 SCL -> ESP32 GPIO22 by default
```

Provide SDA and SCL pull-ups, normally 2.2 kΩ–4.7 kΩ to 3.3 V, unless the
breakout already includes them. Never use 5 V I2C pull-ups with ESP32 pins.
Place the magnet centered over the IC and verify the printed magnet status.

## Build and reproduce

From the repository root:

```bash
source /Users/crisi/.espressif/v5.5.4/esp-idf/export.sh
cd hw/esp32/as5600_example
idf.py --version
idf.py set-target esp32
idf.py build
```

Set `AS5600_SDA_GPIO`, `AS5600_SCL_GPIO`, `AS5600_I2C_FREQ_HZ`, or
`AS5600_SAMPLE_PERIOD_MS` with `idf.py menuconfig` when the board differs from
the defaults. Flashing and live verification require the connected ESP32 and
are intentionally separate from the build check.

The example calls `AS5600::Init()` and read/status functions only. It does not
call `BurnAngle()` or any other OTP programming function, because those writes
are irreversible and are not part of electrical bring-up.

## Evidence to record after board testing

Keep the raw monitor capture, board wiring/pin configuration, ESP-IDF version,
submodule SHA, and a short result summary together under an ignored hardware
results directory. Record whether initialization succeeded, the observed
angle direction, wrap behavior near 0/4095, magnet status, and I2C errors.
