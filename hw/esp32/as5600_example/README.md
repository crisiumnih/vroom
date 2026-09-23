# AS5600 ESP-IDF example

This example exercises the pinned `third_party/as5600_driver` HAL through an
ESP-IDF I2C transport adapter. It reports the AS5600 raw 12-bit angle, degrees,
normalized angle, wrap-aware delta, and magnet diagnostics.

The example does not call OTP-burning APIs. Do not add `BurnAngle()` to a
bring-up loop; AS5600 OTP programming is limited and irreversible.

## Wiring

Use 3.3 V logic so the I2C pull-ups cannot drive ESP32 pins above 3.3 V:

```text
AS5600 VCC -> ESP32 3V3
AS5600 GND -> ESP32 GND
AS5600 SDA -> GPIO21 by default
AS5600 SCL -> GPIO22 by default
```

Use external SDA/SCL pull-ups, typically 2.2 kΩ–4.7 kΩ to 3.3 V, unless the
breakout board already includes them. Keep the magnet centered over the IC and
check the magnet diagnostics printed by the example.

## Build and run

The project uses ESP-IDF 5.5.x:

```bash
cd hw/esp32/as5600_example
source /Users/crisi/.espressif/v5.5.4/esp-idf/export.sh
idf.py --version
idf.py set-target esp32
idf.py menuconfig
idf.py build
idf.py -p PORT flash monitor
```

Configure `AS5600 example` in `menuconfig` if your SDA/SCL pins differ. A
successful run reports an AS5600 response at address `0x36`, then prints lines
similar to:

```text
raw=2048 deg= 180.00 norm=0.50000 delta=+12 magnet=1 weak=0 strong=0
```

## Integration boundary

The HAL remains hardware-independent and is pinned as a Git submodule. The
ESP-IDF component in `hw/esp32/components/as5600_driver` only exposes the HAL
source and headers. The example owns the ESP-IDF I2C callbacks, GPIO mapping,
bus speed, and task timing.
