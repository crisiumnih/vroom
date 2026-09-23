# ACS714 calibration firmware

This is a standalone ESP-IDF calibration utility for the Pololu ACS714 ±5 A
carrier and the proposed 10 kΩ / 10 kΩ output divider. It reads ADC1 through
the ESP-IDF calibration driver and prints raw calibration records over the
console. It does not drive a motor or a power stage.

The project is intended for ESP-IDF 5.5.x. Set the target before configuring:

```bash
cd hw/esp32/acs714_calibration
idf.py --version
idf.py set-target esp32
idf.py menuconfig
idf.py build
idf.py flash monitor
```

In `menuconfig`, set `ACS714 calibration utility -> ADC1 channel` to the
channel mapped to the chosen ADC1 GPIO on the actual ESP32 target. The default
is ADC1 channel 6, which is GPIO34 on the original ESP32; mappings differ on
other ESP32 variants.

The utility accepts these commands in the monitor:

```text
z       capture zero-current voltage
p 0.500 capture a point for 0.500 A
p 1.000 capture a point for 1.000 A
v       print configuration
help    print help
m       start continuous ADC monitor
x       stop continuous ADC monitor
```

Save the monitor output as a raw text capture. From the repository root, fit
the calibration on the host:

```bash
cd /path/to/vroom
python tools/hw_bench/fit_acs714.py capture.txt \
  --output results/hardware/acs714_calibration/calibration.json
```

The JSON contains the fitted intercept, measured sensitivity in mV/A, input
points, residual error, and a completion status. The raw monitor capture is
the source evidence and must be retained with the JSON artifact.

For wiring debug, use `m`. It prints:

```text
ADC,timestamp_us,raw_adc,calibrated_adc_node_mv
```

Use `x` to stop it. The monitor period is configurable in `menuconfig` and
defaults to 100 ms.
