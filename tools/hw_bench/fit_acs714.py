#!/usr/bin/env python3
"""Fit ACS714 voltage/current coefficients from calibration utility output.

Input is the raw text captured from the ESP-IDF monitor. The firmware emits
machine-readable lines such as:

    ZERO,1248.000,0.500,1247,1249,1000
    POINT,1.000000,1341.000,0.600,1340,1342,1000

The fit is V_adc_mV = intercept_mV + slope_mV_per_A * current_A.
The output JSON is a calibration artifact for later firmware integration.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


POINT_RE = re.compile(
    r"POINT,"
    r"([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+),"
    r"([-+0-9]+),([-+0-9]+),([-+0-9]+)"
)
ZERO_RE = re.compile(
    r"ZERO,"
    r"([-+0-9.eE]+),([-+0-9.eE]+),"
    r"([-+0-9]+),([-+0-9]+),([-+0-9]+)"
)


def read_capture(path: str) -> tuple[list[dict[str, float | int]], dict[str, float | int] | None]:
    handle = sys.stdin if path == "-" else Path(path).open("r", encoding="utf-8")
    points: list[dict[str, float | int]] = []
    zero: dict[str, float | int] | None = None
    try:
        for line in handle:
            point_match = POINT_RE.search(line)
            if point_match:
                current_a, voltage_mv, stddev_mv, min_mv, max_mv, samples = point_match.groups()
                points.append(
                    {
                        "current_a": float(current_a),
                        "voltage_mv": float(voltage_mv),
                        "stddev_mv": float(stddev_mv),
                        "min_mv": int(min_mv),
                        "max_mv": int(max_mv),
                        "samples": int(samples),
                    }
                )
                continue

            zero_match = ZERO_RE.search(line)
            if zero_match:
                voltage_mv, stddev_mv, min_mv, max_mv, samples = zero_match.groups()
                zero = {
                    "voltage_mv": float(voltage_mv),
                    "stddev_mv": float(stddev_mv),
                    "min_mv": int(min_mv),
                    "max_mv": int(max_mv),
                    "samples": int(samples),
                }
    finally:
        if handle is not sys.stdin:
            handle.close()
    return points, zero


def fit(points: list[dict[str, float | int]], minimum_span_a: float) -> dict[str, object]:
    if len(points) < 3:
        raise ValueError("at least three POINT records are required")

    currents = [float(point["current_a"]) for point in points]
    voltages = [float(point["voltage_mv"]) for point in points]
    mean_current = sum(currents) / len(currents)
    mean_voltage = sum(voltages) / len(voltages)
    denominator = sum((current - mean_current) ** 2 for current in currents)
    if denominator <= 0.0:
        raise ValueError("POINT records must contain more than one current value")

    slope = sum(
        (current - mean_current) * (voltage - mean_voltage)
        for current, voltage in zip(currents, voltages)
    ) / denominator
    intercept = mean_voltage - slope * mean_current
    if slope <= 0.0:
        raise ValueError("fitted slope is not positive; check current direction and wiring")

    residuals_a = [
        ((voltage - intercept) / slope) - current
        for current, voltage in zip(currents, voltages)
    ]
    rmse_a = math.sqrt(sum(error * error for error in residuals_a) / len(residuals_a))
    max_abs_error_a = max(abs(error) for error in residuals_a)
    span_a = max(currents) - min(currents)

    return {
        "status": "complete" if span_a >= minimum_span_a else "incomplete_span",
        "fit_model": "voltage_mv = intercept_mv + slope_mv_per_a * current_a",
        "intercept_mv": intercept,
        "zero_current_voltage_mv": intercept,
        "slope_mv_per_a": slope,
        "nominal_acs714_slope_after_half_divider_mv_per_a": 92.5,
        "current_span_a": span_a,
        "fit_rmse_a": rmse_a,
        "fit_max_abs_error_a": max_abs_error_a,
        "point_count": len(points),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="raw monitor capture path, or - for stdin")
    parser.add_argument("-o", "--output", required=True, help="output calibration JSON path")
    parser.add_argument("--minimum-span-a", type=float, default=1.0)
    parser.add_argument("--sensor", default="ACS714-5A")
    parser.add_argument("--divider-top-ohms", type=float, default=10000.0)
    parser.add_argument("--divider-bottom-ohms", type=float, default=10000.0)
    args = parser.parse_args()

    points, zero = read_capture(args.input)
    fit_result = fit(points, args.minimum_span_a)
    result = {
        "sensor": args.sensor,
        "divider_top_ohms": args.divider_top_ohms,
        "divider_bottom_ohms": args.divider_bottom_ohms,
        "divider_ratio": args.divider_bottom_ohms
        / (args.divider_top_ohms + args.divider_bottom_ohms),
        "zero_measurement": zero,
        "points": points,
        **fit_result,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
