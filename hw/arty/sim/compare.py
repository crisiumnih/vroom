#!/usr/bin/env python3
"""Compare saved dense outputs and command-cycle traces; never rerun or edit them.

The current suite may append additional output cases. Every baseline output row
must match its prefix exactly; the complete original command timing JSON must
match. This is regression evidence, not a proof for arbitrary input sequences.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    baseline, current = args.baseline.resolve(), args.current.resolve()
    out = Path(tempfile.mkdtemp(prefix=time.strftime("comparison-%Y%m%dT%H%M%S-"), dir=current.parent))
    report = {"baseline": str(baseline), "current": str(current), "status": "failed",
              "python": platform.python_version(), "input_hashes": {}, "cases": []}

    def read_json(path):
        report["input_hashes"][str(path)] = digest(path)
        return json.loads(path.read_text())

    def read_csv(path):
        report["input_hashes"][str(path)] = digest(path)
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    try:
        old_metrics = read_json(baseline / "metrics.json")
        new_metrics = read_json(current / "metrics.json")
        assert old_metrics["status"] == new_metrics["status"] == "passed", "Both suites must pass"
        for lanes in (12, 16):
            for latency in (3, 19):
                name = f"dense_{lanes}_latency{latency}"
                before = read_csv(baseline / f"{name}-outputs.csv")
                after = read_csv(current / f"{name}-outputs.csv")
                old_timing = read_json(baseline / f"{name}-timing.json")
                new_timing = read_json(current / f"{name}-timing.json")
                report["cases"].append({"name": name, "baseline_output_words": len(before),
                    "appended_output_words": len(after)-len(before),
                    "baseline_outputs_identical": bool(before) and before == after[:len(before)],
                    "command_cycles_identical": bool(old_timing["tiles"]) and old_timing == new_timing})
        if all(c["baseline_outputs_identical"] and c["command_cycles_identical"] for c in report["cases"]):
            report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(__file__, out / "compare.py")
    lines = ["# Dense regression comparison", "", f"Status: {report['status']}", "",
             f"Baseline: {baseline}", f"Current: {current}", "",
             "| Configuration | Baseline output words | Appended words | Outputs identical | Original command cycles identical |",
             "| --- | ---: | ---: | --- | --- |"]
    for c in report["cases"]:
        lines.append(f"| {c['name']} | {c['baseline_output_words']} | {c['appended_output_words']} | {c['baseline_outputs_identical']} | {c['command_cycles_identical']} |")
    if "error" in report:
        lines += ["", report["error"]]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "artifact_hashes.json").write_text(json.dumps({p.name: digest(p) for p in out.iterdir() if p.is_file()}, indent=2) + "\n")
    print(out)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
