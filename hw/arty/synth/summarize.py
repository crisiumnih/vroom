#!/usr/bin/env python3
"""Create a separate, immutable analysis directory for an open-flow probe.

Requires matplotlib for figures. Does not rerun tools or rewrite raw evidence.
Resource catalog: AMD Artix 7 product table (XC7A35T). nextpnr LUTX slots are
not treated as vendor LUT counts. See the probe README for measurement limits.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--baseline", type=Path, help="Optional saved probe for a before/after comparison")
    args = parser.parse_args()
    probe = args.probe.resolve()
    metrics = json.loads((probe / "metrics.json").read_text())
    if metrics["status"] == "running":
        raise SystemExit("The probe must finish before analysis")
    out = Path(tempfile.mkdtemp(prefix=time.strftime("analysis-%Y%m%dT%H%M%S-"), dir=probe.parent))
    report = {"probe": str(probe), "probe_metrics_sha256": digest(probe / "metrics.json"),
              "catalog": {"lut6_sites": 20800, "flipflops": 41600, "dsp48e1": 90, "bram36": 50},
              "catalog_source": "https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/artix-7.html",
              "database_mapping_source": "https://github.com/f4pga/prjxray-db/blob/master/artix7/mapping/devices.yaml",
              "matplotlib_version": matplotlib.__version__, "cases": [],
              "scope": "Current RTL plus scan fixture, no FP32 operators, no complete controller timing or fit proof"}
    report["target_mhz"] = metrics["target_mhz"]
    baseline = {}
    if args.baseline:
        baseline_path = args.baseline.resolve()
        old_metrics = json.loads((baseline_path / "metrics.json").read_text())
        if old_metrics["status"] == "running":
            raise SystemExit("The baseline must finish before analysis")
        report["baseline"] = {"probe": str(baseline_path),
                              "metrics_sha256": digest(baseline_path / "metrics.json"), "cases": []}
        for c in old_metrics["cases"]:
            cells = c.get("mapped_cells", {})
            row = {"name": c["name"], "status": c["status"],
                   "lut_cells": sum(n for k, n in cells.items() if re.fullmatch(r"LUT[1-6]", k)),
                   "dsp48e1": cells.get("DSP48E1", 0), "bram36": cells.get("RAMB36E1", 0),
                   "fmax_mhz": min((v["achieved"] for v in c.get("pnr", {}).get("fmax", {}).values()), default=None)}
            baseline[row["name"]] = row
            report["baseline"]["cases"].append(row)
    for case in metrics["cases"]:
        cells = case.get("mapped_cells", {})
        pnr_path = probe / f"{case['name']}-nextpnr.log"
        if not pnr_path.exists():
            pnr_path = probe / f"{case['name']}-apio-report.log"
        text = pnr_path.read_text(errors="replace") if pnr_path.exists() else ""
        clocks = case.get("pnr", {}).get("fmax", {})
        row = {"name": case["name"], "status": case["status"],
               "lut_cells": sum(n for k, n in cells.items() if re.fullmatch(r"LUT[1-6]", k)),
               "lut6": cells.get("LUT6", 0), "dsp48e1": cells.get("DSP48E1", 0),
               "bram36": cells.get("RAMB36E1", 0), "bram18": cells.get("RAMB18E1", 0),
               "fixture": case.get("fixture"), "fixture_flipflops": case.get("fixture_mapped_flipflops"),
               "fmax_mhz": min((c["achieved"] for c in clocks.values()), default=None),
               "errors": re.findall(r"^ERROR: (.*)$", text, re.MULTILINE),
               "database_available": {}, "dsp_source_counts": {}}
        for key, value in case.get("pnr", {}).get("utilization", {}).items():
            if key in ("DSP48E1_DSP48E1", "RAMB36E1_RAMB36E1", "SLICE_FFX", "SLICE_LUTX"):
                row["database_available"][key] = value["available"]
        if "netlist" in case:
            netlist = json.loads((probe / case["netlist"]).read_text())["modules"]["timing_top"]
            row["dsp_source_counts"] = dict(Counter(
                c.get("attributes", {}).get("src", "unknown").split("|")[0]
                for c in netlist["cells"].values() if c["type"] == "DSP48E1"))
        delays = re.findall(r"Info:\s+([0-9.]+) ns logic, ([0-9.]+) ns routing", text)
        if delays:
            row["critical_path_ns_rounded"] = {"logic": float(delays[-1][0]), "routing": float(delays[-1][1])}
        report["cases"].append(row)
    report["database_capacity_mismatch"] = any(
        c["database_available"].get("DSP48E1_DSP48E1", 90) != 90 or
        c["database_available"].get("RAMB36E1_RAMB36E1", 50) != 50 for c in report["cases"])
    (out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(__file__, out / "summarize.py")
    cores = [c for c in report["cases"] if c["name"].startswith("core")]
    cache = [c for c in report["cases"] if c["name"] == "cache"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, rows, key, capacity, title in [
        (axes[0], cores, "lut_cells", None, "Core mapped LUT1–6 cells (excludes LUTRAM)"),
        (axes[1], cores, "dsp48e1", 90, "Core address/control DSP blocks"),
        (axes[2], cache, "bram36", 50, "Pinned cache BRAM36 blocks")]:
        values = [r[key] for r in rows]
        positions = list(range(len(rows)))
        before = [baseline.get(r["name"], {}).get(key, 0) for r in rows]
        if baseline:
            old_bars = ax.bar([x-.18 for x in positions], before, color="#ad363d", width=.34, label="Baseline")
            ax.bar_label(old_bars, labels=[f"{v:,}" for v in before], padding=5, fontsize=8)
        bars = ax.bar([x+.18 if baseline else x for x in positions], values, color="#137c80", width=.34, label="Current")
        ax.set_xticks(positions, [r["name"] for r in rows])
        if capacity:
            ax.axhline(capacity, color="#253b50", linestyle="--", label=f"35T capacity: {capacity:,}")
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=5)
        ax.set_ylim(0, max([capacity or 1] + values + before) * 1.35)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Mapped cells / blocks")
        ax.legend(loc="upper left", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RTL implementation probes — FP32 operators excluded; scan fixture included", fontsize=12)
    fig.savefig(out / "resources.svg")
    fig.savefig(out / "resources.png", dpi=160)
    plt.close(fig)
    fit_failed = any(c["lut6"] > 20800 or c["bram36"] > 50 or c["dsp48e1"] > 90 for c in report["cases"])
    conclusion = "The current mapped RTL exceeds the XC7A35T resource capacity." if fit_failed else "See the per-probe results below; this does not establish full-controller feasibility."
    lines = ["# Arty open-flow findings", "", conclusion, "",
             "| Probe | Mapped LUT cells | LUT6 cells | DSP48E1 | BRAM36 | Outcome |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for c in report["cases"]:
        target = metrics["target_mhz"]
        timing = "meets" if c["fmax_mhz"] is not None and c["fmax_mhz"] >= target else "fails"
        outcome = f"{c['fmax_mhz']:.2f} MHz backend estimate; {timing} {target:g} MHz" if c["fmax_mhz"] is not None else ("Flow failed (see raw log)" if c["errors"] else c["status"])
        lines.append(f"| {c['name']} | {c['lut_cells']:,} | {c['lut6']:,} | {c['dsp48e1']} | {c['bram36']} | {outcome} |")
    lines += ["", "The mapped LUT count sums LUT1–LUT6 primitives and excludes LUTRAM. It is not a post-pack physical-LUT count. LUT6 cells can be compared to the device's 20,800 physical LUT sites. RAMB18 counts, registered scan-fixture widths, and identifiable fixture FF counts are in analysis.json.", ""]
    if any(c["bram36"] == 88 for c in cache):
        lines += ["The pinned cache maps to 88 RAMB36 blocks. Its separate boot write and two read ports induce replication; the payload-only estimate of 44 blocks is not achieved by this RTL.", ""]
    if any(c["dsp48e1"] for c in cores):
        lines += ["Source tags in analysis.json locate inferred DSPs in address/control RTL. These are not FP32 operators; inspect those source tags before attributing them to arithmetic lanes.", ""]
    if baseline:
        lines += ["## Change from saved baseline", "",
                  "| Probe | LUT1–6 cells before → after | DSP before → after | BRAM36 before → after |",
                  "| --- | ---: | ---: | ---: |"]
        for c in report["cases"]:
            if c["name"] in baseline:
                old = baseline[c["name"]]
                lines.append(f"| {c['name']} | {old['lut_cells']:,} → {c['lut_cells']:,} | {old['dsp48e1']} → {c['dsp48e1']} | {old['bram36']} → {c['bram36']} |")
        lines += ["", f"Saved baseline: [{baseline_path.name}]({baseline_path}). Both runs include their scan fixtures. Mapped LUT primitive counts are a synthesis comparison, not physical utilization or proof of full-system fit.", ""]
    if report["database_capacity_mismatch"]:
        lines += ["The downloaded xc7a35tcsg324 chip database exposes capacities that differ from the XC7A35T catalog (see analysis.json). The bundled Project X-Ray mapping maps xc7a35t to xc7a50t. Therefore nextpnr utilization percentages are not XC7A35T utilization, and routed timing is only a backend diagnostic, not validated timing on the 35T's permitted resource set.", ""]
    for c in report["cases"]:
        if "critical_path_ns_rounded" in c:
            delay = c["critical_path_ns_rounded"]
            lines += [f"{c['name']} critical path: approximately {delay['logic']:.1f} ns logic plus {delay['routing']:.1f} ns routing. Inspect the raw log for endpoints; these rounded values are diagnostic, not a guarantee for a different placement.", ""]
    lines += ["FP32 arithmetic, nonlinear activations, the DDR controller/CDC, resident W0/W2, cache routing, and controller scheduling remain excluded. These results identify implementation problems; they do not establish that an optimized FP32 architecture is impossible.", "",
              f"Raw evidence: [{probe.name}]({probe})", "",
              "![Resource comparison](resources.svg)", "",
              "Sources: [AMD Artix 7 resources](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/artix-7.html), [Project X-Ray fabric mapping](https://github.com/f4pga/prjxray-db/blob/master/artix7/mapping/devices.yaml).", ""]
    (out / "findings.md").write_text("\n".join(lines))
    (out / "artifact_hashes.json").write_text(json.dumps({str(p.relative_to(out)): digest(p) for p in out.iterdir() if p.is_file()}, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
