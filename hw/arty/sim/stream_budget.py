#!/usr/bin/env python3
"""Development cycle model for the proposed layer-streaming FP32 engine.

This is NOT RTL, floating-point numerical simulation, DDR simulation or timing
closure evidence. It models II=1 multipliers feeding a bounded product FIFO,
then II=1 adders with a configurable per-row feedback interval. Column order is
checked on every accumulation. Weights, inputs and initialized biases are
assumed available. Other costs are explicitly excluded in the output metrics.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time


def simulate_layer(inputs, outputs, lanes, mul_latency, feedback, depth, trace=False):
    groups = (outputs + lanes - 1) // lanes
    total = inputs * groups
    products, pending = deque(), deque()
    completed_columns = [0] * groups
    row_ready = [0] * groups
    issued = accepted = retired = peak_products = stalls = 0
    cycle = 0
    events = []

    def event(kind, column, group):
        if trace:
            events.append((cycle, kind, column, group, min(lanes, outputs-group*lanes)))

    while retired < total:
        # feedback includes read/write visibility or forwarding. A new add can
        # consume the prior result on exactly its declared ready cycle.
        while pending and pending[0][0] == cycle:
            _, column, group = pending.popleft()
            assert completed_columns[group] == column
            completed_columns[group] += 1
            retired += 1
            event("retire", column, group)
        if products and products[0][0] <= cycle:
            _, column, group = products[0]
            if row_ready[group] <= cycle:
                assert completed_columns[group] == column
                products.popleft()
                row_ready[group] = cycle + feedback
                pending.append((cycle + feedback, column, group))
                accepted += 1
                event("accumulate", column, group)
            else:
                stalls += 1
        # FIFO credits reserve space for multiplier results still in flight.
        if issued < total and len(products) < depth:
            column, group = divmod(issued, groups)
            products.append((cycle + mul_latency, column, group))
            issued += 1
            event("multiply", column, group)
        peak_products = max(peak_products, len(products))
        if retired == total:
            break
        cycle += 1
    assert issued == accepted == retired == total
    assert completed_columns == [inputs] * groups
    return {"inputs": inputs, "outputs": outputs, "groups": groups,
            "cycles": cycle, "scalar_macs": inputs*outputs, "vector_requests": total,
            "peak_product_credits": peak_products, "feedback_stall_cycles": stalls}, events


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", nargs="+", type=int, default=[8, 10, 12, 16])
    parser.add_argument("--feedback", nargs="+", type=int, default=[6, 8, 12, 19])
    parser.add_argument("--mul-latency", type=int, default=4)
    parser.add_argument("--product-depth", type=int, default=16)
    parser.add_argument("--mhz", type=float, default=100)
    args = parser.parse_args()
    assert min(args.lanes + args.feedback + [args.mul_latency, args.product_depth]) >= 1
    assert args.mhz > 0
    repo = Path(__file__).resolve().parents[3]
    root = repo / "results" / "arty_architecture"
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime("stream-%Y%m%dT%H%M%S-"), dir=root))
    report = {"status": "running", "kind": "conditional architectural cycle model",
              "config": vars(args), "seed": None, "deterministic": True,
              "python": platform.python_version(), "source_sha256": digest(Path(__file__)),
              "excluded": ["RTL/FPU resource fit and routed clock timing", "DDR service and bursts",
                           "cache routing stalls", "initialization", "nonlinear functions",
                           "controller adapter", "actor switching and scheduler overhead"],
              "cases": [], "full_deadline_proven": False}
    for key, argv in {"head": ["git", "rev-parse", "HEAD"], "branch": ["git", "branch", "--show-current"],
                      "status": ["git", "status", "--short"], "submodules": ["git", "submodule", "status"]}.items():
        p = subprocess.run(argv, cwd=repo, text=True, capture_output=True)
        report[key] = {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    shutil.copy2(__file__, out / "stream_budget.py")
    try:
        # Analytic cross-check when FIFO credits cover the multiplier pipeline.
        # Hidden groups hide feedback if G>=F; a two-output layer has G=1.
        for lanes in args.lanes:
            for feedback in args.feedback:
                row = {"lanes": lanes, "feedback_cycles": feedback, "actors": {}}
                with (out / f"lanes{lanes}-feedback{feedback}-events.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["actor", "layer", "local_cycle", "event", "column", "group", "active_lanes"])
                    for actor, dims in [("inner", [25, 256, 256, 2]), ("outer", [19, 256, 256, 2])]:
                        layers = []
                        for layer, (nin, nout) in enumerate(zip(dims, dims[1:])):
                            data, events = simulate_layer(nin, nout, lanes, args.mul_latency,
                                                          feedback, args.product_depth, trace=True)
                            if args.product_depth >= args.mul_latency:
                                predicted = (nin-1)*max(data["groups"], feedback) + data["groups"]-1 + args.mul_latency + feedback
                                assert data["cycles"] == predicted, (data, predicted)
                            layers.append(data)
                            writer.writerows((actor, layer, *e) for e in events)
                        row["actors"][actor] = {"layers": layers, "cycles": sum(v["cycles"] for v in layers)}
                inner, outer = row["actors"]["inner"]["cycles"], row["actors"]["outer"]["cycles"]
                row.update(inner_us=inner/args.mhz, outer_us=outer/args.mhz,
                           compute_cycles_per_ms=10*inner+outer,
                           compute_load_percent=(10*inner+outer)/(args.mhz*1000)*100,
                           inner_compute_within_100us=inner <= args.mhz*100,
                           aggregate_compute_within_1ms=10*inner+outer <= args.mhz*1000)
                report["cases"].append(row)
        inner_macs = 25*256 + 256*256 + 256*2
        outer_macs = 19*256 + 256*256 + 256*2
        report["mac_lower_bound"] = {"inner": inner_macs, "outer": outer_macs,
            "required_per_clock": (10*inner_macs+outer_macs)/(args.mhz*1000)}
        report["weight_bandwidth_bytes_per_second"] = {
            "176KiB_inner_W1_resident_only": (inner_macs-176*256)*4*10000 + outer_macs*4*1000,
            "also_inner_W0_W2_resident": (256-176)*256*4*10000 + outer_macs*4*1000,
            "scope": "Weight payload only, excludes biases, burst padding, refresh and all service gaps"}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report["matplotlib_version"] = matplotlib.__version__
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for lanes in args.lanes:
            rows = [r for r in report["cases"] if r["lanes"] == lanes]
            for ax, field in zip(axes, ["inner_us", "compute_load_percent"]):
                ax.plot([r["feedback_cycles"] for r in rows], [r[field] for r in rows], marker="o", label=f"{lanes} lanes")
        for ax, ylabel, title in zip(axes, ["Inner compute time (µs)", "Compute occupancy over 1 ms (%)"],
                                    ["Inner compute only", "Ten inner + one outer inference"]):
            ax.axhline(100, color="black", linestyle="--", label="Budget boundary")
            ax.set(xlabel="Accumulator feedback interval (cycles)", ylabel=ylabel, title=title)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Proposed streaming schedule at {args.mhz:g} MHz — conditional model, memory/other costs excluded")
        fig.savefig(out / "budget.png", dpi=160)
        fig.savefig(out / "budget.svg")
        plt.close(fig)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = ["# Proposed streaming architecture budget", "", f"Model run status: {report['status']}", "",
                 "Conditional compute model only. No claim of RTL implementation, FPGA fit, DDR performance, or full deadlines.", "",
                 "| Lanes | Feedback cycles | Inner compute µs | Outer compute µs | 1 ms compute occupancy |",
                 "| ---: | ---: | ---: | ---: | ---: |"]
        for r in report["cases"]:
            lines.append(f"| {r['lanes']} | {r['feedback_cycles']} | {r['inner_us']:.2f} | {r['outer_us']:.2f} | {r['compute_load_percent']:.2f}% |")
        (out / "summary.md").write_text("\n".join(lines) + "\n")
        (out / "artifact_hashes.json").write_text(json.dumps({p.name: digest(p) for p in out.iterdir() if p.is_file()}, indent=2) + "\n")
        print(out)


if __name__ == "__main__":
    main()
