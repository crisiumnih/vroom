#!/usr/bin/env python3
"""Bounded RTL regression. No vendor-IP generation, checkpoint export, or motor I/O.

Creates a unique evidence directory, preserves every compile/run log and VCD,
records source/tool/config identity, and returns nonzero on any failed check.
Requires Python 3.9+ and Icarus Verilog (iverilog/vvp) on PATH. Dense tests also
require iverilog-vpi and a C11 compiler for a simulation-only arithmetic model.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from dense_vectors import generate as generate_dense_vectors


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(argv, cwd):
    proc = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-root", type=Path, help="Provenance root; default is the checkout containing this script")
    parser.add_argument("--seed", type=int, default=2805)
    parser.add_argument("--suite", choices=["all", "control", "dense"], default="all")
    parser.add_argument("--lint", action="store_true", help="Also require strict Verilator RTL lint")
    parser.add_argument("--synth-check", action="store_true", help="Also run Yosys process/memory structural checks (not FPGA implementation)")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    repo = (args.repo_root or source.parents[1]).resolve()
    output_root = (args.output_root or repo / "results" / "arty_rtl").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=time.strftime("control-%Y%m%dT%H%M%S-"), dir=output_root))
    print(f"Evidence: {output}", flush=True)
    results = {"status": "running", "seed": args.seed, "suite": args.suite, "checks": [],
               "scope": "RTL memory/control and dense execution with a simulation-only FP32 operator; no synthesized arithmetic or board timing claim"}
    metadata = {"source_directory": str(source), "repository": str(repo), "python": sys.version,
                "platform": platform.platform(), "commands": [], "tools": {}, "git": {}, "sources": {}}
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            metadata["sources"][str(path.relative_to(source))] = sha256(path)
    for name, argv in {"head": ["git", "rev-parse", "HEAD"], "branch": ["git", "branch", "--show-current"],
                       "status": ["git", "status", "--short"], "submodules": ["git", "submodule", "status"]}.items():
        code, text = capture(argv, repo)
        metadata["git"][name] = {"exit_code": code, "output": text.strip()}

    def run(name, argv):
        metadata["commands"].append({"name": name, "argv": [str(s) for s in argv], "cwd": str(output)})
        try:
            proc = subprocess.run(argv, cwd=output, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=120)
            log, code = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as exc:
            log = exc.stdout or ""
            if isinstance(log, bytes):
                log = log.decode(errors="replace")
            log += "\nTIMEOUT after 120 seconds\n"
            code = 124
        (output / f"{name}.log").write_text(log)
        if code:
            raise RuntimeError(f"{name} failed ({code}); see {output / (name + '.log')}")
        return log

    try:
        tool_names = ["iverilog", "vvp"] + (["cc"] if args.suite != "control" else []) + (["verilator"] if args.lint else []) + (["yosys"] if args.synth_check else [])
        for tool in tool_names:
            location = shutil.which(tool)
            if not location:
                raise RuntimeError(f"Required tool not found: {tool}")
            metadata["tools"][tool] = {"path": location, "version": run(f"version-{tool}", [location, "--version" if tool in ["verilator", "cc"] else "-V"])}
        rtl = sorted((source / "rtl").glob("*.v"))
        cases = []
        if args.suite != "dense":
            cases += [("weight_tiles_12", "tb_weight_tiles", 12, None), ("weight_tiles_16", "tb_weight_tiles", 16, None),
                      ("pinned_cache", "tb_pinned_cache", None, None), ("rate_monitor", "tb_rate_monitor", None, None)]
        if args.suite != "control":
            helper = shutil.which("iverilog-vpi")
            if not helper:
                raise RuntimeError("Required VPI build helper not found: iverilog-vpi")
            flags = shlex.split(run("vpi-cflags", [helper, "--cflags"]))
            metadata["tools"]["iverilog-vpi"] = {"path": helper, "sha256": sha256(Path(helper).resolve()), "cflags": flags}
            run("fp32-model-compile", ["cc", *flags, "-std=c11", "-ffp-contract=off", "-fno-fast-math", "-frounding-math",
                                       "-c", str(source / "sim" / "fp32_vpi.c"), "-o", str(output / "fp32_vpi.o")])
            run("fp32-model-link", [helper, "--name=fp32_vpi", str(output / "fp32_vpi.o"), "-lm"])
            metadata["dense_fixtures"] = generate_dense_vectors(output, args.seed)
            cases += [(f"dense_{lanes}_latency{latency}", "tb_dense_tile", lanes, latency)
                      for lanes in [12, 16] for latency in [3, 19]]
        for name, top, lanes, latency in cases:
            exe = output / f"{name}.vvp"
            command = ["iverilog", "-g2005", "-Wall", "-s", top, "-o", str(exe)]
            if lanes is not None:
                command += [f"-P{top}.LANES={lanes}"]
            if latency is not None:
                command += [f"-P{top}.MAC_LATENCY={latency}", str(source / "sim" / "fp32_mac_model.v")]
            command += [str(path) for path in rtl] + [str(source / "sim" / f"{top}.v")]
            run(f"{name}-compile", command)
            plugins = ["-M", str(output), "-m", "fp32_vpi"] if latency is not None else []
            log = run(name, ["vvp", *plugins, str(exe), f"+seed={args.seed}", f"+vcd={output / (name+'.vcd')}"])
            records = [json.loads(line.removeprefix("RESULT ")) for line in log.splitlines() if line.startswith("RESULT ")]
            if len(records) != 1 or records[0].get("status") != "passed":
                raise RuntimeError(f"Missing or invalid result from {name}")
            results["checks"].append({"name": name, **records[0]})
            print(f"PASS {name}: {records[0]['assertions']} assertions", flush=True)
            if top == "tb_dense_tile":
                starts, timings = {}, []
                # Isolate numerical cases from deliberate fault/reset tests.
                baseline = log.split("CASE invalid result range")[0]
                for line in baseline.splitlines():
                    if line.startswith("DENSE_EVENT "):
                        cells = line[len("DENSE_EVENT "):].split(",")
                        if cells[1] == "start":
                            starts[(int(cells[2]), int(cells[3]))] = int(cells[0])
                    elif line.startswith("DENSE_COMPLETE "):
                        cycle, actor, layer, tag = map(int, line[len("DENSE_COMPLETE "):].split(","))
                        start = starts.pop((actor, tag))
                        timings.append({"actor": actor, "layer": layer, "tag": tag, "start_cycle": start,
                                        "done_cycle": cycle, "cycles": cycle-start})
                groups = {}
                for item in timings:
                    key = f"actor{item['actor']}_layer{item['layer']}"
                    group = groups.setdefault(key, {"tile_count": 0, "command_cycles_sum": 0, "max_tile_cycles": 0})
                    group["tile_count"] += 1
                    group["command_cycles_sum"] += item["cycles"]
                    group["max_tile_cycles"] = max(group["max_tile_cycles"], item["cycles"])
                results["checks"][-1]["simulated_command_timing"] = groups
                (output / f"{name}-timing.json").write_text(json.dumps({"scope": "Simulated command acceptance to drained completion; excludes between-command DMA/init/dispatch; includes operator stalls", "tiles": timings, "groups": groups}, indent=2)+"\n")
                events = [line[len("DENSE_EVENT "):].split(",") for line in log.splitlines() if line.startswith("DENSE_EVENT ")]
                with (output / f"{name}-events.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["core_cycle", "event", "actor", "tag", "row", "rows", "column", "columns"])
                    writer.writerows(events)
                actual = [line[len("DENSE_RESULT "):].split(",") for line in log.splitlines() if line.startswith("DENSE_RESULT ")]
                with (output / f"{name}-outputs.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["fixture", "row", "fp32_hex"])
                    writer.writerows(actual)
            if top == "tb_rate_monitor":
                events = [list(map(int, line[6:].split(","))) for line in log.splitlines() if line.startswith("EVENT ")]
                with (output / "release-events.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["core_cycle", "inner_release", "outer_release"])
                    writer.writerows(events)
                svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="220" viewBox="0 0 1000 220">',
                       '<rect width="1000" height="220" fill="#fff"/>',
                       '<g font-family="sans-serif" font-size="14" fill="#102030">',
                       '<text x="24" y="28">Observed RTL releases at 100 MHz — 2 ms; initial releases included</text>',
                       '<text x="24" y="82">Inner 10 kHz</text><text x="24" y="132">Outer 1 kHz</text>']
                for cycle, inner, outer in events:
                    x = 160 + cycle / 200000 * 800
                    for flag, y, color in [(inner, 80, "#006d77"), (outer, 130, "#b34921")]:
                        if flag:
                            svg.append(f'<line x1="{x}" y1="{y-14}" x2="{x}" y2="{y+14}" stroke="{color}" stroke-width="3"/>')
                for cycle in [0, 100000, 200000]:
                    x = 160 + cycle / 200000 * 800
                    svg.append(f'<text x="{x-10}" y="172">{cycle/100000:g} ms</text>')
                svg += ['<text x="24" y="205">Simulation evidence only. No inference execution time or physical clock timing measured.</text>', '</g></svg>']
                (output / "release-timeline.svg").write_text("\n".join(svg))
        # Separate mathematical check of the proposed vector-to-bank request map.
        cases_checked = 0
        for lanes in [12, 16]:
            for row in range(176):
                for column in range(256):
                    indices = [(r * 256 + column) for r in range(row, min(176, row + lanes))]
                    counts = collections.Counter(index % 11 for index in indices)
                    assert max(counts.values()) <= 2
                    assert all(index // 11 < 4096 for index in indices)
                    cases_checked += 1
        results["checks"].append({"name": "pinned_bank_mapping", "status": "passed", "vectors_checked": cases_checked,
                                  "lanes": [12, 16], "max_reads_per_bank": 2})
        tops = ["vroom_weight_tiles", "vroom_pinned_cache", "vroom_rate_monitor", "vroom_dense_tile_core"]
        if args.lint:
            for top in tops:
                run(f"lint-{top}", ["verilator", "--lint-only", "--Wall", "--top-module", top] + [str(path) for path in rtl])
                results["checks"].append({"name": f"lint-{top}", "status": "passed"})
        if args.synth_check:
            for top in tops:
                # JSON-quoted strings are Yosys script arguments, not shell text.
                script = "read_verilog " + " ".join(json.dumps(str(path)) for path in rtl)
                script += f"; hierarchy -check -top {top}; proc; opt; memory_dff; memory_collect; opt_clean; check -assert; stat; "
                script += "write_json " + json.dumps(str(output / f"{top}-netlist.json"))
                run(f"structure-{top}", ["yosys", "-p", script])
                results["checks"].append({"name": f"structure-{top}", "status": "passed"})
        results["status"] = "passed"
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        lines = [f"Status: {results['status']}", results["scope"], f"Seed: {args.seed}"]
        lines += [f"{record['name']}: {record['status']}" for record in results["checks"]]
        for record in results["checks"]:
            if "simulated_command_timing" in record:
                groups = record["simulated_command_timing"]
                cycles = sum(group["command_cycles_sum"] for key, group in groups.items() if key.startswith("actor0_"))
                lines.append(f"{record['name']}: inner-shaped dense commands = {cycles} simulated cycles ({cycles/100:g} us at nominal 100 MHz); excludes between-command DMA/init/dispatch and nonlinear/controller work")
        if "error" in results:
            lines.append(results["error"])
        lines.append("Reproduce: python3 hw/arty/sim/run.py --seed " + str(args.seed)
                     + " --suite " + args.suite
                     + (" --lint" if args.lint else "") + (" --synth-check" if args.synth_check else ""))
        (output / "summary.txt").write_text("\n".join(lines) + "\n")
        manifest = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
        (output / "artifact-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if results["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
