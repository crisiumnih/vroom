#!/usr/bin/env python3
"""Reproducible XC7 synthesis and Apio place/route probes; never programs a board.

The scan wrapper preserves variable inputs and observable outputs while reducing
the physical interface to four pins. It is an implementation measurement fixture,
not a controller interface or an FP32 operator. See README.md for evidence limits.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wrapper(top, ports, lanes):
    inputs = [(n, p) for n, p in ports.items() if p["direction"] == "input" and n != "clk"]
    outputs = [(n, p) for n, p in ports.items() if p["direction"] == "output"]
    assert ports["clk"]["direction"] == "input" and len(ports["clk"]["bits"]) == 1
    assert not any(p["direction"] == "inout" for p in ports.values())
    ni = sum(len(p["bits"]) for _, p in inputs)
    no = sum(len(p["bits"]) for _, p in outputs)
    lines = ["`default_nettype none", "// Measurement fixture: no arithmetic or controller behavior is emulated.",
             "module timing_top(input wire clk, scan_in, scan_capture, output wire scan_out);",
             f'  (* keep = "true", shreg_extract = "no" *) reg [{ni-1}:0] launch;',
             f'  (* keep = "true", shreg_extract = "no" *) reg [{no-1}:0] capture;',
             f"  wire [{no-1}:0] observed;", "  always @(posedge clk) begin",
             f"    launch <= {{launch[{ni-2}:0], scan_in}};",
             "    if (scan_capture) capture <= observed;",
             f"    else capture <= {{capture[{no-2}:0], 1'b0}};", "  end",
             f"  assign scan_out = capture[{no-1}];"]
    connections = [".clk(clk)"]
    for entries, bus in [(inputs, "launch"), (outputs, "observed")]:
        offset = 0
        for name, port in entries:
            width = len(port["bits"])
            connections.append(f".{name}({bus}[{offset} +: {width}])")
            offset += width
    params = f" #(.LANES({lanes}))" if lanes else ""
    lines += [f"  {top}{params} dut (", "    " + ",\n    ".join(connections),
              "  );", "endmodule", "`default_nettype wire", ""]
    return "\n".join(lines), {"launch_bits": ni, "capture_bits": no}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=["core12", "core16", "cache", "rates", "tile12", "tile16", "fpmul", "fpadd", "fplane", "fptanh", "stream12", "stream16"],
                   default=["core12", "core16", "cache"])
    p.add_argument("--seed", type=int, default=2805)
    p.add_argument("--mhz", type=float, default=100.0)
    p.add_argument("--timeout", type=int, default=1800, help="Per tool invocation, seconds")
    p.add_argument("--synth-only", action="store_true", help="System Yosys only; no timing claim")
    p.add_argument("--placement-only", action="store_true", help="Diagnostic placement timing only; never a routed timing pass")
    p.add_argument("--abc9", action="store_true", help="Use delay-aware ABC9 LUT mapping")
    p.add_argument("--placer", choices=["heap", "sa"], default="heap")
    p.add_argument("--logic-region", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                   help="Constrain packed logic/DSP cells to a backend coordinate rectangle; diagnostic floorplan")
    p.add_argument("--native-tools", action="store_true",
                   help="System Yosys plus Apio-installed nextpnr/chipdb, without the full Apio CAD bundle")
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()
    if args.placement_only and (not args.native_tools or args.synth_only):
        p.error("--placement-only requires --native-tools without --synth-only")
    if args.logic_region and not args.native_tools:
        p.error("--logic-region requires --native-tools")
    if args.abc9 and not (args.native_tools or args.synth_only):
        p.error("--abc9 requires --native-tools or --synth-only")
    if args.placer != "heap" and not args.native_tools:
        p.error("--placer requires --native-tools")
    mapping = " -abc9" if args.abc9 else ""
    if any(c.startswith(("fp", "stream")) for c in args.cases) and not (args.native_tools or args.synth_only):
        p.error("FP32/stream cases require --native-tools or --synth-only for pinned HardFloat includes")
    source = Path(__file__).resolve().parents[1]
    repo = source.parents[1]
    root = (args.output_root or repo / "results" / "arty_openflow").resolve()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=time.strftime("probe-%Y%m%dT%H%M%S-"), dir=root))
    print(f"Evidence: {out}", flush=True)
    metadata = {"status": "running", "config": vars(args) | {"output_root": str(root)},
                "platform": platform.platform(), "python": sys.version, "commands": [],
                "source_hashes": {}, "git": {}, "tools": {}}
    metrics = {"status": "running", "part": "xc7a35tcsg324-1", "target_mhz": args.mhz,
               "seed": args.seed, "cases": [], "full_controller_proven": False,
               "catalog_capacity": {"lut6_sites": 20800, "flipflops": 41600, "dsp48e1": 90, "bram36": 50},
               "excluded": ["concurrent inner/outer scheduling", "DDR/MIG/CDC",
                            "resident W0/W2", "cache-to-engine routing", "controller adapter"]}
    apio = ["uv", "tool", "run", "--from", "apio==1.6.0", "apio"]
    raw = [] if args.synth_only or args.native_tools else apio + ["raw", "--"]
    apio_home = Path(os.environ.get("APIO_HOME", str(Path.home() / ".apio")))
    packages = apio_home / "packages"
    native_pnr = packages / "openxc7" / "bin" / "nextpnr-xilinx"
    native_chipdb = packages / "openxc7" / "chipdb" / "xc7a35tcsg324.bin"

    def save():
        (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    def run(name, argv, cwd=out):
        record = {"name": name, "argv": argv, "cwd": str(cwd), "status": "running"}
        metadata["commands"].append(record)
        save()
        started = time.monotonic()
        log_path = out / f"{name}.log"
        print(f"Running {name}", flush=True)
        with log_path.open("w") as log:
            proc = subprocess.Popen(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                code = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                log.write(f"\nTIMEOUT after {args.timeout} seconds\n")
                code = 124
        record.update(exit_code=code, seconds=time.monotonic()-started,
                      status="passed" if code == 0 else "failed")
        save()
        return code, log_path.read_text(errors="replace")

    try:
        for key, cmd in {"head": ["git", "rev-parse", "HEAD"],
                         "branch": ["git", "branch", "--show-current"],
                         "status": ["git", "status", "--short"],
                         "submodules": ["git", "submodule", "status"]}.items():
            code, log = run("git-" + key, cmd, repo)
            metadata["git"][key] = {"exit_code": code, "output": log.strip()}
        snapshot = out / "source"
        snapshot.mkdir()
        for path in sorted((source / "rtl").glob("*.v")):
            shutil.copy2(path, snapshot / path.name)
            metadata["source_hashes"][str(path.relative_to(repo))] = digest(path)
        if any(c.startswith(("fp", "stream")) for c in args.cases):
            for path in sorted((source / "fp32/rtl").glob("*.v")):
                shutil.copy2(path, snapshot / path.name)
                metadata["source_hashes"][str(path.relative_to(repo))] = digest(path)
            vendor = repo / "third_party/hardfloat"
            shutil.copytree(vendor, snapshot / "hardfloat")
            for path in sorted(vendor.rglob("*")):
                if path.is_file():
                    metadata["source_hashes"][str(path.relative_to(repo))] = digest(path)
        if "fptanh" in args.cases or any(c.startswith("stream") for c in args.cases):
            for path in sorted((source / "tanh/rtl").glob("*.v")):
                shutil.copy2(path, snapshot / path.name)
                metadata["source_hashes"][str(path.relative_to(repo))] = digest(path)
        if any(c.startswith("stream") for c in args.cases):
            for path in sorted((source / "stream/rtl").glob("*.v")):
                shutil.copy2(path, snapshot / path.name)
                metadata["source_hashes"][str(path.relative_to(repo))] = digest(path)
        shutil.copy2(__file__, out / "runner.py")
        metadata["source_hashes"][str(Path(__file__).resolve().relative_to(repo))] = digest(Path(__file__))
        for name, cmd in {"yosys": raw + ["yosys", "-V"],
                          **({} if args.synth_only else {
                              "apio": apio + ["--version"],
                              "nextpnr": raw + [str(native_pnr) if args.native_tools else "nextpnr-xilinx", "--version"],
                              **({} if args.native_tools else {"apio-info": apio + ["info", "system"]}),
                          })}.items():
            code, log = run("version-" + name, cmd)
            metadata["tools"][name] = log.strip()
            if code:
                raise RuntimeError(f"Tool unavailable: {name}")
        if not args.synth_only:
            records = out / "toolchain"
            records.mkdir()
            for path in [packages / "installed_packages.json", apio_home / "cached-remote-config.json",
                         *packages.glob("*/BUILD-INFO.json"),
                         packages / "openxc7" / "XILINX-PARTS-INDEX.json",
                         packages / "openxc7/share/nextpnr/external/prjxray-db/artix7/mapping/devices.yaml"]:
                if path.exists():
                    name = path.parent.name + "-" + path.name
                    shutil.copy2(path, records / name)
            for path in [packages / "openxc7" / "bin" / "nextpnr-xilinx",
                         packages / "openxc7" / "libexec" / "nextpnr-xilinx",
                         Path(shutil.which("yosys") or "/nonexistent/yosys") if args.native_tools else
                         packages / "oss-cad-suite" / "bin" / "yosys"]:
                if path.exists():
                    metadata["tools"][str(path)] = {"sha256": digest(path)}

        for case in args.cases:
            work = out / case
            work.mkdir()
            lanes = int(case[6:]) if case.startswith("stream") else int(case[4:]) if case.startswith(("core", "tile")) else None
            top = ("vroom_actor_stream" if case.startswith("stream") else
                   "vroom_dense_tile_core" if case.startswith("core") else
                   "vroom_tile_buffer" if case.startswith("tile") else
                   "vroom_pinned_cache" if case == "cache" else
                   {"fpmul": "vroom_fp32_mul", "fpadd": "vroom_fp32_add", "fplane": "vroom_fp32_lane", "fptanh": "vroom_fp32_tanh"}.get(case, "vroom_rate_monitor"))
            result = {"name": case, "module": top, "lanes": lanes, "status": "running",
                      "fp32_operators_included": case.startswith(("fp", "stream")),
                      "binary32_activation_included": case == "fptanh" or case.startswith("stream")}
            metrics["cases"].append(result)
            for path in snapshot.glob("*.v"):
                shutil.copy2(path, work / path.name)
            rtl = " ".join(p.name for p in sorted(work.glob("*.v")))
            include = ""
            if case.startswith(("fp", "stream")):
                shutil.copytree(snapshot / "hardfloat", work / "hardfloat")
                rtl += " " + " ".join(str(p.relative_to(work)) for p in sorted((work / "hardfloat/source").glob("*.v")))
                rtl += " hardfloat/source/RISCV/HardFloat_specialize.v"
                include = "-Ihardfloat/source -Ihardfloat/source/RISCV "
            hierarchy = f"hierarchy -top {top}" + (f" -chparam LANES {lanes}" if lanes else "")
            script = f"read_verilog {include}{rtl}; {hierarchy}; proc; write_json interface.json"
            code, log = run(case + "-interface", raw + ["yosys", "-p", script], work)
            if code:
                result.update(status="elaboration_failed")
                continue
            ports = json.loads((work / "interface.json").read_text())["modules"][top]["ports"]
            verilog, fixture = wrapper(top, ports, lanes)
            (work / "timing_top.v").write_text(verilog)
            result["fixture"] = fixture
            # Pins are from Digilent's Arty-A7-35 master XDC. The wrapper is
            # never uploaded. External scan setup/hold is outside this probe.
            (work / "timing.xdc").write_text(
                "set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports {clk}]\n"
                "set_property -dict { PACKAGE_PIN D9 IOSTANDARD LVCMOS33 } [get_ports {scan_in}]\n"
                "set_property -dict { PACKAGE_PIN C9 IOSTANDARD LVCMOS33 } [get_ports {scan_capture}]\n"
                "set_property -dict { PACKAGE_PIN H5 IOSTANDARD LVCMOS33 } [get_ports {scan_out}]\n"
                f"create_clock -period {1000/args.mhz:.6f} -name core_clk [get_ports {{clk}}]\n")
            (work / "apio.ini").write_text(
                "[env:probe]\nboard = arty-a7-35t\ntop-module = timing_top\n"
                "constraint-file = timing.xdc\nyosys-extra-options = -flatten\n"
                f"nextpnr-extra-options = --freq {args.mhz} --seed {args.seed} --timing-allow-fail\n")
            if case.startswith(("fp", "stream")):
                # Isolate DUT resources without repeating launch/capture fixtures
                # in lane-count projections. This has no routed timing claim.
                script = f"read_verilog {include}{rtl}; {hierarchy}; synth_xilinx -family xc7 -top {top} -flatten -noiopad{mapping}; write_json dut-mapped.json"
                (work / "dut-synth.ys").write_text(script + "\n")
                dut_code, dut_log = run(case + "-dut-synth", ["yosys", "-s", "dut-synth.ys"], work)
                result["dut_synthesis_status"] = "passed" if not dut_code else "failed"
                if not dut_code:
                    cells = json.loads((work / "dut-mapped.json").read_text())["modules"][top]["cells"]
                    result["dut_mapped_cells"] = dict(sorted(Counter(c["type"] for c in cells.values()).items()))
                    lc_estimates = re.findall(r"Estimated number of LCs:\s+(\d+)", dut_log)
                    if lc_estimates:
                        result["dut_yosys_estimated_lcs"] = int(lc_estimates[-1])
            if args.synth_only or args.native_tools:
                script = f"read_verilog {include}{rtl} timing_top.v; synth_xilinx -family xc7 -top timing_top -flatten{mapping}; write_json mapped.json"
                (work / "synth.ys").write_text(script + "\n")
                code, log = run(case + "-synth", ["yosys", "-s", "synth.ys"], work)
                result.update(status="synthesized_no_timing" if not code else "synthesis_failed")
                if args.native_tools and not args.synth_only and not code:
                    if args.logic_region:
                        (work / "floorplan.py").write_text(
                            f"ctx.createRectangularRegion('logic', {', '.join(str(v) for v in args.logic_region)})\n"
                            "count = 0\n"
                            "for name, cell in ctx.cells:\n"
                            "    if cell.type in ('SLICE_LUTX', 'SLICE_FFX', 'CARRY4', 'SELMUX2_1', 'DSP48E1_DSP48E1'):\n"
                            "        ctx.constrainCellToRegion(name, 'logic')\n"
                            "        count += 1\n"
                            "print('Floorplanned logic/DSP cells:', count)\n")
                    code, log = run(case + "-nextpnr", [str(native_pnr), "--chipdb", str(native_chipdb),
                        "--xdc", "timing.xdc", "--json", "mapped.json", "--report", "timing.pnr",
                        "--freq", str(args.mhz), "--seed", str(args.seed), "--placer", args.placer, "--timing-allow-fail"] +
                        (["--no-route"] if args.placement_only else []) +
                        (["--pre-place", "floorplan.py"] if args.logic_region else []), work)
                    result.update(status="flow_failed" if code else "routed")
            else:
                code, log = run(case + "-apio-report", apio + ["report", "--verbose"], work)
                result.update(status="flow_failed" if code else "routed")
            result["errors"] = re.findall(r"^ERROR: (.*)$", log, re.MULTILINE)
            mapped = list(work.glob("**/hardware.json")) + list(work.glob("**/timing_top.json")) + list(work.glob("mapped.json"))
            # Apio's target name may vary by release; select the netlist by
            # content, excluding the unsynthesized interface.json.
            if not mapped:
                mapped = [q for q in work.rglob("*.json") if q.name != "interface.json"]
            for path in mapped:
                data = json.loads(path.read_text())
                if "timing_top" not in data.get("modules", {}):
                    continue
                cells = data["modules"]["timing_top"].get("cells", {})
                result["mapped_cells"] = dict(sorted(Counter(c["type"] for c in cells.values()).items()))
                # Flattening/ABC renames cells; name prefixes cannot reliably
                # separate DUT resources from fixture resources.
                boundary = {b for name in ("launch", "capture")
                            for b in data["modules"]["timing_top"].get("netnames", {}).get(name, {}).get("bits", [])}
                result["fixture_mapped_flipflops"] = sum(
                    1 for c in cells.values() if c["type"].startswith("FD")
                    and any(b in boundary for b in c.get("connections", {}).get("Q", [])))
                result["netlist"] = str(path.relative_to(out))
                break
            reports = list(work.rglob("*.pnr"))
            if reports:
                report = json.loads(reports[0].read_text())
                result["pnr_report"] = str(reports[0].relative_to(out))
                result["pnr"] = report
                capacities = report.get("utilization", {})
                result["database_capacity_mismatch"] = (
                    capacities.get("DSP48E1_DSP48E1", {}).get("available", 90) != 90 or
                    capacities.get("RAMB36E1_RAMB36E1", {}).get("available", 50) != 50)
                fmax = report.get("fmax", {})
                result["timing_pass"] = not args.placement_only and not code and bool(fmax) and all(v["achieved"] >= v["constraint"] for v in fmax.values())
                if not code:
                    result["status"] = "placement_only" if args.placement_only else "timing_pass" if result["timing_pass"] else "timing_failed_or_unreported"
            if not args.synth_only:
                chipdb = packages / "openxc7" / "chipdb" / "xc7a35tcsg324.bin"
                if chipdb.exists() and "chipdb" not in metadata:
                    metadata["chipdb"] = {"path": str(chipdb), "sha256": digest(chipdb)}
            print(f"{case}: {result['status']}", flush=True)
            save()
        metrics["status"] = "completed" if all(c["status"] in ("timing_pass", "synthesized_no_timing", "placement_only") for c in metrics["cases"]) else "completed_with_failures"
        metadata["status"] = metrics["status"]
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc))
        metrics.update(status="failed", error=str(exc))
        raise
    finally:
        save()
        lines = ["# XC7 open-flow probe", "", f"Status: {metrics['status']}", "",
                 f"Target: xc7a35tcsg324-1, {args.mhz} MHz, seed {args.seed}.", "",
                 "This is a registered scan-fixture measurement. Cases fpmul/fpadd/fplane include real FP32 arithmetic and fptanh includes the binary32 CORDIC activation; stream12/stream16 integrate three-layer inference; the existing core/cache/tile cases do not. All cases exclude the complete controller/DDR system. Check database_capacity_mismatch: the current open database maps the 35T to a larger 50T fabric, so these are backend diagnostics, not device sign-off.", "",
                 "| Case | Status | Clock Fmax (MHz) |", "| --- | --- | --- |"]
        for c in metrics["cases"]:
            clocks = c.get("pnr", {}).get("fmax", {})
            values = ", ".join(f"{n}: {v['achieved']:.2f}" for n, v in clocks.items()) or "not available"
            lines.append(f"| {c['name']} | {c['status']} | {values} |")
        lines += ["", "Raw command logs, source snapshots, configurations, generated wrappers, and netlists are retained. See metrics.json and provenance.json.", ""]
        (out / "summary.md").write_text("\n".join(lines))
        (out / "artifact_hashes.json").write_text(json.dumps({str(f.relative_to(out)): digest(f) for f in out.rglob("*") if f.is_file() and f.name != "artifact_hashes.json"}, indent=2) + "\n")
    return 0 if metrics["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
