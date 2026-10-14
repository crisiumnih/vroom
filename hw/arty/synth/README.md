# XC7 open-flow implementation probe

This probe measures the existing RTL with Yosys and nextpnr, provisioned through
the Python package Apio 1.6.0. The requested target is `xc7a35tcsg324-1`, with a
100 MHz / 10 ns clock and deterministic place/route seed 2805. No command in the
runner uploads, flashes, opens a hardware connection, or runs a motor.

```sh
python3 hw/arty/synth/run.py
```

The [first baseline](baseline-20261009.md) fails resource fit. Its single-buffer
timing probe also fails 100 MHz. The downloaded chip database uses the larger
XC7A50T fabric, so routed results are diagnostic rather than 35T sign-off.

The [bank/cache follow-up](banking-20261009.md) achieves 44 cache BRAM36 blocks
and reduces core mapped logic by roughly 68–70%. Functional outputs and original
cycle traces are unchanged. It still fails 100 MHz: the final 12-lane core routes
at 34.74 MHz with seed 2806; seed 2805 has constant-routing failures in both core
sizes. Those failures are retained, and constant-connectivity checks are never
bypassed. See the follow-up for per-case timing and excluded integration work.

The default cases are the 12-lane dense core, the 16-lane dense core, and the
176 KiB pinned cache. Each is a separate implementation probe. Their timing
cannot be combined into timing of a complete controller. Use `--cases core12`,
`--cases core16`, or `--cases cache` to select one. `--cases rates` probes the
release/deadline instrumentation separately.
`--cases tile12` or `--cases tile16` probes one weight buffer independently.
`--native-tools --cases fpmul fpadd fplane` probes the [real FP32 arithmetic](../fp32/README.md).
`--native-tools --cases fptanh` probes the [binary32 CORDIC output activation](../tanh/README.md).
`--native-tools --cases stream12 stream16` probes the [complete serial actor stream](../stream/README.md),
including all FP32 lanes, activation storage, ReLU and output tanh. Add
`--synth-only` for mapped resource evidence without place/route.
These cases also perform DUT-only area synthesis without scan fixtures; they
require native mode (or `--synth-only`) to load the pinned HardFloat sources.

The runner requires Python and `uv`; it invokes the pinned Python package with
`uv tool run --from apio==1.6.0 apio`. Apio installs its native tool packages and
downloads the matching Xilinx chip database on demand. The package version does
not freeze Apio's remotely selected tool releases, so the evidence must record
the installed package metadata and native tool versions as well.

When Apio's `openxc7` package and the matching chipdb are already installed,
use the existing system Yosys without downloading the full Apio CAD bundle:

```sh
python3 hw/arty/synth/run.py --native-tools --cases cache tile16 core12 core16
```

This was the flow used for the first placement/timing baseline. It invokes
Apio-provided nextpnr directly, not `apio report`; the distinction and each native
version are preserved in provenance. The native mode never installs packages.
The initial Apio setup installed `definitions` and `openxc7`; its subsequent
full-CAD-bundle download was stopped once the required Xilinx tools were usable.

For mapped synthesis with the system Yosys, without Apio or place/route:

```sh
python3 hw/arty/synth/run.py --synth-only
```

This mode provides resource evidence only and never reports an Fmax.

For the integrated timing work, `--abc9` selects Yosys's ABC9 LUT mapper for
both DUT-only and scan-fixture synthesis. Record this option with every result:

```sh
python3 hw/arty/synth/run.py --native-tools --abc9 --cases stream12 --timeout 300
```

The retained [complete-stream 100 MHz pass](../stream/README.md#current-100-mhz-diagnostic)
uses the standard mapper, without `--abc9`: 100.65 MHz, seed 2805. Reproduce its
configuration with `python3 hw/arty/synth/run.py --native-tools --cases stream12 --timeout 300`.
The passing source snapshot and matching full inference regression are recorded
in the stream report. The larger-fabric and scan-fixture limits below still apply.

Native diagnostic options include `--placement-only` (runs `--no-route`, always
recorded as placement-only rather than a timing pass), `--placer heap|sa`, and
`--logic-region X0 Y0 X1 Y1` (constrains logic/FF/carry/mux/DSP cells to a rectangle).
These options do not correct the larger chip database or establish a valid 35T
floorplan. Failed and timed-out placement/routing experiments remain in their
own evidence directories. The installed nextpnr revision leaves the JSON
`critical_paths` array empty; inspect the text log's critical-path report for
register, logic and net-delay details.

## Measurement boundary

The core exposes wide DMA, descriptor, result, and external arithmetic buses.
Mapping these directly onto package pins would measure an impossible board
interface. Instead, a generated four-pin scan fixture launches every non-clock
input from a register and captures every output in a register. Serial shifting
makes those registers controllable and observable; no payloads or handshakes are
tied to constants, and no fake multiply/add implementation is supplied.

All fixture and DUT registers share the core clock. The generated XDC uses the
clock pin E3, button pins D9/C9, and LED pin H5 from the official
[Digilent Arty-A7-35 master XDC](https://github.com/Digilent/digilent-xdc/blob/master/Arty-A7-35-Master.xdc).
Those assignments permit placement, but the fixture is not intended for upload.
External scan input setup/hold and output delays are not specified. The reported
clock Fmax concerns internal synchronous paths, including fixture-to-DUT and
DUT-to-fixture paths. Fixture resources and routing congestion are included.
Input/output register bit counts and identifiable mapped boundary flip-flops
are recorded; flattened cell-name prefixes are not a reliable DUT-area split.

`--timing-allow-fail` lets nextpnr finish and retain its report when 100 MHz fails.
The runner independently checks every reported clock's achieved frequency against
its constraint and records a failed/missing clock report as such. A successful
process exit alone is not a timing pass. Failures and timeouts are retained, and
timeouts terminate the process group rather than leaving tools running.

The dense core still has an **external FP32 operator interface**. The core/cache/tile cases exclude
synthesized floating-point operators; the separate fp* cases include them. All
cases exclude MIG/DDR/CDC, concurrent actor scheduling, controller adapter,
resident W0/W2, or cache-to-engine router in this experiment. DSP cells inferred
from the older core/cache/tile RTL, if any, are integer address/control arithmetic;
the fp/stream cases include actual FP32 multiplier DSPs. The stream cases also
include output activation integration. Even a passing 100 MHz result does not establish the
10 kHz inference deadline or fit of the complete accelerator.

## Evidence

Every run creates a unique ignored `results/arty_openflow/probe-*` directory:

- immutable input RTL snapshots and generated fixture/project/XDC files;
- commands, return codes, elapsed time, git/submodule identity, tool versions,
  source hashes, and complete raw tool logs;
- elaborated and mapped netlists, Apio/nextpnr reports when produced;
- machine-readable `metrics.json`, readable `summary.md`, and artifact hashes.

The clock constraint is an assumption, routed timing is an open-tool estimate,
simulation cycle counts are separate evidence, and no hardware measurement is
claimed. The original RTL and failed runs are preserved when a follow-up probe
changes a configuration or implementation.

Generate a separate analysis directory and resource figures without altering the
raw probe directory:

```sh
uv run --no-project --with matplotlib==3.10.8 python hw/arty/synth/summarize.py \
    results/arty_openflow/probe-YYYYMMDDTHHMMSS-XXXXXXXX
```

Add `--baseline PATH_TO_PRIOR_PROBE` for a saved before/after comparison. The
analysis records the baseline metrics hash and preserves raw results. Reported
timing outcomes compare Fmax to the actual requested clock constraint.
