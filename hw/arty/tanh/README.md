# Shared binary32 CORDIC tanh

Implemented and tested on `codex/fp32-arty-cache` for the actor output activation.
The recorded actor architecture has two ReLU hidden layers and **two tanh outputs**
per actor, giving `2*10000 + 2*1000 = 22000` tanh evaluations per second. This
module is also integrated into the [serial actor stream](../stream/README.md).
It is not connected to a checkpoint, DDR or hardware.

One shared iterative unit meets that throughput budget at an assumed 100 MHz.
It preserves binary32 interfaces and does not quantize model weights or stored
activations. Internally it uses a **58-bit signed fixed-point working format with
52 fractional guard bits** for shift/add arithmetic. This is a numerical
approximation, not a sequence of IEEE FP32 add/multiply instructions. Its accuracy
is tested separately from the interface width.

## Current numerical scope and historical isolated probe

The area/timing entries below describe the retained isolated probe before the
stream timing work. Current RTL executes payload updates independently of reset
and applies synchronous reset last to ownership state. Reset still cancels active
and held responses; the change removes reset from payload enable paths without
changing arithmetic. The later separate ROUND/PACK states add one clock per
regular input and preserve every rounded result. All 29,792 numerical/protocol
cases pass in `results/arty_tanh/cordic-20261010T025154-93sb15s_`. Current integrated timing is
reported with the actor stream; the historical isolated Fmax is not transferred
to the changed RTL.

| Quantity | Result |
| --- | ---: |
| Worst request-to-first-response latency | 199 clocks = 1.99 us at 100 MHz |
| Fast paths (tiny/saturated/special) | 1 clock |
| Worst service interval, always-ready sink | 200 clocks |
| Two worst-case outputs, no contention | **399 clocks = 3.99 us**, verified in RTL |
| Shared-unit utilization for 22,000 values/s | At most **4.400%**, excluding output stalls |
| Historical DUT-only logic LUT1–6 cells | **1,406** |
| Historical DUT-only flip-flop cells | **750** |
| Historical DSP48E1 / BRAM36 | **0 / 0** |
| Historical routed diagnostic Fmax | **100.76 MHz**, seed 2805 |

The timing margin is only about 0.075 ns. As with the prior lane probe, the
nominal 35T chip database uses larger 50T fabric (75 BRAM/120 DSP sites rather
than 50/90). This is a passing **open-flow diagnostic**, not Arty-35T timing
sign-off or integrated 100 MHz proof. Timing includes a registered scan fixture;
area is separately synthesized without that fixture. Logical LUT counts are not
packed physical LUT-site counts. No board connection or bitstream is involved.

## Algorithm and numerical contract

For a finite regular input `u = abs(a)`, express:

```
2u = k*ln(2) + r,  0 <= r < ln(2)
exp(-2u) = (cosh(r) - sinh(r))*2^-k
tanh(a) = sign(a)*(1-exp(-2u))/(1+exp(-2u))
```

The input conversion to the Q52 working angle is exact for the regular domain.
Its exponent is 114..130. Admission registers `exponent-113` (1..17); conversion
then computes `{significand,16'b0} << shift`, equal to the original
`significand << (exponent-97)`. Factoring the fixed shift and registering its
amount removes exponent subtraction from the wide barrel-shift path without
adding a cycle or changing any working bits.
Repeated subtraction of a guarded ln(2) constant gives k in 0..28. Hyperbolic
CORDIC then starts with `(x,y,z) = (1/K,0,r)` and rotates:

```
d = +1 if z >= 0, else -1
x_next = x + d*(y >> i)
y_next = y + d*(x >> i)
z_next = z - d*atanh(2^-i)
K = product(sqrt(1-2^(-2i))) over the actual iteration sequence
```

Signed shifts round toward negative infinity. Indices 1..52 repeat 4, 13 and 40,
for 55 rotations. Each rotation takes two clocks: registered shifts/table read,
then add/subtract. Constants use 100-decimal-digit arithmetic and nearest/even
rounding to Q52; `sim/constants.py` regenerates or checks their exact embedded
values. The coordinate gain compensation avoids a run-time multiply.

After CORDIC, subtract x-y, shift by k, form the ratio numerator/denominator,
and generate 52 fractional quotient bits with restoring division. Iterative
normalization and nearest/even packing produce the binary32 output. No full
floating-point divider, multiplier, DSP or BRAM is instantiated. Internal shifts
and quotient truncation are part of the tested approximation error.

Special handling:

- `abs(a) <= 2^-13`: return a, including signed zero and every subnormal. The
  cubic tanh correction is too small to change binary32 rounding in this range.
- `abs(a) >= 10`, including infinity: return signed one. For finite inputs,
  `1-tanh(10) < 2*exp(-20) < 2^-25`, below half the spacing immediately below one.
- NaN: return canonical positive `0x7fc00000`; assert `invalid` only for a
  signaling NaN. NaN detection takes precedence over saturation.

Only the invalid exception is exposed. Inexact/underflow exception flags are
**not provided** by this activation interface. Correct rounding for all 2^32
inputs, formal error bounds and bit parity with a particular PyTorch/libm tanh
implementation are **not established**. The development acceptance limit was
set to one ULP; observed error on the retained tests is zero ULP.

AMD's [CORDIC product guide, PG105](https://www.amd.com/content/dam/xilinx/support/documents/ip_documentation/cordic/v6_0/pg105-cordic.pdf)
describes the serial/parallel architecture tradeoff and hyperbolic functions.
This repository supplies its own open RTL, range reduction and ratio calculation;
it does not instantiate or copy AMD's encrypted core.

## Interface and integration requirements

`valid && ready` accepts a binary32 value and 16-bit opaque tag on a rising edge.
The unit holds one request and captures its inputs. `response_valid` remains high
with stable result/tag/invalid until `response_ready`. The next request can be
accepted one edge after a response transfers. Output backpressure extends service
time indefinitely; the 199-clock bound is to **first response validity**.

Synchronous active-high reset cancels an active request or a held response.
`RESET_GATES_HANDSHAKES=1` is the standalone default and suppresses ready/valid
immediately during reset. The serial actor uses `0`: its parent gates all external
handshakes and resets ownership synchronously, allowing internal status signals
to avoid a reset-to-status-to-controller feedback path. This option does not
disable the activation unit's synchronous ownership reset.
The unit has no input queue, actor arbitration, action scaling or atomic pair
commit. Those belong to the future scheduler/adapter. Two separate response tags
must be collected before committing a controller's action vector. Prioritize
ready inner activation requests between scalar operations if using a shared
unit; an operation already started is not preemptible. Queue waiting and outer
contention are not included in the 3.99 us pair estimate.

For regular inputs the clock model is `171 + k + s`, where k is the ln(2)
reduction count and s the quotient normalization shifts. k=28 occurs near the
upper end, where s=0, giving the 199-clock maximum. Near the lower shortcut
threshold k=0 and s<=12. This is checked against every RTL response trace.

## Historical effect on the earlier dense budget (before ROUND/PACK split)

Using the retained lane-based model (M=7, F=11), with an immediately available
activation unit, initialized data and no memory/adapter/scheduler costs:

| Lanes | Inner dense only | Inner plus tanh tail | Fully serialized aggregate occupancy |
| --- | ---: | ---: | ---: |
| 12 | 90.39 us | **94.36 us** | **103.664%** |
| 16 | 73.53 us | **77.50 us** | **85.154%** |

The last column assumes that no next dense job starts until both tanh outputs
finish; this makes 12 lanes fail the aggregate compute-plus-activation budget.
With separate resources, tanh may overlap another job's dense computation: its
4.378% occupancy must **not** simply be added to dense occupancy in that case.
The 12-lane dense resource still has only 0.703% spare capacity, however, and the
uncontended inner path leaves 5.64 us for all omitted work. These arithmetic
budgets do not prove that an actual arbiter, buffers, DDR and release ordering
meet deadlines. The activation unit also adds area to the already tight design.

## Verification and retained evidence

Historical numerical/protocol run: `results/arty_tanh/cordic-20261010T001455-6povmlt8`.

29,792 inputs include both signs, random full-domain bits, random regular-range
values, a grid, near-zero cases, and 33 neighboring bit patterns at each shortcut,
power-of-two, saturation-rounding and ln(2) range-reduction boundary. The integer
CORDIC model is checked against an independent Decimal exponential-based tanh
reference at **both 120 and 180 decimal digits**. The reference rounds the exact
Decimal rational directly to binary32, without a binary64 intermediate. Native
binary64 `tanh` provides a further check for all 29,755 non-NaN inputs.

All 29,792 RTL results exactly match both the integer model and reference-rounded
binary32 output. Tests also verify tags, input capture, single-request ownership,
104,221 output-stall clocks, exact per-input latency, the 397-clock worst-case
pair, reset during CORDIC and reset of a held output. Strict Verilator lint passes
without waivers. These are development vectors, not consumed RL final tests.

The first passing run, `cordic-20261010T001211-xopbuasl`, is retained. A second run
added the explicit worst-case pair test without changing the RTL or accuracy
threshold. No simulation or route failure was discarded.

Synthesis/route: `results/arty_openflow/probe-20261010T001236-asd_ncwq`.
Joined report: `results/arty_tanh/report-20261010T001554-5igfruq_`.
The joined report checks byte-identical simulated and routed RTL. Each raw run
records sources, tool versions, Git/submodule identity, seed, commands, hashes,
completion status and raw logs. Numerical runs retain all vectors/oracle results,
all response CSVs, figures and the first 32 requests' internal VCD waveform.

```sh
python3 hw/arty/tanh/sim/constants.py
uv run --no-project --with matplotlib==3.10.8 python hw/arty/tanh/sim/run.py \
  --seed 2805 --random-count 10000
python3 hw/arty/synth/run.py --native-tools --cases fptanh
python3 hw/arty/tanh/summarize.py \
  --verification results/arty_tanh/cordic-20261010T001455-6povmlt8 \
  --probe results/arty_openflow/probe-20261010T001236-asd_ncwq \
  --compute results/arty_architecture/stream-20261009T235443-zydjp8rv
```

Use `--write` with the constants script only when intentionally regenerating its
embedded table. Every verification/probe/report invocation creates a new result
directory. Checkpoint output/trajectory parity and integrated scheduling are the
remaining gates before using this unit in the complete controller.
