# Arty A7 FP32 controller RTL

This directory begins the user-authorized Verilog implementation of the cached
two-actor controller. The [complete inference stream](stream/README.md) now runs
three dense layers with real FP32 arithmetic, hidden ReLU and output tanh. Separate
cache/tile building blocks also exist. Cache/DDR integration, concurrent scheduling
and full-device timing/fit remain unfinished. It performs no quantization.

The current requested milestone is the standalone inference stream: prepared
FP32 input vectors plus weights/biases in, tagged two-value FP32 actor outputs
after tanh out. Motor sensing, observation/history construction, action mapping
and actuator interfaces are outside this milestone. See the
[current delivery scope](streaming-architecture.md#current-delivery-scope-inference-input-to-output).

## Implemented

| Module | Behavior |
| --- | --- |
| `stream/rtl/vroom_actor_stream.v` | Complete serial actor job: prepared FP32 inputs and streamed bias/weight groups to a tagged two-output packet; tested real FP32 lanes, activation banks, ReLU, CORDIC tanh, fault/reset handling and output backpressure |
| `rtl/vroom_tile_buffer.v` | 512 FP32 words (2 KiB), 128-bit DMA writes, up to 16 consecutive word reads per cycle, transfer-tag checks, exact byte/last-beat checks, ownership and sticky faults |
| `rtl/vroom_weight_tiles.v` | Four independent tile buffers: inner A/B = IDs 0/1, outer A/B = IDs 2/3; loading and reading different buffers overlap |
| `rtl/vroom_pinned_cache.v` | 176 KiB pinned inner W1 cache, 11 banks of 4,096 words, two synchronous read ports per bank, sequential boot loading and explicit seal |
| `rtl/vroom_rate_monitor.v` | Release and deadline instrumentation at 10,000/100,000 cycles; completion job-ID validation and sticky deadline evidence |
| `rtl/vroom_dense_tile.v` | Independent actor contexts, dense address sequencing, operand buffering, accumulator hazards, ordered FP32 result retirement, and drained tile completion |
| `rtl/vroom_dense_tile_core.v` | Connects the dense executor to the four tile buffers, permitting prefetch during compute |
| `rtl/vroom_bank_rotate.v` | Four combinational mux stages for sixteen bank words; forward reads and reverse retirement routing, with no payload arithmetic |

The older tile-based dense executor's floating-point operators remain external. Its numerical tests use
a simulation-only FP32 C/VPI model, not integer substitutes or a synthesized FPU.
See [dense-engine.md](dense-engine.md) for interfaces, arithmetic ordering, test
coverage, and measured simulation cycle costs. The current core exceeds the
100 us inner budget; passing functional tests is not a 10 kHz timing claim.

The [bank/cache follow-up](synth/banking-20261009.md) reduces the pinned cache
from 88 to 44 BRAM36 blocks and removes the large bank-index arithmetic from the
[first synthesis baseline](synth/baseline-20261009.md). Core logic LUT1–6 cells
drop from 62,614 to 19,873 for 12 lanes and 82,632 to 24,477 for 16 lanes;
address/control DSPs drop from 59/79 to two. FP32 operators remain excluded.
The cache routes at a diagnostic 185.19 MHz, the single tile at 66.22 MHz, and
the 12-lane core at 34.74 MHz. The 16-lane backend run fails constant-connectivity
checks. The core still fails 100 MHz. The open database uses the larger 50T
fabric, so these results are not board timing sign-off. See
[the probe runner](synth/README.md) for commands and measurement boundaries.

The [proposed layer-streaming replacement](streaming-architecture.md) addresses
the control path and command/feedback overhead while keeping binary32 arithmetic
and concurrent weight loading. Its executable cycle model is conditional;
the complete DDR-backed replacement and full-system deadlines remain unverified. A
[real FP32 lane](fp32/README.md) now has tested multiply/add, local accumulator
storage and product queuing. Its diagnostic Fmax is 103.62 MHz, with six-cycle
multiply, nine-cycle add and eleven-cycle accumulator feedback. It is not yet
integrated with the older tile core, but is now used by the complete streaming
engine. Its area and feedback measurements leave the 12/16-lane choice unresolved.

A separate [shared CORDIC tanh unit](tanh/README.md) now implements the output
activation with binary32 interfaces and 52 internal fractional guard bits. It
passes 29,792 numerical/protocol vectors. Its earlier isolated route reached
100.76 MHz; current timing is measured with the integrated stream. Separate
ROUND/PACK stages now make two uncontended worst-case outputs take 399 clocks
(3.99 us at 100 MHz).
The unit is now connected to final sums and tagged output delivery in the
streaming engine. Concurrent actor scheduling remains unimplemented.

The [complete stream regression](stream/README.md#verification-and-observed-cycle-counts)
passes for both widths. With ready data, inner latency is 9,478 clocks at 12
lanes and 7,780 at 16, including all layers and tanh. At 100 MHz this
is 94.78/77.80 us. The serial 12-lane aggregate still exceeds the combined
10 kHz inner plus 1 kHz outer budget; neither version has full-system timing sign-off.
The [current 12-lane integrated open-flow probe](stream/README.md#current-100-mhz-diagnostic)
passes placement, routing, connectivity and the 100 MHz constraint at **100.65 MHz**
with seed 2805 and the standard Yosys mapper. Its roughly 0.065 ns margin is
small, and the larger-fabric database still prevents actual 35T sign-off. The
current 16-lane RTL is functionally tested but has no accepted physical probe.
The stream report retains the original failed probes and all source evidence.

RTL is Verilog-2005. Default lane count is 12; 16 is also tested. Lane count is
a parameter, not a claimed hardware maximum. All clocks in these modules are the
same core clock, intended to be 100 MHz. Reset is synchronous, active high.

`rate_monitor` is **not a dispatcher or a controller adapter**. It neither
preempts inference nor commits motor commands. Its simultaneous initial releases
are instrumentation events, not a decision about outer/inner observation ordering.
The previously discussed one-millisecond outer delay is not implemented or enabled.

## Tile interface and layout

Each buffer moves through `EMPTY -> FILLING -> READY -> READING -> EMPTY`.
Handshake transfers occur at a rising edge with both `valid` and `ready` high;
a producer must hold its payload while backpressured. The load request carries
an opaque 32-bit transaction tag and a word count, from 4 to 512 in multiples of
4. Unsupported counts fail rather than silently padding a transfer.

One beat carries four consecutive words, with the first in `dma_data[31:0]`.
The intended tile ordering is column-major within each tile: for a tile with R
output rows, word index = `local_column * R + local_row`. This provides consecutive
output-neuron weights at a fixed input column. Tile storage maps:

```
bank    = word_index % 16
address = word_index / 16
```

The sixteen 32-word banks each have one write and one asynchronous read port;
compute outputs are registered. A buffer's DMA and compute ownership are exclusive,
while separate buffers operate concurrently. At a read handshake, lane L returns
word `read_base + L` if its mask bit is set; unmasked outputs are zero. Masked
indices must be below the allocated word count. The registered response has no
backpressure: the consumer must accept every `response_valid` cycle.

For starting bank `read_base[3:0]`, a physical bank below that index reads the
next word address; other banks use `read_base[8:4]`. A shared four-stage rotation
aligns those bank words with output lanes. Bounds checking uses an 11-bit
`allocated_words - read_base` subtraction, preserving underflow before comparing
the remaining count to each enabled lane. This retains sparse masks, masked
tail reads and out-of-range faults without wide signed bank-index arithmetic.

The final data beat must assert `dma_last` exactly at the allocated length.
Readiness also requires an independently successful status with matching tag and
byte count. Status may arrive before or after data across an external CDC bridge.
A completion status is accepted only once. Acquisition and release validate the
same transaction tag. The four-buffer wrapper halts all handshakes and suppresses
read-response validity after any buffer fault. A request already accepted on the
edge that reveals a fault may have changed another buffer; the entire episode is
invalid and must be reset.

| Fault code | Meaning |
| --- | --- |
| 1 | Unsupported allocation length |
| 2 | DMA data tag mismatch |
| 3 | Early or missing last beat |
| 4 | Failed, duplicate, unowned, wrong-tag, or wrong-length DMA status |
| 5 / 6 | Wrong acquire / release tag |
| 7 | Masked read outside the tile |

The future DMA bridge must attach buffer and transaction identity to every
beat and status. AXI DataMover does not directly supply this exact custom
interface. Tags must not be reused while old responses could arrive. Reset clears
ownership but not RAM contents; the bridge must either drain/reset its channels or
advance its externally maintained epoch before allocating new tags. Late data with
an old tag is rejected when the new transfer starts. The buffer does not invent
epoch identity, cancel DMA commands, or implement a CDC.

## Pinned cache

The proposed cached region is inner W1 rows 0 through 175, all 256 columns:
`176 * 256 * 4 = 180224 bytes`. Feed precisely 45,056 FP32 words in row-major
matrix order through the sequential boot handshake. The hardware advances its own
bank/address counters; the host does not provide unchecked write addresses.

```
linear_index = output_row * 256 + input_column
bank         = linear_index % 11
address      = linear_index / 11
```

Assert `seal` only after the final boot handshake, on a later cycle. An incomplete
seal faults; a sealed cache backpressures all writes. The future boot controller
must verify model identity/checksum before sealing: a word count alone does not
authenticate a checkpoint. Reset invalidates the complete cache and requires reload.

Read request ports are already banked: each of the 11 banks has two address/enable
pairs and one-cycle responses. Routing a 12- or 16-neuron vector into these ports is
not implemented yet. The regression exhaustively checks the proposed address map:
consecutive output rows at one input column need at most two reads per bank for
both lane counts. Logical storage is equivalent to 44 BRAM36 blocks at 32-bit
payload width. Boot writes now explicitly share physical port A's address with
inference reads; port B remains the second read port. Yosys maps this to 44 BRAM36
blocks, with no replication. This leaves six of the device's fifty blocks before
input storage, other resident weights, queues, and integration logic are budgeted.

## Timing and controller contract

At the intended 100 MHz clock, one cycle is 10 ns:

```
inner release period = 10000 cycles  = 100 us
outer release period = 100000 cycles = 1 ms
```

The monitor releases job 0 of both actors at `start`, then increments the respective
32-bit job ID at each period. A matching completion at the exact next release edge
counts as on time; the new job then becomes pending. A missed deadline or unexpected
completion latches actor, job ID, cycle, and fault code (1 = deadline, 2 = completion
protocol), stops releases, and requires reset. `start` while running is ignored.
Measurements refer to the jobs presented to this monitor, not proven inference
latency. Outputs, histories, normalization, and observation snapshots are not wired.

The intended actors remain inner 25->256->256->2 at 10 kHz and outer
19->256->256->2 at 1 kHz, FP32 with ReLU hidden layers and tanh output.
The checkpoint investigated during architecture design was
`Vroom_2805_Experimental_Controller_Pair/selected_checkpoint.npz`, SHA-256
`f2a8b76db02dc23be07fb944dbe40d99f72b56e4711f2983cc8c7067e5c90b41`.
No checkpoint is copied or exported by this implementation. The stream tests use
independent synthetic inference references, not checkpoint golden vectors.
Training/evaluation controller timing,
history ordering, resets, scaling, and action transforms remain the contract for
the future adapter. A delayed outer-loop schedule needs separate simulation
validation; a 1 kHz release rate alone does not establish timing equivalence.

## Reproduce verification

From the repository root, with Python 3.9+, Icarus Verilog/`iverilog-vpi`, and a C11
compiler on PATH:

```sh
python3 hw/arty/sim/run.py --seed 2805
```

Include strict Verilator lint and generic Yosys process/memory checks:

```sh
python3 hw/arty/sim/run.py --seed 2805 --lint --synth-check
```

Each invocation creates a unique ignored `results/arty_rtl/control-*` directory.
It preserves compiler/run logs, VCD waveforms, machine-readable metrics,
source and tool identities, Git/submodule state, artifact hashes, a readable
summary, and a release timeline SVG generated from observed RTL release events.
Failures exit nonzero and retain the evidence. The fixed seed controls DMA-source
bubbles and synthetic dense fixtures; both lane counts and exhaustive pinned-cache
reads are tested. Dense tests also vary arithmetic latency and input stalls. No plant
simulation or training is launched. Use `--output-root` for another evidence
directory; `--repo-root` is for explicitly recording provenance while testing a
staged source tree. `--suite control` runs the original memory/rate tests without
the C operator; `--suite dense` selects the dense integration tests.

Coverage includes all four buffer contexts, concurrent DMA/compute, source bubbles,
overwrite backpressure, masked tails, buffer reuse, reset during DMA, stale tags,
status/data ordering, malformed transfers, sticky faults, both pinned-cache ports,
10:1 rate spacing, exact-deadline acceptance, overdue jobs, wrong and duplicate
completion IDs. Synthetic payloads include NaN, infinity, signed zero, and subnormal
bit patterns. Their preservation does not establish floating-point arithmetic behavior.

The bank-address regression also checks every base from 0 through 511 for both
full and eight-word tiles, using dense and sparse masks (2,048 window requests per
lane configuration). It checks zero output in mask holes, underflow with a fully
masked request, and an enabled lane that crosses the 512-word boundary. Dense
tests append all accumulator bank alignments and result windows; see
[dense-engine.md](dense-engine.md) for the exact workload and comparison command.

## Next integration steps

1. Close resource fit and timing for the integrated FP32 stream. Reduce duplicated
   lane control/storage as required; keep the tested binary32 arithmetic contract.
2. Connect the pinned cache and weight staging through a registered gather/frontend
   that supplies the stream's explicit metadata/layout. Verify boot identity,
   short reads, source stalls, and reload/reset behavior.
3. Add actor scheduling and output buffering, then prove the 10 kHz/1 kHz stream
   deadlines with bounded memory service. The proposed 203 KiB inner residency
   still needs about 1.103 GB/s of weight reads before overhead; this is a target,
   not a measured result. Inner W0/W2 residency remains unresolved.
4. Integrate the DDR/MIG/CDC backend and check checkpoint numerical parity when
   authorized model fixtures are available. There are no DDR pins, vendor IP
   wrappers, board constraints, Vivado block design or bitstream in this milestone.

Observation/history/action adapters and all motor-side interfaces are outside the
current user-requested inference stream scope.

These are experimental checkpoints, not a motor-qualified controller. This work
does not flash the Arty or connect actuator outputs.
