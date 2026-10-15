# Trained FP32 inference on Arty A7-35T

This board adapter runs the existing `vroom_actor_stream` using actual learned
parameters and prepared observations. It connects only the 100 MHz oscillator
and the onboard FTDI UART. The actor remains binary32, with two 256-wide ReLU
hidden layers and two tanh outputs; there is no quantization or retraining.

For each row, the unchanged stream computes, in ascending input-column order:

```
sum = bias
product = RN32(weight * input)
sum = RN32(sum + product)
```

The package's golden outputs use its original software matrix operations.
`prepare.py` preserves that reference and separately builds exact-order RTL
fixtures. Acceptance against package outputs remains absolute error <= 1e-5,
relative tolerance zero. Intermediate dense-layer RTL values must match the
exact-order reference bit for bit. The host never changes model parameters to
make these checks pass.

## Scope and timing

The inner actor takes 25 prepared FP32 values, the outer takes 19. Both return
two raw FP32 outputs. Normalization, history, controller memory, action mapping,
and motor interfaces remain outside this input-to-output test, as specified in
the [inference contract](../streaming-architecture.md#current-delivery-scope-inference-input-to-output).

The top uses the existing tested 12-lane implementation and a 1 Mbaud 8N1 UART.
Every job streams its complete parameters; no DDR or weight cache is attached.
A request takes roughly three seconds to transport. The returned cycle counter
includes that source starvation and is **not** a ready-source compute-latency
measurement or proof of the 10 kHz/1 kHz control rates.

`build.tcl` targets `xc7a35ticsg324-1L`, checks actual IO constraints and routed
setup/hold timing at 100 MHz, and keeps Vivado's bitstream DRC errors enabled.
The three pin assignments come from the official
[Digilent master constraints](https://github.com/Digilent/digilent-xdc/blob/master/Arty-A7-35-Master.xdc).
Unused IO has no pull resistor configured. `program.tcl` selects the cable
serial and 35T device, loads volatile configuration via JTAG, and checks DONE,
configuration CRC status and the `VRM2` USR_ACCESS signature. SPI flash is
preserved; power cycling restores the previously stored configuration.

## Wire protocol

Requests consist of `VRM1`, actor (u8, inner=0/outer=1), job ID (u32), input
words, then the biases and weights for each of three layers, then CRC32 (u32).
All multi-byte fields and FP32 bits are little endian. Bias groups are ordered
by group then lane. Weight groups are ordered by column, group, then lane.
The lane-zero word occupies the low 32 bits. Tail lanes carry transport zeros
and are masked by the core; no observation or missing trace is padded.
CRC32 uses the standard zlib/IEEE reflected convention and covers actor through
the last weight. The host independently verifies checkpoint SHA256.

Replies are exactly 32 bytes:

| Byte | Field |
| --- | --- |
| 0..3 | `VRO1` |
| 4..7 | version=1, actor, fault, arithmetic flags (one byte each) |
| 8..11 | job ID |
| 12..19 | two raw FP32 outputs |
| 20..23 | cycles from job admission to output, including UART starvation |
| 24..27 | received request CRC32 |
| 28..31 | CRC32 of reply bytes 0..27 |

Only complete, CRC-valid, correctly tagged replies count as tests. Errors stop
the host without retries, flushing or reconstructed replies. Bridge faults are
sticky and require an explicit reprogram/reset; good jobs do not reset the core.
UART framing, buffer overrun, bad CRC, invalid actor, and output identity errors
use codes 0x80..0x85; codes 1..5 come from the inference stream. Ordinary FP32
inexact/underflow flags are reported; invalid/divide-by-zero/overflow fail.

The UART byte spacing is 1,000 core clocks. Core transitions and recurrence
stalls are bounded well below that spacing; a still-owned word/vector on arrival
of another byte produces an explicit overrun. The bridge does not offer general
unbounded source flow control or concurrent actor scheduling.

## Reproduce

Use fresh output directories throughout, with the supplied package extracted
under ignored `results/`. NumPy is pinned to the repo version (2.5.2); RTL checks
need Icarus, Verilator and the vendored HardFloat sources. Vivado 2024.2 was used
for the physical build. No SB3 training or plant simulation is launched.

```
python prepare.py --package PACKAGE --out FIXTURES
python simulate.py --fixtures FIXTURES --out RTL_EVIDENCE
# From a fresh evidence directory:
vivado -mode batch -nojournal -log build.log -source /absolute/path/to/board/build.tcl
vivado -mode batch -nojournal -log program.log -source /absolute/path/to/board/program.tcl -tclargs CABLE_SERIAL /absolute/path/to/build/vroom_inference.bit
python check.py --fixtures FIXTURES --port /dev/serial/by-id/BOARD_UART --out HARDWARE_EVIDENCE
```

The transport-isolation test reads complete request packets as `request0.bin`
and `request1.bin`, checks every stream word and metadata field with sink stalls,
and rejects a corrupted third packet. Its `replies.bin` is decoded by
`check.decode_reply`; test arithmetic is checked by the separate real-FP32
regression and physical board comparisons. The UART regression covers all 256
byte values in both directions at the production divider. Both testbenches live
under `tests/`, outside the synthesized source glob.

`check.py` acquires both POSIX flock (used by pyserial) and kernel TIOCEXCL.
An already-running reader can otherwise consume reply bytes despite TIOCEXCL.
Preserve failed attempts and stop conflicting readers explicitly before a new
run. Each attempt retains reply bytes, per-job JSONL and final metrics. The
prepared request files retain the full raw transmit stream. Result directories
also retain source/tool/checkpoint identities, layer traces, timing/utilization
reports and the bitstream hash.
