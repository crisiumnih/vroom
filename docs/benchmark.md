# Rollout backend benchmark: numpy reference vs device loop

Accepted numbers from the clean-GPU sweep in
`results/reviews/rollout-opt-20260921/` (ignored local evidence; see
`metrics.json` there plus the archived contended runs). Fixture: frozen
inner policy, zero start, fixed outer action 0.1, constant 3 rad/s
reference, no episode ends inside windows; 10 warm-up steps, 3×20-step
windows, synchronized wall timing.

Environment: RTX 3060 12 GB, CUDA 13.0, driver 580.178.04, torch
2.13.0+cu130, Warp 1.17.0, SB3 2.9.0, NumPy 2.5.2, one process, one
PyTorch CPU thread.

## Headline

| Envs (N) | numpy ms/vector step | device ms/vector step | Speedup |
|---:|---:|---:|---:|
| 8 | 7.51 | 8.14 | 0.92× |
| 16 | 10.25 | 8.46 | 1.21× |
| 32 | 15.83 | 8.50 | 1.86× |
| 64 | 25.85 | 8.53 | 3.03× |

The device path is flat at ~8.5 ms/vector step: per-transition cost falls
linearly with N. At N=8 it is at parity (marginal loss, honestly reported);
wins start at N≈16 and grow with batch size.

Reproduce (refuses to overwrite `metrics.json`):

```bash
.venv/bin/python results/reviews/rollout-opt-20260921/bench_opt.py
```

## Where the speedup comes from (all measured, in order)

1. **Python lane loops → batched torch ops.** cProfile of the reference
   showed ~71% of wall time in per-lane `CurrentEncoder.encode`/`action`
   calls (9,600 calls per 30 vector steps at N=32). The device loop keeps
   history, previous outputs, clocks, and masks as `(N, …)` tensors.
2. **~50 host/device transfers + ~20 allocations per outer step → ~zero.**
   Warp arrays are views over preallocated torch storage
   (`wp.from_torch`, created once per reset); voltages, disturbances, and
   outputs reuse the same buffers every step.
3. **10 device-wide synchronizations per outer step → 1.** One shared
   stream (torch `ExternalStream` wrapping the Warp stream handle); a
   single `wp.synchronize()` at the SB3 outer-step boundary.
4. **~1,800 kernel launches per outer step → ~500.** Fused
   `torch.compile` regions (encode + frozen actor + project in one graph,
   reward in another) instead of ~40 tiny eager ops per inner step;
   `torch.where` instead of masked scatters (each scatter had emitted a
   multi-kernel cub select/reduce sequence); roll-based history instead of
   per-lane `np.roll` + fancy-index gathers; hand-written 2-norm instead
   of `linalg.norm`; elementwise trip check instead of `amax`.
5. **Dynamo per-call overhead → graph replay.** `eval_frame` guard
   evaluation cost ~110 µs per compiled call (~3.3 ms/step across 30
   calls); `reduce-overhead` mode replays them as CUDA graphs.

## How we know each claim (instruments, not vibes)

- nsys trace: ~51 ms of real GPU work inside ~250 ms+ of wall time
  across 17 outer steps — the workload was CPU/launcher-bound, so
  launch-count reduction (not kernel tuning) was the correct target.
- nsys runtime table: ~30k `cudaLaunchKernel` calls (~149 ms CPU-side)
  before the fusion work.
- cProfile after fusion: no single hotspot remains; cost is diffuse
  across hundreds of individually cheap dispatches.
- Parity held throughout: old-vs-new rollout observations agree to
  ~1e-7, rewards to ~1e-8 (27-test suite green), so none of the
  restructuring changed the controller math.

## What did NOT help, and what was rejected

- `max-autotune` inductor mode: benchmarks kernel configs by timing, so
  a contended machine can lock bad choices into its cache. Default
  inductor + graphs instead.
- Benchmarking on a shared box: two full sweeps were archived as
  `metrics.json.contended-*` after a rival job was found pinning CPU/VRAM
  mid-run. Timing evidence is only accepted from a free GPU.
- Further fusion (whole 10-step loop in one graph): poor ROI once the
  profile went diffuse, plus real regression risk against the parity
  gates. Parked, not pursued.

## Limits of these numbers

- Rollout only: learner updates, GEM validation, and serialization are
  separate costs, measured separately per AGENTS.md.
- Fixed-command fixture: no resets, failures, or reference events inside
  windows; reset-heavy and failure fixtures are separate checks.
- N=8 shows no win: training at 8 environments should not claim a
  rollout speedup from this work.
