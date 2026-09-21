# vroom working conventions

## Scope
Simulation only. No hardware integration, export, quantization, Arty flashing
without a new user instruction. Sim params are assumed, not measured motor specs.

## Evidence rules
- Document code alongside implementation: purpose, equations, units, assumptions, limits, repro commands.
- Distinguish implemented / tested / proposed / measured.
- Preserve raw traces + machine-readable metrics + readable summaries + figures.
- Record seeds, configs, versions, source + submodule hashes, completion/failure status.
- Never silently reset failed episodes, pad missing traces, overwrite result dirs, or tune from consumed final tests.
- Use identical plant/scenarios/eval for PI/RL comparisons.

## Learning constraints
- Final RL from scratch: no PI demos, imitation pretraining, teacher checkpoints, shadow PI state.
- PI is evaluation baseline only.
- Use pinned stable-baselines3; keep adapters local.
- Preserve observation order/scaling, memory/reset semantics, action limits, sample timing across train/eval.

## Harness correctness before expensive runs
- Accepted device implementation: `warp_backend/rollout.py:FastOuterBackend` (`vecenv.py: backend="device"`); `warp_backend/backend.py:WarpOuterBackend` is the unchanged reference for parity. Contract changes must land in both until the reference is retired.
- Read `plan/plan.md` and `plan/harness-performance.md` before changing the outer loop or rollout backend. Their proposed designs are not implemented contracts or permission to launch training.
- Resolve the actual runner, imported GEM path, interpreter, packages, and source/submodule/checkpoint hashes. Do not infer implementation from README paths or silently use a different working-tree fallback.
- Warp trains; GEM judges. Use a shared versioned controller contract for both, including history, memory gain, action transforms, filtering, reward, and resets. Exercise the real validation callback with nondefault settings before a long run.
- Validate configured plant parameters against kernel parameters. The current Warp kernel has nominal constants; reject unsupported configurations until runtime parameterization is implemented and verified.
- Test automatic episode resets through `step_wait`, with different scenarios/durations per lane. Explicit `reset()` tests alone do not establish training coverage. Preserve scenario IDs, per-lane RNG state, failure records, and terminal observations.
- Freeze each lane at its first termination/truncation, including plant and controller memory, while other lanes continue. Check mixed live/done batches. Use SB3's exact `TimeLimit.truncated` key; time limits bootstrap and physical failures do not.

## Performance engineering
- Profile before optimizing. Separate warmed rollout, learner updates, GEM validation, serialization, and startup/JIT time. Use synchronized wall timing for throughput, CUDA events for stream timing, and CPU/GPU profilers for attribution; profiler overhead is not production throughput.
- Report vector steps, total environment transitions, and executed inner physics steps separately. SB3 `num_timesteps` already counts all environments: do not divide by `n_envs` twice. Sum per-transition physics-step deltas, not cumulative episode counters.
- Compare performance at fixed total transitions and declared updates per transition. Changing `n_envs`, `train_freq`, `gradient_steps`, warm-up, validation cadence, noise, precision, or sample timing can change the learning experiment. Record critic and delayed actor updates separately.
- Target batched, preallocated controller state on the GPU. Avoid adding per-environment Python loops, `np.roll`, fresh device buffers, `.cpu()`, `.numpy()`, `.item()`, or device-wide synchronization inside the ten inner steps. Keep required host transfers at the SB3 outer-step boundary; diagnostic slow paths must be explicit.
- Reuse Warp/PyTorch views with compatible dtype, layout, device, and lifetime. Use a shared CUDA stream or explicit dependency events; removing synchronization without establishing ordering is incorrect. Copy mutable observations into replay/evaluation snapshots before their buffers are reused.
- Preserve all ten closed-loop inner actor/physics decisions and per-physics-step trip checks. Holding voltage for 1 ms, skipping inner inference, or checking limits only at the end is a controller change, not an equivalent speedup.
- Capture CUDA graphs only after warm-up, with stable shapes/addresses, device-side masks, and correct reset/terminal-state handling. Benchmark capture/compile startup separately. Keep FP64 physics and FP32 policy inference unless a separately declared precision study passes trajectory, trip, and gate checks; do not silently enable AMP/TF32/fast math.
- Start GPU benchmarks with one worker and controlled CPU thread counts; measure contention before increasing workers. More processes, environments, or a larger GPU are not automatically faster. Retain pinned dependencies and SB3; require a measured bottleneck and a separate migration comparison before adopting another learner or simulator stack.
- Use immutable policy snapshots and bounded workers for parallel GEM evaluation. Preserve every scheduled validation, failure, and selection barrier. Do not reduce evaluation coverage, trace evidence, or training updates to claim a speedup.
- Before an authorized long run, complete the relevant bounded correctness/parity and performance checks, estimate total transitions/updates/physics work and wall time, and freeze configuration. Record failed checks; do not repeatedly launch long training to debug adapters. Keep raw profiling evidence under ignored `results/`, with source/config hashes and reproduction commands.

## Repo hygiene
- `main` holds decided architecture only. No understanding/logging dumps, no bulk results, no PDFs.
- `third_party/gym-electric-motor` is a submodule; pin SHA per experiment, never edit in place.
- UI shows clean details + key graphs only.
