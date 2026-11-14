# kinloop (`vroom`) — working conventions

Instructions for anyone, human or coding agent, changing this repo. Read
`README.md` first for what the project is; this file says how to work on it.

## What this repo is

One learned speed–current cascade for a defined family of BLDC motors:
- `warp_backend/`, `rl/`, `populations/`: GPU training and the simulation contract.
- `third_party/gym-electric-motor`: the CPU judge for selection and evaluation (submodule).
- `hw/focmini/`: ESP32 + SimpleFOCmini testbed firmware.
- `hw/arty/`: FP32 actor inference on an Arty A7-35T.

All three areas (simulation, ESP32, FPGA) are in scope. Quantization,
retraining for hardware, or a new board are *not*; they need an explicit
decision from the team first.

## Claims and evidence

Every statement in code comments, docs, figures or PRs must say which of these it is:
**implemented**, **tested in simulation**, **measured on hardware**, or **proposed**.
- Simulation parameters are assumed, not measured motor specifications, until
  the identification results say otherwise. Simulation results are never hardware results.
- Numerical agreement on the FPGA is not closed-loop control; a 100 MHz clock is
  not a control-period measurement. ESP32 bring-up is not policy deployment.
- Cite primary sources for external methods (link them, never invent page or
  equation numbers). Label our own design choices as ours.
- Keep README status table, docs and figures in sync with the code. If they
  disagree, write the discrepancy down rather than picking one silently.

## Experiments

- Preserve raw traces, machine-readable metrics, readable summaries and figures.
  Raw evidence lives in ignored `results/`; never commit it.
- Record seeds, full config, package versions, source and submodule hashes, and
  completion/failure status for every run.
- Define metric windows, units, thresholds and missing-value rules before
  looking at results.
- Never silently reset failed episodes, pad missing traces, drop failed cases,
  or overwrite an existing result directory.
- PI vs RL comparisons use the identical plant, scenarios, observations, action
  limits and evaluation code. Tune on development cases only; final test cases
  and populations are used once and never fed back into tuning.
- Different seeds of a deterministic run are repeatability checks, not
  statistical robustness.

## Learning rules

- Final controllers learn from scratch by interacting with simulated motors:
  no PI demonstrations, imitation, teacher checkpoints or shadow PI state.
  PI is an evaluation baseline only.
- Train the current policy first and freeze it while the speed policy learns.
- Use pinned Stable-Baselines3 (`requirements.txt`); keep adapters local.
- Keep observation order and scaling, memory/reset semantics, action scaling
  and sample timing identical across training, evaluation and hardware. Version
  any change to that contract.

## Simulation harness

- **Warp trains, GEM judges.** Training, validation and evaluation share one
  versioned controller contract (history, memory gain, action transforms,
  filtering, reward, resets).
- The accepted device rollout is `warp_backend/rollout.py:FastOuterBackend`
  (`vecenv.py`, `backend="device"`). `warp_backend/backend.py:WarpOuterBackend` is
  the unchanged parity reference; contract changes land in both until it is retired.
- Validate per-lane plant parameters before use. Test automatic resets through
  `step_wait` with mixed live/done lanes; freeze each lane at its first
  termination. Use SB3's `TimeLimit.truncated`: time limits bootstrap, physical
  failures do not.
- Keep all ten inner decisions and per-physics-step trip checks per outer step.
  Holding voltage for 1 ms or checking limits only at the end changes the controller.
- Keep FP64 physics and FP32 policy inference; no silent AMP/TF32/fast math.
- Before any long run: pass the relevant parity and correctness tests, estimate
  transitions/updates/wall time, freeze the config. Do not debug by repeatedly
  launching long training.

## Performance work

- Profile before optimizing. Report warmed rollout, learner updates, GEM
  validation and startup/JIT time separately, using synchronized timing.
- Count vector steps, environment transitions and inner physics steps
  separately; SB3 `num_timesteps` already includes all environments.
- Compare at fixed total transitions and update ratio. Changing `n_envs`,
  `train_freq`, `gradient_steps`, noise, precision or validation cadence changes
  the experiment, not just the speed.
- No per-environment Python loops, fresh device buffers or host syncs
  (`.cpu()`, `.item()`, `.numpy()`) inside the inner steps; host transfers belong
  at the SB3 outer-step boundary.
- Never cut evaluation coverage, trace evidence or training updates to report a speedup.

## Hardware

- **ESP32 (`hw/focmini/`).** Bench supply with a current limit, motor clamped in
  the fixture, firmware safeguards enabled. Spin commands need explicit
  confirmation (see `run_motor.sh`). Run `hw/focmini/tests/` after changing
  control or parsing code. Wiring changes go into the README diagram.
- **FPGA (`hw/arty/`).** Actors stay binary32 with the trained weights; never
  alter parameters to make a check pass. Acceptance is absolute error ≤ 1e-5
  against the software reference. Report timing only from routed results on the
  target part, and keep failed probes.
- Inputs to either board must use the same observation contract as simulation.

## Repo hygiene and PRs

- Work on a branch and open a PR. Keep `main` to decided architecture.
  Before review: tests for the area touched pass and `git diff --check` is clean.
- Never commit `results/`, model zips, `*.pkl`, replay buffers, logs, PDFs or
  bulk data. Small README figures go in `docs/figs/` (prefer SVG).
- `third_party/gym-electric-motor` is a submodule; pin its SHA per experiment
  and never edit it in place. `third_party/hardfloat` is vendored; keep its licence.
- `rl_legacy_cpu/` is superseded reference code; do not extend it.
- Document substantive code alongside it: purpose, equations, symbols with SI
  units, assumptions, configuration, limitations and reproduction commands.
