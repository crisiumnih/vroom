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

## Repo hygiene
- `main` holds decided architecture only. No understanding/logging dumps, no bulk results, no PDFs.
- `third_party/gym-electric-motor` is a submodule; pin SHA per experiment, never edit in place.
- UI shows clean details + key graphs only.
