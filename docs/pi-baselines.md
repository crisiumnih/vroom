# PI baselines B0/B1 (master_plan PR4)

Truth drives simulation; estimates drive PI. Separate records, or the
baseline retunes itself.

## Records

- `warp_backend/estimates.py:ControllerEstimates` — what PI may know
  (`r/l/ke/j_rotor/j_load/load_b` hats). Gain rule `design_gains` is the
  declared `Kp=L·ωc` form, tested key-by-key against
  `benchmarks/bldc/cascaded.py:derived_gains`.
- **B0 (fixed):** `FROZEN_B0` — infra freeze at L0 + 1800 Hz / 5 Hz /
  damping 1.0. Reselected with M0 data before family training (new record).
  Same estimates object for every plant: perturbed truth cannot move it
  (`test_b0_frozen_across_plants`).
- **B1 (parameter-configured):** `b1_from_truth` fills motor hats from the
  actual simulated motor; attached load stays nominal (`j_load_hat` never
  matched to the case — D5-safe, tested).

## fyp threading (additive, legacy default kept)

`cascaded.py` takes optional `estimates` (`None` = historical behavior
bitwise). `DQCurrentController`/`CascadedPIController` feedforward uses
estimate inductance + estimate EMF; pole count stays structural.
`benchmarks/bldc/pi_estimates.py` holds the B0/B1 factories.
`evaluation.py:SpeedPI` untouched (pinned by closed protocols) — outer-PI
B0/B1 wiring is follow-up runner work.

## Benchmark matrix

`warp_backend/bench_matrix.py` freezes motors × cases × conditions ×
controllers × 3 fixed repetitions (seeds 5101–5103, mirrored signs on the
third). No execution — the work list can't silently change.

Repro: `.venv/bin/python -W ignore -m pytest warp_backend/test_estimates.py -q`
