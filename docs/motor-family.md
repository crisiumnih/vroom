# Motor family: simulator support (master_plan PR1)

`warp_backend/plant.py:MotorParams` is the single struct both backends answer
to. Until M0 is measured, tests use Legacy-L0-relative multipliers (infra
fixtures, not the family).

## Field meanings (match the GEM fork)

| Field | Meaning |
|---|---|
| `p` | integer pole pairs (v1: fixed per family) |
| `r_s` | equivalent phase resistance, ohm |
| `l_s` | equivalent phase inductance `L−M`, henry |
| `k_e` | back-EMF amplitude on **mechanical** rad/s, V·s/rad |
| `j_rotor` / `j_load` | split inertias, kg·m²; physics uses `j_total` sum |
| `supply_v` | DC bus, V (`u_nominal` in GEM supply, `0.5·U_SUP` phase factor in Warp) |
| `load_a/b/c` | static / viscous / quadratic coefficients (`DisturbedLoad`) |
| disturbance | signed external torque **subtracted** from shaft torque |

Friction smoothing (1 ms) is fixed by the plan; `W_LIM`/`LIN_F` derive per
lane from (`load_a`, `j_total`). Back-EMF shape is the shared trapezoid
until M0 admission decides otherwise — smoothing it is not membership.

## What changed in Warp

- `fullstep.py`: no nominal constants; `full_rhs` + all three kernels take a
  per-lane `[n, 11]` params table. Torque helpers use per-lane `k_e`.
- `reset(n, case, seed, plants)`: one plant broadcasts; a list sets per-lane
  motors. `VecEnv.reset(plants=...)` threads it through. Per-episode motor
  draws are runner work (later PR), not backend work.
- `validate_plant` is structural only (finite/positive); the nominal-only
  restriction is gone (`test_non_nominal_supply_accepted`).

## Parity scope (measured)

- 4 corners (nominal, fast-electrical, heavy-slow, weak-magnet-low-bus):
  device vs GEM agree within the `test_fullstep` gates **while both stay
  under the 4 A trip**, with matching trip steps (±5). Single-step agreement
  ~1e-4 in all corners — no semantic gap.
- Runaway past trip is out of parity scope: integrators separate in regimes
  no controller may enter.
- dt check at fastest corner (τe ≈ 255 µs): 100 µs RK4 vs 2×50 µs differ
  ~2e-4 relative — 10× below the cross-simulator bar. No substepping needed.

Repro: `.venv/bin/python -W ignore -m pytest warp_backend/test_plant.py -q`
