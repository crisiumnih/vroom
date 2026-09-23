# Sensing pipeline N0/N1/N2 (master_plan PR3)

`warp_backend/sensing.py:SenseConfig` is the single measurement contract.
True simulator state is never controller input.

## Chain (per 100 µs inner step)

- Phase A/B: gain + offset + Gaussian noise, then first-order filter.
- Phase C: `-(A_s + B_s)` two-sensor reconstruction (N1/N2). N0 passes
  ground truth through exactly (legacy bitwise).
- Angle: fixed offset, then age buffer (N1: 5 steps, N2: 10 steps).
- Speed: passes through (no speed-sensor error in the declared table).
- Applied voltage: 1-sample transport delay (N1/N2); N0 applies immediately.
- Trip uses TRUE currents (protection); control, history and reward use
  sensed values (sensor-space MDP). Dead device lanes freeze their sensed
  frame; the numpy reference preserves its advance-dead-lanes behavior.

## Conditions

| | noise σ | gains A/B | offsets | filter τ | angle off/age | volt delay |
|---|---|---|---|---|---|---|
| N0 | 0 | 1, 1 | 0, 0 | 0 | 0 / 0 | 0 |
| N1 | .01·I_B | 1.005, .995 | ±.005·I_B | .10 ms | 0.5° / 5 | 1 |
| N2 | .02·I_B | 1.02, .98 | ±.02·I_B | .20 ms | 2° / 10 | 1 |

Fractions use a frozen `ib_a` (default 0.5 A); the runner sets the real I_B.
Backends take `sense=` (default N0); `VecEnv.reset(plants=...)` threads
motors, `sense` is construction-time.

## Gates

- N0 bitwise legacy (full suite green by default) + `sense == plant` check.
- numpy/torch sense math agrees to 1e-6 on shared inputs (noise injected by
  caller, so parity is exact despite different RNG streams).
- Determinism: same seed + actions → identical trajectories.
- Filter converges to gauged DC; config validation rejects negatives and
  delay > 1 sample.

Repro: `.venv/bin/python -W ignore -m pytest warp_backend/test_sensing.py -q`
