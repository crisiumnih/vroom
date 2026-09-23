# Benchmark waveforms (master_plan PR2)

`warp_backend/waves.py` is the single definition of every reference and
disturbance waveform (plan sect.6). Segments are plain tuples; both
simulators consume dense `to_schedule()` tables through their unchanged
table code, so cross-backend agreement holds by construction
(`test_tables_match_direct_sampling_both_consumers` checks every integer
step through the real GEM `schedule_value`).

- Speed cases S1–S6, disturbances D1–D3/D6, current cases C1–C4 (q/d pairs).
- Right-continuous: event at t0 applies at t >= t0; outside segments: 0.
- S4 = quadratic **reference** in time; D4 = quadratic **load law** in
  speed — never shared labels.
- `scale_waves(amp=0.9, time=1.25)` is the validation transform (sect.5).

Repro: `.venv/bin/python -W ignore -m pytest warp_backend/test_waves.py -q`
