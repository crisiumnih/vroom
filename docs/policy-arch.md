# Policy architecture map (measured, current use)

Shapes read off the frozen artifacts and runner configs — not guesses.
SB3 MLP policies, ReLU hidden, tanh-squashed outputs.

## Inner current loop (frozen DDPG, 10 kHz)

```
25 obs ──▶ [256 ──▶ 256 ──▶ 256] ──▶ tanh ──▶ 2 (v_d, v_q normalized)
```

- Actor: 25 → 256 → 256 → 256 → 2 (`inner_model.zip`: mu.0/2/4/6).
- Obs groups (25): dq current/4 (2), dq reference/4 (2), error/8 (2),
  prev outputs (4), current history (10), speed/20 (1), sin/cos angle (2),
  padding (2).
- Critic (training only): 27 (25 obs + 2 act) → 4×295 → 1.
- Output scaled by `action_norm_limit` → volts; frozen, grad off, eval mode.

## Outer speed loop (TD3, 1 kHz, training)

Direct arm (v2/v3 `direct-td3`, `iasa-l1` is IASA — see below):

```
7 obs ──▶ [128 ──▶ 128] ──▶ tanh ──▶ 1 (iq* / 1.5 A)
```

IASA arms (`iasa-td3`, `iasa-l1`, `iasa-candidate`):

```
19 obs ──▶ [128 ──▶ 128] ──▶ tanh ──▶ 2 (a_P direct, a_I increment)
                                              │            │
                                              ▼            ▼
                                         ×1.5 A      b += 0.015·a_I
                                              └────▶ sum, clip ±1.5,
                                                     anti-windup, 0.5 filter
                                                              │
                                                              ▼
                                                     iq* (id* = 0)
```

- 19 obs: 10 base (speed/ref/error/tanh-err/id/iq/bias/prev-cmd/sin/cos)
  + 9 lagged (5/10/20 ms). Full order: `rl/outer/contract.py:V3_OBS_ORDER`.
- Critics (both arms): obs+act → 2×256 → 1 (twin critics, delay-2 actor).
- Learning: lr 1e-4, γ 0.995, τ 0.005, buffer 300k, batch 128,
  8 envs × train_freq 1 × gradient_steps 8 (= 1 update/transition),
  exploration noise σ 0.05 per output.

## Cascade (rates)

```
19/7 obs @1kHz ──▶ OUTER actor ──▶ iq*, id*=0 ──▶ hold ×10 ──▶ INNER actor @10kHz ──▶ v_d,v_q
       ▲                                                              │
       └────────────── plant @10kHz (RK4, 100µs) ◀─────────────────────┘
```

Outer sees filtered command + bias as obs; inner sees dq currents, refs,
history. Neither sees motor parameters.

## What changes under master_plan

Sizes are development choices, not frozen: family training re-selects
capacity by validation; FPGA section will shrink/quantize the INNER actor
(the 3×256 → ≤25 µs path). This page tracks `main` — update it when the
contract or sizes change.
