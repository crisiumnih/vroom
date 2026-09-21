# The cascaded loop, plainly explained

Goal: spin a BLDC motor at a requested speed. Two controllers in series
("cascade"): a slow outer loop decides *how much current* the motor needs,
a fast inner loop decides *what voltage* produces that current.

```text
speed reference r ──► [OUTER: speed PI or speed RL, 1 kHz]
                           │  outputs iq* (id* = 0 always)
                           ▼
                    [INNER: current PI or current RL, 10 kHz]
                           │  outputs vd, vq (voltage)
                           ▼
                    [GEM BLDC plant: 44.4 V, 21 pole pairs,
                     85 mΩ, 50 µH — ASSUMED sim values, not measured]
                           │  outputs speed ω, currents, torque
                           └──── feedback to both loops
```

The inner loop runs 10× per outer decision (100 µs vs 1 ms). Limits:
`iq*` bounded to ±1.5 A, phase-current trip at 4 A (episode fails).

## The blocks and their exact I/O

### 1. Outer speed PI (baseline)
Classic PI on speed error `e = r − ω`, natural-frequency tuned, with
anti-windup from the inner loop's saturation flag. Outputs `iq* ∈ ±1.5 A`.

### 2. Outer speed RL (the thing we're trying to make work)
A DDPG/TD3 actor, run at 1 kHz, held constant for 10 inner steps.

**Input (7 numbers, all normalized to ±1):**

| # | signal | scale |
|---|--------|-------|
| 1 | speed ω | /25 |
| 2 | reference r | /25 |
| 3 | error r−ω | /25 |
| 4 | d-current id | /4 |
| 5 | q-current iq | /4 |
| 6 | previous own command | /1.5 |
| 7 | bounded error memory z | ±1 (already bounded) |

**Output:** one number `a ∈ [−1, 1]` → command `iq* = 1.5·a`, `id* = 0`.

**Memory update (the "integral" channel), once per outer action:**

```text
z ← clip( z + n·0.0001·clip((r−ω)/25) / τm , −1, 1 )     (n = 10 steps normally)
```

v1 contract: `τm = 0.5`. v2 contract (fix screens): `τm = 0.05` (10× gain).
Measured problem: with τm = 0.5 and real errors (en ≈ 0.01), `z` grows
0.002·en per action — over a full 2000-action episode `|z|` never exceeded
**0.018**, contributing cost ≤ 2e-5/step. The actor effectively never sees it.

**Reward (cost minimized), per 100 µs physics step, averaged over the hold:**

```text
speed:    4·clip(e/25)²            (+ L1 arms add 4·(|en| − en²))
current:  0.5·clip(hypot(id,iq)/4)²
delta:    0.1·clip(|Δcmd|/3)²      (one command change charged over all 10 hold steps)
memory:   0.5·z²  (v1)  /  0.005·z² (v2, identical pre-clip cost by construction)
torque:   0.1·clip(|Δtorque|/0.1)²
reward = −mean(costs) ;  −5200 on current-trip termination
```

qeff arms replace the speed term with `100·en²` (equals L1 cost at en = 0.04,
smooth gradient at zero). Correction economics at a typical 0.26 rad/s bias:
holding costs ~0.083/step; one full corrective swing costs 0.025/step delta
plus torque-transient penalties plus continued speed cost through the
80–180 ms transient — the objective can prefer the bias. Rational actor,
wrong incentives.

**Learners tried:** DDPG then TD3 (SB3 defaults otherwise: lr 1e-4, γ .995,
τ .005, batch 128, buffer 300k, 5k warmup steps, noise σ .05). Actor 2×128,
critic 2×256. Updates on GPU; inner rollouts on CPU (GEM) or Warp-CUDA.

### 3. Inner current PI (baseline)
dq-axis PIs with feedforward on the same 1.5 A refs, common voltage limit.

### 4. Inner current RL (qualified, frozen)
DDPG 3×256, 25-observation direct-v1 encoder (dq currents/refs/errors,
previous outputs, current history, speed, angle sine/cosine). Two raw outputs
scaled ×0.12 and projected onto the radius-0.12 voltage disk. Seed 9/500k
checkpoint frozen (copied + checksummed; any hash move aborts training).
**Passed 6/6 new current cases and 5/5 hybrid speed cases at both seeds.**
This half of the cascade works. Only the outer loop is under development.

### 5. The plant (what "simulation" means here)
GEM phase-variable BLDC, trapezoidal back-EMF, averaged inverter, 100 µs
steps. Parameters are legacy assumed values, NOT measured 2805 specs.
All PI/RL comparisons use identical plant, scenarios, limits, and metrics.

## Scoreboard: everything tried on the outer loop, and why each failed

| Screen | idea | result | why it failed (measured) |
|---|---|---|---|
| capacity (8 runs) | width sweep 128→1024 | 0/6 all | relay/lazy split from the start |
| reward L2/L1 | L1 tracking cost | L1 better (0.75 vs 1.16 RMSE) but 0/6 | L1 kink at zero implicated in chatter |
| horizon v1/v2 | γ .995 vs .999 | stopped v1; v2 winner γ.995 by violations, 0/6 | no horizon effect (0.71 vs 0.72) |
| ceiling (Warp) | 500k, 3 widths | stopped 55% | superseded, not failed — mechanism found first |
| fix (memory gain) | 10× integral signal | GEM run killed ~20%; Warp rerun 0–1/6 | gain helped less than predicted; relay half dominates once bias shrinks |
| td3 | TD3 learner | 0.87/1.07, 0/6 | algorithm not the bottleneck |
| qeff | quadratic speed cost | 0.85/0.88, 0/6 | reward shape not the bottleneck |
| smooth | low-pass on iq* (α=0.5) | **0.70 (18 viol) / 1.64** | best-ever + collapse: smoothing enables tracking (seed30) but finding it is a lottery (seed31) |

**The failure signature across all screens:** the actor converges to one of
two attractors — rail-to-rail relay (up to 823 sign flips, never settles) or
flat-lazy (2% of current authority used, steady bias up to 0.43 rad/s).
Nothing tried so far moves both. Width, budget, algorithm, reward shape, and
discount all read ~0 effect; only smoothing changed the picture (and only for
one seed). Best outer checkpoint ever: smooth seed30, RMSE 0.70, 18
violations — tracks (tail errors 0.02–0.16) but won't sit still
(8 unsettled + 8 tail-bias events).

## What next (in order)

1. **Settling-focused screen**: the frontier is now tracking ✓ / settling ✗.
   Candidates, one variable each: stronger smoothing (α 0.5→0.3),
   explicit tail-stillness reward term, or history-window observations
   (3–4 past error stamps + decaying integrator per thesis Eq.19).
2. **Reliability**: 1-of-2 seeds finding the good attractor is a lottery.
   More seeds per screen to measure hit rate; larger exploration noise
   (σ 0.05 is tiny) to stop premature collapse.
3. **Fresh qualification**: any 6/6 both-seeds triggers frozen fresh cases;
   only a pass there opens the `main` architecture PR.
4. **Standing rules**: GEM never trains (Warp trains, GEM judges); one
   variable per screen; failures block selection; no tuning from consumed
   development cases.
