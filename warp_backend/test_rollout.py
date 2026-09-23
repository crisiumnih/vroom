"""Device rollout parity + contract checks (old backend is the reference)."""
import sys
from pathlib import Path
import numpy as np

import os
_here = Path(__file__).resolve()


def _gem_root():
    sub = _here.parents[1] / "third_party" / "gym-electric-motor"
    if (sub / "benchmarks" / "bldc" / "current_rl" / "core.py").exists():
        return str(sub)
    env = os.environ.get("VROOM_GEM_PATH")
    if env and Path(env).exists():
        return env
    return "/home/sra/prajwal/fyp/gym-electric-motor"


GEM_ROOT = _gem_root()
sys.path.insert(0, GEM_ROOT)
from warp_backend.backend import WarpOuterBackend
from warp_backend.rollout import FastOuterBackend

CASE = dict(name="rollout_parity", duration_s=0.3,
            reference=[[0.0, 0.0], [0.05, 5.0]],
            disturbance=[[0.0, 0.0], [0.1, 0.01]], stress=False)
CFG = f"{GEM_ROOT}/results/bldc/outer-horizon-v2/g0995/config.json"
INNER = f"{GEM_ROOT}/results/bldc/outer-inner-freeze-v1/inner_model.zip"


def _pieces():
    import json
    from stable_baselines3 import DDPG
    from benchmarks.bldc.current_rl.repeat_study import read
    cfg = read(CFG)
    inner = DDPG.load(INNER, device="cuda")
    inner.policy.set_training_mode(False)
    for p in inner.policy.parameters():
        p.requires_grad_(False)
    return cfg["plant"], cfg["inner_study"], inner


def _run(be, acts):
    r0 = be.reset(acts.shape[1] if acts.ndim == 3 else 4, case=CASE)
    obs = r0[0] if isinstance(r0, tuple) else r0
    O, R, D = [np.asarray(obs, dtype=float).reshape(-1, 7)], [], []
    for k in range(len(acts)):
        o, r, tm, tc, info = be.step(acts[k])
        O.append(np.asarray(o, dtype=float).reshape(-1, 7))
        R.append(np.asarray(r, dtype=float))
        D.append(np.asarray(tm) | np.asarray(tc))
        if D[-1].all():
            break
    return np.array(O), np.array(R), np.array(D)


def test_device_matches_numpy_default():
    plant, study, inner = _pieces()
    rng = np.random.default_rng(2)
    acts = rng.uniform(-1, 1, (200, 4, 1))
    old = WarpOuterBackend(plant, study, inner, [CASE], seed=0)
    new = FastOuterBackend(plant, study, inner, [CASE], seed=0)
    go, gr, gd = _run(old, acts)
    wo, wr, wd = _run(new, acts)
    assert len(go) == len(wo)
    eo, er = np.abs(go - wo).max(), np.abs(gr - wr).max()
    print(f"default: obs={eo:.2e} rew={er:.2e} steps={len(go)}")
    assert (gd == wd).all()
    assert eo < 1e-4 and er < 1e-4


def test_device_matches_numpy_nondefault():
    plant, study, inner = _pieces()
    kw = dict(reward_shape="qeff", failure=-5200.0, effort_scale=5.0,
              memory_divisor=0.05, memory_cost=0.2, smooth_alpha=0.5)
    rng = np.random.default_rng(4)
    acts = rng.uniform(-1, 1, (200, 4, 1))
    old = WarpOuterBackend(plant, study, inner, [CASE], seed=0, **kw)
    new = FastOuterBackend(plant, study, inner, [CASE], seed=0, **kw)
    go, gr, gd = _run(old, acts)
    wo, wr, wd = _run(new, acts)
    assert len(go) == len(wo)
    eo, er = np.abs(go - wo).max(), np.abs(gr - wr).max()
    print(f"nondefault: obs={eo:.2e} rew={er:.2e} steps={len(go)}")
    assert (gd == wd).all()
    assert eo < 1e-4 and er < 1e-4


def test_device_deterministic():
    plant, study, inner = _pieces()
    rng = np.random.default_rng(6)
    acts = rng.uniform(-1, 1, (120, 4, 1))
    a = FastOuterBackend(plant, study, inner, [CASE], seed=0)
    b = FastOuterBackend(plant, study, inner, [CASE], seed=0)
    go, gr, _ = _run(a, acts)
    wo, wr, _ = _run(b, acts)
    assert np.array_equal(go, wo) and np.array_equal(gr, wr)


def test_truncation_freezes_terminal_obs():
    """A lane done early must freeze while others advance (reference: advances)."""
    import torch
    plant, study, inner = _pieces()
    acts = np.full((150, 2, 1), 0.1)
    new = FastOuterBackend(plant, study, inner, [CASE], seed=0)
    new.reset(2, case=CASE)
    new.dur[1] = 1000  # physics steps: lane 1 ends after ~100 outer steps
    new._dev["dur"][1] = 1000
    O = [np.asarray(new._read_obs(), dtype=float).reshape(-1, 7)]
    D = []
    for k in range(len(acts)):
        o, _, tm, tc, _ = new.step(acts[k])
        O.append(np.asarray(o, dtype=float).reshape(-1, 7))
        D.append(np.asarray(tm) | np.asarray(tc))
    O, D = np.array(O), np.array(D)
    k1 = int(np.argmax(D[:, 1]))
    assert D[k1, 1] and not D[k1, 0] and k1 < 140
    tail = O[k1 + 1:, 1]
    assert bool((tail == tail[0]).all()), "done lane must freeze"
    assert not bool((O[k1 + 1:, 0] == O[k1 + 1, 0]).all()), "live lane must advance"
    old = WarpOuterBackend(plant, study, inner, [CASE], seed=0)
    o0, _ = old.reset(2, case=CASE)
    old.dur[1] = 1000
    Go = [np.asarray(o0, dtype=float).reshape(-1, 7)]
    for k in range(len(acts)):
        o, _, tm, tc, _ = old.step(acts[k])
        Go.append(np.asarray(o, dtype=float).reshape(-1, 7))
        if (np.asarray(tm) | np.asarray(tc)).all():
            break
    Go = np.array(Go)
    assert not np.array_equal(Go[k1 + 1, 1], Go[k1 + 2, 1]), \
        "reference advances done lanes (repaired defect)"


def test_masked_kernel_freezes_dead_lanes():
    import torch, warp as wp
    from warp_backend import fullstep as fs
    from warp_backend.plant import LEGACY_L0
    wp.init()
    n = 8
    s0 = np.random.default_rng(0).uniform(-1, 1, (n, 5))
    d_s = wp.array(s0.copy(), dtype=wp.float64, device="cuda:0")
    d_a = wp.array(np.full((n, 2), 0.05), dtype=wp.float64, device="cuda:0")
    d_d = wp.zeros(n, dtype=wp.float64, device="cuda:0")
    d_l = wp.array(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32), device="cuda:0")
    d_o = wp.zeros((n, 6), dtype=wp.float64, device="cuda:0")
    d_p = wp.array(np.ascontiguousarray([LEGACY_L0.row() for _ in range(n)]),
                   dtype=wp.float64, device="cuda:0")
    for _ in range(5):
        wp.launch(fs.fullstep_masked, dim=n,
                  inputs=[d_s, d_a, d_d, d_l, wp.float64(1e-4), d_o, d_p], device="cuda:0")
    wp.synchronize()
    s1 = d_s.numpy()
    assert np.array_equal(np.asarray(s1[4:]), s0[4:]), "dead lanes must be bitwise frozen"
    assert not np.array_equal(np.asarray(s1[:4]), s0[:4]), "live lanes must advance"


def test_delta_counters():
    plant, study, inner = _pieces()
    new = FastOuterBackend(plant, study, inner, [CASE], seed=0)
    new.reset(2, case=CASE)
    rng = np.random.default_rng(8)
    total = 0
    for k in range(30):
        _, _, _, _, info = new.step(rng.uniform(-1, 1, (2, 1)))
        for j in range(2):
            assert info[j]["inner_steps"] <= 10
            total += info[j]["inner_steps"]
            assert info[j]["physics_steps"] == new.phys[j]
    assert total == int(new.phys.sum())
    assert total <= 30 * 2 * 10


def test_plant_rejection():
    import copy
    plant, study, inner = _pieces()
    bad = copy.deepcopy(plant)
    bad["motor"]["r_s"] = -0.01
    with __import__("pytest").raises(ValueError, match="r_s"):
        FastOuterBackend(bad, study, inner, [CASE], seed=0)


def test_non_nominal_supply_accepted():
    """Nominal-only restriction removed (master_plan PR1): 48 V is physical."""
    import copy
    plant, study, inner = _pieces()
    ok = copy.deepcopy(plant)
    ok["supply_v"] = 48.0
    FastOuterBackend(ok, study, inner, [CASE], seed=0)
