"""Trip parity: first step with max|i_abc|>4 under abusive dq action, Warp vs GEM."""
import sys
from pathlib import Path
import numpy as np
import warp as wp

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
from warp_backend import fullstep as fs
from warp_backend.plant import LEGACY_L0

wp.init()
DT = 1e-4
STEPS = 3000
RAMP = 0.5 / STEPS  # uq ramps 0 -> 0.5: gradual crossing discriminates solvers


def ramp_actions():
    acts = np.zeros((STEPS, 1, 2))
    acts[:, 0, 1] = RAMP * np.arange(STEPS)
    return acts


def warp_trip():
    d_s = wp.array(np.zeros((1, 5)), dtype=wp.float64, device="cuda:0")
    acts = ramp_actions()
    d_a = wp.array(acts, dtype=wp.float64, device="cuda:0")
    d_d = wp.zeros((STEPS, 1), dtype=wp.float64, device="cuda:0")
    d_tr = wp.zeros((STEPS, 1, 6), dtype=wp.float64, device="cuda:0")
    d_p = wp.array(np.ascontiguousarray([LEGACY_L0.row()]), dtype=wp.float64, device="cuda:0")
    wp.launch(fs.fullstep_kernel, dim=1,
              inputs=[d_s, d_a, d_d, wp.float64(DT), STEPS, d_p, d_tr], device="cuda:0")
    wp.synchronize()
    tr = d_tr.numpy()[:, 0, :]
    peak = np.abs(tr[:, 1:4]).max(axis=1)
    hit = np.argmax(peak > 4.0)
    assert peak[hit] > 4.0
    return hit, tr


def gem_trip():
    import json
    from benchmarks.bldc.environment import make_environment
    cfg = json.load(open(f"{GEM_ROOT}/benchmarks/bldc/config.json"))
    sc = dict(name="warp_trip", duration_s=STEPS * DT, reference=[[0.0, 0.0]], disturbance=[[0.0, 0.0]])
    env, _ = make_environment(cfg, sc)
    env.reset(seed=0)
    ps = env.unwrapped.physical_system
    names, lim = ps.state_names, np.asarray(ps.limits, dtype=float)
    idx = [names.index(k) for k in ["i_a", "i_b", "i_c"]]
    acts = ramp_actions()
    for k in range(STEPS):
        (state, _), *_ = env.step(np.ascontiguousarray(acts[k, 0]))
        if np.abs(np.asarray(state, dtype=float)[idx] * lim[idx]).max() > 4.0:
            env.close()
            return k
    env.close()
    raise AssertionError("GEM never tripped")


def test_trip_time_parity():
    w_hit, tr = warp_trip()
    g_hit = gem_trip()
    print(f"trip step: warp={w_hit} gem={g_hit} (dt=100us)")
    print(f"warp peak trajectory: {np.abs(tr[:w_hit+1, 1:4]).max(axis=1)[-3:]}")
    assert abs(w_hit - g_hit) <= 5
