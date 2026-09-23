"""Phase 2a gate: Warp full control-step loop vs GEM env stepping.

Same open-loop normalized dq action sequence in both; compares
[omega, ia, ib, ic, eps, torque] traces with tolerance-based gates
(adaptive dopri5 vs fixed RK4 cannot be bitwise identical).
"""
import sys
from pathlib import Path
import numpy as np
import warp as wp

from warp_backend import fullstep as fs
from warp_backend.plant import LEGACY_L0

wp.init()
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

DT = 1e-4
STEPS = 500  # 0.05 s open loop


def warp_run(actions, dists):
    n = actions.shape[1]
    d_s = wp.array(np.zeros((n, 5)), dtype=wp.float64, device="cuda:0")
    d_a = wp.array(actions, dtype=wp.float64, device="cuda:0")
    d_d = wp.array(dists, dtype=wp.float64, device="cuda:0")
    d_tr = wp.zeros((STEPS, n, 6), dtype=wp.float64, device="cuda:0")
    d_p = wp.array(np.ascontiguousarray([LEGACY_L0.row() for _ in range(n)]),
                   dtype=wp.float64, device="cuda:0")
    wp.launch(fs.fullstep_kernel, dim=n,
              inputs=[d_s, d_a, d_d, wp.float64(DT), STEPS, d_p, d_tr], device="cuda:0")
    wp.synchronize()
    return d_tr.numpy()


def gem_run(actions):
    import json
    from benchmarks.bldc.environment import make_environment
    cfg = json.load(open(f"{GEM_ROOT}/benchmarks/bldc/config.json"))
    scenario = dict(name="warp_parity", duration_s=STEPS * DT,
                    reference=[[0.0, 0.0]], disturbance=[[0.0, 0.0]])
    env, _ = make_environment(cfg, scenario)
    env.reset(seed=0)
    sys_state = env.unwrapped.physical_system
    names = sys_state.state_names
    lim = np.asarray(sys_state.limits, dtype=float)
    idx = {k: names.index(k) for k in ["omega", "torque", "i_a", "i_b", "i_c", "epsilon"]}
    trace = np.zeros((STEPS, 6))
    for k in range(STEPS):
        (state, _ref), *_ = env.step(np.asarray(actions[k, 0], dtype=float))
        s = np.asarray(state, dtype=float) * lim
        trace[k] = [s[idx["omega"]], s[idx["i_a"]], s[idx["i_b"]], s[idx["i_c"]],
                    s[idx["epsilon"]], s[idx["torque"]]]
    env.close()
    return trace


def test_fullstep_matches_gem_open_loop():
    rng = np.random.default_rng(0)
    actions = np.zeros((STEPS, 1, 2))
    actions[:, 0, 1] = 0.1  # constant q voltage like a speed-demand step
    actions[200:, 0, 1] = -0.05
    dists = np.zeros((STEPS, 1))
    tw = warp_run(actions, dists)[:, 0, :]
    tg = gem_run(actions)
    tw[:, 4] = (tw[:, 4] + np.pi) % (2 * np.pi) - np.pi
    err = np.abs(tw - tg)
    scale = np.maximum(np.abs(tg).max(axis=0), [1.0, 0.1, 0.1, 0.1, 0.1, 1e-3])
    rel = (err / scale).max(axis=0)
    print("max abs :", err.max(axis=0))
    print("scaled  :", rel)
    assert rel[0] < 0.02 and rel[1:4].max() < 0.05 and rel[4] < 0.02 and rel[5] < 0.05
