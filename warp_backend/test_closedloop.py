"""Closed-loop parity: PI speed controller stepping Warp vs GEM independently.

Startup: ref 0 for 0.1 s, then 10 rad/s. PI kp=0.001, ki=0.05, disk 0.12
(same as benchmarks/bldc config + evaluate projection).
"""
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
from benchmarks.bldc.control import PIController
from benchmarks.bldc.interfaces import project_voltage

wp.init()
DT = 1e-4
STEPS = 3000
HOLD = 1000


def run_warp_closed():
    d_s = wp.array(np.zeros((1, 5)), dtype=wp.float64, device="cuda:0")
    d_a = wp.zeros((1, 2), dtype=wp.float64, device="cuda:0")
    d_d = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    d_o = wp.zeros((1, 6), dtype=wp.float64, device="cuda:0")
    pi = PIController(0.001, 0.05, 0.12)
    trace = np.zeros((STEPS, 6))
    for k in range(STEPS):
        om = float(d_s.numpy()[0, 0])
        ref = 0.0 if k < HOLD else 10.0
        act = project_voltage(pi.act({"omega": om}, ref, DT), 0.12)
        d_a = wp.array(np.asarray(act, dtype=np.float64).reshape(1, 2), dtype=wp.float64, device="cuda:0")
        wp.launch(fs.fullstep_once, dim=1, inputs=[d_s, d_a, d_d, wp.float64(DT), d_o], device="cuda:0")
        trace[k] = d_o.numpy()[0]
    wp.synchronize()
    return trace


def run_gem_closed():
    import json
    from benchmarks.bldc.environment import make_environment
    cfg = json.load(open(f"{GEM_ROOT}/benchmarks/bldc/config.json"))
    sc = dict(name="warp_closed", duration_s=STEPS * DT, reference=[[0.0, 0.0]], disturbance=[[0.0, 0.0]])
    env, _ = make_environment(cfg, sc)
    env.reset(seed=0)
    ps = env.unwrapped.physical_system
    names, lim = ps.state_names, np.asarray(ps.limits, dtype=float)
    idx = {k: names.index(k) for k in ["omega", "torque", "i_a", "i_b", "i_c", "epsilon"]}
    pi = PIController(0.001, 0.05, 0.12)
    trace = np.zeros((STEPS, 6))
    for k in range(STEPS):
        om = trace[k - 1, 0] if k else 0.0
        ref = 0.0 if k < HOLD else 10.0
        act = project_voltage(pi.act({"omega": om}, ref, DT), 0.12)
        (state, _), *_ = env.step(np.asarray(act, dtype=float))
        s = np.asarray(state, dtype=float) * lim
        trace[k] = [s[idx["omega"]], s[idx["i_a"]], s[idx["i_b"]], s[idx["i_c"]],
                    (s[idx["epsilon"]] + np.pi) % (2 * np.pi) - np.pi, s[idx["torque"]]]
    env.close()
    return trace


def test_closed_loop_parity():
    tw = run_warp_closed()
    tg = run_gem_closed()
    tw[:, 4] = (tw[:, 4] + np.pi) % (2 * np.pi) - np.pi
    err = np.abs(tw - tg)
    scale = np.maximum(np.abs(tg).max(axis=0), [1.0, 0.5, 0.5, 0.5, 0.1, 0.01])
    rel = (err / scale).max(axis=0)
    print("max abs :", err.max(axis=0))
    print("scaled  :", rel)
    fin = np.abs(tw[-1] - tg[-1])
    print("final   :", fin)
    assert rel[0] < 0.05 and rel[1:4].max() < 0.10 and rel[4] < 0.05 and rel[5] < 0.10
