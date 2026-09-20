"""Backend parity: WarpOuterBackend vs GEM OuterEnv, fixed case+actions."""
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

CASE = dict(name="parity_short", duration_s=0.2,
            reference=[[0.0, 0.0], [0.05, 5.0]],
            disturbance=[[0.0, 0.0], [0.1, 0.01]], stress=False)
INNER_DIR = f"{GEM_ROOT}/results/bldc/outer-inner-freeze-v1"
CFG = f"{GEM_ROOT}/results/bldc/outer-horizon-v2/g0995/config.json"
N_OUTER = 200


def load_pieces():
    import json
    from stable_baselines3 import DDPG
    from benchmarks.bldc.current_rl.repeat_study import read
    cfg = read(CFG)
    return cfg["plant"], cfg["inner_study"], DDPG


def test_backend_matches_outer_env():
    import json
    from stable_baselines3 import DDPG
    from benchmarks.bldc.current_rl.repeat_study import read
    from tools.outer_rl.environment import OuterEnv
    cfg = read(CFG)
    plant, study = cfg["plant"], cfg["inner_study"]
    inner_path = f"{GEM_ROOT}/results/bldc/outer-inner-freeze-v1/inner_model.zip"
    inner_cpu = DDPG.load(inner_path, device="cpu")
    inner_cuda = DDPG.load(inner_path, device="cuda")
    inner_cuda.policy.set_training_mode(False)
    for p in inner_cuda.policy.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(2)
    acts = rng.uniform(-1, 1, (N_OUTER, 1)).astype(float)

    genv = OuterEnv(plant, study, inner_cpu, [CASE])
    o, _ = genv.reset(seed=0, options={"case": CASE})
    go, gr, gt = [np.asarray(o, dtype=float)], [], []
    for k in range(N_OUTER):
        o, r, tm, tc, _ = genv.step(acts[k])
        go.append(np.asarray(o, dtype=float))
        gr.append(r)
        gt.append(tm or tc)
        if tm or tc:
            break
    go, gr = np.array(go), np.array(gr)

    be = WarpOuterBackend(plant, study, inner_cuda, [CASE], seed=0)
    bo, _ = be.reset(1, case=CASE)
    wo, wr = [np.asarray(bo, dtype=float).reshape(7)], []
    for k in range(N_OUTER):
        o, r, tm, tc, _ = be.step(acts[k].reshape(1, 1))
        wo.append(np.asarray(o, dtype=float).reshape(7))
        wr.append(float(r[0]))
        if tm[0] or tc[0]:
            break
    wo, wr = np.array(wo), np.array(wr)

    m = min(len(go), len(wo))
    print(f"steps: gem={len(go)} warp={len(wo)} term_gem={bool(gt and gt[-1])}")
    eo = np.abs(go[:m] - wo[:m])
    er = np.abs(gr[:m - 1] - wr[:m - 1])
    print("obs max abs:", eo.max(axis=0).round(6))
    print("rew max abs:", er.max() if len(er) else 0.0)
    assert m > 50, "episode ended too early on both sides"
    assert eo.max() < 0.05 and er.max() < 0.05


def test_backend_matches_horizon_env_l1():
    from unittest.mock import patch
    from stable_baselines3 import DDPG
    from benchmarks.bldc.current_rl.repeat_study import read
    from tools import outer_horizon_study as hz
    from tools.outer_rl import study as original
    from warp_backend.backend import WarpOuterBackend
    cfg = read(CFG)
    plant, study = cfg["plant"], cfg["inner_study"]
    inner_path = f"{GEM_ROOT}/results/bldc/outer-inner-freeze-v1/inner_model.zip"
    inner_cpu = DDPG.load(inner_path, device="cpu")
    inner_cuda = DDPG.load(inner_path, device="cuda")
    inner_cuda.policy.set_training_mode(False)
    for p in inner_cuda.policy.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(4)
    acts = rng.uniform(-1, 1, (150, 1)).astype(float)
    with hz.environment("g0995"):
        genv = original.OuterEnv(plant, study, inner_cpu, [CASE])
        o, _ = genv.reset(seed=0, options={"case": CASE})
        go, gr = [np.asarray(o, dtype=float)], []
        for k in range(150):
            o, r, tm, tc, _ = genv.step(acts[k])
            go.append(np.asarray(o, dtype=float))
            gr.append(r)
            if tm or tc:
                break
    be = WarpOuterBackend(plant, study, inner_cuda, [CASE], seed=0,
                          reward_shape="l1", failure=-5200.0)
    bo, _ = be.reset(1, case=CASE)
    wo, wr = [np.asarray(bo, dtype=float).reshape(7)], []
    for k in range(150):
        o, r, tm, tc, _ = be.step(acts[k].reshape(1, 1))
        wo.append(np.asarray(o, dtype=float).reshape(7))
        wr.append(float(r[0]))
        if tm[0] or tc[0]:
            break
    go, gr, wo, wr = (np.array(x) for x in (go, gr, wo, wr))
    m = min(len(go), len(wo))
    eo = np.abs(go[:m] - wo[:m])
    er = np.abs(gr[:m - 1] - wr[:m - 1])
    print(f"L1 steps: gem={len(go)} warp={len(wo)} obs={eo.max():.2e} rew={er.max():.2e}")
    assert m > 50
    assert eo.max() < 0.05 and er.max() < 0.5
