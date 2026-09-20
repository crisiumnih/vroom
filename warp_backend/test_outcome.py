"""Outcome parity: selected g0995 outer actor scores alike on both backends.

Same policy + same case, GEM OuterEnv (HorizonEnv patch) vs WarpOuterBackend.
Metric code identical both sides (outer-step obs traces). Recorded GEM
selection scores serve as ballpark anchor only.
"""
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

V2 = f"{GEM_ROOT}/results/bldc/outer-horizon-v2"
ACTOR = f"{V2}/g0995/128x2-seed18/best_model.zip"


def run_gem(plant, study, inner_cpu, case, actor_cpu, limit=1200):
    from tools.outer_rl.environment import OuterEnv
    env = OuterEnv(plant, study, inner_cpu, [case])
    o, _ = env.reset(seed=0, options={"case": case})
    sp, rf = [], []
    for _ in range(limit):
        a, _ = actor_cpu.predict(np.asarray(o, dtype=np.float32), deterministic=True)
        o, _, tm, tc, _ = env.step(a)
        o = np.asarray(o, dtype=float)
        sp.append(o[0] * 25)
        rf.append(o[1] * 25)
        if tm or tc:
            break
    env.close()
    return np.array(sp), np.array(rf)


def run_warp(plant, study, inner_cuda, case, actor_cuda, limit=1200):
    be = WarpOuterBackend(plant, study, inner_cuda, [case], seed=0,
                          reward_shape="l1", failure=-5200.0)
    o, _ = be.reset(1, case=case)
    import torch
    sp, rf = [], []
    for _ in range(limit):
        with torch.no_grad():
            a = actor_cuda.actor.forward(
                torch.as_tensor(np.asarray(o, dtype=np.float32).reshape(1, 7), device="cuda")).cpu().numpy()
        o, _, tm, tc, _ = be.step(a.reshape(1, 1))
        o = np.asarray(o, dtype=float).reshape(7)
        sp.append(o[0] * 25)
        rf.append(o[1] * 25)
        if tm[0] or tc[0]:
            break
    return np.array(sp), np.array(rf)


def test_selected_actor_scores_match():
    from stable_baselines3 import DDPG
    from benchmarks.bldc.current_rl.repeat_study import read
    from tools import outer_horizon_study as hz
    cfg = read(f"{V2}/g0995/config.json")
    plant, study = cfg["plant"], cfg["inner_study"]
    inner_path = f"{GEM_ROOT}/results/bldc/outer-inner-freeze-v1/inner_model.zip"
    inner_cpu = DDPG.load(inner_path, device="cpu")
    inner_cuda = DDPG.load(inner_path, device="cuda")
    inner_cuda.policy.set_training_mode(False)
    for p in inner_cuda.policy.parameters():
        p.requires_grad_(False)
    actor_cpu = DDPG.load(ACTOR, device="cpu")
    actor_cuda = DDPG.load(ACTOR, device="cuda")
    for cname in ["startup", "outer_dev_negative"]:
        case = next(c for c in cfg["validation_cases"] if c["name"] == cname)
        with hz.environment("g0995"):
            sp_g, rf_g = run_gem(plant, study, inner_cpu, case, actor_cpu)
        sp_w, rf_w = run_warp(plant, study, inner_cuda, case, actor_cuda)
        m = min(len(sp_g), len(sp_w))
        rmse_g = float(np.sqrt(np.mean((rf_g[:m] - sp_g[:m]) ** 2)))
        rmse_w = float(np.sqrt(np.mean((rf_w[:m] - sp_w[:m]) ** 2)))
        print(f"{cname}: steps gem={len(sp_g)} warp={len(sp_w)} "
              f"rmse gem={rmse_g:.4f} warp={rmse_w:.4f}")
        assert abs(len(sp_g) - len(sp_w)) <= 5, "episode length diverged"
        assert abs(rmse_g - rmse_w) / max(rmse_g, 0.1) < 0.05
