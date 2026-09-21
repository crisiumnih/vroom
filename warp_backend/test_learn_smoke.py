"""Learning smoke: DDPG learns (runs, updates, stays finite) on Warp VecEnv."""
import sys
from pathlib import Path
import numpy as np
import torch

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
from warp_backend.vecenv import WarpOuterVecEnv

CASE = dict(name="smoke", duration_s=0.5,
            reference=[[0.0, 0.0], [0.05, 5.0]],
            disturbance=[[0.0, 0.0]], stress=False)
INNER_DIR = f"{GEM_ROOT}/results/bldc/outer-inner-freeze-v1"
CFG = f"{GEM_ROOT}/results/bldc/outer-horizon-v2/g0995/config.json"


def test_ddpg_learns_on_warp_vecenv():
    import time
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from benchmarks.bldc.current_rl.repeat_study import read
    cfg = read(CFG)
    inner = DDPG.load(f"{INNER_DIR}/inner_model.zip", device="cuda")
    inner.policy.set_training_mode(False)
    for p in inner.policy.parameters():
        p.requires_grad_(False)
    venv = WarpOuterVecEnv(cfg["plant"], cfg["inner_study"], inner, CASE, n_envs=4, seed=0)
    t = time.perf_counter()
    model = DDPG("MlpPolicy", venv, seed=0, device="cuda", verbose=0,
                 learning_rate=1e-4, gamma=0.995, tau=0.005, buffer_size=5000,
                 learning_starts=200, batch_size=128, train_freq=1, gradient_steps=1,
                 action_noise=NormalActionNoise(np.zeros(1), np.ones(1) * 0.05),
                 policy_kwargs=dict(net_arch=dict(pi=[128, 128], qf=[256, 256]),
                                    activation_fn=torch.nn.ReLU))
    model.learn(800)
    dt = time.perf_counter() - t
    assert model.num_timesteps == 800, model.num_timesteps
    assert model.replay_buffer.size() > 0
    obs = venv.reset()
    with torch.no_grad():
        act = model.actor.forward(torch.as_tensor(obs, device="cuda"))
    assert torch.isfinite(act).all()
    print(f"smoke: 800 outer steps x4 envs in {dt:.1f}s "
          # SB3 num_timesteps already counts all envs: per-transition time is dt/800.
          f"({dt / 800 * 1000:.2f} ms/transition)")
