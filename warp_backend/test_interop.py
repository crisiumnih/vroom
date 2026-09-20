"""Zero-copy torch<->warp interop + frozen inner actor on CUDA.

Proves the training-loop data path: warp physics arrays shared with torch
(no copies), real frozen inner actor forward batched on GPU.
"""
import time
import sys
from pathlib import Path
import numpy as np
import torch
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

wp.init()
INNER = "/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip"


def test_shared_memory_no_copy():
    assert torch.cuda.is_available()
    w = wp.zeros(1024, dtype=wp.float64, device="cuda:0")
    t = torch.as_tensor(w)  # shares memory via __cuda_array_interface__
    assert t.device.type == "cuda" and t.data_ptr() == w.ptr
    t.fill_(3.0)
    wp.synchronize()
    assert bool((t.cpu() == 3.0).all())  # torch write visible, no copy
    w.fill_(7.0)
    wp.synchronize()
    assert float(t[100]) == 7.0 and bool((t.cpu() == 7.0).all())  # warp write visible


def test_frozen_inner_actor_batched_cuda():
    from stable_baselines3 import DDPG
    model = DDPG.load(INNER, device="cuda")
    assert next(model.actor.parameters()).is_cuda
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.uniform(-1, 1, (256, 25)).astype(np.float32), device="cuda")
    with torch.no_grad():
        a1 = model.actor.forward(obs)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(100):
            a2 = model.actor.forward(obs)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 100
    assert torch.isfinite(a1).all()
    assert torch.equal(a1, a2), "actor must be deterministic"
    print(f"inner actor 256-batch forward: {dt*1e6:.1f} us ({model.actor.mu}")
    print(f"action range: [{a1.min():.3f}, {a1.max():.3f}]")
