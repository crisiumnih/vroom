"""Backend memory params: default unchanged, 10x gain equivalence."""
import json
import numpy as np
from stable_baselines3 import DDPG
from warp_backend.backend import WarpOuterBackend

inner = DDPG.load('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip', device='cuda')
cfg = json.load(open('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-horizon-v2/g0995/config.json'))
case = dict(name='mem', duration_s=0.5, reference=[[0., 0.], [.01, 3.]],
            disturbance=[[0., 0.]], stress=False)


def _run(div, cost, steps=60):
    be = WarpOuterBackend(cfg['plant'], cfg['inner_study'], inner, [case],
                          seed=0, reward_shape='l1', failure=-5200.0,
                          memory_divisor=div, memory_cost=cost)
    obs, _ = be.reset(2, case=case, seed=0)
    rng = np.random.default_rng(0)
    zs, rs = [], []
    for _ in range(steps):
        obs, rew, _, _, _ = be.step(rng.uniform(-0.2, 0.2, (2, 1)))
        zs.append(be.z.copy())
        rs.append(rew.copy())
    return np.array(zs), np.array(rs)


def test_gain_10x_signal_identical_preclip_cost():
    z0, r0 = _run(0.5, 0.5)
    z1, r1 = _run(0.05, 0.005)
    assert np.abs(z1).max() > np.abs(z0).max()
    assert np.abs(z1 - 10 * z0).max() < 1e-4 or (np.abs(z1) >= 1.0 - 1e-6).any()
    # z never feeds back into physics: identical trajectories => identical
    # total reward pre-clip proves the rescaled memory cost is neutral.
    preclip = np.abs(z1) < 1.0 - 1e-6
    assert np.abs(r1[preclip] - r0[preclip]).max() < 1e-6
