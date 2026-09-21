"""Automatic-reset coverage: step_wait auto-resets resample scenarios per lane."""
import json
import numpy as np
import pytest
from stable_baselines3 import DDPG
from warp_backend.vecenv import WarpOuterVecEnv

inner = DDPG.load('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip', device='cuda')
cfg = json.load(open('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-horizon-v2/g0995/config.json'))
CASES = [dict(name=f'auto{i}', duration_s=d, reference=[[0., 0.], [.01, 2.]],
              disturbance=[[0., 0.]], stress=False)
         for i, d in enumerate([0.02, 0.03, 0.05, 0.08])]


@pytest.mark.parametrize('backend', ['numpy', 'device'])
def test_auto_reset_resamples_scenarios_per_lane(backend):
    venv = WarpOuterVecEnv(cfg['plant'], cfg['inner_study'], inner, CASES[0],
                           n_envs=4, seed=0, cases=CASES, backend=backend)
    venv.reset()
    rng = np.random.default_rng(0)
    seen, dones = set(), 0
    for _ in range(300):
        venv.step_async(rng.uniform(-1, 1, (4, 1)).astype(np.float32))
        obs, rew, ds, infos = venv.step_wait()
        for j, d in enumerate(ds):
            if d:
                dones += 1
                seen.add(infos[j]['scenario'])
        assert np.isfinite(obs).all() and np.isfinite(rew).all()
        for i in infos:
            assert 'scenario' in i and 'physics_steps' in i
    assert dones > 8, 'episodes never completed'
    assert len(seen) >= 2, f'auto-resets stuck on one scenario: {seen}'
    venv.close()
