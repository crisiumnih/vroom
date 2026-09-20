"""VecEnv case-list cycling: distribution over resets + single-case compat."""
import json
import numpy as np
from stable_baselines3 import DDPG
from warp_backend.vecenv import WarpOuterVecEnv

inner = DDPG.load('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip', device='cuda')
cfg = json.load(open('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-horizon-v2/g0995/config.json'))
cases = [dict(name=f'c{i}', duration_s=0.05, reference=[[0., 0.], [.01, float(v)]],
              disturbance=[[0., 0.]], stress=False) for i, v in enumerate([2., -3., 5., -7.])]


def test_cycles_cases_over_resets():
    venv = WarpOuterVecEnv(cfg['plant'], cfg['inner_study'], inner, cases[0], n_envs=2, seed=0, cases=cases)
    seen = {venv.case['name']}
    for _ in range(30):
        venv.reset()
        seen.add(venv.case['name'])
    assert seen == {'c0', 'c1', 'c2', 'c3'}, seen
    venv.close()


def test_single_case_backward_compat():
    venv = WarpOuterVecEnv(cfg['plant'], cfg['inner_study'], inner, cases[0], n_envs=2, seed=0)
    for _ in range(5):
        venv.reset()
        assert venv.case['name'] == 'c0'
    venv.close()
