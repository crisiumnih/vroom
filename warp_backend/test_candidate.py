"""V4 candidate-reward parity: device tensor math + closed-loop vs GEM adapter.

Reference: rl.outer.reward (shared contract owner). Fixture uses a constant
reference so the device per-hold reference (fixed at hold start) and the GEM
per-sample reference agree exactly.
"""
import json
import sys
import numpy as np
import pytest
import torch
from stable_baselines3 import DDPG

from warp_backend.rollout import FastOuterBackend, _candidate_reward_c
from rl.outer import contract, reward as rw

sys.path.insert(0, '/home/sra/prajwal/fyp/gym-electric-motor')

INNER = '/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip'
CFG = '/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-horizon-v2/g0995/config.json'
CASE = dict(name='candidate', duration_s=0.05, reference=[[0., 3.]],
            disturbance=[[0., 0.]], stress=False)


def test_contract_v4_shape():
    d3, d4 = contract.describe('bldc-outer-speed-v3'), contract.describe('bldc-outer-speed-v4')
    assert d4['reward_shape'] == 'candidate' and d3['reward_shape'] == 'l1'
    assert d4['obs'] == d3['obs'] and d4['action_gains'] == d3['action_gains']
    assert contract.digest('bldc-outer-speed-v4') != contract.digest('bldc-outer-speed-v3')
    assert rw.reward_bound_check()['penalty_exceeds_bound'] is True


def test_candidate_control_validation():
    inner = DDPG.load(INNER, device='cuda')
    cfg = json.load(open(CFG))
    with pytest.raises(ValueError):
        FastOuterBackend(cfg['plant'], cfg['inner_study'], inner, [CASE], control='direct',
                         reward_shape='candidate')
    be = FastOuterBackend(cfg['plant'], cfg['inner_study'], inner, [CASE], control='iasa',
                          smooth_alpha=0.5, reward_shape='candidate', failure=-5200.0)
    assert be.shape == 'candidate' and be.failure == -5200.0


def test_candidate_tensor_math_matches_reference():
    rng = np.random.default_rng(1)
    dev = torch.device('cuda:0')
    with torch.no_grad():
        for _ in range(5):
            n = 4
            ref = torch.as_tensor(rng.uniform(-5, 30, n), dtype=torch.float64, device=dev)
            om = torch.as_tensor(rng.uniform(-5, 30, n), dtype=torch.float64, device=dev)
            c_now = torch.as_tensor(rng.uniform(-1.5, 1.5, n), dtype=torch.float64, device=dev)
            c_prev = torch.as_tensor(rng.uniform(-1.5, 1.5, n), dtype=torch.float64, device=dev)
            a_p = torch.as_tensor(rng.uniform(-1, 1, n), dtype=torch.float64, device=dev)
            a_i = torch.as_tensor(rng.uniform(-1, 1, n), dtype=torch.float64, device=dev)
            got = _candidate_reward_c(ref, om, c_now, c_prev, a_p, a_i).cpu().numpy()
            for j in range(n):
                e = float(np.clip(float(ref[j]) - float(om[j]), -25., 25.))
                want = float(rw.candidate_tracking(e)) + float(
                    rw.candidate_command_costs(float(c_now[j]), float(c_prev[j]),
                                               float(a_i[j]), float(a_p[j])))
                assert float(got[j]) == pytest.approx(want, abs=1e-9)


def test_candidate_closed_loop_device_vs_gem():
    from tools.outer_env_candidate import env_for_contract, V4
    inner_cu = DDPG.load(INNER, device='cuda')
    inner_cpu = DDPG.load(INNER, device='cpu')
    cfg = json.load(open(CFG))
    be = FastOuterBackend(cfg['plant'], cfg['inner_study'], inner_cu, [CASE], seed=7,
                          control='iasa', smooth_alpha=0.5, reward_shape='candidate', failure=-5200.0)
    be.reset(1, case=CASE, seed=7)
    env = env_for_contract(V4, cfg['plant'], cfg['inner_study'], inner_cpu, [CASE])
    env.reset(seed=7, options={'case': CASE})
    rng = np.random.default_rng(3)
    acts = [rng.uniform(-1, 1, 2) for _ in range(20)]
    for a in acts:
        be_obs, be_rew, be_term, be_trunc, _ = be.step(a.reshape(1, -1))
        assert np.isfinite(be_obs).all() and not be_term[0] and not be_trunc[0]
        _, rew, term, trunc, _ = env.step(a)
        assert not term and not trunc
        # Device reward = mean over executed inner samples; GEM identical by
        # construction; fp32 filter state vs fp64 reference leaves ~1e-4.
        assert float(be_rew[0]) == pytest.approx(float(rew), abs=2e-4)
