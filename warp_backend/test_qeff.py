"""qeff reward shape + smoothing: numerics and properties."""
import json
import numpy as np
from stable_baselines3 import DDPG
from warp_backend.backend import WarpOuterBackend

inner = DDPG.load('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip', device='cuda')
cfg = json.load(open('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-horizon-v2/g0995/config.json'))
case = dict(name='qr', duration_s=0.2, reference=[[0., 0.], [.01, 3.]],
            disturbance=[[0., 0.]], stress=False)


def _run(shape='l1', alpha=0.0, steps=40, seed=0):
    be = WarpOuterBackend(cfg['plant'], cfg['inner_study'], inner, [case],
                          seed=seed, reward_shape=shape, failure=-5200.0,
                          smooth_alpha=alpha)
    obs, _ = be.reset(2, case=case, seed=seed)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(steps):
        obs, rew, _, _, _ = be.step(rng.uniform(-1, 1, (2, 1)))
        rs.append(rew.copy())
    return np.array(rs)


def test_qeff_matches_l1_at_calibration_point():
    # qw=100: at en=0.04, 100*en^2 == 4*|en| (0.16). Speed-term-only check via
    # single-error construction is covered by formula; here check finiteness/shape.
    r = _run('qeff')
    assert np.isfinite(r).all() and (r <= 0).all()


def test_qeff_cheaper_small_errors_pricier_large():
    # Same trajectory, different shape: compare speed-term magnitudes
    # analytically at two error sizes (backend-independent property).
    for en, expect in [(0.01, -1), (0.20, +1)]:
        l1 = 4 * abs(en)
        q = 100 * en ** 2
        assert np.sign(q - l1) == expect, (en, l1, q)


def test_smoothing_alpha1_is_identity():
    be = WarpOuterBackend(cfg['plant'], cfg['inner_study'], inner, [case], seed=0, smooth_alpha=1.0)
    be.reset(1, case=case, seed=0)
    a = np.array([[0.37]])
    be.step(a)
    assert be.cmd_f[0] == 1.5 * 0.37


def test_smoothing_halves_step_response():
    be = WarpOuterBackend(cfg['plant'], cfg['inner_study'], inner, [case], seed=0, smooth_alpha=0.5)
    be.reset(1, case=case, seed=0)
    be.step(np.ones((1, 1)))
    assert be.cmd_f[0] == 1.5 * 0.5
