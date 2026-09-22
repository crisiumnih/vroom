"""IASA device parity: _iasa_command/_assemble_iasa_obs vs shared reference."""
import json
import numpy as np
import pytest
import torch
from stable_baselines3 import DDPG
from warp_backend.rollout import FastOuterBackend
from rl.outer import controller as ref

inner = DDPG.load('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-inner-freeze-v1/inner_model.zip', device='cuda')
cfg = json.load(open('/home/sra/prajwal/fyp/gym-electric-motor/results/bldc/outer-horizon-v2/g0995/config.json'))
CASE = dict(name='iasa', duration_s=0.2, reference=[[0., 0.], [.01, 3.]],
            disturbance=[[0., 0.]], stress=False)


def _be(n=2, seed=0):
    be = FastOuterBackend(cfg['plant'], cfg['inner_study'], inner, [CASE], seed=seed, control='iasa', smooth_alpha=0.5,
                          reward_shape='l1')
    be.reset(n, case=CASE, seed=seed)
    return be


def test_iasa_command_matches_reference():
    be = _be()
    d = be._dev
    rng = np.random.default_rng(0)
    st = ref.init_state()
    for b0 in [0.0, 0.4, -1.2, 1.5]:
        d["bias"][:] = b0
        d["cmd_f"][:] = 0.3
        st['b'] = b0
        st['c'] = st['prev_applied'] = 0.3
        a = rng.uniform(-1, 1, (2, 2))
        with torch.no_grad():
            cmdf, delta = be._iasa_command(a)
        # reference per-lane independent application (device lanes share start state)
        st2 = ref.init_state()
        st2['b'] = b0
        st2['c'] = st2['prev_applied'] = 0.3
        ref.apply_action(st2, a[0, 0], a[0, 1])
        assert float(d["bias"][0]) == pytest.approx(st2['b'], abs=1e-5)
        assert float(cmdf[0]) == pytest.approx(st2['c'], abs=1e-5)
        assert float(d["pcmd"][0]) == pytest.approx(0.3)


def test_iasa_obs_matches_reference():
    be = _be(n=1)
    d = be._dev
    st = ref.init_state()
    om, rr, isd, isq, eps = 5.0, 6.0, 0.1, 0.2, 0.5
    frames = [(om + 0.01 * k, rr, 0.1 * k) for k in range(25)]
    for w_, r_, c_ in frames:
        ref.record_history(st, w_, r_, c_)
    d["plant"][0] = torch.tensor([om, 0.05, -0.03, 0.02, eps], dtype=torch.float64)
    d["clk"][0] = 0
    d["dur"][0] = 10 ** 9
    d["bias"][0] = 0.2
    d["pcmd"][0] = 0.1
    h = np.array([w_ for w_, _, _ in frames], dtype=np.float32)
    d["h_spd"][0] = torch.as_tensor(h[-20:])
    d["h_err"][0] = torch.as_tensor(np.array([(r_ - w_) for w_, r_, _ in frames], dtype=np.float32)[-20:])
    d["h_cmd"][0] = torch.as_tensor(np.array([c_ for _, _, c_ in frames], dtype=np.float32)[-20:])
    st['b'] = 0.2
    st['prev_applied'] = 0.1
    got = be._assemble_iasa_obs()[0].cpu().numpy()
    want = ref.observe(st, om, rr, isd, isq, float(np.sin(eps)), float(np.cos(eps)))
    # isd/isq inside device obs come from plant currents; compare structure only
    assert got.shape == (19,) and np.isfinite(got).all()
    assert got[0] == pytest.approx(want[0], abs=1e-6)
    assert got[6] == pytest.approx(want[6], abs=1e-5)
    assert got[7] == pytest.approx(want[7], abs=1e-5)
    assert got[10] == pytest.approx(want[10], abs=1e-4)
    assert got[13] == pytest.approx(want[13], abs=1e-4)
    assert got[16] == pytest.approx(want[16], abs=1e-4)
