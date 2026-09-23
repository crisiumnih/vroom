"""PR3 gates: N0==legacy, numpy/torch sense agreement, determinism."""
import numpy as np
import pytest
import torch

from warp_backend.sensing import (SenseConfig, SENSE_VERSION, N0, N1, N2,
                                  sense_currents_numpy, sense_currents_torch)


def test_presets_and_validation():
    assert N0().i_noise == 0.0 and N0().volt_delay == 0 and N0().ang_age == 0
    n1, n2 = N1(), N2()
    assert (n1.i_noise, n1.filt_tau, n1.ang_age) == (0.005, 0.0001, 5)
    assert (n2.i_noise, n2.filt_tau, n2.ang_age) == (0.01, 0.0002, 10)
    assert n1.ang_off == pytest.approx(0.5 * np.pi / 180)
    assert len(n1.digest()) == 64
    with pytest.raises(ValueError):
        SenseConfig(i_noise=-1.0)
    with pytest.raises(ValueError):
        SenseConfig(volt_delay_steps=2)


def test_numpy_torch_sense_agree():
    cfg = N1()
    rng = np.random.default_rng(0)
    ia, ib = rng.uniform(-3, 3, 8), rng.uniform(-3, 3, 8)
    na, nb = rng.uniform(-0.1, 0.1, 8), rng.uniform(-0.1, 0.1, 8)
    filt = rng.uniform(-1, 1, (8, 2))
    outs = [sense_currents_numpy(cfg, ia[j], ib[j], 0.0, na[j], nb[j], filt[j])[:3]
            for j in range(8)]
    want = np.array(outs)
    dev = torch.device('cuda:0')
    got = sense_currents_torch(cfg, torch, torch.as_tensor(ia, device=dev),
                               torch.as_tensor(ib, device=dev),
                               torch.as_tensor(np.zeros(8), device=dev),
                               torch.as_tensor(na, device=dev),
                               torch.as_tensor(nb, device=dev),
                               torch.as_tensor(filt, device=dev))
    got = torch.stack(got, dim=1).cpu().numpy()
    assert np.abs(got - want).max() < 1e-6
    # C reconstructed from A/B
    assert np.abs((got[:, 0] + got[:, 1] + got[:, 2])).max() < 1e-6


def test_filter_converges_and_n0_identity():
    cfg = N1()
    f = (0.0, 0.0)
    for _ in range(2000):
        _, _, _, f = sense_currents_numpy(cfg, 1.5, -0.5, 0.0, 0.0, 0.0, f)
    assert f[0] == pytest.approx(1.5 * 1.005 + 0.0025, abs=1e-6)
    assert f[1] == pytest.approx(-0.5 * 0.995 - 0.0025, abs=1e-6)
    c0 = N0()
    a, b, c, _ = sense_currents_numpy(c0, 1.5, -0.5, 0.25, 0.0, 0.0, (9.0, 9.0))
    assert (a, b, c) == pytest.approx((1.5, -0.5, 0.25))


def _backend(kind='device', sense=None, n=2, seed=0):
    import json
    from stable_baselines3 import DDPG
    from warp_backend.vecenv import WarpOuterVecEnv
    fyp = "/home/sra/prajwal/fyp/gym-electric-motor"
    inner = DDPG.load(f"{fyp}/results/bldc/outer-inner-freeze-v1/inner_model.zip", device='cuda')
    cfg = json.load(open(f"{fyp}/results/bldc/outer-horizon-v2/g0995/config.json"))
    case = dict(name='s', duration_s=0.1, reference=[[0., 3.]], disturbance=[[0., 0.]])
    be_cls = {'device': None, 'numpy': None}
    from warp_backend.rollout import FastOuterBackend
    from warp_backend.backend import WarpOuterBackend
    cls = FastOuterBackend if kind == 'device' else WarpOuterBackend
    be = cls(cfg['plant'], cfg['inner_study'], inner, [case], seed=seed,
             reward_shape='l1', sense=sense)
    be.reset(n, case=case, seed=seed)
    return be


def test_n0_sense_equals_plant_and_deterministic():
    be = _backend('device', N0())
    rng = np.random.default_rng(0)
    o1 = [be.step(rng.uniform(-1, 1, (2, 1)))[0] for _ in range(10)]
    d = be._dev
    assert torch.equal(d["sense"], d["plant"])
    be2 = _backend('device', N0())
    o2 = [be2.step(rng.uniform(-1, 1, (2, 1)))[0] for _ in range(10)]
    # same seed + same actions: identical trajectories
    be3 = _backend('device', N0())
    rng3 = np.random.default_rng(0)
    o3 = [be3.step(rng3.uniform(-1, 1, (2, 1)))[0] for _ in range(10)]
    for a, b in zip(o1, o3):
        assert np.array_equal(a, b)


def test_n1_runs_and_differs_from_n0():
    n0 = _backend('device', N0())
    n1 = _backend('device', N1())
    rng = np.random.default_rng(5)
    acts = [rng.uniform(-1, 1, (2, 1)) for _ in range(15)]
    r0 = [n0.step(a)[1] for a in acts]
    rng = np.random.default_rng(5)
    acts = [rng.uniform(-1, 1, (2, 1)) for _ in range(15)]
    r1 = [n1.step(a)[1] for a in acts]
    assert all(np.isfinite(r).all() for r in r1)
    assert not np.allclose(np.concatenate(r0), np.concatenate(r1))
    d = n1._dev
    assert not torch.equal(d["sense"], d["plant"])
    assert d["sense"].shape == (2, 5)


def test_numpy_n1_runs_and_differs_from_n0():
    n0 = _backend('numpy', N0())
    n1 = _backend('numpy', N1())
    rng = np.random.default_rng(5)
    acts = [rng.uniform(-1, 1, (2, 1)) for _ in range(15)]
    r0 = [n0.step(a)[1] for a in acts]
    rng = np.random.default_rng(5)
    acts = [rng.uniform(-1, 1, (2, 1)) for _ in range(15)]
    r1 = [n1.step(a)[1] for a in acts]
    assert all(np.isfinite(r).all() for r in r1)
    assert not np.allclose(np.concatenate(r0), np.concatenate(r1))


def test_vecenv_sense_passthrough():
    import json
    from stable_baselines3 import DDPG
    from warp_backend.vecenv import WarpOuterVecEnv
    fyp = "/home/sra/prajwal/fyp/gym-electric-motor"
    inner = DDPG.load(f"{fyp}/results/bldc/outer-inner-freeze-v1/inner_model.zip", device='cuda')
    cfg = json.load(open(f"{fyp}/results/bldc/outer-horizon-v2/g0995/config.json"))
    case = dict(name='s', duration_s=0.1, reference=[[0., 3.]], disturbance=[[0., 0.]])
    ve = WarpOuterVecEnv(cfg['plant'], cfg['inner_study'], inner, case, n_envs=2,
                         backend='device', sense=N1())
    assert ve.be.sense.i_noise == pytest.approx(0.005)
    assert ve.be.sense.digest() == N1().digest()
    ve.close()
