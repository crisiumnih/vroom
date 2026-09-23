"""PR1 gates: plant struct, per-lane tables, corner parity vs GEM.

Corners are L0-relative F1-spirit multipliers (M0 unmeasured; multipliers
are infra fixtures, not the family). GEM meanings assumed per master_plan
sect.3: l_s = L-M, k_e on mechanical rad/s, j_total = j_rotor + j_load.
"""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import warp as wp

from warp_backend import fullstep as fs
from warp_backend.plant import MotorParams, LEGACY_L0, resolve_plants

wp.init()

_here = Path(__file__).resolve()
_sub = _here.parents[1] / "third_party" / "gym-electric-motor"
GEM_ROOT = (str(_sub) if (_sub / "benchmarks" / "bldc" / "current_rl" / "core.py").exists()
            else "/home/sra/prajwal/fyp/gym-electric-motor")
sys.path.insert(0, GEM_ROOT)
FYP = "/home/sra/prajwal/fyp/gym-electric-motor"

DT = 1e-4
STEPS = 500

CORNERS = {
    'nominal': {},
    'fast_electrical': dict(r_s=1.5 * 85e-3, l_s=0.65 * 50e-6),
    'heavy_slow': dict(r_s=0.8 * 85e-3, l_s=1.35 * 50e-6, j_rotor=2 * 0.0001,
                       j_load=3 * 0.0029),
    'weak_magnet_low_bus': dict(k_e=0.85 * 0.0955, supply_v=0.9 * 44.4, load_b=3 * 0.01),
}


def _params(name):
    base = dict(p=21, r_s=85e-3, l_s=50e-6, k_e=0.0955, j_rotor=0.0001,
                supply_v=44.4, load_a=0.01, load_b=0.01, load_c=0.0, j_load=0.0029)
    base.update(CORNERS[name])
    return MotorParams(**base)


def test_struct_validation():
    LEGACY_L0.validated()
    assert LEGACY_L0.row()[1] == pytest.approx(85e-3)
    assert LEGACY_L0.j_total == pytest.approx(0.003)
    with pytest.raises(ValueError):
        MotorParams(p=0).validated()
    with pytest.raises(ValueError):
        MotorParams(r_s=-1.0).validated()
    with pytest.raises(ValueError):
        MotorParams(l_s=float('nan')).validated()
    with pytest.raises(ValueError):
        resolve_plants([LEGACY_L0], 2)
    assert len(resolve_plants(LEGACY_L0, 3)) == 3
    assert LEGACY_L0.digest() == LEGACY_L0.digest()


def _device_trace(params, actions, dists, dt=DT):
    n = 1
    d_s = wp.array(np.zeros((n, 5)), dtype=wp.float64, device="cuda:0")
    d_a = wp.array(actions, dtype=wp.float64, device="cuda:0")
    d_d = wp.array(dists, dtype=wp.float64, device="cuda:0")
    d_tr = wp.zeros((actions.shape[0], n, 6), dtype=wp.float64, device="cuda:0")
    d_p = wp.array(np.ascontiguousarray([params.row()]),
                   dtype=wp.float64, device="cuda:0")
    wp.launch(fs.fullstep_kernel, dim=n,
              inputs=[d_s, d_a, d_d, wp.float64(dt), np.int32(actions.shape[0]), d_p, d_tr],
              device="cuda:0")
    wp.synchronize()
    return d_tr.numpy()[:, 0, :]


def _gem_trace(params, actions):
    from benchmarks.bldc.environment import make_environment
    cfg = json.load(open(f"{GEM_ROOT}/benchmarks/bldc/config.json"))
    cfg['motor'].update(p=params.p, r_s=params.r_s, l_s=params.l_s,
                        k_e=params.k_e, j_rotor=params.j_rotor)
    cfg['supply_v'] = params.supply_v
    cfg['load'].update(a=params.load_a, b=params.load_b, c=params.load_c,
                       j_load=params.j_load)
    scenario = dict(name="plant_parity", duration_s=STEPS * DT,
                    reference=[[0.0, 0.0]], disturbance=[[0.0, 0.0]])
    env, _ = make_environment(cfg, scenario)
    env.reset(seed=0)
    sys_state = env.unwrapped.physical_system
    names = sys_state.state_names
    lim = np.asarray(sys_state.limits, dtype=float)
    idx = {k: names.index(k) for k in ["omega", "torque", "i_a", "i_b", "i_c", "epsilon"]}
    trace = np.zeros((STEPS, 6))
    for k in range(STEPS):
        (state, _ref), *_ = env.step(np.asarray(actions[k, 0], dtype=float))
        s = np.asarray(state, dtype=float) * lim
        trace[k] = [s[idx["omega"]], s[idx["i_a"]], s[idx["i_b"]], s[idx["i_c"]],
                    s[idx["epsilon"]], s[idx["torque"]]]
    env.close()
    return trace


def _actions():
    rng = np.random.default_rng(0)
    a = np.zeros((STEPS, 1, 2))
    a[:, 0, 1] = 0.1
    a[200:, 0, 1] = -0.05
    a[:, 0, 0] = 0.02 * np.sin(np.arange(STEPS) / 25.0)
    return a


@pytest.mark.parametrize("name", sorted(CORNERS))
def test_corner_matches_gem(name):
    """Trip-gated parity: compare while both stay in-envelope (< 4 A), and
    require matching trip steps. Open-loop runaway past trip is outside any
    operating envelope and out of parity scope (cf. test_trip tolerance)."""
    params = _params(name)
    acts = _actions()
    dists = np.zeros((STEPS, 1))
    tw = _device_trace(params, acts, dists)
    tg = _gem_trace(params, acts)
    trip_w = int(np.argmax(np.abs(tw[:, 1:4]).max(axis=1) > 4.0)) if (np.abs(tw[:, 1:4]).max(axis=1) > 4.0).any() else STEPS
    trip_g = int(np.argmax(np.abs(tg[:, 1:4]).max(axis=1) > 4.0)) if (np.abs(tg[:, 1:4]).max(axis=1) > 4.0).any() else STEPS
    if trip_w != STEPS or trip_g != STEPS:
        assert abs(trip_w - trip_g) <= 5, (name, trip_w, trip_g)
    cut = max(50, min(trip_w, trip_g))
    tw, tg = tw[:cut], tg[:cut]
    tw[:, 4] = (tw[:, 4] + np.pi) % (2 * np.pi) - np.pi
    err = np.abs(tw - tg)
    scale = np.maximum(np.abs(tg).max(axis=0), [1.0, 0.1, 0.1, 0.1, 0.1, 1e-3])
    rel = (err / scale).max(axis=0)
    print(name, "cut:", cut, "trip:", (trip_w, trip_g), "scaled:", rel)
    assert rel[0] < 0.02 and rel[1:4].max() < 0.05 and rel[4] < 0.02 and rel[5] < 0.05


def test_dt_refinement_at_fastest_corner():
    """100 us RK4 vs 2x50 us at tau_e ~= 255 us: integration adequacy.

    Measured max rel disagreement ~2e-4 (currents ~1e-3 A abs). Gate 1e-3
    keeps dt error 10x below the cross-simulator parity bar (0.01):
    no substepping needed at the fastest F1 corner.
    """
    params = _params('fast_electrical')
    acts = _actions()
    d1 = _device_trace(params, acts, np.zeros((STEPS, 1)), dt=DT)
    half = np.repeat(acts, 2, axis=0)
    d2 = _device_trace(params, half, np.zeros((2 * STEPS, 1)), dt=DT / 2)[1::2]
    err = np.abs(d1 - d2)
    scale = np.maximum(np.abs(d1).max(axis=0), [1.0, 0.1, 0.1, 0.1, 0.1, 1e-3])
    assert ((err / scale).max(axis=0) < 1e-3).all()


def test_broadcast_matches_single_lane():
    import torch
    from warp_backend.rollout import FastOuterBackend
    from stable_baselines3 import DDPG
    inner = DDPG.load(f"{FYP}/results/bldc/outer-inner-freeze-v1/inner_model.zip", device='cuda')
    cfg = json.load(open(f"{FYP}/results/bldc/outer-horizon-v2/g0995/config.json"))
    case = dict(name='b', duration_s=0.05, reference=[[0., 3.]], disturbance=[[0., 0.]])
    be = FastOuterBackend(cfg['plant'], cfg['inner_study'], inner, [case], seed=0)
    be.reset(2, case=case, seed=0)
    assert torch.equal(be._dev["params"][0], be._dev["params"][1])
    assert float(be._dev["ke"][0]) == pytest.approx(0.0955)
