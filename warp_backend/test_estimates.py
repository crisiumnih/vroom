"""PR4 gates: one gain rule, B0 frozen, B1 tracks truth, legacy default kept."""
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest

from warp_backend.estimates import (ControllerEstimates, FROZEN_B0, DESIGN,
                                    b1_from_truth, design_gains)

_here = Path(__file__).resolve()
_sub = _here.parents[1] / "third_party" / "gym-electric-motor"
GEM_ROOT = (str(_sub) if (_sub / "benchmarks" / "bldc" / "current_rl" / "core.py").exists()
            else "/home/sra/prajwal/fyp/gym-electric-motor")
sys.path.insert(0, GEM_ROOT)
FYP = "/home/sra/prajwal/fyp/gym-electric-motor"


def _plant():
    cfg = json.load(open(f"{FYP}/results/bldc/outer-iasa-v2/inner_config.json"))
    return cfg['plant']


def test_rule_equality_with_fyp():
    from benchmarks.bldc.cascaded import derived_gains
    plant = _plant()
    design = plant['controller']
    a = design_gains(FROZEN_B0, DESIGN)
    b = derived_gains(plant, design, asdict(FROZEN_B0))
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9), k


def test_legacy_default_unchanged():
    from benchmarks.bldc.cascaded import derived_gains, DQCurrentController
    plant = _plant()
    design = plant['controller']
    cfg = {'motor': dict(plant['motor']), 'load': dict(plant['load']),
           'supply_v': plant['supply_v'], 'action_norm_limit': 0.12}
    assert derived_gains(cfg, design) == derived_gains(cfg, design, None)
    c = DQCurrentController(cfg, {**design, 'current_reference_limit_a': 1.5})
    v = c.act_current({'i_sd': 0.1, 'i_sq': 0.2, 'omega': 5.0}, [0.0, 0.5], 1e-4)
    assert np.isfinite(v).all()


def test_b0_frozen_across_plants():
    from benchmarks.bldc.cascaded import derived_gains
    plant = _plant()
    design = plant['controller']
    est = asdict(FROZEN_B0)
    mk = lambda rs, ls: {'motor': {**plant['motor'], 'r_s': rs, 'l_s': ls},
                         'load': dict(plant['load']), 'supply_v': plant['supply_v'],
                         'action_norm_limit': 0.12}
    g1 = derived_gains(mk(0.085, 50e-6), design, est)
    g2 = derived_gains(mk(0.13, 32e-6), design, est)  # perturbed truth
    assert g1 == g2  # B0 cannot be retuned by the plant
    assert FROZEN_B0.digest() == FROZEN_B0.digest()


def test_b1_tracks_truth_but_not_load_case():
    from benchmarks.bldc.cascaded import derived_gains
    from benchmarks.bldc.pi_estimates import b1_estimates
    plant = _plant()
    design = plant['controller']
    pert = copy.deepcopy(plant)
    pert['motor']['l_s'] = 0.65 * 50e-6
    pert['load']['j_load'] = 3 * 0.0029  # D5-like case must NOT leak in
    est = b1_estimates(pert['motor'], pert['load'])
    assert est['l_hat'] == pytest.approx(0.65 * 50e-6)
    assert est['j_load_hat'] == pytest.approx(0.0029)
    g0 = derived_gains({**plant, 'action_norm_limit': 0.12}, design, asdict(FROZEN_B0))
    g1 = derived_gains({**pert, 'action_norm_limit': 0.12}, design, est)
    assert g1['current_kp_v_per_a'] == pytest.approx(0.65 * g0['current_kp_v_per_a'])
    rec = b1_from_truth(pert)
    assert rec.l_hat == pytest.approx(est['l_hat'])
    with pytest.raises(ValueError):
        ControllerEstimates(r_hat=-1.0).validated()


def test_bench_matrix_fixed():
    from warp_backend.bench_matrix import build_matrix
    motors = [{'name': 'm0'}, {'name': 'm1'}]
    rows, d = build_matrix(motors, ['S1', 'D1'], ('N1',), ('B0', 'R0'))
    assert len(rows) == 2 * 2 * 1 * 2 * 3 == 24
    _, d2 = build_matrix(motors, ['S1', 'D1'], ('N1',), ('B0', 'R0'))
    assert d == d2 and rows[0]['noise_seed'] == 5101 and rows[2]['mirror_signs']
