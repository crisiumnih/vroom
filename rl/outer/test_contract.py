"""Shared-contract unit checks (CPU, no simulator)."""
import numpy as np
import pytest
from rl.outer import contract, controller, reward


def test_contract_versions_digest_stable():
    d1 = contract.digest('bldc-outer-speed-v3')
    assert d1 == contract.digest('bldc-outer-speed-v3') and len(d1) == 64
    assert contract.describe('bldc-outer-speed-v1')['outputs'] == 1
    with pytest.raises(ValueError):
        contract.describe('nope')


def test_accumulator_slew_and_antiwindup():
    s = controller.init_state()
    for _ in range(100):
        controller.apply_action(s, 0.0, 1.0)
    assert s['b'] == pytest.approx(1.5, abs=1e-9)  # 100 ms at max increment
    s2 = controller.init_state()
    c, _ = controller.apply_action(s2, 1.0, 1.0)  # v = 1.5+0.015 -> clipped
    assert c <= 1.5 and s2['b'] < 0.015 + 1e-12  # saturation residual fed back


def test_zero_increment_preserves_bias():
    s = controller.init_state()
    s['b'] = 0.4
    controller.apply_action(s, 0.0, 0.0)
    assert s['b'] == pytest.approx(0.4)


def test_filter_and_obs_contract():
    s = controller.init_state()
    c, dq = controller.apply_action(s, 1.0, 0.0)
    assert dq == 0.0 and c == pytest.approx(0.75)  # 0.5*0 + 0.5*1.5
    controller.record_history(s, 5.0, 5.0, c)
    o = controller.observe(s, 5.0, 5.0, 0.1, 0.2, 0.0, 1.0)
    assert o.shape == (19,) and o.dtype == np.float32
    assert o[0] == pytest.approx(0.2) and o[3] == pytest.approx(0.0, abs=1e-6)
    assert o[6] == pytest.approx(0.0) and o[7] == pytest.approx(0.5)


def test_reward_bound_check():
    chk = reward.reward_bound_check()
    assert chk['worst_step_cost'] == pytest.approx(5.274, abs=1e-3)
    assert chk['penalty_exceeds_bound'] is True
    assert reward.candidate_reward([0.0] * 10, 0.0, 0.0, 0.0, 0.0, False) == pytest.approx(0.0)
    assert reward.candidate_reward([1.0] * 10, 0.0, 0.0, 0.0, 0.0, True) == -5200.0
