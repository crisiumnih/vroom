"""Versioned outer-controller contract (single source of truth).

Warp trains; GEM judges; both implement THIS contract. Any semantic change
means a new version string. Runners record the contract hash in protocols
and checkpoint metadata; evaluation dispatches the controller from the saved
contract instead of hardcoding the original semantics.

Versions:
- bldc-outer-speed-v1: 7-obs scalar absolute-current actor, z divisor 0.5.
- bldc-outer-speed-v2: v1 with memory divisor/cost parameters (fix screens).
- bldc-outer-speed-v3: IASA two-output TD3 (direct + integral accumulator),
  19 observations, command filter state, candidate reward (plan/plan.md sect.2-3).
"""
import hashlib
import json

VERSION_V1 = 'bldc-outer-speed-v1'
VERSION_V2 = 'bldc-outer-speed-v2'
VERSION_V3 = 'bldc-outer-speed-v3'

# v3 action/state constants (plan sect.2; starting choices, not optima).
V3_DIRECT_GAIN_A = 1.5
V3_INCREMENT_GAIN_A = 0.015
V3_COMMAND_LIMIT_A = 1.5
V3_ANTIWINDUP_GAIN = 0.1
V3_FILTER_ALPHA = 0.5

# v3 observation order (plan sect.2): 10 base + 9 lagged (5/10/20 ms).
V3_OBS_ORDER = [
    'omega', 'reference', 'error', 'tanh_error', 'id', 'iq',
    'bias', 'prev_applied', 'sin_angle', 'cos_angle',
    'speed_lag5', 'error_lag5', 'cmd_lag5',
    'speed_lag10', 'error_lag10', 'cmd_lag10',
    'speed_lag20', 'error_lag20', 'cmd_lag20',
]
V3_OBS_SCALES = {
    'omega': 25.0, 'reference': 25.0, 'error': 25.0, 'tanh_error': 0.05,
    'id': 4.0, 'iq': 4.0, 'bias': 1.5, 'prev_applied': 1.5,
    'speed_lag5': 25.0, 'error_lag5': 25.0, 'cmd_lag5': 1.5,
    'speed_lag10': 25.0, 'error_lag10': 25.0, 'cmd_lag10': 1.5,
    'speed_lag20': 25.0, 'error_lag20': 25.0, 'cmd_lag20': 1.5,
}

# v3 candidate reward weights (plan sect.3).
V3_REWARD = {
    'quad_weight': 4.0, 'log_weight': 0.1, 'log_scale': 0.05,
    'command_change': 0.02, 'integral_increment': 0.01, 'direct_output': 0.001,
    'failure': -5200.0,
}

TIMING = {'outer_dt_s': 0.001, 'inner_dt_s': 0.0001, 'hold_steps': 10,
          'reference_limit_a': 1.5, 'phase_limit_a': 4.0}


def describe(version):
    """Canonical contract dict for hashing/recording. Raises on unknown."""
    if version == VERSION_V1:
        return {'version': version, 'obs': '7-obs scalar', 'memory_divisor': 0.5,
                'memory_cost': 0.5, 'outputs': 1, **TIMING}
    if version == VERSION_V2:
        return {'version': version, 'obs': '7-obs scalar',
                'memory_divisor': 'parameter', 'memory_cost': 'parameter',
                'outputs': 1, **TIMING}
    if version == VERSION_V3:
        return {'version': version, 'obs': V3_OBS_ORDER,
                'action_gains': {'direct': V3_DIRECT_GAIN_A, 'increment': V3_INCREMENT_GAIN_A},
                'command_limit': V3_COMMAND_LIMIT_A, 'antiwindup': V3_ANTIWINDUP_GAIN,
                'filter_alpha': V3_FILTER_ALPHA, 'reward': V3_REWARD, 'outputs': 2, **TIMING}
    raise ValueError(f'Unknown contract {version}')


def digest(version):
    return hashlib.sha256(json.dumps(describe(version), sort_keys=True).encode()).hexdigest()
