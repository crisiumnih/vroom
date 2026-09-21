"""Shared outer-reward components (plan sect.3 candidate + legacy shapes).

Pure functions over per-inner-sample errors and outer commands, so GEM and
Warp adapters compute identical rewards. Units documented per term.
"""
import numpy as np

from rl.outer.contract import V3_REWARD


def candidate_tracking(e_bar):
    """Plan sect.3 tracking cost per inner sample. e_bar: clipped error (rad/s)."""
    e = np.asarray(e_bar, dtype=float)
    return (V3_REWARD['quad_weight'] * (e / 25.0) ** 2
            + V3_REWARD['log_weight'] * np.log1p((e / V3_REWARD['log_scale']) ** 2))


def candidate_command_costs(c_now, c_prev, a_i, a_p):
    w = V3_REWARD
    return (w['command_change'] * ((c_now - c_prev) / 3.0) ** 2
            + w['integral_increment'] * float(a_i) ** 2
            + w['direct_output'] * float(a_p) ** 2)


def candidate_reward(e_samples, c_now, c_prev, a_i, a_p, terminated):
    """Mean tracking over executed inner samples + outer command costs."""
    e = np.asarray(e_samples, dtype=float)
    cost = candidate_tracking(e).mean() + candidate_command_costs(c_now, c_prev, a_i, a_p)
    if terminated:
        return float(V3_REWARD['failure'])
    return float(-cost)


def reward_bound_check(gamma=0.995):
    """Plan sect.3 incentive check: worst ordinary cost and discounted sum."""
    worst = 4.0 * 1.0 + 0.1 * np.log1p((25.0 / 0.05) ** 2) + 0.02 + 0.01 + 0.001
    return {'worst_step_cost': float(worst),
            'worst_discounted_sum': float(worst / (1.0 - gamma)),
            'failure_penalty': float(V3_REWARD['failure']),
            'penalty_exceeds_bound': bool(-V3_REWARD['failure'] > worst / (1.0 - gamma))}
