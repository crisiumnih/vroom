"""IASA outer-controller arithmetic (plan sect.2), backend-independent.

Reference implementation in NumPy; device backends must reproduce it bit-near
(tested by parity). State per lane: bias b (A), filtered command c (A),
previous applied command, lag histories, bounded error memory z.
"""
import numpy as np

from rl.outer.contract import (
    V3_DIRECT_GAIN_A, V3_INCREMENT_GAIN_A, V3_COMMAND_LIMIT_A,
    V3_ANTIWINDUP_GAIN, V3_FILTER_ALPHA,
)

LAG_STEPS = (5, 10, 20)  # outer steps = ms at 1 kHz


def init_state():
    return {'b': 0.0, 'c': 0.0, 'prev_applied': 0.0, 'z': 0.0,
            'hist_speed': [], 'hist_error': [], 'hist_cmd': []}


def apply_action(state, a_p, a_i):
    """One outer decision. Returns (iq_ref, id_ref); mutates state in place."""
    a_p = float(np.clip(a_p, -1, 1))
    a_i = float(np.clip(a_i, -1, 1))
    p = V3_DIRECT_GAIN_A * a_p
    b_trial = state['b'] + V3_INCREMENT_GAIN_A * a_i
    v = p + b_trial
    lim = V3_COMMAND_LIMIT_A
    s = float(np.clip(v, -lim, lim))
    state['b'] = float(np.clip(b_trial + V3_ANTIWINDUP_GAIN * (s - v), -lim, lim))
    c = V3_FILTER_ALPHA * state['c'] + (1.0 - V3_FILTER_ALPHA) * s
    state['c'] = float(c)
    state['prev_applied'] = float(c)
    return float(c), 0.0


def observe(state, omega, reference, i_sd, i_sq, sin_angle, cos_angle):
    """19-dim observation per contract order. History must be recorded
    separately each outer step via record_history (below) BEFORE observe."""
    err = reference - omega
    base = [omega / 25.0, reference / 25.0, err / 25.0, float(np.tanh(err / 0.05)),
            i_sd / 4.0, i_sq / 4.0, state['b'] / 1.5, state['prev_applied'] / 1.5,
            sin_angle, cos_angle]
    lags = []
    for lag in LAG_STEPS:
        if len(state['hist_speed']) >= lag:
            lags += [state['hist_speed'][-lag] / 25.0, state['hist_error'][-lag] / 25.0,
                     state['hist_cmd'][-lag] / 1.5]
        else:
            lags += [base[0], base[2], base[7]]
    v = np.asarray(base + lags, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite IASA observation')
    return np.clip(v, -1, 1).astype(np.float32)


def record_history(state, omega, reference, applied_cmd):
    state['hist_speed'].append(float(omega))
    state['hist_error'].append(float(reference - omega))
    state['hist_cmd'].append(float(applied_cmd))
