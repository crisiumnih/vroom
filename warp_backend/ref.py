"""Float32 numpy transcription of BrushlessDCMotor RHS (spike reference).

Mirrors `src/gym_electric_motor/.../brushless_dc_motor.py`:
bemf_phase_a / bemf_shape / electrical_ode RHS / torque, defaults
p=21, r_s=85e-3, l_s=50e-6, k_e=0.0955, bemf_smoothing=0.0.
"""
import numpy as np

P = 21
R_S = np.float32(85e-3)
L_S = np.float32(50e-6)
K_E = np.float32(0.0955)
TWO_PI = np.float32(2 * np.pi)


def bemf_phase_a(theta):
    theta = np.mod(theta + np.float32(5 * np.pi / 6), TWO_PI).astype(np.float32)
    return np.where(
        theta < TWO_PI / 3,
        np.float32(1.0),
        np.where(
            theta < np.float32(np.pi),
            np.float32(1.0) + (np.float32(-1.0) - np.float32(1.0)) * (theta - TWO_PI / 3) / (np.float32(np.pi) - TWO_PI / 3),
            np.where(
                theta < np.float32(5 * np.pi / 3),
                np.float32(-1.0),
                np.float32(-1.0) + (np.float32(1.0) - np.float32(-1.0)) * (theta - np.float32(5 * np.pi / 3)) / (TWO_PI - np.float32(5 * np.pi / 3)),
            ),
        ),
    ).astype(np.float32)


def bemf_shape(epsilon):
    theta = np.mod(epsilon, TWO_PI).astype(np.float32)
    return np.stack((
        bemf_phase_a(theta),
        bemf_phase_a(np.mod(theta - TWO_PI / 3, TWO_PI).astype(np.float32)),
        bemf_phase_a(np.mod(theta - np.float32(4 * np.pi / 3), TWO_PI).astype(np.float32)),
    ))


def electrical_rhs(state, u_abc, omega):
    state = np.asarray(state, dtype=np.float32)
    single = state.ndim == 1
    s = state[None] if single else state
    u = np.asarray(u_abc, dtype=np.float32)
    u = u[None] if single else u
    w = np.asarray(omega, dtype=np.float32).reshape(-1, 1)
    i = s[..., 0:3]
    f = bemf_shape(s[..., 3]).T  # (N,3)
    e = K_E * w * f
    u_n = (np.sum(u, axis=-1, keepdims=True) - R_S * np.sum(i, axis=-1, keepdims=True)
           - np.sum(e, axis=-1, keepdims=True)) / np.float32(3.0)
    di = (u - u_n - R_S * i - e) / L_S
    deps = np.float32(P) * np.asarray(omega, dtype=np.float32)
    if single:
        return di[0].astype(np.float32), np.float32(deps[0]) if np.ndim(deps) else np.float32(P * float(w[0, 0]))
    return di.astype(np.float32), deps


def torque(state):
    state = np.asarray(state, dtype=np.float32)
    return np.float32(K_E * np.sum(bemf_shape(state[..., 3]) * state[..., 0:3], axis=-1))
