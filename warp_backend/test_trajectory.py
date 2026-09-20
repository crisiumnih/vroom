"""Phase 1b test: persistent Warp RK4 vs numpy RK4 + throughput bench."""
import time
import numpy as np
import warp as wp
from warp_backend import trajectory as traj
from warp_backend import ref as ref32  # noqa: F401 (kept for lineage)

P, R_S, L_S, K_E = 21.0, 85e-3, 50e-6, 0.0955
DT = 1e-4
STEPS = 2000  # 0.2 s, one inner episode length
N = 64

wp.init()


def numpy_rollout(states, voltages, omegas):
    s = states.copy()
    trace = np.zeros((STEPS, s.shape[0], 4))
    h = DT
    for k in range(STEPS):
        def rhs(x):
            i = x[:, 0:3]
            th = np.mod(x[:, 3] + 5 * np.pi / 6, 2 * np.pi)  # alignment shift, as GEM/Warp
            ths = np.stack([th, np.mod(th - 2 * np.pi / 3, 2 * np.pi), np.mod(th - 4 * np.pi / 3, 2 * np.pi)])
            f = np.where(ths < 2 * np.pi / 3, 1.0,
                 np.where(ths < np.pi, 1.0 - 2.0 * (ths - 2 * np.pi / 3) / (np.pi - 2 * np.pi / 3),
                 np.where(ths < 5 * np.pi / 3, -1.0, -1.0 + 2.0 * (ths - 5 * np.pi / 3) / (2 * np.pi - 5 * np.pi / 3))))
            e = K_E * omegas[:, None] * f.T
            u_n = (voltages.sum(1, keepdims=True) - R_S * i.sum(1, keepdims=True) - e.sum(1, keepdims=True)) / 3.0
            di = (voltages - u_n - R_S * i - e) / L_S
            return np.concatenate([di, (P * omegas)[:, None]], axis=1)
        k1 = rhs(s)
        k2 = rhs(s + h / 2 * k1)
        k3 = rhs(s + h / 2 * k2)
        k4 = rhs(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        trace[k] = s
    return trace


def warp_rollout(states, voltages, omegas):
    n = states.shape[0]
    d_s = wp.array(states, dtype=wp.float64, device="cuda:0")
    d_u = wp.array(voltages, dtype=wp.float64, device="cuda:0")
    d_om = wp.array(omegas, dtype=wp.float64, device="cuda:0")
    d_tr = wp.zeros((STEPS, n, 4), dtype=wp.float64, device="cuda:0")
    wp.launch(traj.rollout_kernel, dim=n,
              inputs=[d_s, d_u, d_om, wp.float64(DT), STEPS, d_tr], device="cuda:0")
    wp.synchronize()
    return d_tr.numpy()


def make_inputs():
    rng = np.random.default_rng(5)
    states = np.zeros((N, 4))
    states[:, 0:3] = rng.uniform(-1.5, 1.5, (N, 3))
    states[:, 3] = rng.uniform(0, 2 * np.pi, N)
    voltages = rng.uniform(-22, 22, (N, 3))
    omegas = rng.uniform(-10, 10, N)
    return states, voltages, omegas


def test_persistent_rollout_matches_numpy():
    states, voltages, omegas = make_inputs()
    tw = warp_rollout(states, voltages, omegas)
    tn = numpy_rollout(states, voltages, omegas)
    err = np.abs(tw - tn)
    print(f"rollout: max_abs={err.max():.2e} mean={err.mean():.2e}")
    assert err.max() < 1e-9


def test_rollout_bench():
    states, voltages, omegas = make_inputs()
    warp_rollout(states, voltages, omegas)  # compile
    t = time.perf_counter()
    warp_rollout(states, voltages, omegas)
    dt_w = time.perf_counter() - t
    t = time.perf_counter()
    numpy_rollout(states, voltages, omegas)
    dt_n = time.perf_counter() - t
    sim_s = STEPS * DT * N
    print(f"warp {dt_w:.2f}s vs numpy {dt_n:.2f}s for {sim_s:.1f} env-seconds "
          f"(warp {sim_s/dt_w:.0f}x realtime, numpy {sim_s/dt_n:.0f}x)")
