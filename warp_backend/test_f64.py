"""Phase 1a: float64 Warp RHS vs GEM float64 — tight parity."""
import numpy as np
import warp as wp
from warp_backend import kernels64

wp.init()


def run_warp64(states, voltages, omegas, device="cuda:0"):
    n = states.shape[0]
    d_states = wp.array(states.astype(np.float64), dtype=wp.float64, device=device)
    d_u = wp.array(voltages.astype(np.float64), dtype=wp.float64, device=device)
    d_om = wp.array(omegas.astype(np.float64), dtype=wp.float64, device=device)
    d_di = wp.zeros((n, 3), dtype=wp.float64, device=device)
    d_deps = wp.zeros(n, dtype=wp.float64, device=device)
    d_tq = wp.zeros(n, dtype=wp.float64, device=device)
    wp.launch(kernels64.electrical_rhs_kernel, dim=n,
              inputs=[d_states, d_u, d_om, d_di, d_deps, d_tq], device=device)
    wp.synchronize()
    return d_di.numpy(), d_deps.numpy(), d_tq.numpy()


def test_f64_matches_gem_tight():
    import sys
    sys.path.insert(0, "src")
    from gym_electric_motor.physical_systems.electric_motors.brushless_dc_motor import BrushlessDCMotor
    motor = BrushlessDCMotor()
    rng = np.random.default_rng(21)
    n = 2000
    states = np.zeros((n, 4))
    states[:, 0:3] = rng.uniform(-4, 4, (n, 3))
    states[:, 3] = rng.uniform(-50, 50, n)
    voltages = rng.uniform(-22, 22, (n, 3))
    omegas = rng.uniform(-300, 300, n)
    di_w, deps_w, tq_w = run_warp64(states, voltages, omegas)
    worst_di, worst_tq, worst_deps = 0.0, 0.0, 0.0
    for i in range(n):
        g = motor.electrical_ode(states[i], list(voltages[i]), float(omegas[i]))
        worst_di = max(worst_di, np.abs(g[:3] - di_w[i]).max() / max(np.abs(g[:3]).max(), 1.0))
        worst_deps = max(worst_deps, abs(g[3] - deps_w[i]) / max(abs(g[3]), 1e-9))
        worst_tq = max(worst_tq, abs(motor.torque(states[i]) - tq_w[i]) / max(abs(motor.torque(states[i])), 1e-9))
    print(f"f64 vs GEM: di_rel={worst_di:.2e} deps_rel={worst_deps:.2e} tq_rel={worst_tq:.2e}")
    assert worst_di < 1e-9 and worst_deps < 1e-12 and worst_tq < 1e-9
