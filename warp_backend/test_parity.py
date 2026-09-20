"""Parity: Warp batched RHS vs numpy transcription and vs GEM (float64)."""
import numpy as np
import warp as wp
from warp_backend import ref, kernels

wp.init()


def run_warp(states, voltages, omegas, device="cuda:0"):
    n = states.shape[0]
    d_states = wp.array(states.astype(np.float32), dtype=wp.float32, device=device)
    d_u = wp.array(voltages.astype(np.float32), dtype=wp.float32, device=device)
    d_om = wp.array(omegas.astype(np.float32), dtype=wp.float32, device=device)
    d_di = wp.zeros((n, 3), dtype=wp.float32, device=device)
    d_deps = wp.zeros(n, dtype=wp.float32, device=device)
    d_tq = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(kernels.electrical_rhs_kernel, dim=n,
              inputs=[d_states, d_u, d_om, d_di, d_deps, d_tq], device=device)
    wp.synchronize_device(device) if hasattr(wp, "synchronize_device") else wp.synchronize()
    return d_di.numpy(), d_deps.numpy(), d_tq.numpy()


def test_warp_matches_numpy_transcription():
    rng = np.random.default_rng(7)
    n = 10000
    states = np.zeros((n, 4), dtype=np.float32)
    states[:, 0:3] = rng.uniform(-4, 4, (n, 3)).astype(np.float32)
    states[:, 3] = rng.uniform(-50, 50, n).astype(np.float32)  # incl. negative eps
    voltages = rng.uniform(-22, 22, (n, 3)).astype(np.float32)
    omegas = rng.uniform(-300, 300, n).astype(np.float32)
    di_ref = np.stack([ref.electrical_rhs(s, u, w)[0] for s, u, w in zip(states, voltages, omegas)])
    deps_ref = np.float32(21) * omegas
    tq_ref = np.array([ref.torque(s) for s in states], dtype=np.float32)
    di_w, deps_w, tq_w = run_warp(states, voltages, omegas)
    # NOTE: di/torque suffer float32 catastrophic cancellation in near-balanced
    # residuals (20 V terms cancelling to ~1e-3 V), plus GPU FMA contraction.
    # Absolute bounds cover rounding; tight MEAN bounds catch systematic bias.
    assert np.abs(di_w - di_ref).max() < 2.0
    assert (np.abs(di_w - di_ref) / np.maximum(np.abs(di_ref), 100.0)).mean() < 5e-6
    assert np.abs(deps_w - deps_ref).max() == 0.0
    assert np.abs(tq_w - tq_ref).max() < 2e-3
    assert (np.abs(tq_w - tq_ref) / np.maximum(np.abs(tq_ref), 1e-3)).mean() < 5e-6


def test_transcription_matches_gem():
    from gym_electric_motor.physical_systems.electric_motors.brushless_dc_motor import BrushlessDCMotor
    import sys
    sys.path.insert(0, "src")
    motor = BrushlessDCMotor()
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        s = np.array([rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-10, 10)])
        u = [rng.uniform(-22, 22) for _ in range(3)]
        w = float(rng.uniform(-300, 300))
        gem_rhs = motor.electrical_ode(s, u, w)
        my_di, my_deps = ref.electrical_rhs(s.astype(np.float32), np.float32(u), np.float32(w))
        gem_tq = motor.torque(s)
        my_tq = ref.torque(s.astype(np.float32))
        worst = max(worst, np.abs(gem_rhs[:3] - my_di).max(), abs(gem_tq - my_tq) * 1000)
        assert np.abs(gem_rhs[:3] - my_di).max() < 2.0
        assert abs(gem_rhs[3] - my_deps) / max(abs(gem_rhs[3]), 1e-3) < 1e-6
        assert abs(gem_tq - my_tq) < 2e-3
    print("gem transcription worst (di-abs scale):", worst)
    assert worst < 2.0
