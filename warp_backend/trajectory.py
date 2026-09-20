"""Phase 1b: persistent-device fixed-step RK4 rollout (electrical + epsilon).

One kernel launch integrates K steps; state never leaves the GPU.
Frozen omega + constant voltage per env (matches inner current-loop setup).
Reference: identical RK4 in numpy float64.
"""
import warp as wp

P = wp.float64(21.0)
R_S = wp.float64(85e-3)
L_S = wp.float64(50e-6)
K_E = wp.float64(0.0955)
TWO_PI = wp.float64(2.0 * 3.141592653589793)
PI = wp.float64(3.141592653589793)
SHIFT = wp.float64(5.0 * 3.141592653589793 / 6.0)


@wp.func
def bemf_phase_a(theta: wp.float64):
    t = theta + SHIFT - wp.floor((theta + SHIFT) / TWO_PI) * TWO_PI
    lo1 = TWO_PI / wp.float64(3.0)
    hi5 = wp.float64(5.0) * PI / wp.float64(3.0)
    ramp_down = wp.float64(1.0) + wp.float64(-2.0) * (t - lo1) / (PI - lo1)
    ramp_up = wp.float64(-1.0) + wp.float64(2.0) * (t - hi5) / (TWO_PI - hi5)
    mid = wp.where(t < PI, ramp_down, wp.float64(-1.0))
    tail = wp.where(t < hi5, mid, ramp_up)
    return wp.where(t < lo1, wp.float64(1.0), tail)


@wp.func
def rhs(
    ia: wp.float64, ib: wp.float64, ic: wp.float64, eps: wp.float64,
    ua: wp.float64, ub: wp.float64, uc: wp.float64, om: wp.float64,
):
    th = eps - wp.floor(eps / TWO_PI) * TWO_PI
    fa = bemf_phase_a(th)
    fb = bemf_phase_a(th - TWO_PI / wp.float64(3.0))
    fc = bemf_phase_a(th - wp.float64(4.0) * PI / wp.float64(3.0))
    ea = K_E * om * fa
    eb = K_E * om * fb
    ec = K_E * om * fc
    un = (ua + ub + uc - R_S * (ia + ib + ic) - (ea + eb + ec)) / wp.float64(3.0)
    dia = (ua - un - R_S * ia - ea) / L_S
    dib = (ub - un - R_S * ib - eb) / L_S
    dic = (uc - un - R_S * ic - ec) / L_S
    deps = P * om
    return dia, dib, dic, deps


@wp.kernel
def rollout_kernel(
    states: wp.array(dtype=wp.float64, ndim=2),
    voltages: wp.array(dtype=wp.float64, ndim=2),
    omegas: wp.array(dtype=wp.float64),
    dt: wp.float64,
    steps: wp.int32,
    trace: wp.array(dtype=wp.float64, ndim=3),
):
    tid = wp.tid()
    ia = states[tid, 0]
    ib = states[tid, 1]
    ic = states[tid, 2]
    eps = states[tid, 3]
    ua = voltages[tid, 0]
    ub = voltages[tid, 1]
    uc = voltages[tid, 2]
    om = omegas[tid]
    h = dt
    for k in range(steps):
        k1a, k1b, k1c, k1e = rhs(ia, ib, ic, eps, ua, ub, uc, om)
        k2a, k2b, k2c, k2e = rhs(
            ia + h * k1a / wp.float64(2.0), ib + h * k1b / wp.float64(2.0),
            ic + h * k1c / wp.float64(2.0), eps + h * k1e / wp.float64(2.0), ua, ub, uc, om)
        k3a, k3b, k3c, k3e = rhs(
            ia + h * k2a / wp.float64(2.0), ib + h * k2b / wp.float64(2.0),
            ic + h * k2c / wp.float64(2.0), eps + h * k2e / wp.float64(2.0), ua, ub, uc, om)
        k4a, k4b, k4c, k4e = rhs(
            ia + h * k3a, ib + h * k3b, ic + h * k3c, eps + h * k3e, ua, ub, uc, om)
        ia = ia + h * (k1a + wp.float64(2.0) * k2a + wp.float64(2.0) * k3a + k4a) / wp.float64(6.0)
        ib = ib + h * (k1b + wp.float64(2.0) * k2b + wp.float64(2.0) * k3b + k4b) / wp.float64(6.0)
        ic = ic + h * (k1c + wp.float64(2.0) * k2c + wp.float64(2.0) * k3c + k4c) / wp.float64(6.0)
        eps = eps + h * (k1e + wp.float64(2.0) * k2e + wp.float64(2.0) * k3e + k4e) / wp.float64(6.0)
        trace[k, tid, 0] = ia
        trace[k, tid, 1] = ib
        trace[k, tid, 2] = ic
        trace[k, tid, 3] = eps
