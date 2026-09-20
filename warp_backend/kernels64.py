"""Batched Warp kernels for the BLDC electrical RHS + torque (spike)."""
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
    ramp_down = wp.float64(1.0) + (wp.float64(-2.0)) * (t - lo1) / (PI - lo1)
    ramp_up = wp.float64(-1.0) + wp.float64(2.0) * (t - hi5) / (TWO_PI - hi5)
    mid = wp.where(t < PI, ramp_down, wp.float64(-1.0))
    tail = wp.where(t < hi5, mid, ramp_up)
    return wp.where(t < lo1, wp.float64(1.0), tail)


@wp.kernel
def electrical_rhs_kernel(
    states: wp.array(dtype=wp.float64, ndim=2),
    voltages: wp.array(dtype=wp.float64, ndim=2),
    omegas: wp.array(dtype=wp.float64),
    di_out: wp.array(dtype=wp.float64, ndim=2),
    deps_out: wp.array(dtype=wp.float64),
    torque_out: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    ia = states[tid, 0]
    ib = states[tid, 1]
    ic = states[tid, 2]
    eps = states[tid, 3]
    om = omegas[tid]
    th = eps - wp.floor(eps / TWO_PI) * TWO_PI
    fa = bemf_phase_a(th)
    fb = bemf_phase_a(th - TWO_PI / wp.float64(3.0))
    fc = bemf_phase_a(th - wp.float64(4.0) * PI / wp.float64(3.0))
    # warp % on negative floats follows C fmod; replicate numpy.mod via floor
    # by rewrapping shifted angles into [0, 2pi)
    ea = K_E * om * fa
    eb = K_E * om * fb
    ec = K_E * om * fc
    ua = voltages[tid, 0]
    ub = voltages[tid, 1]
    uc = voltages[tid, 2]
    un = (ua + ub + uc - R_S * (ia + ib + ic) - (ea + eb + ec)) / wp.float64(3.0)
    dia = (ua - un - R_S * ia - ea) / L_S
    dib = (ub - un - R_S * ib - eb) / L_S
    dic = (uc - un - R_S * ic - ec) / L_S
    di_out[tid, 0] = wp.float64(dia)
    di_out[tid, 1] = wp.float64(dib)
    di_out[tid, 2] = wp.float64(dic)
    deps_out[tid] = wp.float64(P * om)
    torque_out[tid] = wp.float64(K_E * (fa * ia + fb * ib + fc * ic))
