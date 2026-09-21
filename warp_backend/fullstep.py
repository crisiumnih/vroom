"""Phase 2a: full 100us control step in Warp (float64).

Chain per step (mirrors GEM Cont-SC-BLDC-v0, control_space='dq',
ContB6BridgeConverter, interlocking_time=0, IdealVoltageSupply):
  abc_duty = t32 . q(u_dq, eps);  u_abc = abc_duty/2 * U_SUP
  coupled RK4 over [omega, ia, ib, ic, eps] with held u_dq/disturbance.
Load: a=0.01, b=0.01, c=0, j_total=0.003, tau_decay=1e-3 (+DisturbedLoad).
"""
import warp as wp

P = wp.float64(21.0)
R_S = wp.float64(85e-3)
L_S = wp.float64(50e-6)
K_E = wp.float64(0.0955)
U_SUP = wp.float64(44.4)
J_TOTAL = wp.float64(0.003)
LOAD_A = wp.float64(0.01)
LOAD_B = wp.float64(0.01)
LOAD_C = wp.float64(0.0)
W_LIM = wp.float64(0.01 / 0.003 * 1e-3)
LIN_F = wp.float64(0.003 / 1e-3)
SQRT3_2 = wp.float64(0.8660254037844386)
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
def bemf3(eps: wp.float64):
    th = eps - wp.floor(eps / TWO_PI) * TWO_PI
    fa = bemf_phase_a(th)
    fb = bemf_phase_a(th - TWO_PI / wp.float64(3.0))
    fc = bemf_phase_a(th - wp.float64(4.0) * PI / wp.float64(3.0))
    return fa, fb, fc


@wp.func
def full_rhs(
    om: wp.float64, ia: wp.float64, ib: wp.float64, ic: wp.float64, eps: wp.float64,
    ud: wp.float64, uq: wp.float64, dist: wp.float64,
):
    ce = wp.cos(eps)
    se = wp.sin(eps)
    alpha = ce * ud - se * uq
    beta = se * ud + ce * uq
    ua = alpha * wp.float64(0.5) * U_SUP
    ub = (wp.float64(-0.5) * alpha + SQRT3_2 * beta) * wp.float64(0.5) * U_SUP
    uc = (wp.float64(-0.5) * alpha - SQRT3_2 * beta) * wp.float64(0.5) * U_SUP
    fa, fb, fc = bemf3(eps)
    ea = K_E * om * fa
    eb = K_E * om * fb
    ec = K_E * om * fc
    un = (ua + ub + uc - R_S * (ia + ib + ic) - (ea + eb + ec)) / wp.float64(3.0)
    dia = (ua - un - R_S * ia - ea) / L_S
    dib = (ub - un - R_S * ib - eb) / L_S
    dic = (uc - un - R_S * ic - ec) / L_S
    tq = K_E * (fa * ia + fb * ib + fc * ic)
    sgn = wp.where(om > wp.float64(0.0), wp.float64(1.0),
          wp.where(om < wp.float64(0.0), wp.float64(-1.0), wp.float64(0.0)))
    a_term = wp.where(wp.abs(om) > W_LIM, sgn * LOAD_A, LIN_F * om)
    static = sgn * LOAD_C * om * om + LOAD_B * om + a_term
    dom = (tq - static - dist) / J_TOTAL
    deps = P * om
    return dom, dia, dib, dic, deps, tq


@wp.kernel
def fullstep_kernel(
    states: wp.array(dtype=wp.float64, ndim=2),  # [om, ia, ib, ic, eps]
    actions: wp.array(dtype=wp.float64, ndim=3),  # [step, env, 2] normalized dq
    dists: wp.array(dtype=wp.float64, ndim=2),  # [step, env] disturbance Nm
    dt: wp.float64,
    steps: wp.int32,
    trace: wp.array(dtype=wp.float64, ndim=3),  # [step, env, 6]: om,ia,ib,ic,eps,tq
):
    tid = wp.tid()
    om = states[tid, 0]
    ia = states[tid, 1]
    ib = states[tid, 2]
    ic = states[tid, 3]
    eps = states[tid, 4]
    h = dt
    half = wp.float64(0.5)
    sixth = wp.float64(1.0) / wp.float64(6.0)
    two = wp.float64(2.0)
    for k in range(steps):
        ud = actions[k, tid, 0]
        uq = actions[k, tid, 1]
        dist = dists[k, tid]
        m1, a1, b1, c1, e1, _t1 = full_rhs(om, ia, ib, ic, eps, ud, uq, dist)
        m2, a2, b2, c2, e2, _t2 = full_rhs(om + h * half * m1, ia + h * half * a1, ib + h * half * b1,
                      ic + h * half * c1, eps + h * half * e1, ud, uq, dist)
        m3, a3, b3, c3, e3, _t3 = full_rhs(om + h * half * m2, ia + h * half * a2, ib + h * half * b2,
                      ic + h * half * c2, eps + h * half * e2, ud, uq, dist)
        m4, a4, b4, c4, e4, _t4 = full_rhs(om + h * m3, ia + h * a3, ib + h * b3,
                      ic + h * c3, eps + h * e3, ud, uq, dist)
        om = om + h * (m1 + two * m2 + two * m3 + m4) * sixth
        ia = ia + h * (a1 + two * a2 + two * a3 + a4) * sixth
        ib = ib + h * (b1 + two * b2 + two * b3 + b4) * sixth
        ic = ic + h * (c1 + two * c2 + two * c3 + c4) * sixth
        eps = eps + h * (e1 + two * e2 + two * e3 + e4) * sixth
        _m5, _a5, _b5, _c5, _e5, tq = full_rhs(om, ia, ib, ic, eps, ud, uq, dist)
        trace[k, tid, 0] = om
        trace[k, tid, 1] = ia
        trace[k, tid, 2] = ib
        trace[k, tid, 3] = ic
        trace[k, tid, 4] = eps
        trace[k, tid, 5] = tq


@wp.kernel
def fullstep_once(
    states: wp.array(dtype=wp.float64, ndim=2),
    actions: wp.array(dtype=wp.float64, ndim=2),
    dists: wp.array(dtype=wp.float64),
    dt: wp.float64,
    out: wp.array(dtype=wp.float64, ndim=2),
):
    tid = wp.tid()
    om = states[tid, 0]
    ia = states[tid, 1]
    ib = states[tid, 2]
    ic = states[tid, 3]
    eps = states[tid, 4]
    ud = actions[tid, 0]
    uq = actions[tid, 1]
    dist = dists[tid]
    h = dt
    half = wp.float64(0.5)
    sixth = wp.float64(1.0) / wp.float64(6.0)
    two = wp.float64(2.0)
    m1, a1, b1, c1, e1, _t1 = full_rhs(om, ia, ib, ic, eps, ud, uq, dist)
    m2, a2, b2, c2, e2, _t2 = full_rhs(om + h * half * m1, ia + h * half * a1, ib + h * half * b1,
                  ic + h * half * c1, eps + h * half * e1, ud, uq, dist)
    m3, a3, b3, c3, e3, _t3 = full_rhs(om + h * half * m2, ia + h * half * a2, ib + h * half * b2,
                  ic + h * half * c2, eps + h * half * e2, ud, uq, dist)
    m4, a4, b4, c4, e4, _t4 = full_rhs(om + h * m3, ia + h * a3, ib + h * b3,
                  ic + h * c3, eps + h * e3, ud, uq, dist)
    om = om + h * (m1 + two * m2 + two * m3 + m4) * sixth
    ia = ia + h * (a1 + two * a2 + two * a3 + a4) * sixth
    ib = ib + h * (b1 + two * b2 + two * b3 + b4) * sixth
    ic = ic + h * (c1 + two * c2 + two * c3 + c4) * sixth
    eps = eps + h * (e1 + two * e2 + two * e3 + e4) * sixth
    _m5, _a5, _b5, _c5, _e5, tq = full_rhs(om, ia, ib, ic, eps, ud, uq, dist)
    states[tid, 0] = om
    states[tid, 1] = ia
    states[tid, 2] = ib
    states[tid, 3] = ic
    states[tid, 4] = eps
    out[tid, 0] = om
    out[tid, 1] = ia
    out[tid, 2] = ib
    out[tid, 3] = ic
    out[tid, 4] = eps
    out[tid, 5] = tq


@wp.kernel
def reset_row_kernel(states: wp.array(dtype=wp.float64, ndim=2), row: wp.int32):
    states[row, 0] = wp.float64(0.0)
    states[row, 1] = wp.float64(0.0)
    states[row, 2] = wp.float64(0.0)
    states[row, 3] = wp.float64(0.0)
    states[row, 4] = wp.float64(0.0)


@wp.kernel
def fullstep_masked(
    states: wp.array(dtype=wp.float64, ndim=2),
    actions: wp.array(dtype=wp.float64, ndim=2),
    dists: wp.array(dtype=wp.float64),
    live: wp.array(dtype=wp.int32),
    dt: wp.float64,
    out: wp.array(dtype=wp.float64, ndim=2),
):
    """One RK4 step like fullstep_once, but frozen lanes keep their state.

    Lanes with live==0 still write their (unchanged) state to out, so the
    caller can use out unconditionally; executed-step accounting stays host
    side via the same mask.
    """
    tid = wp.tid()
    ud = actions[tid, 0]
    uq = actions[tid, 1]
    dist = dists[tid]
    if live[tid] != 0:
        om = states[tid, 0]
        ia = states[tid, 1]
        ib = states[tid, 2]
        ic = states[tid, 3]
        eps = states[tid, 4]
        h = dt
        half = wp.float64(0.5)
        sixth = wp.float64(1.0) / wp.float64(6.0)
        two = wp.float64(2.0)
        m1, a1, b1, c1, e1, _t1 = full_rhs(om, ia, ib, ic, eps, ud, uq, dist)
        m2, a2, b2, c2, e2, _t2 = full_rhs(om + h * half * m1, ia + h * half * a1, ib + h * half * b1,
                      ic + h * half * c1, eps + h * half * e1, ud, uq, dist)
        m3, a3, b3, c3, e3, _t3 = full_rhs(om + h * half * m2, ia + h * half * a2, ib + h * half * b2,
                      ic + h * half * c2, eps + h * half * e2, ud, uq, dist)
        m4, a4, b4, c4, e4, _t4 = full_rhs(om + h * m3, ia + h * a3, ib + h * b3,
                      ic + h * c3, eps + h * e3, ud, uq, dist)
        om = om + h * (m1 + two * m2 + two * m3 + m4) * sixth
        ia = ia + h * (a1 + two * a2 + two * a3 + a4) * sixth
        ib = ib + h * (b1 + two * b2 + two * b3 + b4) * sixth
        ic = ic + h * (c1 + two * c2 + two * c3 + c4) * sixth
        eps = eps + h * (e1 + two * e2 + two * e3 + e4) * sixth
        states[tid, 0] = om
        states[tid, 1] = ia
        states[tid, 2] = ib
        states[tid, 3] = ic
        states[tid, 4] = eps
    out[tid, 0] = states[tid, 0]
    out[tid, 1] = states[tid, 1]
    out[tid, 2] = states[tid, 2]
    out[tid, 3] = states[tid, 3]
    out[tid, 4] = states[tid, 4]
    _m5, _a5, _b5, _c5, _e5, tq = full_rhs(states[tid, 0], states[tid, 1], states[tid, 2],
                                          states[tid, 3], states[tid, 4], ud, uq, dist)
    out[tid, 5] = tq
