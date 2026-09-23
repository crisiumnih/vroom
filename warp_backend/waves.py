"""Analytic benchmark waveforms (master_plan sect.6, PR2).

One language for steps, squares, ramps, quadratic ramps, sinusoids,
reversals and disturbances. Segments are plain JSON-able tuples; both
simulators consume them as dense event tables through their UNCHANGED
table code (Warp build_tables / GEM schedule_value: last event with
round(t/dt) <= k), so agreement holds by construction. Bench code can
evaluate segments directly via sample().

Right-continuous convention: an event at t0 applies at t >= t0. Tables
are sampled at integer steps t = k*dt.

Label discipline: S4 is a quadratic REFERENCE in time; D4 is a quadratic
LOAD LAW in speed. Different questions, never shared labels.
"""
import math

# Segment forms: ("hold", t0, t1, x)
#                ("ramp", t0, t1, x0, x1)              linear in time
#                ("qramp", t0, t1, x0, x1)             quadratic rise: x0+(x1-x0)*u^2
#                ("qramp_fall", t0, t1, x0, x1)        quadratic fall: x0+(x1-x0)*(1-(1-u)^2)
#                ("sine", t0, t1, amp, freq_hz, phase, offset)
#                ("square", t0, t1, amp, half_period)  amp*(-1)^floor((t-t0)/half)


def sample(segments, t):
    """Value at time t (right-continuous). Outside all segments: 0.0."""
    v = 0.0
    for seg in segments:
        kind = seg[0]
        _, t0, t1 = seg[0], seg[1], seg[2]
        if t < t0 or t >= t1:
            continue
        if kind == "hold":
            v = seg[3]
        elif kind == "ramp":
            _, _, _, x0, x1 = seg
            u = (t - t0) / (t1 - t0)
            v = x0 + (x1 - x0) * u
        elif kind == "qramp":
            _, _, _, x0, x1 = seg
            u = (t - t0) / (t1 - t0)
            v = x0 + (x1 - x0) * u * u
        elif kind == "qramp_fall":
            _, _, _, x0, x1 = seg
            u = (t - t0) / (t1 - t0)
            v = x0 + (x1 - x0) * (1.0 - (1.0 - u) * (1.0 - u))
        elif kind == "sine":
            _, _, _, amp, freq, phase, offset = seg
            v = offset + amp * math.sin(2.0 * math.pi * freq * (t - t0) + phase)
        elif kind == "square":
            _, _, _, amp, half = seg
            v = amp * (-1.0) ** math.floor((t - t0) / half)
        else:
            raise ValueError(f"Unknown segment {kind}")
    return v


def to_schedule(segments, dt, duration_s):
    """Dense [[t, v]] event table sampled at integer steps (both backends)."""
    n = int(round(duration_s / dt))
    return [[k * dt, sample(segments, k * dt)] for k in range(n + 1)]


def scale_waves(segments, amp=0.9, time=1.25):
    """Validation transform (master_plan sect.5): amplitudes x0.9, time x1.25."""
    out = []
    for seg in segments:
        kind = seg[0]
        t0, t1 = seg[1] * time, seg[2] * time
        if kind == "hold":
            out.append(("hold", t0, t1, seg[3] * amp))
        elif kind in ("ramp", "qramp", "qramp_fall"):
            out.append((kind, t0, t1, seg[3] * amp, seg[4] * amp))
        elif kind == "sine":
            out.append(("sine", t0, t1, seg[3] * amp, seg[4] / time, seg[5], seg[6] * amp))
        elif kind == "square":
            out.append(("square", t0, t1, seg[3] * amp, seg[4] * time))
        else:
            raise ValueError(f"Unknown segment {kind}")
    return out


# --- Coupled speed cases S1-S6 (normalized x = omega*/Omega_B, 8 s) ---

def S1():
    return [("hold", 0, 1, 0.0), ("hold", 1, 3, 0.25), ("hold", 3, 5, 0.75),
            ("hold", 5, 7, 0.50), ("hold", 7, 8, 0.0)]


def S2():
    return [("hold", 0, 1, 0.0), ("square", 1, 7, 0.50, 1.0), ("hold", 7, 8, 0.0)]


def S3():
    return [("hold", 0, 1, 0.0), ("ramp", 1, 3, 0.0, 0.75), ("hold", 3, 4, 0.75),
            ("ramp", 4, 6, 0.75, 0.0), ("hold", 6, 8, 0.0)]


def S4():
    return [("hold", 0, 1, 0.0), ("qramp", 1, 3, 0.0, 0.75), ("hold", 3, 4, 0.75),
            ("qramp_fall", 4, 6, 0.75, 0.0), ("hold", 6, 8, 0.0)]


def S5():
    return [("hold", 0, 1, 0.0), ("sine", 1, 7, 0.50, 0.50, 0.0, 0.0),
            ("hold", 7, 8, 0.0)]


def S6():
    return [("hold", 0, 1, 0.0), ("hold", 1, 2.5, 0.75), ("hold", 2.5, 4, -0.75),
            ("hold", 4, 5.5, 0.25), ("hold", 5.5, 8, 0.0)]


# --- Disturbance cases D1-D6 (normalized torque / bus, 8 s) ---

def D1():
    return [("hold", 0, 2, 0.0), ("hold", 2, 4, 0.30), ("hold", 4, 8, 0.0)]


def D2():
    segs = [("hold", 0, 2, 0.0)]
    t = 2.0
    loaded = True
    while t < 6.0:
        segs.append(("hold", t, min(t + 0.5, 6.0), 0.40 if loaded else 0.0))
        loaded = not loaded
        t += 0.5
    segs.append(("hold", 6, 8, 0.0))
    return segs


def D3():
    """Supply multiplier: 0.90 sag on [2,4), 1.0 otherwise."""
    return [("hold", 0, 2, 1.0), ("hold", 2, 4, 0.90), ("hold", 4, 8, 1.0)]


def D6_text():
    """Combined (on S1): Text=.20 on [3.5,4.5), Vdc=.90 on [3.25,4.75)."""
    return ([("hold", 0, 3.5, 0.0), ("hold", 3.5, 4.5, 0.20), ("hold", 4.5, 8, 0.0)],
            [("hold", 0, 3.25, 1.0), ("hold", 3.25, 4.75, 0.90), ("hold", 4.75, 8, 1.0)])


# --- Inner current cases C1-C4 (normalized q/d, 0.25 s) ---

def C1():
    return ([("hold", 0, 0.02, 0.0), ("hold", 0.02, 0.08, 0.50),
             ("hold", 0.08, 0.14, -0.50), ("hold", 0.14, 0.20, 0.80),
             ("hold", 0.20, 0.25, 0.0)],
            [("hold", 0, 0.25, 0.0)])


def C2():
    return ([("hold", 0, 0.02, 0.0), ("square", 0.02, 0.22, 0.50, 0.02),
             ("hold", 0.22, 0.25, 0.0)],
            [("hold", 0, 0.25, 0.0)])


def C3():
    return ([("hold", 0, 0.02, 0.0), ("sine", 0.02, 0.22, 0.50, 20.0, 0.0, 0.0),
             ("hold", 0.22, 0.25, 0.0)],
            [("hold", 0, 0.25, 0.0)])


def C4():
    return ([("hold", 0, 0.02, 0.0), ("hold", 0.02, 0.22, 0.40),
             ("hold", 0.22, 0.25, 0.0)],
            [("hold", 0, 0.08, 0.0), ("hold", 0.08, 0.14, 0.20),
             ("hold", 0.14, 0.20, -0.20), ("hold", 0.20, 0.25, 0.0)])


CASES = {'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5, 'S6': S6,
         'D1': D1, 'D2': D2, 'D3': D3, 'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4}
