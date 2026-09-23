"""Measurement pipeline N0/N1/N2 (master_plan sect.7, PR3).

True simulator state is NEVER controller input. Each inner step maps true
(om, ia, ib, ic, eps) to sensed values through the declared chain:

  phase A/B: gain + offset + Gaussian noise, then first-order filter;
  phase C: reconstructed as -(A_s + B_s) (two-sensor chain);
  angle: fixed offset, then age (delay buffer);
  speed: passes through (no speed-sensor error in the declared table);
  applied voltage: transport delay (previous command while delay=1).

N0 is exactly identity (legacy behavior, gated by test). Noise is injected
by the caller: production draws from the backend RNG, tests pass explicit
arrays, so device-vs-numpy parity is exact. I_B (amps) parameterizes the
fractional table entries; training/validation freeze one value.
"""

SENSE_VERSION = 'sense-v1'


class SenseConfig:
    def __init__(self, ib_a=0.5, i_noise=0.0, gain_a=1.0, gain_b=1.0,
                 off_a=0.0, off_b=0.0, filt_tau=0.0, ang_off_deg=0.0,
                 ang_age_steps=0, volt_delay_steps=0, ideal=False):
        self.ib_a = float(ib_a)
        self.i_noise = float(i_noise)
        self.gain_a = float(gain_a)
        self.gain_b = float(gain_b)
        self.off_a = float(off_a)
        self.off_b = float(off_b)
        self.filt_tau = float(filt_tau)
        self.ang_off = float(ang_off_deg) * 3.141592653589793 / 180.0
        self.ang_age = int(ang_age_steps)
        self.volt_delay = int(volt_delay_steps)
        self.ideal = bool(ideal)
        if self.i_noise < 0 or self.filt_tau < 0 or self.ang_age < 0 or self.volt_delay < 0:
            raise ValueError('negative sensing parameter')
        if self.volt_delay > 1:
            raise ValueError('transport delay above 1 inner sample is undeclared')

    def digest(self):
        import hashlib, json
        return hashlib.sha256(json.dumps([SENSE_VERSION, self.__dict__], sort_keys=True,
                                         default=float).encode()).hexdigest()


def N0(ib_a=0.5):
    return SenseConfig(ib_a=ib_a, ideal=True)


def N1(ib_a=0.5):
    return SenseConfig(ib_a=ib_a, i_noise=0.01 * ib_a, gain_a=1.005, gain_b=0.995,
                       off_a=0.005 * ib_a, off_b=-0.005 * ib_a, filt_tau=0.10e-3,
                       ang_off_deg=0.5, ang_age_steps=5, volt_delay_steps=1)


def N2(ib_a=0.5):
    return SenseConfig(ib_a=ib_a, i_noise=0.02 * ib_a, gain_a=1.02, gain_b=0.98,
                       off_a=0.02 * ib_a, off_b=-0.02 * ib_a, filt_tau=0.20e-3,
                       ang_off_deg=2.0, ang_age_steps=10, volt_delay_steps=1)


def sense_currents_numpy(cfg, ia, ib, ic, noise_a, noise_b, filt_state, dt=1e-4):
    """Returns (sa, sb, sc, new_filt). filt_state: (2,) previous filtered A/B.
    Ideal (N0): ground truth passes through exactly (legacy)."""
    if cfg.ideal:
        return ia, ib, ic, filt_state
    xa = ia * cfg.gain_a + cfg.off_a + noise_a
    xb = ib * cfg.gain_b + cfg.off_b + noise_b
    if cfg.filt_tau <= 0:
        fa, fb = xa, xb
    else:
        k = dt / (cfg.filt_tau + dt)
        fa = filt_state[0] + k * (xa - filt_state[0])
        fb = filt_state[1] + k * (xb - filt_state[1])
    return fa, fb, -(fa + fb), (fa, fb)


def sense_currents_torch(cfg, torch_mod, ia, ib, ic, noise_a, noise_b, filt_state, dt=1e-4):
    if cfg.ideal:
        return ia, ib, ic
    xa = ia * cfg.gain_a + cfg.off_a + noise_a
    xb = ib * cfg.gain_b + cfg.off_b + noise_b
    if cfg.filt_tau <= 0:
        fa, fb = xa, xb
    else:
        k = dt / (cfg.filt_tau + dt)
        fa = filt_state[:, 0] + k * (xa - filt_state[:, 0])
        fb = filt_state[:, 1] + k * (xb - filt_state[:, 1])
    return fa, fb, -(fa + fb)
