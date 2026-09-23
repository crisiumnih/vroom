"""Controller estimates: what PI may know (master_plan sect.8, PR4).

Truth (MotorParams / plant dicts) drives simulation. Estimates drive PI
gains and feedforward. The two records are separate objects; passing a
perturbed plant into a PI constructor must NEVER change a B0 baseline.

- B0 (fixed): estimates frozen from development motors (infra freeze: L0
  values + the legacy 1800 Hz / 5 Hz / damping-1.0 design; reselected with
  M0 data before family training, producing a new frozen record).
- B1 (parameter-configured): estimates filled from the ACTUAL simulated
  motor by rule, except attached load stays nominal (never matched to D5).
"""
import hashlib
import json
import math
from dataclasses import dataclass, asdict

DESIGN = {'current_bandwidth_hz': 1800.0, 'speed_natural_frequency_hz': 5.0,
          'speed_damping': 1.0, 'speed_setpoint_weight': 0.0}


@dataclass(frozen=True)
class ControllerEstimates:
    r_hat: float = 85e-3
    l_hat: float = 50e-6
    ke_hat: float = 0.0955
    j_rotor_hat: float = 0.0001
    j_load_hat: float = 0.0029
    load_b_hat: float = 0.01

    def validated(self):
        for k in ('r_hat', 'l_hat', 'ke_hat', 'j_rotor_hat', 'j_load_hat'):
            v = getattr(self, k)
            if not math.isfinite(v) or not v > 0:
                raise ValueError(f'{k} must be finite positive, got {v}')
        if not math.isfinite(self.load_b_hat) or self.load_b_hat < 0:
            raise ValueError(f'load_b_hat must be finite non-negative, got {self.load_b_hat}')
        return self

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


# Infra freeze: L0 development values. Reselected with M0 data later.
FROZEN_B0 = ControllerEstimates()


def b1_from_truth(plant, j_load_hat=0.0029):
    """Parameter-configured estimates: truth for motor, nominal load.

    j_load_hat defaults to the nominal fixture (never secretly matched to
    the case under test). Returns validated ControllerEstimates.
    """
    m, lo = plant['motor'], plant['load']
    return ControllerEstimates(
        r_hat=m['r_s'], l_hat=m['l_s'], ke_hat=m['k_e'],
        j_rotor_hat=m['j_rotor'], j_load_hat=j_load_hat,
        load_b_hat=lo['b'],
    ).validated()


def design_gains(est, design=DESIGN):
    """The declared PI gain rule, computed from ESTIMATES only.

    Mirrors benchmarks/bldc/cascaded.py:derived_gains core (tested for
    equality in test_estimates.py). One rule for B0 and B1; only the
    estimate source differs.
    """
    est.validated()
    wc = 2 * math.pi * design['current_bandwidth_hz']
    wn = 2 * math.pi * design['speed_natural_frequency_hz']
    inertia = est.j_rotor_hat + est.j_load_hat
    emf_fund = 12 / math.pi**2 * est.ke_hat
    tq_per_iq = 1.5 * emf_fund
    return {
        'current_kp_v_per_a': est.l_hat * wc,
        'current_ki_v_per_as': est.r_hat * wc,
        'current_aw_per_s': wc,
        'speed_kp_a_per_rad_s': (2 * design['speed_damping'] * wn * inertia
                                 - est.load_b_hat) / tq_per_iq,
        'speed_ki_a_per_rad': inertia * wn**2 / tq_per_iq,
        'emf_fundamental_vs_per_rad': emf_fund,
        'torque_per_iq_nm_per_a': tq_per_iq,
    }
