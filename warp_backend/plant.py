"""Motor-plant parameters: one struct for both backends (master_plan PR1).

Legacy-L0 is the retained assumed plant (regression only). F1 members are
expressed as multipliers of the M0 anchor; until M0 is measured, infra tests
use L0-relative multipliers. Field meanings match the GEM fork conventions
(master_plan sect.3): l_s is equivalent phase inductance (L-M), k_e uses
mechanical rad/s, j_total = j_rotor + j_load, T_ext subtracts shaft torque.

Friction smoothing time (1 ms) is fixed by the plan, not a motor parameter;
W_LIM/LIN_F are derived per lane from (load_a, j_total).
"""
import hashlib
import json
import math
from dataclasses import dataclass, asdict

FRICTION_SMOOTH_S = 1e-3

# Column order for the per-lane device table (warp_backend/fullstep).
COLS = ('p', 'r_s', 'l_s', 'k_e', 'u_sup', 'j_total',
        'load_a', 'load_b', 'load_c', 'w_lim', 'lin_f')


@dataclass(frozen=True)
class MotorParams:
    p: int = 21
    r_s: float = 85e-3
    l_s: float = 50e-6
    k_e: float = 0.0955
    j_rotor: float = 0.0001
    supply_v: float = 44.4
    load_a: float = 0.01
    load_b: float = 0.01
    load_c: float = 0.0
    j_load: float = 0.0029

    def validated(self):
        if not isinstance(self.p, int) or self.p <= 0:
            raise ValueError(f'p must be a positive int, got {self.p}')
        for k in ('r_s', 'l_s', 'k_e', 'j_rotor', 'supply_v', 'j_load'):
            v = getattr(self, k)
            if not math.isfinite(v) or not v > 0:
                raise ValueError(f'{k} must be finite positive, got {v}')
        for k in ('load_a', 'load_b', 'load_c'):
            v = getattr(self, k)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f'{k} must be finite non-negative, got {v}')
        return self

    @property
    def j_total(self):
        return self.j_rotor + self.j_load

    def row(self):
        """Per-lane device table row in COLS order (derived W_LIM/LIN_F)."""
        self.validated()
        jt = self.j_total
        return (float(self.p), self.r_s, self.l_s, self.k_e, self.supply_v, jt,
                self.load_a, self.load_b, self.load_c,
                self.load_a / jt * FRICTION_SMOOTH_S, jt / FRICTION_SMOOTH_S)

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


LEGACY_L0 = MotorParams()


def from_legacy_dict(plant):
    """Build MotorParams from a legacy plant dict (motor/load/supply_v keys)."""
    m, lo = plant['motor'], plant['load']
    return MotorParams(
        p=int(m['p']), r_s=m['r_s'], l_s=m['l_s'], k_e=m['k_e'],
        j_rotor=m['j_rotor'], supply_v=plant['supply_v'],
        load_a=lo['a'], load_b=lo['b'], load_c=lo['c'], j_load=lo['j_load'],
    ).validated()


def resolve_plants(plant_or_list, n):
    """Broadcast one legacy dict / MotorParams to n lanes, or validate a list.

    Returns a list of MotorParams length n. Training-time per-episode motor
    draws are the runner's job; this only resolves storage.
    """
    if isinstance(plant_or_list, (list, tuple)):
        if len(plant_or_list) != n:
            raise ValueError(f'{len(plant_or_list)} plants for {n} lanes')
        out = [p if isinstance(p, MotorParams) else from_legacy_dict(p)
               for p in plant_or_list]
    else:
        p = plant_or_list if isinstance(plant_or_list, MotorParams) else from_legacy_dict(plant_or_list)
        out = [p] * n
    return [p.validated() for p in out]
