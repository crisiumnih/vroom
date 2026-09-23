"""Benchmark matrix: what gets evaluated, fixed before execution (PR4).

Cartesian product of motors x cases x sensing conditions x controllers x
repetitions, with the plan's fixed repetition seeds (5101-5103) and angles.
No execution here — this only freezes the work list so runs can't silently
drop or add combinations. Motor populations (train/val/test) arrive with
the M0 anchor; this builder takes plain motor records as input.
"""
import hashlib
import json

REPETITIONS = [
    {'angle': 0.0, 'noise_seed': 5101, 'mirror_signs': False},
    {'angle': 2.0943951023931953, 'noise_seed': 5102, 'mirror_signs': False},
    {'angle': 4.1887902047863905, 'noise_seed': 5103, 'mirror_signs': True},
]
CONDITIONS = ('N0', 'N1', 'N2')
CONTROLLERS = ('B0', 'B1', 'B2', 'R0')


def build_matrix(motors, cases, conditions=CONDITIONS, controllers=CONTROLLERS):
    """motors: [{name, ...}], cases: [case ids]. Returns row list + digest."""
    rows = []
    for m in motors:
        for c in cases:
            for cond in conditions:
                for ctl in controllers:
                    for r, rep in enumerate(REPETITIONS):
                        rows.append({'motor': m['name'], 'case': c, 'condition': cond,
                                     'controller': ctl, 'repetition': r, **rep})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return rows, digest
