"""Frozen normalized motor populations (master_plan sect.5).

F1 intervals as multipliers of the unmeasured M0 anchor:
  R/R0 in [0.80, 1.50], L/L0 in [0.65, 1.35],
  Ke/Ke0 in [0.85, 1.15], Jrot/J0 in [0.50, 2.00].
Ordered dimensions: (R, L, Ke, J). Deterministic LHS per partition;
M0 = (1,1,1,1) is a training member (development, never unseen test).
Boundary = all 16 min/max corners (edge tests, not extrapolation).
No tuple belongs to more than one partition.
"""
import hashlib
import json

BOUNDS = [(0.80, 1.50), (0.65, 1.35), (0.85, 1.15), (0.50, 2.00)]
DIMS = ('R_R0', 'L_L0', 'Ke_Ke0', 'Jrot_J0')
PARTITIONS = {'training': (64, 1001), 'validation': (16, 2001), 'final_interp': (32, 3001)}
TRAIN_SEEDS = (101, 102, 103, 104)
VAL_NOISE_SEEDS = (4101, 4102, 4103)
TEST_REPS = ({'angle': 0.0, 'noise_seed': 5101},
             {'angle': 2.0943951023931953, 'noise_seed': 5102},
             {'angle': 4.1887902047863905, 'noise_seed': 5103})


def _lhs(n, seed):
    """Deterministic Latin hypercube in [0,1]^4 (stratified permutations)."""
    import random
    out = []
    for d in range(4):
        rng = random.Random(seed * 10 + d)
        cuts = [(i + rng.random()) / n for i in range(n)]
        perm = list(range(n))
        rng.shuffle(perm)
        out.append([cuts[perm[i]] for i in range(n)])
    return out


def _scale(cols):
    return [[lo + u * (hi - lo) for (lo, hi), u in zip(BOUNDS, row)] for row in zip(*cols)]


def _corners():
    import itertools
    return [list(c) for c in itertools.product(*[(lo, hi) for lo, hi in BOUNDS])]


def build_manifest():
    pops = {}
    for name, (n, seed) in PARTITIONS.items():
        rows = _scale(_lhs(n, seed))
        if name == 'training':
            rows[0] = [1.0, 1.0, 1.0, 1.0]  # M0 anchor member
        pops[name] = {'count': n, 'seed': seed, 'dims': list(DIMS),
                      'bounds': [list(b) for b in BOUNDS], 'tuples': rows}
    pops['boundary'] = {'count': 16, 'seed': None, 'dims': list(DIMS),
                        'bounds': [list(b) for b in BOUNDS], 'tuples': _corners()}
    manifest = {'version': 'vroom-motor-population-v1', 'dims': list(DIMS),
                'bounds': [list(b) for b in BOUNDS],
                'training_seeds': list(TRAIN_SEEDS),
                'validation_noise_seeds': list(VAL_NOISE_SEEDS),
                'final_repetitions': TEST_REPS, 'populations': pops}
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest['sha256'] = hashlib.sha256(blob).hexdigest()
    return manifest
