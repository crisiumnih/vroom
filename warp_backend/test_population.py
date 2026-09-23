"""Population manifest gates (master_plan sect.5)."""
from warp_backend.population import build_manifest, BOUNDS


def test_manifest_structure():
    m1, m2 = build_manifest(), build_manifest()
    assert m1['sha256'] == m2['sha256'] and len(m1['sha256']) == 64
    pops = m1['populations']
    assert {k: pops[k]['count'] for k in pops} == {
        'training': 64, 'validation': 16, 'final_interp': 32, 'boundary': 16}
    assert pops['training']['tuples'][0] == [1.0, 1.0, 1.0, 1.0]
    assert m1['training_seeds'] == [101, 102, 103, 104]


def test_bounds_and_disjoint():
    m = build_manifest()
    seen = set()
    for name, pop in m['populations'].items():
        assert len(pop['tuples']) == pop['count']
        for t in pop['tuples']:
            for (lo, hi), v in zip(BOUNDS, t):
                assert lo - 1e-12 <= v <= hi + 1e-12, (name, t)
            key = tuple(round(v, 12) for v in t)
            assert key not in seen, ('shared tuple', name, t)
            seen.add(key)
    corners = m['populations']['boundary']['tuples']
    assert len({tuple(c) for c in corners}) == 16
    for i, (lo, hi) in enumerate(BOUNDS):
        assert any(c[i] == lo for c in corners) and any(c[i] == hi for c in corners)
