"""PR2 gates: analytic values match master_plan sect.6, tables agree."""
import math
import sys
from pathlib import Path
import pytest

from warp_backend import waves as w

_here = Path(__file__).resolve()
_sub = _here.parents[1] / "third_party" / "gym-electric-motor"
GEM_ROOT = (str(_sub) if (_sub / "benchmarks" / "bldc" / "current_rl" / "core.py").exists()
            else "/home/sra/prajwal/fyp/gym-electric-motor")
sys.path.insert(0, GEM_ROOT)


def test_plan_spot_values():
    assert w.sample(w.S1(), 2.0) == pytest.approx(0.25)
    assert w.sample(w.S1(), 4.0) == pytest.approx(0.75)
    assert w.sample(w.S2(), 1.5) == pytest.approx(0.5)
    assert w.sample(w.S2(), 2.5) == pytest.approx(-0.5)
    assert w.sample(w.S3(), 2.0) == pytest.approx(0.375)   # .75(t-1)/2
    assert w.sample(w.S3(), 5.0) == pytest.approx(0.375)   # .75(6-t)/2
    assert w.sample(w.S4(), 2.0) == pytest.approx(0.1875)  # .75((t-1)/2)^2
    assert w.sample(w.S4(), 5.0) == pytest.approx(0.1875)  # .75((6-t)/2)^2
    assert w.sample(w.S5(), 1.5) == pytest.approx(0.5)     # .5 sin(pi/2)
    assert w.sample(w.S5(), 2.5) == pytest.approx(-0.5)
    assert w.sample(w.S6(), 3.0) == pytest.approx(-0.75)
    assert w.sample(w.D1(), 3.0) == pytest.approx(0.30)
    assert w.sample(w.D3(), 3.0) == pytest.approx(0.90)
    q, _ = w.C1()
    assert w.sample(q, 0.05) == pytest.approx(0.50)
    assert w.sample(q, 0.10) == pytest.approx(-0.50)
    assert w.sample(q, 0.17) == pytest.approx(0.80)
    qc, _ = w.C3()
    assert w.sample(qc, 0.0325) == pytest.approx(0.50)     # first sine peak
    _, d = w.C4()
    assert w.sample(d, 0.10) == pytest.approx(0.20)
    assert w.sample(d, 0.17) == pytest.approx(-0.20)


def test_right_continuous_and_zero_outside():
    assert w.sample(w.S1(), 1.0) == pytest.approx(0.25)  # event applies at t0
    assert w.sample(w.S1(), 7.0) == pytest.approx(0.0)
    assert w.sample(w.S1(), 99.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        w.sample([("nope", 0, 1, 0.0)], 0.5)


def test_tables_match_direct_sampling_both_consumers():
    """Dense tables through real GEM schedule_value == sample() at every k."""
    from benchmarks.bldc.environment import schedule_value
    for name in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'D1', 'D2', 'D3']:
        segs = w.CASES[name]()
        for dt in (1e-3, 1e-4):
            tab = w.to_schedule(segs, dt, 8.0)
            n = len(tab) - 1
            for k in (0, 1, 999, n // 2, n - 1, n):
                got = schedule_value([[t, v] for t, v in tab], k, dt)
                assert got == pytest.approx(w.sample(segs, k * dt), abs=1e-12), (name, dt, k)


def test_validation_transform():
    s = w.scale_waves(w.S1(), amp=0.9, time=1.25)
    assert w.sample(s, 2.5) == pytest.approx(0.225)   # .25*.9 at t=1*1.25..3*1.25
    assert w.sample(s, 1.0) == pytest.approx(0.0)
    s5 = w.scale_waves(w.S5(), amp=0.9, time=1.25)
    assert w.sample(s5, 1.25 + 0.625) == pytest.approx(0.45)  # quarter period
