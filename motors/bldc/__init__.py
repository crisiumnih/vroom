"""BLDC motor entry: thin wrapper over GEM submodule plant (assumed sim params)."""
import sys
from pathlib import Path

SUBMODULE = Path(__file__).resolve().parents[2] / "third_party" / "gym-electric-motor"
if str(SUBMODULE) not in sys.path:
    sys.path.insert(0, str(SUBMODULE))

NOMINAL = {
    "tau_s": 0.0001,
    "supply_v": 44.4,
    "pole_pairs": 21,
    "r_ohm": 0.085,
    "l_h": 50e-6,
    "current_reference_limit_a": 1.5,
    "phase_trip_a": 4.0,
    "note": "Assumed legacy sim values, not measured 2805 specs.",
}
