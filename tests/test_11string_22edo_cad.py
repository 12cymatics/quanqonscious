import importlib.util
import sys
from decimal import Decimal
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "11string_22edo_acoustic_electric_FULL_CAD.py"
SPEC = importlib.util.spec_from_file_location("guitar_cad", SCRIPT)
cad = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cad
SPEC.loader.exec_module(cad)


def test_octave_and_rear_scale_identities_are_exact():
    proof = cad.verify_exact_geometry()
    assert cad.fret_distance(22) == Decimal("355.60")
    assert proof["rear_scale_squared_mm2"] == proof["specified_scale_squared_mm2"]


def test_hardware_contract():
    assert len(cad.FRONT_TUNER_CENTERS) == 10
    assert cad.P.front_strings + cad.P.rear_strings == 11


def test_all_rear_frets_lie_at_their_speaking_line_distance():
    origin = (Decimal(0), Decimal(0), cad.dec("rear_line_z0"))
    for n in range(1, cad.P.fret_count + 1):
        distance = cad.fret_distance(n)
        # Decimal.sqrt and exp are correctly rounded, so an irrational point can
        # differ from its algebraic identity only in the final context digit.
        error = abs(cad.squared_distance(origin, cad.rear_fret_point(n)) - distance**2)
        assert error < Decimal("1e-43")
