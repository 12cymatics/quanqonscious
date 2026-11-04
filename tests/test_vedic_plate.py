import json
import subprocess
import sys
from pathlib import Path

import pytest
import sympy as sp

from vedic_book import evaluate
from vedic_book_cases import EXAMPLE_PARAMETERS
from vedic_plate import Interval, PlateField, sine_bounds
from vedic_sutra_cymatics import exact_parameters


def test_all_modes_satisfy_the_plate_equation_and_simply_supported_edges():
    x, y, t, nu = sp.symbols("x y t nu", real=True)
    u = PlateField((3, 0, -2, sp.Rational(1, 7))).expression(x, y, t)
    residual = sp.diff(u, t, 2) + sp.diff(u, x, 4) + 2 * sp.diff(u, x, 2, y, 2) + sp.diff(u, y, 4)
    assert sp.expand(residual) == 0
    for edge in (0, 1):
        assert u.subs(x, edge) == 0
        assert u.subs(y, edge) == 0
        assert (sp.diff(u, x, 2) + nu * sp.diff(u, y, 2)).subs(x, edge) == 0
        assert (sp.diff(u, y, 2) + nu * sp.diff(u, x, 2)).subs(y, edge) == 0
    assert sp.diff(u, t).subs(t, 0) == 0


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_sine_intervals_contain_exact_algebraic_values(n):
    for k, interval in enumerate(sine_bounds(n, 32)):
        exact = sp.sin(sp.pi * sp.Rational(k, n))
        assert sp.Rational(interval.lower, interval.scale) <= exact
        assert exact <= sp.Rational(interval.upper, interval.scale)


def test_interval_operations_round_outward():
    a, b = Interval(-3, 7, 16), Interval(-9, 2, 16)
    for av in (sp.Rational(-3, 16), sp.Rational(7, 16)):
        for bv in (sp.Rational(-9, 16), sp.Rational(2, 16)):
            product = a * b
            assert sp.Rational(product.lower, 16) <= av * bv <= sp.Rational(product.upper, 16)
    r = Interval(2, 3, 16).sqrt()
    assert sp.Rational(r.lower, 16) ** 2 <= sp.Rational(2, 16)
    assert sp.Rational(r.upper, 16) ** 2 >= sp.Rational(3, 16)


def test_quantized_levels_match_exact_expression():
    field = PlateField((1, -2, 3))
    for y in range(9):
        for x in range(17):
            value = field.expression(sp.Rational(x, 16), sp.Rational(y, 8)).doit()
            expected = int(sp.floor(9 * sp.Abs(value) / field.amplitude_bound))
            assert field.level(x, y, 16, 8, 10) == expected


@pytest.mark.parametrize("sid", range(1, 30))
def test_every_sutra_coefficient_is_retained_in_order(sid):
    result = evaluate(sid, **EXAMPLE_PARAMETERS[sid])
    field = PlateField.from_evaluation(result)
    assert field.coefficients == result.spectral_coefficients
    assert field.maximum_mode == len(result.spectral_coefficients)
    assert field.default_intervals >= 2 * field.maximum_mode
    assert field.metadata()["mapping"].startswith("authored_")


def test_nyquist_grid_is_rejected_and_higher_modes_are_not_dropped():
    field = PlateField(tuple(range(1, 65)))
    with pytest.raises(ValueError, match="Aliasing"):
        field.render(64)
    assert field.validate_grid(128) == 128
    assert field.maximum_mode == 64


def test_different_sutra_operands_change_geometry():
    first = PlateField.from_evaluation(evaluate(2, a=98, b=97, base=100))
    second = PlateField.from_evaluation(evaluate(2, a=94, b=91, base=100))
    assert first.render(16) != second.render(16)
    assert PlateField((0, 0, 0)).render(4) == "\n".join(["@" * 9] * 5)


def test_refining_the_grid_preserves_every_existing_sample():
    field = PlateField((7, -4, 3, 2, -11))
    for y in range(9):
        for x in range(17):
            assert field.level(x, y, 16, 8, 10) == field.level(2 * x, 2 * y, 32, 16, 10)


def test_cli_all_29_and_errors():
    root = Path(__file__).resolve().parents[1]
    script = root / "vedic_sutra_cymatics.py"
    result = subprocess.run([sys.executable, str(script), "--json"], cwd=root, capture_output=True, text=True, check=True)
    records = json.loads(result.stdout)
    assert [r["sutra"] for r in records] == list(range(1, 30))
    assert all(r["pages"] and r["book"]["pagination"] == "printed" for r in records)
    for record in records:
        restored = evaluate(record["sutra"], **exact_parameters(record["parameters"]))
        assert str(restored.value) == record["value"]
    bad = subprocess.run([sys.executable, str(script), "--sutra", "1", "--intervals", "16"], cwd=root, capture_output=True, text=True)
    assert bad.returncode != 0
    assert "Aliasing" in bad.stderr
