"""Smoke tests for the classical (NumPy) Vedic sutra engine adapter.

These tests verify that every callable in ``VedicSutraEngine.sutras`` and
``.sub_sutras`` (16 + 13 = 29) is invocable on representative inputs and
returns a finite NumPy / scalar value. No epsilons, no float tolerance —
shape and finiteness only.
"""
from __future__ import annotations

import numpy as np

from vedic.external import ProofTester, VedicSutraEngine


def test_engine_counts() -> None:
    eng = VedicSutraEngine()
    assert len(eng.sutras) == 16
    assert len(eng.sub_sutras) == 13


def test_engine_smoke_runs() -> None:
    tester = ProofTester()
    tester.verify_sutra_engine()
    res = tester.results["vedic_sutras"]
    assert res["main_sutras"] == 16
    assert res["sub_sutras"] == 13
    assert res["executed"] == 29
    assert res["verified"] is True


def test_engine_outputs_are_finite() -> None:
    """Concrete invocation: every classical sutra produces finite output."""
    eng = VedicSutraEngine()
    rng = np.random.default_rng(0xDEADBEEF)
    X = np.abs(rng.standard_normal((6, 6))) + 0.5
    Y = np.abs(rng.standard_normal((6, 6))) + 0.5
    V = np.abs(rng.standard_normal(16)) + 0.5

    out = eng.ekadhikena_purvena(X)
    assert np.all(np.isfinite(out))

    out = eng.urdhva_tiryagbhyam(X, Y)
    assert out.shape == (6, 6)
    assert np.all(np.isfinite(out))

    sym, diff = eng.sankalana_vyavakalanabhyam(X, Y)
    assert np.allclose(sym + diff, 2 * X)
    assert np.allclose(sym - diff, 2 * Y)

    rolled = eng.chalana_kalanabyham(V, steps=3, direction=1)
    assert rolled.shape == V.shape

    sum_prod, prod_sum = eng.gunitasamuccayah_samuccayagunitah(V, V)
    assert np.isfinite(sum_prod)
    assert np.isfinite(prod_sum)
