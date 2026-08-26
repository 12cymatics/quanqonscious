"""Callability, count, shape and finiteness checks on the classical engine adapter.

What these three tests actually assert, in full:

- ``VedicSutraEngine`` declares 16 sutras and 13 sub-sutras, and
  ``ProofTester.verify_sutra_engine`` invokes all 29 on representative inputs
  without any returning empty or non-finite values. That harness derives its
  ``verified`` key from the outputs it collected, so the assertion on it can
  fail; it checks emptiness and finiteness, not correctness of any value.
- Five operators are then invoked directly and checked for finiteness and
  declared shape: ``ekadhikena_purvena``, ``urdhva_tiryagbhyam``,
  ``chalana_kalanabyham`` and ``gunitasamuccayah_samuccayagunitah``.
- One genuine algebraic identity is checked, on
  ``sankalana_vyavakalanabhyam``: that ``sym + diff == 2X`` and
  ``sym - diff == 2Y``, under ``np.allclose``. This is the only test in the
  file that constrains what a sutra computes rather than that it computed
  something.

For the other 28 operators nothing here compares an output against a
reference value or an independent implementation, so their numerical results
are unverified by this file.
"""
from __future__ import annotations

import numpy as np

from vedic.external import ProofTester, VedicSutraEngine


def test_engine_counts() -> None:
    eng = VedicSutraEngine()
    assert len(eng.sutras) == 16
    assert len(eng.sub_sutras) == 13


def test_engine_executes_all_29_sutras() -> None:
    tester = ProofTester()
    tester.verify_sutra_engine()
    res = tester.results["vedic_sutras"]
    assert res["main_sutras"] == 16
    assert res["sub_sutras"] == 13
    assert res["executed"] == 29
    assert res["verified"] is True
    assert res["main_sutras"] + res["sub_sutras"] == res["executed"] == 29


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
