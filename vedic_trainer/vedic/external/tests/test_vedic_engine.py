"""Callability, count, shape and exactness checks on the classical engine adapter.

What these tests assert, in full:

- ``VedicSutraEngine`` declares 16 sutras and 13 sub-sutras, and
  ``ProofTester.verify_sutra_engine`` invokes all 29 on representative inputs
  without any returning empty or non-finite values. That harness derives its
  ``verified`` key from the outputs it collected, so the assertion on it can
  fail; it checks emptiness and finiteness, not correctness of any value.
- Five operators are invoked directly and checked for declared shape and
  finiteness: ``ekadhikena_purvena``, ``urdhva_tiryagbhyam``,
  ``chalana_kalanabyham`` and ``gunitasamuccayah_samuccayagunitah``.
- Three algebraic identities are checked exactly on
  ``sankalana_vyavakalanabhyam`` and ``chalana_kalanabyham``.

For the other operators nothing here compares an output against a reference
value or an independent implementation, so their numerical results are
unverified by this file. That is stated rather than implied.

Exact inputs, exact comparisons
-------------------------------
The inputs were ``np.abs(rng.standard_normal(...)) + 0.5`` — synthetic random
floats with full 53-bit mantissas. Nothing about the identities being checked
requires that, and it forced the comparisons to be ``np.allclose``: with such
inputs ``(a+b) + (a-b)`` is not bitwise ``2a``, so a tolerance was the only
way to assert the identity at all.

The inputs are now small integers, which float64 represents exactly and whose
sums and differences stay exact. ``sym + diff == 2X`` then holds bitwise and
is asserted with ``np.array_equal``: an exact statement about the operator
rather than an approximate one about float rounding. Integers are also
reproducible without a generator, so there is no seed to trust.
"""
from __future__ import annotations

import numpy as np

from vedic.external import ProofTester, VedicSutraEngine


def _exact_matrix(n: int, offset: int) -> np.ndarray:
    """A deterministic integer-valued matrix, exact in float64.

    Values are distinct and nonzero so an operator that returns a constant,
    the input unchanged, or zeros is distinguishable from one that does not.
    """
    return np.array([[float(offset + i * n + j + 1) for j in range(n)]
                     for i in range(n)], dtype=np.float64)


def _exact_vector(n: int, offset: int) -> np.ndarray:
    return np.array([float(offset + i + 1) for i in range(n)], dtype=np.float64)


def test_the_inputs_are_exact_in_float64() -> None:
    """Guards every exact comparison below.

    If an input were not exactly representable, ``array_equal`` would be
    asserting that rounding happened to cancel, and this file would be back
    to testing float64 behaviour rather than the operators.
    """
    for arr in (_exact_matrix(6, 0), _exact_matrix(6, 100), _exact_vector(16, 0)):
        assert np.all(arr == np.floor(arr)), "input is not integer-valued"
        assert np.all(np.abs(arr) < 2 ** 53), "input exceeds exact integer range"
        assert np.all(arr != 0), "a zero component makes several checks vacuous"


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


def test_engine_outputs_have_the_declared_shape_and_are_finite() -> None:
    eng = VedicSutraEngine()
    X = _exact_matrix(6, 0)
    Y = _exact_matrix(6, 100)
    V = _exact_vector(16, 0)

    out = eng.ekadhikena_purvena(X)
    assert out.shape == X.shape
    assert np.all(np.isfinite(out))

    out = eng.urdhva_tiryagbhyam(X, Y)
    assert out.shape == (6, 6)
    assert np.all(np.isfinite(out))

    rolled = eng.chalana_kalanabyham(V, steps=3, direction=1)
    assert rolled.shape == V.shape
    assert np.all(np.isfinite(rolled))

    sum_prod, prod_sum = eng.gunitasamuccayah_samuccayagunitah(V, V)
    assert np.isfinite(sum_prod)
    assert np.isfinite(prod_sum)


def test_sankalana_vyavakalanabhyam_recovers_both_arguments_exactly() -> None:
    """(sym + diff) = 2X and (sym − diff) = 2Y, bitwise.

    On integer inputs every intermediate is exact, so this is an equality
    rather than a proximity claim.
    """
    eng = VedicSutraEngine()
    X = _exact_matrix(6, 0)
    Y = _exact_matrix(6, 100)
    sym, diff = eng.sankalana_vyavakalanabhyam(X, Y)
    assert np.array_equal(sym + diff, 2 * X)
    assert np.array_equal(sym - diff, 2 * Y)


def test_sankalana_vyavakalanabhyam_is_not_trivially_symmetric() -> None:
    """Guards the identity above.

    An operator returning ``(a, a)`` or ``(a, 0)`` would satisfy one half of
    the pair of identities. Both halves must be non-degenerate.
    """
    eng = VedicSutraEngine()
    X = _exact_matrix(6, 0)
    Y = _exact_matrix(6, 100)
    sym, diff = eng.sankalana_vyavakalanabhyam(X, Y)
    assert not np.array_equal(sym, diff)
    assert np.any(diff != 0)
    assert not np.array_equal(sym, X)


def test_chalana_kalanabyham_is_an_exact_rotation() -> None:
    """A roll by k then by −k returns the input bitwise, for every k.

    The old file checked only that the output had the same shape, which a
    function returning zeros satisfies.
    """
    eng = VedicSutraEngine()
    V = _exact_vector(16, 0)
    for steps in range(1, 17):
        forward = eng.chalana_kalanabyham(V, steps=steps, direction=1)
        back = eng.chalana_kalanabyham(forward, steps=steps, direction=-1)
        assert np.array_equal(back, V), f"roll by {steps} is not invertible"
        if steps % 16 != 0:
            assert not np.array_equal(forward, V), \
                f"roll by {steps} is the identity"


def test_vyashtisamanstih_reports_an_exact_discrepancy() -> None:
    """whole − Σparts, exactly zero when the parts do sum to the whole.

    ``np.sum(parts)`` totals every element of every part, so this is a scalar
    identity — vyashti-samashti, the individual and the total — not an
    elementwise one.

    The operator used to return ``np.isclose(whole, np.sum(parts))``, a Bool
    decided under numpy's default rtol=1e-5. That could not express the
    identity: on values near 1e6 a discrepancy of 10 came back as True, and a
    caller had no way to see how far off the sum was. Now it is checkable
    exactly, so it is checked.
    """
    eng = VedicSutraEngine()
    X = _exact_matrix(6, 0)
    Y = _exact_matrix(6, 100)
    total = float(np.sum(X) + np.sum(Y))
    assert eng.vyashtisamanstih(total, [X, Y]) == 0.0


def test_vyashtisamanstih_does_not_report_a_real_difference_as_zero() -> None:
    """The case numpy's default rtol=1e-5 silently accepted.

    Two values near 1e6 differing by 10 are ``np.isclose`` to each other. The
    difference is returned, so the discrepancy is visible rather than decided.
    """
    eng = VedicSutraEngine()
    whole = 1_000_010.0
    parts = [np.array([1_000_000.0])]
    diff = eng.vyashtisamanstih(whole, parts)
    assert diff == 10.0, diff
    # And the tolerance that used to stand here would have called them equal.
    assert np.isclose(whole, np.sum(parts)), \
        "the demonstration case no longer demonstrates anything"


def test_vyashtisamanstih_is_sensitive_to_every_part() -> None:
    """Dropping any single part must change the discrepancy.

    Guards against an implementation that ignores its ``parts`` argument, or
    sums only the first one — either of which would return 0 for the identity
    case above and pass that test.
    """
    eng = VedicSutraEngine()
    parts = [_exact_matrix(3, 0), _exact_matrix(3, 50), _exact_matrix(3, 900)]
    total = float(sum(float(np.sum(p)) for p in parts))
    assert eng.vyashtisamanstih(total, parts) == 0.0
    for drop in range(len(parts)):
        fewer = [p for k, p in enumerate(parts) if k != drop]
        assert eng.vyashtisamanstih(total, fewer) != 0.0, \
            f"dropping part {drop} did not change the discrepancy"
