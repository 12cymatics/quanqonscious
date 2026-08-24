"""Conservation laws: R1..R4 hold exactly over ℚ on every legal input."""
from __future__ import annotations

from fractions import Fraction

import pytest

from vedic.kernel import conservation_exact as ce
from vedic.kernel.q import Q16
from vedic.kernel.z2_primitives import s29_mean_drive


def test_r1_zero_on_multiples_of_435() -> None:
    """R1 = trace_sum mod 435 is zero exactly when trace is a multiple of T(29)."""
    for k in (0, 1, 2, 7, 100):
        assert ce.r1_trace_closure(Fraction(435 * k)) == Fraction(0)


def test_r1_nonzero_off_grid() -> None:
    for off in (1, 2, 217, 434):
        assert ce.r1_trace_closure(Fraction(off)) != Fraction(0)


def test_r1_rejects_non_integer() -> None:
    with pytest.raises(ValueError):
        ce.r1_trace_closure(Fraction(1, 2))


def test_r2_zero_for_every_psi(q16_corpus: list[Q16]) -> None:
    """R2 ≡ 0 by construction: every vertex appears in exactly one (v, v̄) pair."""
    for psi in q16_corpus:
        assert ce.r2_complement_pair_sum(psi) == Fraction(0)


def test_r3_zero_for_every_psi(q16_corpus: list[Q16]) -> None:
    """R3 ≡ 0: S29 preserves the mean exactly."""
    for psi in q16_corpus:
        assert ce.r3_s29_mean_preservation(psi) == Fraction(0)


@pytest.mark.parametrize("weight", [
    Fraction(1, 2),      # the canonical weight
    Fraction(0),         # identity
    Fraction(1),         # full drive to the mean
    Fraction(1, 3),
    Fraction(7, 11),
    Fraction(-2, 5),     # outside [0, 1]: still affine, still mean-preserving
    Fraction(13, 4),
])
def test_r3_vanishes_at_every_weight(weight: Fraction,
                                     q16_corpus: list[Q16]) -> None:
    """R3 = 0 is a property of the affine family, not of w = 1/2.

    (S29 Ψ)_v = (1−w)·Ψ_v + w·mean(Ψ) has mean (1−w)·mean + w·mean = mean
    for any w. ``conservation_exact`` states this; a test that only ever
    exercised the canonical weight would leave the general claim unchecked.
    """
    for psi in q16_corpus:
        driven = s29_mean_drive(psi, weight)
        mean_in = sum(psi, Fraction(0)) / len(psi)
        mean_out = sum(driven, Fraction(0)) / len(driven)
        assert mean_out - mean_in == Fraction(0), \
            f"S29 at weight {weight} moved the mean on {psi}"


def test_r4_zero_for_every_psi(q16_corpus: list[Q16]) -> None:
    """R4 ≡ 0: ⟨S(Ψ), A(Ψ)⟩ vanishes because S, A live on opposite eigenspaces of σ."""
    for psi in q16_corpus:
        assert ce.r4_beltrami_orthogonality(psi) == Fraction(0)


def test_all_residuals_zero_on_canonical_input() -> None:
    psi = tuple(Fraction(v - 8, 16) for v in range(16))
    r1, r2, r3, r4 = ce.all_residuals(psi, Fraction(435))
    assert r1 == Fraction(0)
    assert r2 == Fraction(0)
    assert r3 == Fraction(0)
    assert r4 == Fraction(0)


def test_cons_l2_squared_zero_on_canonical_input() -> None:
    psi = tuple(Fraction(v - 8, 16) for v in range(16))
    assert ce.cons_l2_squared(psi, Fraction(435)) == Fraction(0)
