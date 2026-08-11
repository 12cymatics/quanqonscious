"""Four conservation residuals in exact ℚ arithmetic.

R1: T(29) closure mod 435.
    The triangular number T(29) = 29 · 30 / 2 = 435. The residual takes a
    Q-valued accumulator ``trace_sum`` and reduces it mod 435; the residual
    is zero exactly when the accumulator is an integer multiple of 435.

R2: complement-pair sum equals total sum.
    Σ_{v < v̄} (Ψ_v + Ψ_{v̄}) = Σ_v Ψ_v.
    This is identically zero by construction (every vertex appears in
    exactly one (v, v̄) pair), so R2 ≡ 0 for any Ψ. We still return the
    residual so the loss can be observed numerically.

R3: S29 mean preservation.
    mean(S29(Ψ)) − mean(Ψ) = 0  for all Ψ.
    Direct from the simplified S29 formula (Ψ_v + mean(Ψ)) / 2.

R4: S/A (Beltrami) orthogonality.
    ⟨S(Ψ), A(Ψ)⟩ = 0 because S/A decompose Ψ on the eigenspaces of the
    complement involution (eigenvalues +1 / −1) which are orthogonal under
    the standard inner product on ℚ^16.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

from .q import Q16
from .sutras_exact import s7_sankalana_vyavakalana, s29_mean_drive
from .tesseract import COMPLEMENT, NUM_VERTICES, pairs_v_lt_complement

# T(29) = 1 + 2 + ... + 29 = 435.
T29: int = 29 * 30 // 2


def r1_trace_closure(trace_sum: Fraction) -> Fraction:
    """R1 = trace_sum mod 435   (zero ⇔ trace_sum is an integer multiple of T(29))."""
    if trace_sum.denominator != 1:
        # Allow Fraction inputs but fold the integer reduction over the
        # numerator once the denominator has been reduced. The residual is
        # only meaningful for integer trace sums; non-integer inputs are an
        # invariant violation and assert loudly.
        raise ValueError(f"R1 expects integer trace_sum, got {trace_sum}")
    return Fraction(trace_sum.numerator % T29)


def r2_complement_pair_sum(psi: Q16) -> Fraction:
    """R2 = Σ_{v < v̄}(Ψ_v + Ψ_{v̄}) − Σ_v Ψ_v."""
    pair_total = sum(
        (psi[v] + psi[c] for v, c in pairs_v_lt_complement()),
        Fraction(0),
    )
    grand_total = sum(psi, Fraction(0))
    return pair_total - grand_total


def r3_s29_mean_preservation(psi: Q16) -> Fraction:
    """R3 = mean(S29(Ψ)) − mean(Ψ).

    Algebraically zero: mean((Ψ + m)/2) = (mean(Ψ) + m)/2 = m where m = mean(Ψ).
    """
    psi_after = s29_mean_drive(psi)
    mean_before = sum(psi, Fraction(0)) / Fraction(NUM_VERTICES)
    mean_after = sum(psi_after, Fraction(0)) / Fraction(NUM_VERTICES)
    return mean_after - mean_before


def r4_beltrami_orthogonality(psi: Q16) -> Fraction:
    """R4 = ⟨S(Ψ), A(Ψ)⟩.

    Identically zero because the complement involution has orthogonal
    +1/−1 eigenspaces.
    """
    sym, anti = s7_sankalana_vyavakalana(psi)
    return sum(
        (sym[v] * anti[v] for v in range(NUM_VERTICES)),
        Fraction(0),
    )


def all_residuals(psi: Q16, trace_sum: Fraction) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return (R1, R2, R3, R4) — the canonical conservation tuple."""
    return (
        r1_trace_closure(trace_sum),
        r2_complement_pair_sum(psi),
        r3_s29_mean_preservation(psi),
        r4_beltrami_orthogonality(psi),
    )


def cons_l2_squared(psi: Q16, trace_sum: Fraction) -> Fraction:
    """ΣᵢRᵢ² — the Q-version of the conservation loss."""
    r1, r2, r3, r4 = all_residuals(psi, trace_sum)
    return r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4
