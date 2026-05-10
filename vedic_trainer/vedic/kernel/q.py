"""Exact ℚ reference type (`Fraction[16]`) and conversion helpers.

The kernel uses ``fractions.Fraction`` everywhere as the ground truth. All 29
sutras and 4 conservation residuals are implemented twice — once over Q
(this file's tuple-of-Fraction representation) and once over torch.float32 —
and a bit-exact test compares them. No epsilons, no clamps, no try/except.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence, Tuple

# A length-16 tuple of Fraction values represents Ψ ∈ ℚ^{Z₂⁴}. The tuple is
# immutable so it can be hashed and compared directly. We expose Q (a single
# rational) and Q16 (a length-16 tuple) as type aliases for clarity.
Q = Fraction
Q16 = Tuple[Fraction, ...]


def q_zeros() -> Q16:
    """Return the zero vector in ℚ^16."""
    return tuple(Fraction(0) for _ in range(16))


def q_from_floats(values: Sequence[float], denom_limit: int = 1_000_000) -> Q16:
    """Convert a length-16 float sequence to Q16 via ``limit_denominator``.

    The denominator cap exists so that round-trips through float (e.g. when
    reading JSON fixtures) cannot blow the rationals up to arbitrary
    precision. ``denom_limit`` of one million is far above any legitimate
    simulator export and preserves bit-exactness for our test vectors.
    """
    if len(values) != 16:
        raise ValueError(f"expected length 16, got {len(values)}")
    return tuple(Fraction(v).limit_denominator(denom_limit) for v in values)


def q_to_floats(psi: Q16) -> Tuple[float, ...]:
    """Convert Q16 to a length-16 tuple of Python floats."""
    return tuple(float(x) for x in psi)


def q_eq(a: Q16, b: Q16) -> bool:
    """Bit-exact equality on Q16 (Fraction comparison is exact)."""
    return tuple(a) == tuple(b)


def q_close(a: Q16, b: Sequence[float], rtol: float = 1e-7, atol: float = 1e-9) -> bool:
    """Compare a Q16 reference against a float sequence within tolerance.

    Used by the bit-exact test harness when one side is the torch.float32
    output. ``rtol`` is relative-to-max(|a|), ``atol`` is the absolute floor
    so that comparisons against zero do not blow up.
    """
    if len(a) != len(b):
        return False
    a_floats = q_to_floats(a)
    scale = max((abs(x) for x in a_floats), default=1.0)
    if scale == 0.0:
        scale = 1.0
    for x, y in zip(a_floats, b):
        if abs(x - y) > atol + rtol * scale:
            return False
    return True


def q_iter(psi: Q16) -> Iterable[Tuple[int, Fraction]]:
    """Iterate (vertex, value) pairs over a Q16."""
    for v, x in enumerate(psi):
        yield v, x
