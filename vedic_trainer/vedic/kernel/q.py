"""Exact ℚ reference type (`Fraction[16]`) and conversion helpers.

The kernel uses ``fractions.Fraction`` everywhere as the ground truth. The 29
sutras and the 4 conservation residuals are each implemented twice — once over
ℚ (this file's tuple-of-Fraction representation) and once over torch — but the
two ports are NOT compared numerically operator by operator, and no such
comparison could be "bit-exact": a float port cannot reproduce a rational
exactly, so any cross-port check is a tolerance check by construction.

What is actually verified, and by what:

* ``tests/test_simulator_match.py`` and ``scripts/verify_bit_exact.py`` —
  genuinely bit-exact, but ℚ-against-ℚ: the kernel's rationals are compared
  as (numerator, denominator) integer pairs against the committed
  ``fixtures/*.json``. No float is involved on either side.
* ``tests/test_conservation_torch.py`` — the only numeric cross-port test.
  It covers the 4 residuals only, in float64, at a 1e-7 relative tolerance.
  It says nothing about the 29 sutras.
* ``tests/test_torch_buffers.py`` — covers the 29 torch sutras, but
  structurally: it compares their integer index/mask/permutation buffers
  against the ℚ definition. It deliberately performs no float arithmetic and
  no output comparison, so the torch operators' numeric outputs are checked
  nowhere.

Not verified: that any torch sutra's output matches its ℚ counterpart at any
tolerance.

The ℚ functions here contain no epsilons, no clamps, no try/except and no
tolerances of any kind. ``q_eq`` is exact Fraction equality; ``q_to_floats``
is a one-way conversion for display and for the torch port's buffers, and
nothing converts back.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

# A length-16 tuple of Fraction values represents Ψ ∈ ℚ^{Z₂⁴}. The tuple is
# immutable so it can be hashed and compared directly. We expose Q (a single
# rational) and Q16 (a length-16 tuple) as type aliases for clarity.
Q = Fraction
Q16 = Tuple[Fraction, ...]


def q_zeros() -> Q16:
    """Return the zero vector in ℚ^16."""
    return tuple(Fraction(0) for _ in range(16))


def q_to_floats(psi: Q16) -> Tuple[float, ...]:
    """Convert Q16 to a length-16 tuple of Python floats."""
    return tuple(float(x) for x in psi)


def q_eq(a: Q16, b: Q16) -> bool:
    """Bit-exact equality on Q16 (Fraction comparison is exact)."""
    return tuple(a) == tuple(b)
