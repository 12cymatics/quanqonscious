"""16×16 Walsh-Hadamard matrix and 4×16 axis matrix in exact ℚ.

The Walsh matrix is the natural Sylvester-ordered Hadamard:
    H₂ⁿ = H₂ ⊗ H₂ⁿ⁻¹  with  H₂ = [[1, 1], [1, -1]].

Rows are indexed v ∈ range(16) (4-bit binary). The axis matrix
WHT_AXIS has 4 rows: W[k, v] = (−1)^{((v >> k) & 1)} for k ∈ {0, 1, 2, 3}.
These are the four single-axis Walsh characters; the dual-basis loss
projects Ψ onto the span of {1, W[0], W[1], W[2], W[3]} (5 dimensions).
The full exact transform below retains all 16 characters as a diagnostic;
it does not change that training-loss definition. Torch functions are
explicit output adapters and do not import numerical states into the exact core.
"""
from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import Sequence, Tuple

import numpy as np
import torch

from .tesseract import BIT_WIDTH, NUM_VERTICES


def _build_hadamard_q() -> Tuple[Tuple[Fraction, ...], ...]:
    """Build the 16×16 Sylvester Hadamard over ℚ."""
    rows: list[list[Fraction]] = []
    for r in range(NUM_VERTICES):
        row: list[Fraction] = []
        for v in range(NUM_VERTICES):
            sign = 1
            for k in range(BIT_WIDTH):
                if (r >> k) & 1 and (v >> k) & 1:
                    sign = -sign
            row.append(Fraction(sign))
        rows.append(row)
    return tuple(tuple(r) for r in rows)


def _build_axis_q() -> Tuple[Tuple[Fraction, ...], ...]:
    """Build the 4×16 single-axis Walsh matrix over ℚ.

    Row k corresponds to bit k: W[k, v] = (−1)^{((v >> k) & 1)}.
    """
    rows: list[list[Fraction]] = []
    for k in range(BIT_WIDTH):
        row = [Fraction(-1) if ((v >> k) & 1) else Fraction(1) for v in range(NUM_VERTICES)]
        rows.append(row)
    return tuple(tuple(r) for r in rows)


HADAMARD_16_Q: Tuple[Tuple[Fraction, ...], ...] = _build_hadamard_q()
"""16×16 Sylvester Hadamard matrix over ℚ."""

WHT_AXIS_Q: Tuple[Tuple[Fraction, ...], ...] = _build_axis_q()
"""4×16 single-axis Walsh matrix over ℚ."""


def _exact_vector(values: Sequence[int | Fraction], name: str) -> Tuple[Fraction, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} requires an ordered sequence of exact rational values")
    if len(values) != NUM_VERTICES:
        raise ValueError(f"{name} requires all {NUM_VERTICES} values; got {len(values)}")
    out: list[Fraction] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Rational):
            raise TypeError(f"{name}[{index}] must be an exact rational (integer or Fraction)")
        out.append(Fraction(int(value.numerator), int(value.denominator)))
    return tuple(out)


def walsh_transform_exact(values: Sequence[int | Fraction]) -> Tuple[Fraction, ...]:
    """Return every unnormalized coefficient c[r] = Σ_v (−1)^popcount(r&v) f[v].

    The 16 input entries and all 16 output coefficients are exact rationals.
    Character indices use the same Sylvester order as ``HADAMARD_16_Q``.
    Floating-point inputs are rejected without rationalizing them.
    """
    values = _exact_vector(values, "Walsh transform")
    return tuple(
        sum((character * value for character, value in zip(row, values)), Fraction(0))
        for row in HADAMARD_16_Q
    )


def walsh_inverse_exact(coefficients: Sequence[int | Fraction]) -> Tuple[Fraction, ...]:
    """Invert the full transform by Hc/16, retaining every character.

    This normalization is exact because Hᵀ = H and H² = 16I.
    """
    coefficients = _exact_vector(coefficients, "Walsh inverse")
    return tuple(value / NUM_VERTICES for value in walsh_transform_exact(coefficients))


def hadamard_16_torch(dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> torch.Tensor:
    """Output adapter: return the 16×16 Sylvester Hadamard as a Torch tensor."""
    h = np.array([[float(x) for x in row] for row in HADAMARD_16_Q], dtype=np.float32)
    return torch.from_numpy(h).to(dtype=dtype, device=device)


def wht_axis_torch(dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> torch.Tensor:
    """Output adapter: return the 4×16 single-axis Walsh matrix as a Torch tensor."""
    a = np.array([[float(x) for x in row] for row in WHT_AXIS_Q], dtype=np.float32)
    return torch.from_numpy(a).to(dtype=dtype, device=device)
