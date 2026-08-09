"""16×16 Walsh-Hadamard matrix and 16×4 axis matrix in exact ℚ.

The Walsh matrix is the natural Sylvester-ordered Hadamard:
    H₂ⁿ = H₂ ⊗ H₂ⁿ⁻¹  with  H₂ = [[1, 1], [1, -1]].

Rows are indexed v ∈ range(16) (4-bit binary). The axis matrix
WHT_AXIS has 4 rows: W[k, v] = (−1)^{((v >> k) & 1)} for k ∈ {0, 1, 2, 3}.
These are the four single-axis Walsh characters; the dual-basis loss
projects Ψ onto the span of {1, W[0], W[1], W[2], W[3]} (5 dimensions).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

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


def hadamard_16_torch(dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the 16×16 Sylvester Hadamard as a torch tensor."""
    h = np.array([[float(x) for x in row] for row in HADAMARD_16_Q], dtype=np.float32)
    return torch.from_numpy(h).to(dtype=dtype, device=device)


def wht_axis_torch(dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the 4×16 single-axis Walsh matrix as a torch tensor."""
    a = np.array([[float(x) for x in row] for row in WHT_AXIS_Q], dtype=np.float32)
    return torch.from_numpy(a).to(dtype=dtype, device=device)
