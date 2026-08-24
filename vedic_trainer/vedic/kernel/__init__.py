"""29-sutra kernel: exact ℚ reference + torch autograd port."""
from __future__ import annotations

from .q import Q, Q16, q_zeros, q_to_floats, q_eq
from .tesseract import (
    NUM_VERTICES,
    BIT_WIDTH,
    COMPLEMENT,
    POPCOUNT,
    SHELLS,
    enumerate_vertices,
    complement,
    popcount,
    rotate_left_1,
    pairs_v_lt_complement,
)

__all__ = [
    "Q",
    "Q16",
    "q_zeros",
    "q_to_floats",
    "q_eq",
    "NUM_VERTICES",
    "BIT_WIDTH",
    "COMPLEMENT",
    "POPCOUNT",
    "SHELLS",
    "enumerate_vertices",
    "complement",
    "popcount",
    "rotate_left_1",
    "pairs_v_lt_complement",
]
