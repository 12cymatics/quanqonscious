"""Z₂⁴ Boolean cube structure: 16 vertices, complement, Hamming weight, shells.

Vertices are integers ``v ∈ range(16)`` with their 4-bit binary encoding
giving the position on the tesseract. Bit 0 is the least-significant axis.
"""
from __future__ import annotations

from typing import List, Tuple

NUM_VERTICES: int = 16
BIT_WIDTH: int = 4

# Pre-compute structural lookup tables. These are the only place where these
# integers appear; everything else imports from here.
COMPLEMENT: Tuple[int, ...] = tuple(v ^ 0b1111 for v in range(NUM_VERTICES))


def _popcount(v: int) -> int:
    n = 0
    while v:
        n += v & 1
        v >>= 1
    return n


POPCOUNT: Tuple[int, ...] = tuple(_popcount(v) for v in range(NUM_VERTICES))

# SHELLS[k] is the tuple of vertices with Hamming weight k.
SHELLS: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(v for v in range(NUM_VERTICES) if POPCOUNT[v] == k)
    for k in range(BIT_WIDTH + 1)
)


def enumerate_vertices() -> List[int]:
    """Return [0, 1, ..., 15]."""
    return list(range(NUM_VERTICES))


def complement(v: int) -> int:
    """The complement vertex v̄ = v XOR 0b1111."""
    if not 0 <= v < NUM_VERTICES:
        raise ValueError(f"vertex out of range: {v}")
    return v ^ 0b1111


def popcount(v: int) -> int:
    """Hamming weight (number of set bits) of vertex v."""
    if not 0 <= v < NUM_VERTICES:
        raise ValueError(f"vertex out of range: {v}")
    return POPCOUNT[v]


def rotate_left_1(v: int) -> int:
    """Bit-rotate-left-1 on the 4-bit field, used by S25 (VestanaCircular).

    Example: 0b1011 -> 0b0111 (the high bit wraps around to bit 0).
    """
    if not 0 <= v < NUM_VERTICES:
        raise ValueError(f"vertex out of range: {v}")
    return ((v << 1) & 0b1111) | ((v >> 3) & 0b1)


def pairs_v_lt_complement() -> Tuple[Tuple[int, int], ...]:
    """Return the 8 pairs (v, v̄) with v < v̄, indexed in ascending v.

    These are the canonical S22 outputs and the index set used by L_cons R2.
    """
    out: List[Tuple[int, int]] = []
    for v in range(NUM_VERTICES):
        c = COMPLEMENT[v]
        if v < c:
            out.append((v, c))
    return tuple(out)
