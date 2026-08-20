"""Exact Q4 cell complex: V4 = Fin 4 → Bool ≅ Z₂⁴, cochains, d² = 0, Laplacian.

Source: *Exact Kernel Evolution Blueprint*, 31 July 2026, §Exact Q4 and
toroidal geometry.

This replaces a ring stencil with the actual Boolean four-cube adjacency.
The blueprint's Gate B obligations, each of which is a test here:

    toggle involution
    commutation of distinct coordinate toggles
    injectivity of the axis-to-neighbour map
    exactly four distinct unsigned neighbours per vertex
    exactly eight oriented axis incidences
    rational zero-, one-, and two-cochains
    d1(d0(f)) = 0 for every zero-cochain
    the four-axis graph Laplacian
    annihilation of constants
    zero total Laplacian sum

Everything is exact ℚ.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

N_AXES: int = 4
N_VERTICES: int = 1 << N_AXES          # 16
FULL_MASK: int = N_VERTICES - 1        # 0b1111

# A 0-cochain is a value per vertex; a 1-cochain a value per oriented edge;
# a 2-cochain a value per oriented square (plaquette).
Cochain0 = Tuple[Fraction, ...]
Cochain1 = Dict[Tuple[int, int], Fraction]
Cochain2 = Dict[Tuple[int, int, int], Fraction]


def toggle(v: int, j: int) -> int:
    """Flip exactly bit j of the vertex label."""
    if not 0 <= v < N_VERTICES:
        raise ValueError(f"vertex out of range: {v}")
    if not 0 <= j < N_AXES:
        raise ValueError(f"axis out of range: {j}")
    return v ^ (1 << j)


def neighbours(v: int) -> Tuple[int, ...]:
    """The four unsigned neighbours of v, one per axis."""
    return tuple(toggle(v, j) for j in range(N_AXES))


def edges() -> Tuple[Tuple[int, int], ...]:
    """The 32 unsigned edges (u < w), each an axis toggle."""
    out: List[Tuple[int, int]] = []
    for v in range(N_VERTICES):
        for j in range(N_AXES):
            w = toggle(v, j)
            if v < w:
                out.append((v, w))
    return tuple(out)


def oriented_incidences(v: int) -> Tuple[Tuple[int, int], ...]:
    """The eight oriented axis incidences at v: outgoing and incoming per axis."""
    out: List[Tuple[int, int]] = []
    for j in range(N_AXES):
        w = toggle(v, j)
        out.append((v, w))   # outgoing
        out.append((w, v))   # incoming
    return tuple(out)


def plaquettes() -> Tuple[Tuple[int, int, int], ...]:
    """Oriented 2-cells, keyed (v, j, k) with j < k, base vertex v having both
    bits clear. Each square is v → v⊕2ʲ → v⊕2ʲ⊕2ᵏ → v⊕2ᵏ → v."""
    out: List[Tuple[int, int, int]] = []
    for v in range(N_VERTICES):
        for j, k in combinations(range(N_AXES), 2):
            if not (v >> j) & 1 and not (v >> k) & 1:
                out.append((v, j, k))
    return tuple(out)


# ----------------------------------------------------------------------
# Coboundary operators
# ----------------------------------------------------------------------


def d0(f: Cochain0) -> Cochain1:
    """(d⁰f)(u,w) = f(w) − f(u) on every oriented edge."""
    _check0(f)
    out: Cochain1 = {}
    for v in range(N_VERTICES):
        for j in range(N_AXES):
            w = toggle(v, j)
            out[(v, w)] = f[w] - f[v]
    return out


def d1(g: Cochain1) -> Cochain2:
    """(d¹g) on a plaquette = the signed sum around its boundary loop.

    For the square v → a → c → b → v with a = v⊕2ʲ, b = v⊕2ᵏ, c = v⊕2ʲ⊕2ᵏ:

        g(v,a) + g(a,c) − g(b,c) − g(v,b)
    """
    out: Cochain2 = {}
    for (v, j, k) in plaquettes():
        a = toggle(v, j)
        b = toggle(v, k)
        c = toggle(a, k)
        out[(v, j, k)] = g[(v, a)] + g[(a, c)] - g[(b, c)] - g[(v, b)]
    return out


def laplacian(f: Cochain0) -> Cochain0:
    """The four-axis graph Laplacian: (Δf)(v) = Σ_j (f(v⊕2ʲ) − f(v)).

    Sign convention matches the engine's DIFF stencil.
    """
    _check0(f)
    return tuple(
        sum((f[toggle(v, j)] - f[v] for j in range(N_AXES)), Fraction(0))
        for v in range(N_VERTICES)
    )


def constant(value) -> Cochain0:
    q = Fraction(value)
    return tuple(q for _ in range(N_VERTICES))


def total(f: Cochain0) -> Fraction:
    _check0(f)
    return sum(f, Fraction(0))


def _check0(f: Sequence[Fraction]) -> None:
    if len(f) != N_VERTICES:
        raise ValueError(f"0-cochain needs {N_VERTICES} values; got {len(f)}")
