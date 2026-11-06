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
    rational cochains in every degree zero through four
    d1(d0(f)) = 0, d2(d1(g)) = 0, and d3(d2(h)) = 0
    the four-axis graph Laplacian
    annihilation of constants
    zero total Laplacian sum

Everything is exact ℚ.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from numbers import Integral, Rational
from typing import Dict, List, Mapping, Sequence, Tuple

N_AXES: int = 4
N_VERTICES: int = 1 << N_AXES          # 16
FULL_MASK: int = N_VERTICES - 1        # 0b1111

# A 0-cochain is a value per vertex; a 1-cochain a value per oriented edge;
# higher cochains use (base vertex, increasing active axes) as their keys.
Cochain0 = Tuple[Fraction, ...]
Cochain1 = Dict[Tuple[int, int], Fraction]
Cochain2 = Dict[Tuple[int, int, int], Fraction]
Cochain3 = Dict[Tuple[int, int, int, int], Fraction]
Cochain4 = Dict[Tuple[int, int, int, int, int], Fraction]


def toggle(v: int, j: int) -> int:
    """Flip exactly bit j of the vertex label."""
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise TypeError("vertex must be an integer")
    if isinstance(j, bool) or not isinstance(j, Integral):
        raise TypeError("axis must be an integer")
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


def cubes() -> Tuple[Tuple[int, int, int, int], ...]:
    """All eight oriented 3-cells, keyed (base, i, j, k), i < j < k.

    The base has all three active-axis bits clear. Orientation follows the
    increasing axis order, consistently with ``plaquettes``.
    """
    return tuple(
        (v, i, j, k)
        for v in range(N_VERTICES)
        for i, j, k in combinations(range(N_AXES), 3)
        if all(not (v >> axis) & 1 for axis in (i, j, k))
    )


def four_cells() -> Tuple[Tuple[int, int, int, int, int], ...]:
    """The oriented 4-cell, based at 0 with axes (0, 1, 2, 3)."""
    return ((0, 0, 1, 2, 3),)


# ----------------------------------------------------------------------
# Coboundary operators
# ----------------------------------------------------------------------


def d0(f: Cochain0) -> Cochain1:
    """(d⁰f)(u,w) = f(w) − f(u) on every oriented edge."""
    f = _check0(f)
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

    Every canonical edge must be supplied. Reverse edges may also be
    supplied, but must be the negative of their canonical orientation.
    """
    g = _check1(g)
    out: Cochain2 = {}
    for (v, j, k) in plaquettes():
        a = toggle(v, j)
        b = toggle(v, k)
        c = toggle(a, k)
        out[(v, j, k)] = g[(v, a)] + g[(a, c)] - g[(b, c)] - g[(v, b)]
    return out


def d2(h: Cochain2) -> Cochain3:
    """Signed six-face boundary sum on each oriented cube.

    For axes i < j < k, take upper-minus-lower faces with signs +, −, +:
    h(v⊕2ⁱ,j,k)−h(v,j,k) − h(v⊕2ʲ,i,k)+h(v,i,k)
    + h(v⊕2ᵏ,i,j)−h(v,i,j).
    All 24 plaquette values are required, including explicitly given zeros.
    """
    h = _check_cells(h, plaquettes(), "2-cochain")
    out: Cochain3 = {}
    for v, i, j, k in cubes():
        out[(v, i, j, k)] = (
            h[(toggle(v, i), j, k)] - h[(v, j, k)]
            - h[(toggle(v, j), i, k)] + h[(v, i, k)]
            + h[(toggle(v, k), i, j)] - h[(v, i, j)]
        )
    return out


def d3(k: Cochain3) -> Cochain4:
    """Signed eight-cube boundary sum on the oriented 4-cell.

    Removing active axis r gives sign (−1)^r on its upper-minus-lower
    faces. All eight cube values must be explicitly supplied.
    """
    k = _check_cells(k, cubes(), "3-cochain")
    out: Cochain4 = {}
    for cell in four_cells():
        v, *axes = cell
        value = Fraction(0)
        for position, axis in enumerate(axes):
            face_axes = tuple(axes[:position] + axes[position + 1:])
            upper = (toggle(v, axis), *face_axes)
            lower = (v, *face_axes)
            sign = -1 if position % 2 else 1
            value += sign * (k[upper] - k[lower])
        out[cell] = value
    return out


def laplacian(f: Cochain0) -> Cochain0:
    """The four-axis graph Laplacian: (Δf)(v) = Σ_j (f(v⊕2ʲ) − f(v)).

    Sign convention matches the engine's DIFF stencil.
    """
    f = _check0(f)
    return tuple(
        sum((f[toggle(v, j)] - f[v] for j in range(N_AXES)), Fraction(0))
        for v in range(N_VERTICES)
    )


def constant(value: int | Fraction) -> Cochain0:
    """An exact constant zero-cochain; floating-point values are rejected."""
    q = _rational(value, "constant value")
    return tuple(q for _ in range(N_VERTICES))


def total(f: Cochain0) -> Fraction:
    f = _check0(f)
    return sum(f, Fraction(0))


def _rational(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError(f"{name} must be an exact rational (integer or Fraction)")
    return Fraction(int(value.numerator), int(value.denominator))


def _check0(f: Sequence[Fraction]) -> Cochain0:
    if not isinstance(f, Sequence) or isinstance(f, (str, bytes, bytearray)):
        raise TypeError("0-cochain must be an ordered sequence of exact rational values")
    if len(f) != N_VERTICES:
        raise ValueError(f"0-cochain needs {N_VERTICES} values; got {len(f)}")
    return tuple(_rational(value, f"0-cochain[{index}]") for index, value in enumerate(f))


def _check_key(key: object, arity: int, name: str) -> None:
    if not isinstance(key, tuple) or len(key) != arity:
        raise ValueError(f"{name} keys must be tuples with {arity} integer entries")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in key):
        raise TypeError(f"{name} cell indices must be integers")


def _check1(g: Mapping[Tuple[int, int], Fraction]) -> Cochain1:
    if not isinstance(g, Mapping):
        raise TypeError("1-cochain must be a mapping of oriented edges to exact rationals")
    canonical = set(edges())
    allowed = canonical | {(w, v) for v, w in canonical}
    for key in g:
        _check_key(key, 2, "1-cochain")
    missing = canonical.difference(g)
    extra = set(g).difference(allowed)
    if missing or extra:
        raise ValueError(f"1-cochain has missing canonical edges {sorted(missing)} or invalid edges {sorted(extra)}")
    checked = {key: _rational(value, f"1-cochain[{key}]") for key, value in g.items()}
    for v, w in canonical:
        if (w, v) in checked and checked[(w, v)] != -checked[(v, w)]:
            raise ValueError(f"1-cochain reverse edge {(w, v)} must negate {(v, w)}")
    return checked


def _check_cells(
    values: Mapping[Tuple[int, ...], Fraction],
    cells: Tuple[Tuple[int, ...], ...],
    name: str,
) -> Dict[Tuple[int, ...], Fraction]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of canonical cells to exact rationals")
    for key in values:
        _check_key(key, len(cells[0]), name)
    missing = set(cells).difference(values)
    extra = set(values).difference(cells)
    if missing or extra:
        raise ValueError(f"{name} has missing cells {sorted(missing)} or invalid cells {sorted(extra)}")
    return {key: _rational(value, f"{name}[{key}]") for key, value in values.items()}
