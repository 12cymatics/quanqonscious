"""g_ab is positive definite, exactly, so L_curv's only minimiser is Ψ = 0.

Why this exists
---------------
L_curv = mean_b ⟨Ψ, g_ab Ψ⟩. Whether that term can be lowered only by changing
Ψ's *direction*, or also by shrinking Ψ, depends on g_ab's definiteness. The
pre-flight audit of the experiment found that every auxiliary term can be
driven toward zero by turning the projection away from the data, at no cost
to cross-entropy; for L_curv that is guaranteed, and this pins why: g_ab is
positive definite, so ⟨Ψ, g Ψ⟩ > 0 for all Ψ ≠ 0 and the infimum is reached
only at Ψ = 0.

The check is exact. An LDLᵀ elimination over ℚ, without pivoting, of a
symmetric matrix succeeds with every pivot positive if and only if the matrix
is positive definite (Sylvester's criterion: the pivots are ratios of
consecutive leading principal minors).
"""
from __future__ import annotations

from fractions import Fraction

import torch

from vedic.kernel.hessian import HessianModule, hessian_dense_torch, hessian_exact

#: The smallest eigenvalue of g_ab, bracketed exactly below.
LAMBDA_MIN = Fraction(73, 16)


def _pivots(m: list[list[Fraction]]) -> list[Fraction]:
    a = [row[:] for row in m]
    n = len(a)
    pivots = []
    for k in range(n):
        p = a[k][k]
        pivots.append(p)
        if p == 0:
            break
        for i in range(k + 1, n):
            f = a[i][k] / p
            for j in range(k, n):
                a[i][j] -= f * a[k][j]
    return pivots


def _shifted(shift: Fraction) -> list[list[Fraction]]:
    g = [list(r) for r in hessian_exact()]
    return [[g[i][j] - (shift if i == j else 0) for j in range(16)] for i in range(16)]


def test_g_ab_is_symmetric():
    g = [list(r) for r in hessian_exact()]
    assert all(g[i][j] == g[j][i] for i in range(16) for j in range(16))


def _non_positive(m: list[list[Fraction]]) -> list[tuple[int, Fraction]]:
    pivots = _pivots(m)
    assert len(pivots) == 16, f"elimination stopped at pivot {len(pivots) - 1}"
    return [(i, p) for i, p in enumerate(pivots) if p <= 0]


def test_g_ab_is_positive_definite():
    bad = _non_positive([list(r) for r in hessian_exact()])
    assert not bad, f"non-positive LDL pivots (index, value): {bad}"


def test_the_matrices_l_curv_uses_are_positive_definite():
    """L_curv reads g from HessianModule's buffer, built by
    hessian_dense_torch, not from hessian_exact(). A defect in that path left
    the test above green (found by regeneration testing). Each float is
    converted to the Fraction it exactly represents, so the elimination is
    still exact; the margin, 73/16, dwarfs float32 rounding."""
    for name, g in (("HessianModule().g", HessianModule().g),
                    ("hessian_dense_torch(float64)",
                     hessian_dense_torch(dtype=torch.float64))):
        m = [[Fraction(v) for v in row] for row in g.tolist()]
        bad = _non_positive(m)
        assert not bad, f"{name}: non-positive LDL pivots (index, value): {bad}"


def test_the_smallest_eigenvalue_is_73_over_16():
    """g − λI is positive definite exactly when λ is below the smallest
    eigenvalue. Just below 73/16 it is; just above it is not. This is also
    the criterion seen to fail, so the gate above is not vacuous."""
    eps = Fraction(1, 10**6)
    below = _pivots(_shifted(LAMBDA_MIN - eps))
    assert len(below) == 16 and all(p > 0 for p in below), \
        f"g − (73/16 − 1e-6)·I is not positive definite: pivots {below}"
    assert not all(p > 0 for p in _pivots(_shifted(LAMBDA_MIN + eps))), \
        "g − (73/16 + 1e-6)·I is still positive definite: 73/16 is not the minimum"
