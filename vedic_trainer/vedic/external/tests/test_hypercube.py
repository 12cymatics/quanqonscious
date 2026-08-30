"""Shape, finiteness and construction-invariant checks on the hypercube adapter.

What these four tests actually assert, in full:

- ``lambda_operator`` returns an (8, 8) array equal to its own transpose
  under ``np.allclose``. Λ is built as ``0.5 * (result + result.T)``, so the
  symmetry holds by construction; the test pins the construction, it does not
  discover the property.
- ``weighted_hypercube``, ``omega_operator`` and ``upsilon_operator`` return
  (8, 8) arrays whose entries are all finite.
- ``ProofTester.record_hypercube_shapes`` writes the four expected ``*_shape``
  keys, each (8, 8).

No output of any operator is compared against a reference value, an
independent implementation, or an analytic result. Nothing here establishes
that Λ, Ω, Υ or the weighted hypercube compute the intended quantity — only
that they run, return the declared shape, and stay finite.
"""
from __future__ import annotations

import numpy as np

from vedic.external import Hypercube, ProofTester


def test_lambda_operator_is_symmetric() -> None:
    cube = Hypercube(size=8)
    alpha = np.ones(16)
    Lambda = cube.lambda_operator(alpha)
    assert Lambda.shape == (8, 8)
    # Λ is built as 0.5 * (result + result.T). Float addition is exactly
    # commutative, so Λ[i][j] and Λ[j][i] are computed from the same two
    # values and are bitwise identical — this is an exact equality, and
    # np.allclose was hiding that by admitting a tolerance it never needed.
    assert np.array_equal(Lambda, Lambda.T)


def test_weighted_hypercube_shape() -> None:
    cube = Hypercube(size=8)
    P = cube.weighted_hypercube(chi=0.5)
    assert P.shape == (8, 8)
    assert np.all(np.isfinite(P))


def test_omega_upsilon_are_well_typed() -> None:
    cube = Hypercube(size=8)
    alpha = np.ones(16)
    Omega = cube.omega_operator(chi=0.5, alpha_vector=alpha)
    Upsilon = cube.upsilon_operator(chi=0.5, alpha_vector=alpha)
    assert Omega.shape == (8, 8)
    assert Upsilon.shape == (8, 8)
    assert np.all(np.isfinite(Omega))
    assert np.all(np.isfinite(Upsilon))


def test_prooftester_records_the_four_hypercube_shapes() -> None:
    tester = ProofTester()
    tester.record_hypercube_shapes()
    res = tester.results["hypercube"]
    assert res["P_shape"] == (8, 8)
    assert res["Lambda_shape"] == (8, 8)
    assert res["Omega_shape"] == (8, 8)
    assert res["Upsilon_shape"] == (8, 8)
