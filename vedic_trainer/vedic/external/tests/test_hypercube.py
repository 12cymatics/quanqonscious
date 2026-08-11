"""Smoke tests for the hypercube operators adapter."""
from __future__ import annotations

import numpy as np

from vedic.external import Hypercube, ProofTester


def test_lambda_operator_is_symmetric() -> None:
    cube = Hypercube(size=8)
    alpha = np.ones(16)
    Lambda = cube.lambda_operator(alpha)
    assert Lambda.shape == (8, 8)
    assert np.allclose(Lambda, Lambda.T)


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


def test_prooftester_hypercube_invocations() -> None:
    tester = ProofTester()
    tester.verify_hypercube()
    res = tester.results["hypercube"]
    assert res["P_shape"] == (8, 8)
    assert res["Lambda_shape"] == (8, 8)
    assert res["Omega_shape"] == (8, 8)
    assert res["Upsilon_shape"] == (8, 8)
