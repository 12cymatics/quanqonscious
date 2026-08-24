"""The torch residuals agree with the exact-ℚ ones.

`conservation_torch.py` claimed such a test existed for its whole life while
nothing imported the module and no test referenced it. This is that test.
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import torch

from vedic.kernel.conservation_exact import all_residuals
from vedic.kernel.conservation_torch import (
    cons_l2_torch,
    r1_torch,
    r2_torch,
    r3_torch,
    r4_torch,
)

TOL = 1e-7


def _psi(seed: int) -> tuple[Fraction, ...]:
    r = random.Random(seed)
    return tuple(Fraction(r.randint(-999, 999), r.randint(1, 97)) for _ in range(16))


def _to_tensor(psi) -> torch.Tensor:
    return torch.tensor([float(x) for x in psi], dtype=torch.float64).unsqueeze(0)


SEEDS = list(range(100))


@pytest.mark.parametrize("seed", SEEDS)
def test_torch_residuals_match_exact_within_tolerance(seed: int):
    """The 100 randomized Q16 inputs the docstring promised."""
    psi = _psi(seed)
    scale = max(abs(float(x)) for x in psi) or 1.0
    t = _to_tensor(psi)
    _, r2, r3, r4 = all_residuals(psi, Fraction(0))
    for name, exact, got in (("R2", r2, r2_torch(t)),
                             ("R3", r3, r3_torch(t)),
                             ("R4", r4, r4_torch(t))):
        assert abs(float(got.squeeze()) - float(exact)) <= TOL * scale, (
            f"{name} drifted beyond {TOL}·scale on seed {seed}")


def test_the_exact_residuals_really_are_identities():
    """Guards the test above: if R2..R4 were not identically zero over ℚ,
    matching them within a tolerance would be a much weaker statement."""
    for seed in SEEDS[:20]:
        _, r2, r3, r4 = all_residuals(_psi(seed), Fraction(0))
        assert (r2, r3, r4) == (0, 0, 0)


@pytest.mark.parametrize("trace", [0, 1, 434, 435, 436, 870])
def test_r1_is_the_trace_counter_modulo_435(trace: int):
    got = float(r1_torch(torch.tensor([trace], dtype=torch.float64)).squeeze())
    assert got == pytest.approx(trace % 435)


def test_cons_l2_is_dominated_by_the_trace_term():
    """Why L_cons no longer uses this: R2..R4 vanish, so the sum is R1².

    At trace_sum = 336 the term is ~112,896 against a true CE of ~1.7 -- and
    it carries no gradient w.r.t. Psi at all.
    """
    psi = _to_tensor(_psi(1))
    val = float(cons_l2_torch(psi, torch.tensor([336.0], dtype=torch.float64)))
    assert val == pytest.approx(336.0 ** 2, rel=1e-6)


def test_cons_l2_has_no_gradient_to_psi():
    psi = _to_tensor(_psi(2)).requires_grad_(True)
    val = cons_l2_torch(psi, torch.tensor([336.0], dtype=torch.float64))
    g, = torch.autograd.grad(val, psi, allow_unused=True)
    assert g is None or float(g.abs().sum()) == 0.0


# ─────────────────────────────────── the g_ab dtype contract

def test_hessian_is_built_at_the_requested_dtype():
    """float64 must mean float64, not float32 wearing a float64 dtype.

    `hessian_dense_torch` used to build a hardcoded np.float32 intermediate
    and cast afterwards, so requesting float64 returned float32 precision --
    measured max error 6.358e-07 against exact ℚ, which exceeds the TOL this
    file compares against. HESSIAN_Q contains denominator 96, not a power of
    two, so the loss was real rather than notional.
    """
    from fractions import Fraction

    from vedic.kernel.hessian import HESSIAN_Q, hessian_dense_torch

    exact = [[Fraction(x) for x in row] for row in HESSIAN_Q]
    got = hessian_dense_torch(dtype=torch.float64)
    assert got.dtype is torch.float64
    err = max(abs(float(got[i][j]) - float(exact[i][j]))
              for i in range(16) for j in range(16))
    assert err == 0.0, f"float64 request lost precision: max error {err:.3e}"


def test_the_hessian_has_a_non_dyadic_denominator():
    """Guards the test above: with only powers of two, float32 would be exact
    and the dtype defect would have been invisible."""
    from fractions import Fraction

    from vedic.kernel.hessian import HESSIAN_Q

    dens = {Fraction(x).denominator for row in HESSIAN_Q for x in row}
    assert any(d & (d - 1) for d in dens), (
        f"all denominators {sorted(dens)} are powers of two; the dtype "
        f"contract cannot be observed on this matrix")
