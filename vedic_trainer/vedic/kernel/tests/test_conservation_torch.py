"""The torch residuals agree with the exact-ℚ ones — exactly, with no tolerance.

`conservation_torch.py` claimed such a test existed for its whole life while
nothing imported the module and no test referenced it. This is that test.

Why there is no tolerance any more
----------------------------------
This file used to compare ``abs(float(got) - float(exact)) <= 1e-7 * scale``
on inputs with arbitrary denominators (``randint(1, 97)``). Those rationals
have no exact float64 representation, so the comparison was measuring how
much error IEEE-754 rounding introduced, not whether the torch code computes
the residual. The tolerance absorbed a real maximum of 1.819e-11 — and would
equally have absorbed a genuine error four orders of magnitude larger.

The inputs are dyadic rationals, k/2^m, which float64 represents exactly.
Every operation the residuals perform on them — addition, subtraction,
multiplication, and division by 16 and by 2, both powers of two — is then
exact in float64 as well. The torch residuals come out as exactly 0.0, so
the assertion is ``== 0`` and no tolerance is needed or permitted.

That is a stronger statement than the old one, not a weaker one: it says the
torch implementation computes the same function, rather than that it lands
somewhere near it. What it deliberately does not claim is anything about
non-dyadic inputs, where any disagreement is float64 rounding rather than a
property of the code — see ``test_non_dyadic_inputs_are_outside_what_float64_represents``.
"""
from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from vedic.kernel.conservation_exact import all_residuals
from vedic.kernel.tests.psi_corpus import DYADIC, PSI_CASES
from vedic.kernel.conservation_torch import (
    cons_l2_torch,
    r1_torch,
    r2_torch,
    r3_torch,
    r4_torch,
)

#: Enumerated dyadic vectors, from psi_corpus. Every component is k/2^m, so
#: float64 represents them — and every sum, difference and halving of them —
#: exactly. Nothing is drawn from a generator: the set is the enumeration of
#: 2^-m for m = 0..10 plus four structural cases.
DYADIC_CASES = DYADIC

#: The structured corpus, for the non-dyadic boundary test below. Its
#: fine_denominators entry has denominator 100003, which is not a power of two.
NON_DYADIC_CASES = tuple(
    (label, psi) for label, psi in PSI_CASES
    if any(Fraction(float(x)) != x for x in psi)
)


def _to_tensor(psi) -> torch.Tensor:
    return torch.tensor([float(x) for x in psi], dtype=torch.float64).unsqueeze(0)


DYADIC_LABELS = [label for label, _ in DYADIC_CASES]


@pytest.mark.parametrize("label,psi", DYADIC_CASES, ids=DYADIC_LABELS)
def test_torch_residuals_equal_the_exact_ones(label, psi):
    """Exact equality on dyadic inputs. No tolerance, no scale factor."""
    t = _to_tensor(psi)
    _, r2, r3, r4 = all_residuals(psi, Fraction(0))
    for name, exact, got in (("R2", r2, r2_torch(t)),
                             ("R3", r3, r3_torch(t)),
                             ("R4", r4, r4_torch(t))):
        assert float(got.squeeze()) == float(exact) == 0.0, (
            f"{name} on {label}: torch gave {float(got.squeeze())!r}, "
            f"exact ℚ gives {exact}")


@pytest.mark.parametrize("label,psi", DYADIC_CASES, ids=DYADIC_LABELS)
def test_the_inputs_really_are_exact_in_float64(label, psi):
    """Guards the test above.

    If a component were not exactly representable, ``== 0.0`` would be
    asserting that rounding happened to cancel, and the file would be back to
    testing float64 behaviour instead of the residual formulas.
    """
    for i, x in enumerate(psi):
        assert Fraction(float(x)) == x, \
            f"{label} component {i} = {x} is not exact in float64"
        assert x.denominator & (x.denominator - 1) == 0, \
            f"{label} component {i} = {x} has a non-dyadic denominator"


def test_non_dyadic_inputs_are_outside_what_float64_represents():
    """States the boundary this file draws, rather than leaving it implied.

    On denominators with odd factors the torch residual is nonzero — 1.819e-11
    at its worst over the old input set. That number is IEEE-754 rounding, not
    a property of ``conservation_torch``, and the 1e-7 tolerance that used to
    stand here absorbed it along with anything else up to four orders of
    magnitude larger.
    """
    assert NON_DYADIC_CASES, \
        "no non-dyadic corpus vector; this test would prove nothing"
    worst = 0.0
    for label, psi in NON_DYADIC_CASES:
        t = _to_tensor(psi)
        for fn in (r2_torch, r3_torch, r4_torch):
            worst = max(worst, abs(float(fn(t).squeeze())))
    assert worst > 0.0, (
        "non-dyadic inputs produced exactly zero residuals, so float64 is "
        "representing them exactly and the dyadic restriction above is "
        "unnecessary — reconsider it rather than keeping a distinction that "
        "does not exist")


def test_the_exact_residuals_really_are_identities():
    """Guards the test above: if R2..R4 were not identically zero over ℚ,
    matching them within a tolerance would be a much weaker statement."""
    # All 100 seeds, the same set the tolerance test above uses. Checking a
    # fifth of them made this guard weaker than the thing it guards: a seed
    # where R2..R4 were not identically zero would sit in the untested 80 and
    # the tolerance comparison above would still pass on it.
    for label, psi in DYADIC_CASES + PSI_CASES:
        _, r2, r3, r4 = all_residuals(psi, Fraction(0))
        assert (r2, r3, r4) == (0, 0, 0), f"{label}: R2..R4 not identically zero"


@pytest.mark.parametrize("trace", [0, 1, 434, 435, 436, 870])
def test_r1_is_the_trace_counter_modulo_435(trace: int):
    # Integers below 2^53 are exact in float64, so this is an exact equality.
    got = float(r1_torch(torch.tensor([trace], dtype=torch.float64)).squeeze())
    assert got == float(trace % 435), f"trace {trace}: got {got!r}"


def test_cons_l2_is_dominated_by_the_trace_term():
    """Why L_cons no longer uses this: R2..R4 vanish, so the sum is R1².

    At trace_sum = 336 the term is ~112,896 against a true CE of ~1.7 -- and
    it carries no gradient w.r.t. Psi at all.
    """
    # R2..R4 are exactly zero on a dyadic Ψ, so the sum is exactly R1² and
    # 336² = 112896 is exact in float64. No rel= tolerance is needed.
    psi = _to_tensor(DYADIC_CASES[0][1])
    val = float(cons_l2_torch(psi, torch.tensor([336.0], dtype=torch.float64)))
    assert val == 336.0 ** 2 == 112896.0, f"got {val!r}"


def test_cons_l2_has_no_gradient_to_psi():
    psi = _to_tensor(DYADIC_CASES[1][1]).requires_grad_(True)
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
