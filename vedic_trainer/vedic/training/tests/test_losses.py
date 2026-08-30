"""The four auxiliary losses, pinned to the definitions they now have.

Why this exists
---------------
`losses.py` had no tests. Two of its four terms have since had a
normalisation removed — `L_chi` divided the antisymmetric energy by the total
energy, `L_curv` formed a Rayleigh quotient and then hinged it against
`kappa.detach().mean()`, a statistic of whichever examples shared the batch —
and a change to a loss definition with nothing asserting the old or the new
form is a change nobody can see.

Every check here is exact. The Ψ inputs are dyadic (every component is
k/2^m), so they and every sum, difference and halving of them are represented
exactly in float32, and no comparison needs a tolerance. Nothing is sampled.

The two tests that would have failed before the change are
`test_l_chi_is_not_scale_free` and `test_l_curv_is_non_zero_on_a_single_row`:
the ratio was invariant under Ψ → 2Ψ, and `relu(κ − κ.mean())` is identically
zero on a batch of one whatever Ψ is.
"""
from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.kernel.tesseract import NUM_VERTICES
from vedic.kernel.wht import wht_axis_torch
from vedic.kernel.z2_primitives import s7_sankalana_vyavakalana
from vedic.training.losses import L_chi, L_cons, L_curv, L_dual, total_loss

FULL_MASK = NUM_VERTICES - 1

#: Four dyadic Ψ rows. Components are quarters in [−2, 2], so every quantity
#: below — halves, squares, sums of sixteen of them, a mean over four — is
#: exactly representable in float32 and every assertion can use `==`.
ROWS_Q: tuple[tuple[Fraction, ...], ...] = (
    tuple(Fraction(v - 8, 4) for v in range(NUM_VERTICES)),
    tuple(Fraction((-1) ** v * (v % 5), 4) for v in range(NUM_VERTICES)),
    tuple(Fraction(1, 4) for _ in range(NUM_VERTICES)),
    tuple(Fraction(0) for _ in range(NUM_VERTICES)),
)
#: A second, deliberately small pair of rows for L_dual. Its identity below
#: subtracts two comparable quantities, so it loses significance in float32
#: when ‖p‖² is large — with the rows above, `norm_sq` is tens and the
#: residual is tenths. These rows keep every intermediate small enough that
#: the subtraction is exact and the assertion can stay an equality rather
#: than acquiring a tolerance.
DUAL_ROWS_Q: tuple[tuple[Fraction, ...], ...] = (
    tuple(Fraction(1) if v == 0 else Fraction(0) for v in range(NUM_VERTICES)),
    tuple(Fraction(v % 3 - 1, 4) for v in range(NUM_VERTICES)),
)


def _tensor(rows: tuple[tuple[Fraction, ...], ...]) -> torch.Tensor:
    return torch.tensor([[float(x) for x in row] for row in rows],
                        dtype=torch.float32)


PSI = _tensor(ROWS_Q)
DUAL_PSI = _tensor(DUAL_ROWS_Q)


@pytest.fixture(scope="module")
def modules() -> tuple[S5, S7, S11, HessianModule, torch.Tensor]:
    return S5(), S7(), S11(), HessianModule(), wht_axis_torch(device="cpu")


# ---------------------------------------------------------------- L_chi


def test_l_chi_is_the_antisymmetric_energy(modules) -> None:
    """Exact, against the ℚ kernel rather than against the torch port.

    ``A(Ψ)_v = (Ψ_v − Ψ_{v⊕15})/2``; the loss is the mean over the batch of
    ‖A‖². Computing the reference through ``s7_sankalana_vyavakalana`` — the
    exact-ℚ primitive, not ``sutras_torch.S7`` — means agreement here is a
    statement about both the formula and the port, not a restatement of the
    implementation.
    """
    _, s7, _, _, _ = modules
    expected = Fraction(0)
    for row in ROWS_Q:
        _, anti = s7_sankalana_vyavakalana(row, mask=FULL_MASK)
        expected += sum((a * a for a in anti), Fraction(0))
    expected /= Fraction(len(ROWS_Q))

    assert float(L_chi(PSI, s7)) == float(expected)


def test_l_chi_is_not_scale_free(modules) -> None:
    """The regression test for the removed denominator.

    ‖A‖²/‖Ψ‖² is invariant under Ψ → 2Ψ. ‖A‖² is quadratic in it. This
    would have read `== value` before the change and reads `== 4 * value`
    now; doubling is exact in binary floating point, so it is an equality.
    """
    _, s7, _, _, _ = modules
    value = float(L_chi(PSI, s7))
    assert value != 0.0, "the corpus row set gives no antisymmetric energy"
    assert float(L_chi(2.0 * PSI, s7)) == 4.0 * value
    assert float(L_chi(PSI / 2.0, s7)) == value / 4.0


def test_l_chi_is_zero_on_an_all_zero_batch(modules) -> None:
    """No denominator means no degenerate input: the value is simply 0.

    This used to raise ``ValueError`` because ‖Ψ‖² sat under a division bar.
    """
    _, s7, _, _, _ = modules
    assert float(L_chi(torch.zeros(2, NUM_VERTICES), s7)) == 0.0


def test_l_chi_reaches_psi(modules) -> None:
    _, s7, _, _, _ = modules
    psi = PSI.clone().requires_grad_(True)
    grad, = torch.autograd.grad(L_chi(psi, s7), psi)
    assert float(grad.abs().sum()) != 0.0


# --------------------------------------------------------------- L_curv


def test_l_curv_is_non_zero_on_a_single_row(modules) -> None:
    """The regression test for the removed batch-relative baseline.

    ``relu(κ − κ.detach().mean())`` on a batch of one is ``relu(κ − κ)`` — a
    hard zero for every Ψ, so the term contributed nothing whenever the batch
    size was one and contributed only above-average rows otherwise.
    """
    _, _, _, hessian, _ = modules
    single = PSI[:1]
    assert float(L_curv(single, hessian)) != 0.0


def test_l_curv_does_not_depend_on_the_rest_of_the_batch(modules) -> None:
    """A per-example loss must not change because its neighbours changed.

    The batch here is one row repeated, so the mean over two rows is that
    row's own value exactly (v + v = 2v and 2v/2 = v are both exact). Under
    the old form the answer was 0 for *any* such batch, since every row sat
    exactly at the batch mean.
    """
    _, _, _, hessian, _ = modules
    one = float(L_curv(PSI[:1], hessian))
    doubled = float(L_curv(PSI[:1].repeat(2, 1), hessian))
    assert doubled == one


def test_l_curv_scales_quadratically(modules) -> None:
    """⟨Ψ, g_ab Ψ⟩ is a quadratic form; the Rayleigh quotient was scale-free."""
    _, _, _, hessian, _ = modules
    value = float(L_curv(PSI, hessian))
    assert value != 0.0
    assert float(L_curv(2.0 * PSI, hessian)) == 4.0 * value


def test_l_curv_is_zero_on_an_all_zero_batch(modules) -> None:
    """Also used to raise: the Rayleigh denominator was ⟨Ψ, Ψ⟩."""
    _, _, _, hessian, _ = modules
    assert float(L_curv(torch.zeros(2, NUM_VERTICES), hessian)) == 0.0


def test_l_curv_reaches_psi(modules) -> None:
    """It did not, for the whole of the first ablation: the power-iteration
    form had no ``grad_fn`` at all."""
    _, _, _, hessian, _ = modules
    psi = PSI.clone().requires_grad_(True)
    grad, = torch.autograd.grad(L_curv(psi, hessian), psi)
    assert float(grad.abs().sum()) != 0.0


# --------------------------------------------------------------- L_dual


def test_l_dual_residual_is_orthogonal_to_the_span(modules) -> None:
    """Why the two divisions by 16 stay: they are projection coefficients.

    {𝟙, h₀, h₁, h₂, h₃} are pairwise orthogonal with norm² 16, so by
    Pythagoras the squared residual of the orthogonal projection is
    ‖p‖² − (⟨p,𝟙⟩² + Σ_k ⟨p,h_k⟩²)/16. That identity holds *only* for the
    orthogonal coefficients: drop the /16 and the reconstruction is a
    different vector whose residual is not orthogonal and does not satisfy
    it. Comparing L_dual against the identity therefore checks the
    coefficients rather than restating the code.
    """
    s5, _, s11, _, wht = modules
    p = s11(s5(DUAL_PSI))
    norm_sq = (p * p).sum(dim=-1)
    along_one = p.sum(dim=-1) ** 2
    along_axes = ((p.unsqueeze(1) * wht.unsqueeze(0)).sum(dim=-1) ** 2).sum(dim=-1)
    expected = (norm_sq - (along_one + along_axes) / float(NUM_VERTICES)).mean()
    value = float(L_dual(DUAL_PSI, wht, s5, s11))
    assert value != 0.0, "these rows already lie in the span; the test is vacuous"
    assert value == float(expected)


def test_l_dual_is_zero_on_an_all_zero_batch(modules) -> None:
    s5, _, s11, _, wht = modules
    assert float(L_dual(torch.zeros(2, NUM_VERTICES), wht, s5, s11)) == 0.0


# --------------------------------------------------------------- L_cons


def test_l_cons_is_zero_on_an_all_zero_batch() -> None:
    assert float(L_cons(torch.zeros(2, NUM_VERTICES))) == 0.0


def test_l_cons_takes_no_trace_counter() -> None:
    """It summed R1..R4, and R1 is a step counter with no Ψ in it."""
    import inspect
    assert list(inspect.signature(L_cons).parameters) == ["psi"]


# ------------------------------------------------------------ total_loss


def test_total_loss_is_the_weighted_sum(modules) -> None:
    s5, s7, s11, hessian, wht = modules
    ce = torch.tensor(1.5)
    weights = (0.25, 0.5, 0.125, 2.0)
    total, parts = total_loss(ce_loss=ce, psi=PSI, weights=weights, s5=s5,
                              s7=s7, s11=s11, hessian=hessian, wht_axis=wht)
    # Rebuilt in torch, in the same order and the same dtype the function
    # uses. Re-adding the components as Python floats would compare a
    # float64 sum against a float32 one and fail on the dtype, not on the
    # arithmetic.
    rebuilt = (parts["L_CE"]
               + weights[0] * parts["L_chi"]
               + weights[1] * parts["L_cons"]
               + weights[2] * parts["L_curv"]
               + weights[3] * parts["L_dual"])
    assert float(total) == float(parts["L_total"])
    assert float(total) == float(rebuilt)


def test_total_loss_rejects_an_all_zero_batch(modules) -> None:
    """The one guard left in the file, and it is about pooling, not arithmetic.

    Each term is well defined at Ψ = 0 and returns 0 there — which is exactly
    what a broken pooling layer looks like in the log, four auxiliary terms
    all reporting 0.0 and nothing saying why.
    """
    s5, s7, s11, hessian, wht = modules
    with pytest.raises(ValueError, match="identically zero"):
        total_loss(ce_loss=torch.tensor(1.0),
                   psi=torch.zeros(2, NUM_VERTICES), weights=(1.0,) * 4,
                   s5=s5, s7=s7, s11=s11, hessian=hessian, wht_axis=wht)
