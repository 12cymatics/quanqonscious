"""Four sutra-derived auxiliary losses applied to the Tesseract memory.

    L_χ    : contradiction (S7 antisymmetric energy)
    L_cons : conservation (drift of mass, ‖S‖² and ‖A‖² under S29)
    L_curv : curvature   (energy ⟨Ψ, g_ab Ψ⟩ at Ψ)
    L_dual : dual-basis  ((S5∘S11) Ψ vs WHT-axis reconstruction)

L_cons and L_curv are NOT what an earlier version of this header
specified. It named ``Σᵢ Rᵢ²`` over R1..R4 for the first and a top
eigenvalue of g_ab obtained by Lanczos for the second; neither body does
that, and neither could — R1..R4 are a step counter plus three algebraic
identities, and g_ab is constant in Ψ, so both of the specified
quantities have identically zero gradient. Each function's own docstring
derives its replacement and is the authoritative description; this list
summarises them. There is no Lanczos and no eigensolve in this file.

Total loss is composed by the trainer:

    L = L_CE + α·L_χ + β·L_cons + γ·L_curv + δ·L_dual

The kernel paths (sutras, Hessian, WHT) are exact-rational reference
operations; the autograd path uses torch float buffers built from those
exact rationals (every constant in this module traces back to a Fraction
in vedic.kernel — see SUTRA_CATALOGUE.md). No epsilons in the kernel
sutras, and none here either: this file contains no numerical
stabilisers at all.

**No normalisation either.** A division that exists to make a quantity
comparable — by its own scale, or by a statistic of whichever examples share
its batch — is not in this file. Two were:

* ``L_chi`` divided the antisymmetric energy by the total energy, making it
  a dimensionless *fraction* of energy;
* ``L_curv`` divided by ⟨Ψ,Ψ⟩ to form a Rayleigh quotient, then subtracted
  ``kappa.detach().mean()`` — a statistic of the batch, which made a
  per-example loss depend on its neighbours and reduced the term to a hinge
  against a shifting, gradient-free baseline.

Both are gone. The consequence is stated rather than hidden: without those
denominators both terms are **scale-dependent**, quadratic in ‖Ψ‖, so a model
can reduce either by shrinking Ψ rather than by changing its structure. That
is a real property of this objective.

What remains in ``L_dual`` is not a normalisation. ⟨Ψ, h_k⟩ / 16 and
mean(Ψ) are the orthogonal-projection coefficients onto basis vectors of
known norm — ‖h_k‖² = ‖𝟙‖² = 16 is a fixed property of the 16-vertex
tesseract, not a measurement of the data. Dropping them would not make the
term stricter; it would make the reconstruction a different vector and the
residual meaningless. 16 is a power of two, so the division is exact in
float64.

With no denominators there is nothing to floor and no degenerate Ψ to guard
against numerically: an all-zero Ψ gives each term the value 0, which is
correct for these forms. The one check that remains is in ``total_loss``,
and it is about upstream pooling rather than about arithmetic.
"""
from __future__ import annotations

import torch
from torch import Tensor

# NOTE: conservation_torch is deliberately NOT imported here. L_cons no
# longer sums R1..R4 -- see its docstring for why that sum was a constant --
# so importing the residuals would suggest a dependency that does not exist.
from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11, S29
from vedic.kernel.tesseract import NUM_VERTICES

# ---------- L_χ : contradiction ---------------------------------------


def L_chi(psi: Tensor, s7: S7) -> Tensor:
    """Antisymmetric energy of the S7 decomposition.

        L_χ = mean_b ‖A(Ψ)‖²

    **Not a fraction.** This read ``mean_b ‖A‖² / (‖S‖² + ‖A‖²)`` — which is
    ‖A‖²/‖Ψ‖², since S and A are orthogonal — the *share* of Ψ's energy
    sitting in the antisymmetric (contradictory) part. That denominator is a
    normalisation by the input's own scale, and it is gone.

    The trade-off, stated rather than buried: the ratio was scale-free, so
    the only way to reduce it was to move energy out of A and into S. ‖A‖²
    alone is quadratic in ‖Ψ‖, so it can also be reduced by shrinking Ψ
    entirely. Nothing here prevents that. The term measures antisymmetric
    energy, and that is now all it claims to measure.

    ‖S‖² is no longer computed; it only ever appeared in the denominator.
    """
    _, anti = s7(psi)
    return (anti * anti).sum(dim=-1).mean()


# ---------- L_cons : conservation ------------------------------------


def L_cons(psi: Tensor) -> Tensor:
    """Conservation penalty: drift of the conserved quantities under S29.

    **Why this is not Σ Rᵢ².** The four specified residuals cannot serve as a
    loss, and the earlier implementation that summed them was a constant:

    - ``R1 = trace_sum mod 435`` takes no Ψ at all. It is a step counter, and
      squaring it added ``trace_sum²`` — growing quadratically to ~10⁶ — to a
      loss whose true CE term is ~1.7, with gradient exactly zero.
    - ``R2``, ``R3`` and ``R4`` are **algebraic identities**, exactly zero for
      every Ψ (verified over ℚ in ``test_conservation_laws.py``). R2 restates
      that each vertex lies in one complement pair; R3 that S29 is mean
      preserving by construction; R4 that the symmetric and antisymmetric
      parts of an involution are orthogonal. An identity constrains nothing,
      so its gradient is identically zero.

    A conservation *loss* needs a residual that is zero only on a subspace.
    The quantities R1..R4 name are genuinely conserved by S29, so we penalise
    the drift of those quantities when the operator is applied — which is
    non-zero exactly when Ψ leaves the conserved subspace:

        L_cons = (Δmass)² + (Δ‖S‖²)² + (Δ‖A‖²)²,   Δq = q(S29 Ψ) − q(Ψ)

    averaged over the batch. The mean is the reduction from per-example
    residuals to one scalar — the standard definition of a batch loss — and
    not a normalisation: no example's value depends on which others share
    its batch. It is a function of Ψ alone; the R1 step counter is not an
    input here.
    """
    s29 = S29().to(psi.device)
    s7 = S7().to(psi.device)
    out = s29(psi)

    mass = psi.sum(dim=-1)
    mass_out = out.sum(dim=-1)

    sym, anti = s7(psi)
    sym_o, anti_o = s7(out)
    e_sym = (sym * sym).sum(dim=-1)
    e_anti = (anti * anti).sum(dim=-1)
    e_sym_o = (sym_o * sym_o).sum(dim=-1)
    e_anti_o = (anti_o * anti_o).sum(dim=-1)

    d_mass = mass_out - mass
    d_sym = e_sym_o - e_sym
    d_anti = e_anti_o - e_anti
    return (d_mass * d_mass + d_sym * d_sym + d_anti * d_anti).mean()


# ---------- L_curv : curvature spike ---------------------------------


def L_curv(psi: Tensor, hessian: HessianModule) -> Tensor:
    """Curvature energy of g_ab **at Ψ**.

        L_curv = mean_b ⟨Ψ, g_ab Ψ⟩

    **Why not power iteration.** The earlier implementation power-iterated
    from ``torch.randn_like(psi)`` — a random vector, not Ψ — toward the top
    eigenvector of ``g_ab``. But ``hessian.py`` states, and
    ``test_conservation_laws.py`` verifies, that *g_ab is independent of Ψ*:
    every contributing operator is linear. So the iterate, the eigenvalue and
    hence κ were all constant in Ψ, identical across the batch, and
    ``relu(κ − κ.mean())`` was identically zero with no grad_fn at all.

    **Why not the Rayleigh quotient either.** Its replacement,
    ``κ(Ψ) = ⟨Ψ, g_ab Ψ⟩ / ⟨Ψ, Ψ⟩``, carried two normalisations. The
    denominator divides by the input's own scale. Worse, the penalty was
    ``relu(κ − κ.detach().mean())``: a per-example loss shifted by a
    statistic of whichever examples happened to share its batch, with the
    baseline detached so it contributed no gradient. Re-batching the same
    data changed the loss, and an example was penalised for where it sat in
    the batch rather than for what it was.

    ⟨Ψ, g_ab Ψ⟩ is the quadratic form itself — the curvature energy along the
    current state, with no division and no reference to any other example. It
    is differentiable in Ψ with gradient 2·g_ab Ψ.

    The trade-off: unlike the quotient this is quadratic in ‖Ψ‖, so it can be
    reduced by shrinking Ψ as well as by flattening it. There is no epsilon
    and no clamp, because there is no longer anything being divided.
    """
    H = hessian(psi)  # (B, 16, 16) — constant in Ψ: every operator is linear.
    HPsi = torch.einsum("bij,bj->bi", H, psi)
    return (psi * HPsi).sum(dim=-1).mean()


# ---------- L_dual : dual-basis coherence ----------------------------


def L_dual(psi: Tensor, wht_axis: Tensor, s5: S5, s11: S11) -> Tensor:
    """Mismatch between (S5∘S11) Ψ and its WHT-axis reconstruction.

    On the image of S5∘S11, Ψ should lie in span{𝟙, h₀, h₁, h₂, h₃} where
    {h_k} are the four single-axis Walsh rows. We compute the rank-5
    reconstruction and penalise the L² residual.

    The two divisions by 16 here are **projection coefficients, not
    normalisations**. The Walsh rows and 𝟙 are pairwise orthogonal with
    ‖h_k‖² = ‖𝟙‖² = 16, so the coefficient of Ψ along h_k is ⟨Ψ, h_k⟩/‖h_k‖²
    = ⟨Ψ, h_k⟩/16, and along 𝟙 it is ⟨Ψ, 𝟙⟩/16 = mean(Ψ). Both divisors are
    fixed properties of the 16-vertex tesseract; neither is measured from the
    data, and removing them would not make this stricter — it would compute a
    different vector and call the difference a residual. 16 is a power of
    two, so both divisions are exact in float64.
    """
    psi_proj = s11(s5(psi))
    star = psi_proj.mean(dim=-1, keepdim=True)          # ⟨Ψ, 𝟙⟩ / ‖𝟙‖²
    coeffs = (psi_proj.unsqueeze(1) * wht_axis.unsqueeze(0)).sum(dim=-1) / float(NUM_VERTICES)
    reconstruction = star + (coeffs.unsqueeze(-1) * wht_axis.unsqueeze(0)).sum(dim=1)
    residual = psi_proj - reconstruction
    return (residual * residual).sum(dim=-1).mean()


# ---------- total loss --------------------------------------------------


def total_loss(
    ce_loss: Tensor,
    psi: Tensor,
    *,
    weights: tuple[float, float, float, float],
    s5: S5,
    s7: S7,
    s11: S11,
    hessian: HessianModule,
    wht_axis: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine cross-entropy with the four sutra losses.

    Returns the scalar total and a dict of per-component values for logging.

    The all-zero check is the one guard left in this file. It is not
    numerical — none of the four terms divides by anything derived from Ψ any
    more, and all four are perfectly well defined at Ψ = 0, where they are
    simply zero. It is here because a batch of identically-zero Ψ means the
    upstream pooling that produces Ψ has stopped working, and every auxiliary
    term reporting 0.0 is exactly what that failure looks like from the log.
    Silence there would read as "the auxiliary losses are satisfied".
    """
    if not psi.any():
        raise ValueError(
            "total_loss: every Ψ in the batch is identically zero, so all "
            "four auxiliary terms are 0.0 by construction and none of them "
            "is measuring anything. Check the pooling that produces Ψ.")
    alpha, beta, gamma, delta = weights
    chi = L_chi(psi, s7)
    cons = L_cons(psi)
    curv = L_curv(psi, hessian)
    dual = L_dual(psi, wht_axis, s5, s11)
    total = ce_loss + alpha * chi + beta * cons + gamma * curv + delta * dual
    return total, {
        "L_CE": ce_loss.detach(),
        "L_chi": chi.detach(),
        "L_cons": cons.detach(),
        "L_curv": curv.detach(),
        "L_dual": dual.detach(),
        "L_total": total.detach(),
    }
