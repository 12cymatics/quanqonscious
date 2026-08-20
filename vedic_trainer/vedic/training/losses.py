"""Four sutra-derived auxiliary losses applied to the Tesseract memory.

    L_χ    : contradiction (S7 antisymmetric energy fraction)
    L_cons : conservation (Σᵢ Rᵢ² over R1..R4)
    L_curv : curvature   (top eigenvalue of g_ab via Lanczos)
    L_dual : dual-basis  ((S5∘S11) Ψ vs WHT-axis reconstruction)

Total loss is composed by the trainer:

    L = L_CE + α·L_χ + β·L_cons + γ·L_curv + δ·L_dual

The kernel paths (sutras, Hessian, WHT) are exact-rational reference
operations; the autograd path uses torch float buffers built from those
exact rationals (every constant in this module traces back to a Fraction
in vedic.kernel — see SUTRA_CATALOGUE.md). No epsilons in the kernel
sutras. The only numerical stabilisers in this file are documented and
constrained to the loss aggregation boundary.
"""
from __future__ import annotations

import torch
from torch import Tensor

from vedic.kernel.conservation_torch import (
    cons_l2_torch,
    r1_torch,
    r2_torch,
    r3_torch,
    r4_torch,
)
from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11, S29
from vedic.kernel.tesseract import NUM_VERTICES
from vedic.kernel.wht import wht_axis_torch

CONS_TRACE_KEY = "vedic_trace_sum"
"""Trainer-state key for the running integer trace counter feeding R1.

The trainer increments this by 1 per training example; once 435 examples
have been processed, R1 closes (i.e., evaluates to 0) deterministically.
"""


# ---------- L_χ : contradiction ---------------------------------------


def L_chi(psi: Tensor, s7: S7) -> Tensor:
    """Antisymmetric-energy fraction over the S7 decomposition.

    L_χ = mean_b  ‖A(Ψ)‖² / (‖S(Ψ)‖² + ‖A(Ψ)‖²)

    The denominator is ‖Ψ‖² (S/A are orthogonal), which is non-zero on any
    non-zero Ψ. We assert that invariant rather than add an epsilon.
    """
    sym, anti = s7(psi)
    nS = (sym * sym).sum(dim=-1)
    nA = (anti * anti).sum(dim=-1)
    total = nS + nA
    if (total == 0).any():
        raise ValueError("L_chi: encountered an all-zero Ψ; check upstream pooling.")
    return (nA / total).mean()


# ---------- L_cons : conservation ------------------------------------


def L_cons(psi: Tensor, trace_sum: Tensor) -> Tensor:
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

    normalised by the batch. ``trace_sum`` is retained in the signature and
    reported for the audit chain, but it is a diagnostic, not a loss term.
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


def L_curv(psi: Tensor, hessian: HessianModule, lanczos_iters: int = 16) -> Tensor:
    """Penalise the Rayleigh quotient of g_ab **at Ψ** spiking above the batch mean.

        κ(Ψ) = ⟨Ψ, g_ab Ψ⟩ / ⟨Ψ, Ψ⟩

    **Why not power iteration.** The earlier implementation power-iterated
    from ``torch.randn_like(psi)`` — a random vector, not Ψ — toward the top
    eigenvector of ``g_ab``. But ``hessian.py`` states, and
    ``test_conservation_laws.py`` verifies, that *g_ab is independent of Ψ*:
    every contributing operator is linear. So the iterate, the eigenvalue and
    hence κ were all constant in Ψ, identical across the batch, and
    ``relu(κ − κ.mean())`` was identically zero with no grad_fn at all.

    The Rayleigh quotient evaluated at Ψ is what the curvature of the energy
    *along the current state* actually means, it is what this docstring always
    claimed, and it is genuinely differentiable in Ψ. ``lanczos_iters`` is
    retained for signature compatibility and is unused.

    No epsilon: the denominator is asserted non-zero rather than clamped.
    """
    H = hessian(psi)  # (B, 16, 16) — constant in Ψ: every operator is linear.
    norm_sq = (psi * psi).sum(dim=-1)
    if (norm_sq == 0).any():
        raise ValueError("L_curv: zero-norm Ψ has no Rayleigh quotient.")
    HPsi = torch.einsum("bij,bj->bi", H, psi)
    kappa = (psi * HPsi).sum(dim=-1) / norm_sq
    # Penalise excess over the batch mean (centred, scale-free).
    excess = torch.relu(kappa - kappa.detach().mean())
    return excess.mean()


# ---------- L_dual : dual-basis coherence ----------------------------


def L_dual(psi: Tensor, wht_axis: Tensor, s5: S5, s11: S11) -> Tensor:
    """Mismatch between (S5∘S11) Ψ and its WHT-axis reconstruction.

    On the image of S5∘S11, Ψ should lie in span{1, h₀, h₁, h₂, h₃} where
    {h_k} are the four single-axis Walsh rows. We compute the rank-5
    reconstruction and penalise the L² residual.
    """
    psi_proj = s11(s5(psi))
    star = psi_proj.mean(dim=-1, keepdim=True)
    # lambdas[k] = ⟨psi_proj, h_k⟩ / 16
    coeffs = (psi_proj.unsqueeze(1) * wht_axis.unsqueeze(0)).sum(dim=-1) / float(NUM_VERTICES)
    reconstruction = star + (coeffs.unsqueeze(-1) * wht_axis.unsqueeze(0)).sum(dim=1)
    residual = psi_proj - reconstruction
    return (residual * residual).sum(dim=-1).mean()


# ---------- total loss --------------------------------------------------


def total_loss(
    ce_loss: Tensor,
    psi: Tensor,
    trace_sum: Tensor,
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
    """
    alpha, beta, gamma, delta = weights
    chi = L_chi(psi, s7)
    cons = L_cons(psi, trace_sum)
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
