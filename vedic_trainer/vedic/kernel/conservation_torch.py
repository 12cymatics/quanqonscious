"""Four conservation residuals as torch tensors (autograd-compatible).

Bit-exact mirror of ``conservation_exact.py``. The four residuals are:

    R1 = trace_sum (mod 435)        T(29) closure
    R2 = Σ_{v<v̄}(Ψ_v + Ψ_{v̄}) − Σ_v Ψ_v   complement-pair sum
    R3 = mean(S29 Ψ) − mean(Ψ)      S29 mean preservation
    R4 = ⟨S(Ψ), A(Ψ)⟩               Beltrami orthogonality

R2, R3, R4 are algebraic identities over ℚ — they evaluate to exact zero on
the Fraction path and to a tiny float-rounding residual on the torch path.

**Not on the training path.** ``L_cons`` does not use these; summing R1..R4
produced a constant with zero gradient (see ``vedic/training/losses.py``).
This module is the torch mirror of ``conservation_exact.py``, kept as a
reference implementation and verified against it.

This docstring previously asserted "the bit-exact test verifies that the
torch residual stays below 1e-7 … on 100 randomized Q16 inputs". No such test
existed. ``vedic/training/losses.py`` did *import* five names from here, but
never used them -- their only occurrence was the import block itself -- so
the claim was unbacked for its whole life.
``vedic/kernel/tests/test_conservation_torch.py`` now makes it true.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .sutras_torch import S7, S29
from .tesseract import NUM_VERTICES, pairs_v_lt_complement

T29: int = 29 * 30 // 2


def r1_torch(trace_sum: Tensor) -> Tensor:
    """R1 = trace_sum mod 435.

    ``trace_sum`` is expected to be an integer-valued tensor (typically
    accumulated by the trainer). The residual is computed in float and
    is exactly zero when trace_sum is a multiple of 435.
    """
    if trace_sum.dim() == 0:
        trace_sum = trace_sum.unsqueeze(0)
    return torch.remainder(trace_sum.float(), float(T29))


def r2_torch(psi: Tensor) -> Tensor:
    """R2 = Σ_{v<v̄}(Ψ_v + Ψ_{v̄}) − Σ_v Ψ_v.   Shape: (B,)."""
    if psi.dim() != 2 or psi.size(-1) != NUM_VERTICES:
        raise ValueError(f"psi must be (B, 16); got {tuple(psi.shape)}")
    pairs = pairs_v_lt_complement()
    first = torch.tensor([v for v, _ in pairs], dtype=torch.long, device=psi.device)
    second = torch.tensor([c for _, c in pairs], dtype=torch.long, device=psi.device)
    pair_sum = (psi.index_select(-1, first) + psi.index_select(-1, second)).sum(dim=-1)
    grand = psi.sum(dim=-1)
    return pair_sum - grand


def r3_torch(psi: Tensor) -> Tensor:
    """R3 = mean(S29 Ψ) − mean(Ψ).   Shape: (B,)."""
    if psi.dim() != 2 or psi.size(-1) != NUM_VERTICES:
        raise ValueError(f"psi must be (B, 16); got {tuple(psi.shape)}")
    s29 = S29()
    s29.to(psi.device)
    return s29(psi).mean(dim=-1) - psi.mean(dim=-1)


def r4_torch(psi: Tensor) -> Tensor:
    """R4 = ⟨S(Ψ), A(Ψ)⟩.   Shape: (B,)."""
    if psi.dim() != 2 or psi.size(-1) != NUM_VERTICES:
        raise ValueError(f"psi must be (B, 16); got {tuple(psi.shape)}")
    s7 = S7()
    s7.to(psi.device)
    sym, anti = s7(psi)
    return (sym * anti).sum(dim=-1)


def cons_l2_torch(psi: Tensor, trace_sum: Tensor) -> Tensor:
    """Σᵢ Rᵢ²  averaged over the batch.   Shape: scalar (0-dim)."""
    r1 = r1_torch(trace_sum)
    r2 = r2_torch(psi)
    r3 = r3_torch(psi)
    r4 = r4_torch(psi)
    # Broadcast r1 (which is per-batch trace) against the per-example residuals.
    if r1.shape != r2.shape:
        r1 = r1.expand_as(r2)
    return (r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4).mean()
