"""All 29 sutras as ``torch.nn.Module`` with autograd.

Conventions:
- Input: ``psi`` of shape ``(B, 16)``, dtype ``torch.float32``.
- Output: ``(B, 16)`` for unary sutras, ``(B, 8)`` for S22, scalar ``(B,)``
  for S18 / S27, and a 2-tuple for S7.

Every operator that has a fixed permutation index pre-computes a ``LongTensor``
buffer at module init so the forward path is pure tensor advanced indexing
(no Python loops over the 16 vertices). Bit-exactness against ``sutras_exact``
is verified by ``tests/test_bit_exact.py``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .tesseract import (
    BIT_WIDTH,
    COMPLEMENT,
    NUM_VERTICES,
    POPCOUNT,
    SHELLS,
    pairs_v_lt_complement,
    rotate_left_1,
)
from .wht import hadamard_16_torch


# ----------------------------------------------------------------------
# Pre-computed integer index tensors (registered as buffers in modules).
# ----------------------------------------------------------------------


def _bit0_xor_idx() -> Tensor:
    return torch.tensor([v ^ 0b0001 for v in range(NUM_VERTICES)], dtype=torch.long)


def _complement_idx() -> Tensor:
    return torch.tensor(list(COMPLEMENT), dtype=torch.long)


def _popcount_tensor(dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor(list(POPCOUNT), dtype=dtype)


def _shell_membership_matrix(dtype: torch.dtype = torch.float32) -> Tensor:
    """16×16 matrix with M[v, u] = 1/|shell(v)| if popcount(u) = popcount(v) else 0.

    Multiplying by Ψ this gives the per-vertex shell mean.
    """
    M = torch.zeros((NUM_VERTICES, NUM_VERTICES), dtype=dtype)
    for shell in SHELLS:
        if not shell:
            continue
        scale = 1.0 / float(len(shell))
        for v in shell:
            for u in shell:
                M[v, u] = scale
    return M


def _xor_neighbor_idx() -> Tensor:
    """(16, 4) tensor: row v lists v ⊕ (1<<k) for k = 0..3."""
    out = torch.zeros((NUM_VERTICES, BIT_WIDTH), dtype=torch.long)
    for v in range(NUM_VERTICES):
        for k in range(BIT_WIDTH):
            out[v, k] = v ^ (1 << k)
    return out


def _high_half_mask(dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor([1.0 if (v & 0b1000) else 0.0 for v in range(NUM_VERTICES)], dtype=dtype)


def _top_quad_mask(dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor([1.0 if (v & 0b1100) == 0b1100 else 0.0
                         for v in range(NUM_VERTICES)], dtype=dtype)


def _ekanyunena_idx() -> Tensor:
    return torch.tensor([(v - 1) % NUM_VERTICES for v in range(NUM_VERTICES)], dtype=torch.long)


def _two_pow_popcount(dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor([float(1 << POPCOUNT[v]) for v in range(NUM_VERTICES)], dtype=dtype)


def _v_lt_complement_first() -> Tensor:
    return torch.tensor([v for v, _ in pairs_v_lt_complement()], dtype=torch.long)


def _v_lt_complement_second() -> Tensor:
    return torch.tensor([c for _, c in pairs_v_lt_complement()], dtype=torch.long)


def _multiples_of_7_mask(dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor([0.0 if (v % 7 == 0) else 1.0 for v in range(NUM_VERTICES)], dtype=dtype)


def _rotate_left_1_idx() -> Tensor:
    return torch.tensor([rotate_left_1(v) for v in range(NUM_VERTICES)], dtype=torch.long)


def _h1_pattern(dtype: torch.dtype = torch.float32) -> Tensor:
    """First non-constant Walsh row keyed by bit 0."""
    return torch.tensor([1.0 if not (v & 1) else -1.0 for v in range(NUM_VERTICES)], dtype=dtype)


def _popcount_even_mask(dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor([1.0 if POPCOUNT[v] % 2 == 0 else 0.0
                         for v in range(NUM_VERTICES)], dtype=dtype)


def _bit0_set_idx() -> Tensor:
    return torch.tensor([v | 0b0001 for v in range(NUM_VERTICES)], dtype=torch.long)


def _bit0_clear_idx() -> Tensor:
    return torch.tensor([v & 0b1110 for v in range(NUM_VERTICES)], dtype=torch.long)


# ----------------------------------------------------------------------
# Sutra modules
# ----------------------------------------------------------------------


def _check_psi(psi: Tensor) -> None:
    if psi.dim() != 2 or psi.size(-1) != NUM_VERTICES:
        raise ValueError(f"expected psi shape (B, 16); got {tuple(psi.shape)}")


class S1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _bit0_xor_idx())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.idx)


class S2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _complement_idx())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.idx)


class S3(nn.Module):
    """XOR convolution. Computed via Walsh-Hadamard transform: Ψ ⊛ Φ = H⁻¹(HΨ ⊙ HΦ)."""

    def __init__(self) -> None:
        super().__init__()
        H = hadamard_16_torch()
        self.register_buffer("H", H)
        # H is symmetric and HH = 16·I, so H⁻¹ = H/16.
        self.register_buffer("Hinv", H / float(NUM_VERTICES))

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:
        _check_psi(psi)
        _check_psi(phi)
        Hpsi = psi @ self.H
        Hphi = phi @ self.H
        return (Hpsi * Hphi) @ self.Hinv


class S4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _bit0_xor_idx())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi - psi.index_select(dim=-1, index=self.idx)


class S5(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi - psi.mean(dim=-1, keepdim=True)


class S6(nn.Module):
    """Subtract Ψ_0 from index 0 only."""

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        out = psi.clone()
        out[..., 0] = out[..., 0] - psi[..., 0]
        return out


class S7(nn.Module):
    """Returns (sym, anti) tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _complement_idx())

    def forward(self, psi: Tensor) -> Tuple[Tensor, Tensor]:
        _check_psi(psi)
        bar = psi.index_select(dim=-1, index=self.idx)
        sym = (psi + bar) / 2.0
        anti = (psi - bar) / 2.0
        return sym, anti


class S8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _complement_idx())
        mask = torch.tensor([1.0 if v < COMPLEMENT[v] else 0.0
                             for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("mask", mask)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        bar = psi.index_select(dim=-1, index=self.idx)
        return self.mask * (psi + bar)


class S9(nn.Module):
    """Discrete tesseract Laplacian."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("nbrs", _xor_neighbor_idx())  # (16, 4)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        # gather neighbors: (B, 16, 4) along last dim
        gathered = psi.unsqueeze(-1).expand(-1, NUM_VERTICES, BIT_WIDTH).gather(
            dim=1, index=self.nbrs.unsqueeze(0).expand(psi.size(0), -1, -1)
        )
        return gathered.sum(dim=-1) - float(BIT_WIDTH) * psi


class S10(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return (psi - 1.0) ** 2


class S11(nn.Module):
    """Subtract (1/4) shell-mean per vertex."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shell_M", _shell_membership_matrix())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        shell_means = psi @ self.shell_M.t()
        return psi - 0.25 * shell_means


class S12(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", _high_half_mask())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.mask


class S13(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", _top_quad_mask())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.mask


class S14(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _ekanyunena_idx())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.idx)


class S15(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", _two_pow_popcount())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.scale


class S16(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", _two_pow_popcount())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi / self.scale


class S17(nn.Module):
    """Ψ · Φ_0 / Ψ_0. Precondition: Ψ[:, 0] ≠ 0."""

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:
        _check_psi(psi)
        _check_psi(phi)
        psi0 = psi[..., 0:1]
        if torch.any(psi0 == 0):
            raise ValueError("S17 precondition: Ψ_0 must be non-zero everywhere")
        phi0 = phi[..., 0:1]
        return psi * (phi0 / psi0)


class S18(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi[..., 0] * psi[..., NUM_VERTICES - 1]


class S19(nn.Module):
    """(S19 Ψ)_v = Ψ_v − Ψ_{v & 0b1110} + Ψ_{v | 0b0001}."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("clear_idx", _bit0_clear_idx())
        self.register_buffer("set_idx", _bit0_set_idx())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return (
            psi
            - psi.index_select(dim=-1, index=self.clear_idx)
            + psi.index_select(dim=-1, index=self.set_idx)
        )


class S20(nn.Module):
    """Rank-1 projection onto h₁."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("h1", _h1_pattern())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        inner = (psi * self.h1).sum(dim=-1, keepdim=True)
        return (inner / float(NUM_VERTICES)) * self.h1


class S21(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.abs()


class S22(nn.Module):
    """Length-8 parity-complement vector."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("first", _v_lt_complement_first())
        self.register_buffer("second", _v_lt_complement_second())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.first) - psi.index_select(
            dim=-1, index=self.second
        )


class S23(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _complement_idx())

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:
        _check_psi(psi)
        _check_psi(phi)
        psi_bar = psi.index_select(dim=-1, index=self.idx)
        phi_bar = phi.index_select(dim=-1, index=self.idx)
        return psi * phi_bar + psi_bar * phi


class S24(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mask", _multiples_of_7_mask())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.mask


class S25(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idx", _rotate_left_1_idx())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.idx)


class S26(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * psi


class S27(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("even_mask", _popcount_even_mask())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        even_mask = self.even_mask
        odd_mask = 1.0 - even_mask
        # Use log-sum trick replaced with direct prod for bit-exact match.
        # Replace zeros with 1.0 in the OFF positions before product so the masked
        # vertices contribute 1 (multiplicative identity).
        psi_even = torch.where(even_mask.bool(), psi, torch.ones_like(psi))
        psi_odd = torch.where(odd_mask.bool(), psi, torch.ones_like(psi))
        prod_even = psi_even.prod(dim=-1)
        prod_odd = psi_odd.prod(dim=-1)
        return prod_even - prod_odd


class S28(nn.Module):
    """Inverse of S19 on im(S19): zero on bit-0-clear, divide by 2 on bit-0-set."""

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        bit0 = torch.tensor(
            [(v & 1) for v in range(NUM_VERTICES)],
            dtype=psi.dtype,
            device=psi.device,
        )
        return psi * 0.5 * bit0  # bit-0-clear → 0; bit-0-set → psi/2


class S29(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        m = psi.mean(dim=-1, keepdim=True)
        return (psi + m) * 0.5


# ----------------------------------------------------------------------
# Convenience composite + factory
# ----------------------------------------------------------------------


class S5ThenS11(nn.Module):
    """(S11 ∘ S5) used by L_dual."""

    def __init__(self) -> None:
        super().__init__()
        self.s5 = S5()
        self.s11 = S11()

    def forward(self, psi: Tensor) -> Tensor:
        return self.s11(self.s5(psi))


def all_torch_sutras() -> dict[str, nn.Module]:
    """Return a dict {name: module} for every torch sutra (registered modules)."""
    return {
        "S1": S1(), "S2": S2(), "S3": S3(), "S4": S4(), "S5": S5(),
        "S6": S6(), "S7": S7(), "S8": S8(), "S9": S9(), "S10": S10(),
        "S11": S11(), "S12": S12(), "S13": S13(), "S14": S14(), "S15": S15(),
        "S16": S16(), "S17": S17(), "S18": S18(), "S19": S19(), "S20": S20(),
        "S21": S21(), "S22": S22(), "S23": S23(), "S24": S24(), "S25": S25(),
        "S26": S26(), "S27": S27(), "S28": S28(), "S29": S29(),
    }
