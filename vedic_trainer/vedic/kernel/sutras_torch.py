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
    rotate_left_k,
)
from .wht import hadamard_16_torch
from .sutras_exact import (
    FULL_MASK,
    S1_MASK,
    S4_MASK,
    S6_REF,
    S9_AXES,
    S10_BASE,
    S11_WEIGHT,
    S12_MASK,
    S13_MASK,
    S14_SHIFT,
    S15_BASE,
    S16_BASE,
    S17_REF,
    S18_I,
    S18_J,
    S19_MASK,
    S20_AXIS,
    S24_MODULUS,
    S25_ROT,
    S29_WEIGHT,
)


# ----------------------------------------------------------------------
# Pre-computed integer index tensors (registered as buffers in modules).
# ----------------------------------------------------------------------


def _xor_idx(mask: int) -> Tensor:
    """Index map v ↦ v ⊕ mask. The pairing operand as a gather index."""
    if not 0 <= mask < NUM_VERTICES:
        raise ValueError(f"mask out of range for Z₂⁴: {mask}")
    return torch.tensor([v ^ mask for v in range(NUM_VERTICES)], dtype=torch.long)


def _bit0_xor_idx() -> Tensor:
    return _xor_idx(S1_MASK)


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


def _xor_neighbor_idx(axes: Tuple[int, ...] = S9_AXES) -> Tensor:
    """(16, |axes|) tensor: row v lists v ⊕ (1<<k) for each k in ``axes``."""
    for k in axes:
        if not 0 <= k < BIT_WIDTH:
            raise ValueError(f"axis out of range: {k}")
    out = torch.zeros((NUM_VERTICES, len(axes)), dtype=torch.long)
    for v in range(NUM_VERTICES):
        for j, k in enumerate(axes):
            out[v, j] = v ^ (1 << k)
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
    """(S1 Ψ)_v = Ψ_{v ⊕ mask}. Canonical mask = 0b0001."""

    def __init__(self, mask: int = S1_MASK) -> None:
        super().__init__()
        self.mask = mask
        self.register_buffer("idx", _xor_idx(mask))

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.idx)


class S2(nn.Module):
    """(S2 Ψ)_v = Ψ_{v ⊕ mask}. Canonical mask = 0b1111 (complement)."""

    def __init__(self, mask: int = FULL_MASK) -> None:
        super().__init__()
        self.mask = mask
        self.register_buffer("idx", _xor_idx(mask))

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
    """(S4 Ψ)_v = Ψ_v − Ψ_{v ⊕ mask}. Canonical mask = 0b0001."""

    def __init__(self, mask: int = S4_MASK) -> None:
        super().__init__()
        self.mask = mask
        self.register_buffer("idx", _xor_idx(mask))

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi - psi.index_select(dim=-1, index=self.idx)


class S5(nn.Module):
    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi - psi.mean(dim=-1, keepdim=True)


class S6(nn.Module):
    """Subtract Ψ_ref at index ``ref`` only. Canonical ref = 0."""

    def __init__(self, ref: int = S6_REF) -> None:
        super().__init__()
        if not 0 <= ref < NUM_VERTICES:
            raise ValueError(f"S6 ref out of range: {ref}")
        self.ref = ref

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        out = psi.clone()
        out[..., self.ref] = out[..., self.ref] - psi[..., self.ref]
        return out


class S7(nn.Module):
    """Returns (sym, anti) tuple."""

    def __init__(self, mask: int = FULL_MASK) -> None:
        super().__init__()
        self.mask = mask
        self.register_buffer("idx", _xor_idx(mask))

    def forward(self, psi: Tensor) -> Tuple[Tensor, Tensor]:
        _check_psi(psi)
        bar = psi.index_select(dim=-1, index=self.idx)
        sym = (psi + bar) / 2.0
        anti = (psi - bar) / 2.0
        return sym, anti


class S8(nn.Module):
    """(S8 Ψ)_v = Ψ_v + Ψ_{v⊕mask} on pair leads. Canonical mask = 0b1111."""

    def __init__(self, mask: int = FULL_MASK) -> None:
        super().__init__()
        self.pair_mask = mask
        self.register_buffer("idx", _xor_idx(mask))
        lead = torch.tensor([1.0 if v < (v ^ mask) else 0.0
                             for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("mask", lead)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        bar = psi.index_select(dim=-1, index=self.idx)
        return self.mask * (psi + bar)


class S9(nn.Module):
    """Discrete tesseract Laplacian."""

    def __init__(self, axes: Tuple[int, ...] = S9_AXES) -> None:
        super().__init__()
        self.axes = tuple(axes)
        self.register_buffer("nbrs", _xor_neighbor_idx(self.axes))  # (16, |axes|)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        # gather neighbors: (B, 16, 4) along last dim
        deg = len(self.axes)
        gathered = psi.unsqueeze(-1).expand(-1, NUM_VERTICES, deg).gather(
            dim=1, index=self.nbrs.unsqueeze(0).expand(psi.size(0), -1, -1)
        )
        return gathered.sum(dim=-1) - float(deg) * psi


class S10(nn.Module):
    """(S10 Ψ)_v = (Ψ_v − base)². Canonical base = 1."""

    def __init__(self, base: float = float(S10_BASE)) -> None:
        super().__init__()
        self.base = float(base)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return (psi - self.base) ** 2


class S11(nn.Module):
    """Subtract ``weight`` × shell-mean per vertex. Canonical weight = 1/4."""

    def __init__(self, weight: float = float(S11_WEIGHT)) -> None:
        super().__init__()
        self.weight = float(weight)
        self.register_buffer("shell_M", _shell_membership_matrix())

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        shell_means = psi @ self.shell_M.t()
        return psi - self.weight * shell_means


class S12(nn.Module):
    """Keep Ψ_v where (v & mask). Canonical mask = 0b1000."""

    def __init__(self, mask: int = S12_MASK) -> None:
        super().__init__()
        self.bit_mask = mask
        keep = torch.tensor([1.0 if (v & mask) else 0.0
                             for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("mask", keep)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.mask


class S13(nn.Module):
    """Keep Ψ_v where (v & mask) == mask. Canonical mask = 0b1100."""

    def __init__(self, mask: int = S13_MASK) -> None:
        super().__init__()
        self.bit_mask = mask
        keep = torch.tensor([1.0 if (v & mask) == mask else 0.0
                             for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("mask", keep)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.mask


class S14(nn.Module):
    """(S14 Ψ)_v = Ψ_{(v + shift) mod 16}. Canonical shift = −1."""

    def __init__(self, shift: int = S14_SHIFT) -> None:
        super().__init__()
        self.shift = shift
        idx = torch.tensor([(v + shift) % NUM_VERTICES for v in range(NUM_VERTICES)],
                           dtype=torch.long)
        self.register_buffer("idx", idx)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.idx)


class S15(nn.Module):
    """(S15 Ψ)_v = base^popcount(v) · Ψ_v. Canonical base = 2."""

    def __init__(self, base: float = float(S15_BASE)) -> None:
        super().__init__()
        self.base = float(base)
        sc = torch.tensor([float(base) ** POPCOUNT[v] for v in range(NUM_VERTICES)],
                          dtype=torch.float32)
        self.register_buffer("scale", sc)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.scale


class S16(nn.Module):
    """(S16 Ψ)_v = Ψ_v / base^popcount(v). Canonical base = 2."""

    def __init__(self, base: float = float(S16_BASE)) -> None:
        super().__init__()
        if base == 0:
            raise ValueError("S16 base must be non-zero")
        self.base = float(base)
        sc = torch.tensor([float(base) ** POPCOUNT[v] for v in range(NUM_VERTICES)],
                          dtype=torch.float32)
        self.register_buffer("scale", sc)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi / self.scale


class S17(nn.Module):
    """Ψ · Φ_ref / Ψ_ref. Precondition: Ψ[:, ref] ≠ 0. Canonical ref = 0."""

    def __init__(self, ref: int = S17_REF) -> None:
        super().__init__()
        if not 0 <= ref < NUM_VERTICES:
            raise ValueError(f"S17 ref out of range: {ref}")
        self.ref = ref

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:
        _check_psi(psi)
        _check_psi(phi)
        r = self.ref
        psi_r = psi[..., r:r + 1]
        if torch.any(psi_r == 0):
            raise ValueError(f"S17 precondition: Ψ_{r} must be non-zero everywhere")
        phi_r = phi[..., r:r + 1]
        return psi * (phi_r / psi_r)


class S18(nn.Module):
    """Ψ_i · Ψ_j (scalar). Canonical (i, j) = (0, 15)."""

    def __init__(self, i: int = S18_I, j: int = S18_J) -> None:
        super().__init__()
        for name, idx in (("i", i), ("j", j)):
            if not 0 <= idx < NUM_VERTICES:
                raise ValueError(f"S18 {name} out of range: {idx}")
        self.i, self.j = i, j

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi[..., self.i] * psi[..., self.j]


class S19(nn.Module):
    """(S19 Ψ)_v = Ψ_v − Ψ_{v & 0b1110} + Ψ_{v | 0b0001}."""

    def __init__(self, mask: int = S19_MASK) -> None:
        super().__init__()
        if not 0 <= mask < NUM_VERTICES:
            raise ValueError(f"S19 mask out of range: {mask}")
        self.bit_mask = mask
        inv = (~mask) & FULL_MASK
        self.register_buffer("clear_idx", torch.tensor(
            [v & inv for v in range(NUM_VERTICES)], dtype=torch.long))
        self.register_buffer("set_idx", torch.tensor(
            [v | mask for v in range(NUM_VERTICES)], dtype=torch.long))

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return (
            psi
            - psi.index_select(dim=-1, index=self.clear_idx)
            + psi.index_select(dim=-1, index=self.set_idx)
        )


class S20(nn.Module):
    """Rank-1 projection onto the Walsh row h_axis. Canonical axis = 0."""

    def __init__(self, axis: int = S20_AXIS) -> None:
        super().__init__()
        if not 0 <= axis < BIT_WIDTH:
            raise ValueError(f"S20 axis out of range: {axis}")
        self.axis = axis
        h = torch.tensor([1.0 if not ((v >> axis) & 1) else -1.0
                          for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("h1", h)

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

    def __init__(self, mask: int = FULL_MASK) -> None:
        super().__init__()
        if not 0 <= mask < NUM_VERTICES:
            raise ValueError(f"S22 mask out of range: {mask}")
        self.pair_mask = mask
        pairs = [(v, v ^ mask) for v in range(NUM_VERTICES) if v < (v ^ mask)]
        self.register_buffer("first", torch.tensor([a for a, _ in pairs], dtype=torch.long))
        self.register_buffer("second", torch.tensor([b for _, b in pairs], dtype=torch.long))

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi.index_select(dim=-1, index=self.first) - psi.index_select(
            dim=-1, index=self.second
        )


class S23(nn.Module):
    """Duplex: Ψ_v·Φ_{v⊕mask} + Ψ_{v⊕mask}·Φ_v. Canonical mask = 0b1111."""

    def __init__(self, mask: int = FULL_MASK) -> None:
        super().__init__()
        self.pair_mask = mask
        self.register_buffer("idx", _xor_idx(mask))

    def forward(self, psi: Tensor, phi: Tensor) -> Tensor:
        _check_psi(psi)
        _check_psi(phi)
        psi_bar = psi.index_select(dim=-1, index=self.idx)
        phi_bar = phi.index_select(dim=-1, index=self.idx)
        return psi * phi_bar + psi_bar * phi


class S24(nn.Module):
    """Zero the v ≡ 0 (mod modulus) class. Canonical modulus = 7."""

    def __init__(self, modulus: int = S24_MODULUS) -> None:
        super().__init__()
        if modulus <= 0:
            raise ValueError(f"S24 modulus must be positive: {modulus}")
        self.modulus = modulus
        keep = torch.tensor([0.0 if (v % modulus == 0) else 1.0
                             for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("mask", keep)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * self.mask


class S25(nn.Module):
    """(S25 Ψ)_v = Ψ_{σ_k(v)}, σ_k = bit-rotate-left-k. Canonical k = 1."""

    def __init__(self, k: int = S25_ROT) -> None:
        super().__init__()
        self.k = k
        self.register_buffer("idx", torch.tensor(
            [rotate_left_k(v, k) for v in range(NUM_VERTICES)], dtype=torch.long))

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
    """Inverse of S19 on im(S19) for the same ``mask``: zero on mask-clear,
    halve on mask-set. Canonical mask = 0b0001."""

    def __init__(self, mask: int = S19_MASK) -> None:
        super().__init__()
        if not 0 <= mask < NUM_VERTICES:
            raise ValueError(f"S28 mask out of range: {mask}")
        self.bit_mask = mask
        sel = torch.tensor([1.0 if (v & mask) else 0.0
                            for v in range(NUM_VERTICES)], dtype=torch.float32)
        self.register_buffer("sel", sel)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        return psi * 0.5 * self.sel.to(psi.dtype)


class S29(nn.Module):
    """(1 − weight)·Ψ + weight·mean(Ψ). Canonical weight = 1/2."""

    def __init__(self, weight: float = float(S29_WEIGHT)) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, psi: Tensor) -> Tensor:
        _check_psi(psi)
        m = psi.mean(dim=-1, keepdim=True)
        return (1.0 - self.weight) * psi + self.weight * m


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
