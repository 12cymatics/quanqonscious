"""g_ab from S3 + S5 + S9 + S11 quadratic energies (exact + torch).

We define a quadratic-form energy

    E(Ψ) = ½ · ⟨Ψ, K Ψ⟩ + ½ · ‖S5 Ψ‖² + ½ · ‖S9 Ψ‖² + ½ · ‖S11 Ψ‖²

where K is the symmetric XOR-convolution kernel induced by the bilinear S3
acting against a fixed reference vector ω = e_0 + e_15 (the principal
diagonal of the tesseract). Each summand contributes a 16×16 symmetric
positive-semidefinite block, and the total Hessian is

    g_ab(Ψ) = ∂²E / ∂Ψ_a ∂Ψ_b
            = K_S3 + Mᵀ_S5 M_S5 + Mᵀ_S9 M_S9 + Mᵀ_S11 M_S11

This is the exact `g_ab` invoked by L_curv. Because every contributing
operator is linear, g_ab is independent of Ψ — but L_curv still uses Ψ to
form ⟨Ψ, g Ψ⟩ when computing the maximum eigenvalue / Rayleigh quotient.
We expose ``hessian_exact`` (Fraction tuple-of-tuples) and
``HessianModule`` (torch buffer).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

import torch
from torch import Tensor, nn

from .tesseract import BIT_WIDTH, COMPLEMENT, NUM_VERTICES, SHELLS

# ----------------------------------------------------------------------
# Linear-operator matrices (16x16) over ℚ
# ----------------------------------------------------------------------
QMatrix = Tuple[Tuple[Fraction, ...], ...]


def _zeros_q() -> list[list[Fraction]]:
    return [[Fraction(0) for _ in range(NUM_VERTICES)] for _ in range(NUM_VERTICES)]


def _identity_q() -> list[list[Fraction]]:
    M = _zeros_q()
    for i in range(NUM_VERTICES):
        M[i][i] = Fraction(1)
    return M


def _matmul_q(A: list[list[Fraction]], B: list[list[Fraction]]) -> list[list[Fraction]]:
    n = NUM_VERTICES
    C = _zeros_q()
    for i in range(n):
        for j in range(n):
            acc = Fraction(0)
            for k in range(n):
                acc += A[i][k] * B[k][j]
            C[i][j] = acc
    return C


def _transpose_q(A: list[list[Fraction]]) -> list[list[Fraction]]:
    n = NUM_VERTICES
    T = _zeros_q()
    for i in range(n):
        for j in range(n):
            T[j][i] = A[i][j]
    return T


def _add_q(A: list[list[Fraction]], B: list[list[Fraction]]) -> list[list[Fraction]]:
    n = NUM_VERTICES
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


# ----- Building blocks for each operator's matrix -----


def _matrix_S5() -> list[list[Fraction]]:
    """S5: Ψ_v − (1/16) Σ_u Ψ_u   →   M = I − (1/16) J."""
    M = _identity_q()
    sixteenth = Fraction(1, NUM_VERTICES)
    for i in range(NUM_VERTICES):
        for j in range(NUM_VERTICES):
            M[i][j] -= sixteenth
    return M


def _matrix_S9() -> list[list[Fraction]]:
    """S9: Σ_k (Ψ_{v⊕(1<<k)} − Ψ_v)   →  M[v, v]=−4, M[v, v⊕(1<<k)]=+1."""
    M = _zeros_q()
    for v in range(NUM_VERTICES):
        M[v][v] = Fraction(-BIT_WIDTH)
        for k in range(BIT_WIDTH):
            M[v][v ^ (1 << k)] += Fraction(1)
    return M


def _matrix_S11() -> list[list[Fraction]]:
    """S11: Ψ_v − (1/4) shell_mean(v) → M[v, v] = 1 − 1/(4|shell|),
    M[v, u in same shell, u≠v] = −1/(4|shell|)."""
    M = _identity_q()
    for shell in SHELLS:
        if not shell:
            continue
        s = Fraction(1, 4 * len(shell))
        for v in shell:
            for u in shell:
                M[v][u] -= s
    return M


def _matrix_S3_against_omega() -> list[list[Fraction]]:
    """S3(Ψ, ω) as a linear map in Ψ, with ω = e_0 + e_15.

    (S3(Ψ, ω))_v = Σ_{a⊕b=v} Ψ_a · ω_b. With ω_b = δ_{b,0} + δ_{b,15} we get

        (M Ψ)_v = Ψ_v + Ψ_{v ⊕ 15} = Ψ_v + Ψ_{v̄}

    so M = I + C where C is the complement permutation matrix. The
    resulting K_S3 = MᵀM = (I + C)ᵀ(I + C) = 2I + 2C is symmetric.
    """
    M = _identity_q()
    for v in range(NUM_VERTICES):
        M[v][COMPLEMENT[v]] += Fraction(1)
    return M


def _hessian_q() -> list[list[Fraction]]:
    """g_ab = MᵀM aggregated across S3, S5, S9, S11."""
    blocks: list[list[list[Fraction]]] = [
        _matmul_q(_transpose_q(_matrix_S3_against_omega()), _matrix_S3_against_omega()),
        _matmul_q(_transpose_q(_matrix_S5()), _matrix_S5()),
        _matmul_q(_transpose_q(_matrix_S9()), _matrix_S9()),
        _matmul_q(_transpose_q(_matrix_S11()), _matrix_S11()),
    ]
    H = _zeros_q()
    for B in blocks:
        H = _add_q(H, B)
    return H


HESSIAN_Q: QMatrix = tuple(tuple(row) for row in _hessian_q())
"""Pre-computed exact ℚ Hessian g_ab (16×16, symmetric, PSD)."""


def hessian_exact() -> QMatrix:
    """Return the exact ℚ Hessian g_ab (independent of Ψ)."""
    return HESSIAN_Q


def hessian_dense_torch(dtype: torch.dtype = torch.float32,
                        device: str | torch.device = "cpu") -> Tensor:
    """Return g_ab as a (16, 16) torch tensor with the requested dtype/device.

    Built AT the requested dtype. It used to construct a hardcoded
    ``np.float32`` intermediate and cast afterwards, so asking for float64
    returned float32 precision wearing a float64 dtype -- measured max error
    6.358e-07 against exact ℚ, where a correct conversion gives 0.0. That
    exceeds the 1e-7 tolerance ``test_conservation_torch.py`` compares
    against. ``HESSIAN_Q`` contains denominator 96, which is not a power of
    two, so the loss is real rather than notional.

    Production passes float32 and was unaffected; the parameter was a trap
    for the first caller who trusted it.
    """
    return torch.tensor([[float(x) for x in row] for row in HESSIAN_Q],
                        dtype=dtype, device=device)


class HessianModule(nn.Module):
    """g_ab as a pre-computed buffer; ``forward(psi)`` returns (B, 16, 16)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("g", hessian_dense_torch(dtype=torch.float32))

    def forward(self, psi: Tensor) -> Tensor:
        if psi.dim() != 2 or psi.size(-1) != NUM_VERTICES:
            raise ValueError(f"psi must be (B, 16); got {tuple(psi.shape)}")
        # Hessian is independent of Ψ; broadcast the (16, 16) buffer to (B, 16, 16).
        return self.g.unsqueeze(0).expand(psi.size(0), NUM_VERTICES, NUM_VERTICES)
