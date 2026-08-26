"""Hypercube utilities with Vedic-inspired operators.

Vendored from ``codex/replace-blocks-with-fixed-implementations:src/quanqonscious/hypercube.py``
and re-pointed at the local ``vedic_engine`` adapter. Operates over
NumPy float64 — a classical-arithmetic sidecar; the bit-exact ℚ kernel
lives separately in ``vedic.kernel``.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from .vedic_engine import VedicSutraEngine

class Hypercube:
    """Hypercube utilities with Vedic-inspired operators."""

    def __init__(self, size: int, min_lambda: float = 1e-6, max_lambda: float = 1e6):
        """Validate the geometry once, here, rather than guarding every use.

        ``size`` was accepted unchecked and then defended piecemeal:
        ``lambda_operator`` divided by ``max(1, self.size)`` in two adapters.
        That guard does not rescue a zero size, it only hides it -- with
        size 0 the adjacency is a 0x0 matrix and every operator returns an
        empty array, so the caller gets shapes and no error. A hypercube of
        size 0 or a negative size is not a thing to compute with, so it is
        rejected at construction and the use sites can just say self.size.
        """
        size = int(size)
        if size < 1:
            raise ValueError(
                f"Hypercube size must be a positive integer; got {size}. "
                f"A size of 0 yields empty operators, not small ones.")
        min_lambda = float(min_lambda)
        max_lambda = float(max_lambda)
        if not np.isfinite(min_lambda) or min_lambda <= 0.0:
            raise ValueError(
                f"min_lambda must be finite and positive; got {min_lambda!r}")
        if not np.isfinite(max_lambda) or max_lambda < min_lambda:
            raise ValueError(
                f"max_lambda must be finite and >= min_lambda; got "
                f"max_lambda={max_lambda!r}, min_lambda={min_lambda!r}")
        self.size = size
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.adjacency = np.ones((self.size, self.size), dtype=np.float64)

    def compute_r4_tensor(self, r: np.ndarray, lambda_values: np.ndarray) -> np.ndarray:
        """Return ∏_{k<4} λ_k⁴ / (r⁴ + λ_k⁴), elementwise over ``r``.

        The output lies in [0, 1] by construction, for every real r and every
        non-zero λ: each factor is λ_k⁴/(r⁴ + λ_k⁴) with both r⁴ ≥ 0 and
        λ_k⁴ > 0, so each factor is in (0, 1], and a product of four such
        factors is in (0, 1]. That is a one-line algebraic fact about the
        formula, not a property of any particular input.

        A ``verify_r4_bounds`` method used to assert the same bound by drawing
        100,000 unseeded ``np.random.uniform`` samples, widening the interval
        by ``eps = 1e-15``, and reporting ``'verified': violations == 0``. It
        had no callers, and sampling cannot establish a bound that holds for
        all inputs — a Monte-Carlo miss is not a proof, and the argument above
        is the proof. It was deleted rather than re-seeded.
        """
        if lambda_values.shape[-1] != 4:
            raise ValueError("Lambda must have 4 components")
        r_expanded = np.expand_dims(r, axis=-1).astype(np.float64)
        r4 = np.power(r_expanded, 4)
        lambda4 = np.power(lambda_values.astype(np.float64), 4)
        result = np.ones_like(r, dtype=np.float64)
        for k in range(4):
            lamk4 = lambda4[..., k:k+1]
            factor = lamk4 / (r4 + lamk4)
            result *= np.squeeze(factor, axis=-1)
        return result

    def lambda_operator(self, alpha_vector: np.ndarray) -> np.ndarray:
        if len(alpha_vector) != 16:
            raise ValueError("Alpha vector must have 16 components")
        A = np.eye(self.size, dtype=np.float64)
        E = VedicSutraEngine()
        adapters: List[Callable[[np.ndarray], np.ndarray]] = [
            lambda M: E.ekadhikena_purvena(M),
            lambda M: E.paravartya_yojayet(M, divisor=1.0),
            lambda M: E.sopaantyadvayamantyam(M.copy()),
            lambda M: E.chalana_kalanabyham(M, steps=1, direction=1),
            lambda M: E.puranapuranabyham(M, complement_base=self.size),
            lambda M: E.ekanyunena_purvena(M),
            lambda M: np.real(E.vilokanam(M)),
            lambda M: E.sankalana_vyavakalanabhyam(M, M)[0],
            # Symmetrisation, unclamped. This was wrapped in
            # np.clip(..., -1e12, 1e12), which turned an overflowed or
            # infinite entry into the finite-looking 1e12 and let it sail
            # past the np.all(np.isfinite(T)) gate below. That gate is the
            # check; the clamp existed only to defeat it.
            lambda M: M * 0.5 + M.T * 0.5,
            lambda M: np.where(np.eye(self.size, dtype=bool), M, 0.0),
            lambda M: np.flip(M, axis=1),
            lambda M: np.roll(M, shift=1, axis=0),
            lambda M: np.roll(M, shift=1, axis=1),
            lambda M: (M + 0.1*np.ones_like(M)),
            lambda M: M * (1.0 + 0.05*np.sin(np.pi*np.arange(M.shape[0])[:,None]/self.size)),
            lambda M: M * (1.0 + 0.05*np.sin(np.pi*np.arange(M.shape[1])[None,:]/self.size)),
        ]
        alpha = np.array(alpha_vector, dtype=np.float64)
        if not np.all(np.isfinite(alpha)):
            raise ValueError("Alpha vector must be finite")
        denom = np.sum(alpha)
        # Dividing by 1.0 on a zero sum is not a normalisation. It returned
        # the raw alpha as though it had been normalised, so the operator was
        # built from weights summing to 0 instead of 1 and came back a
        # perfectly finite matrix of the wrong thing. There is no scale that
        # normalises a zero-sum vector.
        if denom == 0.0:
            raise ValueError(
                "Alpha vector sums to 0; there is no normalisation of a "
                "zero-sum vector. Pass weights with a nonzero sum.")
        alpha_norm = alpha / denom
        result = np.zeros_like(A)
        for k in range(16):
            T = adapters[k](A)
            if T.shape != result.shape or not np.all(np.isfinite(T)):
                raise ValueError(f"Adapter {k} produced invalid matrix")
            result += alpha_norm[k] * T
        return 0.5 * (result + result.T)

    def _compute_P32(self, chi: float) -> np.ndarray:
        base_size = 32
        i = np.arange(base_size)[:, None]
        j = np.arange(base_size)[None, :]
        enhanced_tile = (
            1.0
            + 0.25*np.sin(np.pi*(i + j)/base_size)
            + 0.15*np.cos(2*np.pi*(i - j)/base_size)
            + 0.10*np.sin(chi*(i + 1)*(j + 1)/base_size)
        )
        if enhanced_tile.shape[0] > self.size:
            reduced = enhanced_tile[:self.size, :self.size]
        else:
            reduced = np.zeros((self.size, self.size))
            reduced[:enhanced_tile.shape[0], :enhanced_tile.shape[1]] = enhanced_tile
        return self.adjacency * reduced

    def weighted_hypercube(self, chi: float) -> np.ndarray:
        return self._compute_P32(chi)

    def omega_operator(self, chi: float, alpha_vector: np.ndarray) -> np.ndarray:
        P = self.weighted_hypercube(chi)
        L = self.lambda_operator(alpha_vector)
        return P @ L

    def upsilon_operator(self, chi: float, alpha_vector: np.ndarray) -> np.ndarray:
        P = self.weighted_hypercube(chi)
        L = self.lambda_operator(alpha_vector)
        return L @ P
