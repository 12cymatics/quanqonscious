import numpy as np
from typing import List, Callable, Dict

try:  # support both package and standalone execution
    from .vedic_sutra_engine import VedicSutraEngine
except Exception:  # pragma: no cover - fallback for script usage
    from vedic_sutra_engine import VedicSutraEngine

class Hypercube:
    """Hypercube utilities with Vedic-inspired operators."""

    def __init__(self, size: int, min_lambda: float = 1e-6, max_lambda: float = 1e6):
        self.size = int(size)
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)
        self.adjacency = np.ones((self.size, self.size), dtype=np.float64)

    def compute_r4_tensor(self, r: np.ndarray, lambda_values: np.ndarray) -> np.ndarray:
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

    def verify_r4_bounds(self, num_samples: int = 100000) -> Dict:
        violations = 0
        min_val = 1.0
        max_val = 0.0
        eps = 1e-15
        for _ in range(num_samples):
            r = np.random.uniform(0.0, 1000, size=(10, 10))
            lambda_vals = np.random.uniform(self.min_lambda, self.max_lambda, size=(10, 10, 4))
            r4 = self.compute_r4_tensor(r, lambda_vals)
            if np.any(r4 < -eps) or np.any(r4 > 1 + eps):
                violations += 1
            min_val = min(min_val, float(np.min(r4)))
            max_val = max(max_val, float(np.max(r4)))
        return {
            'violations': violations,
            'samples': num_samples,
            'min': min_val,
            'max': max_val,
            'verified': violations == 0
        }

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
            lambda M: np.clip(M * 0.5 + M.T * 0.5, -1e12, 1e12),
            lambda M: np.where(np.eye(self.size, dtype=bool), M, 0.0),
            lambda M: np.flip(M, axis=1),
            lambda M: np.roll(M, shift=1, axis=0),
            lambda M: np.roll(M, shift=1, axis=1),
            lambda M: (M + 0.1*np.ones_like(M)),
            lambda M: M * (1.0 + 0.05*np.sin(np.pi*np.arange(M.shape[0])[:,None]/max(1,self.size))),
            lambda M: M * (1.0 + 0.05*np.sin(np.pi*np.arange(M.shape[1])[None,:]/max(1,self.size))),
        ]
        alpha = np.array(alpha_vector, dtype=np.float64)
        denom = np.sum(alpha)
        alpha_norm = alpha / (denom if denom != 0 else 1.0)
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
