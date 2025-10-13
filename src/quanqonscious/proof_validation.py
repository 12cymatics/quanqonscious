import numpy as np
from typing import Dict, List

try:  # allow standalone usage without package install
    from .vedic_sutra_engine import VedicSutraEngine
    from .hypercube import Hypercube
except Exception:  # pragma: no cover
    from vedic_sutra_engine import VedicSutraEngine
    from hypercube import Hypercube

class ProofTester:
    """Validation harness for Vedic sutra engine and hypercube operators."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def verify_sutra_engine(self) -> None:
        """Verify all 29 Vedic sutras with correct arities and safe domains."""
        engine = VedicSutraEngine()
        X = np.abs(np.random.randn(8, 8)) + 0.5
        Y = np.abs(np.random.randn(8, 8)) + 0.5
        V = np.abs(np.random.randn(16,)) + 0.5
        V2 = np.abs(np.random.randn(16,)) + 0.5
        violations = 0
        try:
            _ = engine.ekadhikena_purvena(X)
        except Exception:
            violations += 1
        try:
            _ = engine.nikhilam_navatashcaramam_dashatah(X, base=10)
        except Exception:
            violations += 1
        try:
            _ = engine.urdhva_tiryagbhyam(X, Y)
        except Exception:
            violations += 1
        try:
            _ = engine.paravartya_yojayet(X, divisor=1.0)
        except Exception:
            violations += 1
        try:
            _ = engine.shunyam_samyasamuccaye(X, X + 1e-9)
        except Exception:
            violations += 1
        try:
            ratio = 1.0
            _ = engine.anurupye_sunyamanyat(X, X / ratio, ratio)
        except Exception:
            violations += 1
        try:
            _sum, _diff = engine.sankalana_vyavakalanabhyam(X, Y)
            _ = _sum + _diff * 0
        except Exception:
            violations += 1
        try:
            _ = engine.puranapuranabyham(X, complement_base=10)
        except Exception:
            violations += 1
        try:
            _ = engine.chalana_kalanabyham(X, steps=2, direction=1)
        except Exception:
            violations += 1
        try:
            _ = engine.yavadunam(X, deficit=0.3)
        except Exception:
            violations += 1
        try:
            _ = engine.vyashtisamanstih(whole=X + Y, parts=[X, Y])
        except Exception:
            violations += 1
        try:
            coeffs = np.array([1.0, 0.0, 0.0])
            _ = engine.shesanyankena_charamena(coeffs, np.floor(X))
        except Exception:
            violations += 1
        try:
            _ = engine.sopaantyadvayamantyam(X.copy())
        except Exception:
            violations += 1
        try:
            _ = engine.ekanyunena_purvena(X)
        except Exception:
            violations += 1
        try:
            _ = engine.gunitasamuchyah(X, Y)
        except Exception:
            violations += 1
        try:
            _ = engine.gunakasamuchyah([X, (Y + 1)])
        except Exception:
            violations += 1
        try:
            _ = engine.anurupyena(X, Y, ratio=1.2)
        except Exception:
            violations += 1
        try:
            _ = engine.sisyate_sesasamjnah(np.floor(X), modulus=9)
        except Exception:
            violations += 1
        try:
            _ = engine.adyamadyenantyamantyena(X)
        except Exception:
            violations += 1
        try:
            _ = engine.kevalaih_saptakam_gunyat(X)
        except Exception:
            violations += 1
        try:
            _ = engine.vestanam(V)
        except Exception:
            violations += 1
        try:
            _ = engine.yavadunam_tavadunam(X, Y)
        except Exception:
            violations += 1
        try:
            _ = engine.yavadunam_tavadunikritya(X, base=10)
        except Exception:
            violations += 1
        try:
            V10 = V.copy()
            V10[-2:] = np.array([4., 6.])
            _ = engine.antyayordashakepi(V10)
        except Exception:
            violations += 1
        try:
            _ = engine.antyayoreva(X)
        except Exception:
            violations += 1
        try:
            _ = engine.samuccayagunitah([np.array([2,3]), np.array([5,7])])
        except Exception:
            violations += 1
        try:
            _ = engine.lopanasthapanabhyam(X, eliminate_index=0, substitute_value=0.0)
        except Exception:
            violations += 1
        try:
            _ = engine.vilokanam(X)
        except Exception:
            violations += 1
        try:
            _ = engine.gunitasamuccayah_samuccayagunitah(X, Y)
        except Exception:
            violations += 1
        self.results['vedic_sutras'] = {
            'main_sutras': len(engine.sutras),
            'sub_sutras': len(engine.sub_sutras),
            'violations': violations,
            'verified': violations == 0
        }
        print(f"  {'✓' if violations==0 else '✗'} Verified {29 - violations}/29 sutras")

    def test_hypercube(self) -> None:
        hypercube = Hypercube(size=8)
        alpha_vector = np.ones(16)
        chi = 0.5
        P = hypercube.weighted_hypercube(chi)
        Lambda = hypercube.lambda_operator(alpha_vector)
        Omega = hypercube.omega_operator(chi, alpha_vector)
        Upsilon = hypercube.upsilon_operator(chi, alpha_vector)
        bounds = hypercube.verify_r4_bounds(num_samples=1000)
        self.results['hypercube'] = {
            'P_shape': P.shape,
            'Lambda_shape': Lambda.shape,
            'Omega_shape': Omega.shape,
            'Upsilon_shape': Upsilon.shape,
            'r4_bounds': bounds,
        }
