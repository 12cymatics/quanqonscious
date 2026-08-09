"""Smoke test harness for the classical Vedic sutra engine and hypercube.

Adapted from
``codex/replace-blocks-with-fixed-implementations:src/quanqonscious/proof_validation.py``.

The original version wrapped every sutra invocation in a bare ``try/except``
and counted failures; the kernel forbids that pattern. We rewrite the
harness so each sutra invocation is direct — any failure raises and the
test framework surfaces the real traceback rather than a silent counter.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .hypercube import Hypercube
from .vedic_engine import VedicSutraEngine


def _make_inputs(seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "X": np.abs(rng.standard_normal((8, 8))) + 0.5,
        "Y": np.abs(rng.standard_normal((8, 8))) + 0.5,
        "V": np.abs(rng.standard_normal(16)) + 0.5,
    }


def _sutra_invocations(engine: VedicSutraEngine,
                       data: Dict[str, np.ndarray]) -> List[Callable[[], object]]:
    X, Y, V = data["X"], data["Y"], data["V"]
    coeffs = np.array([1.0, 0.0, 0.0])
    V10 = V.copy()
    V10[-2:] = np.array([4.0, 6.0])
    return [
        lambda: engine.ekadhikena_purvena(X),
        lambda: engine.nikhilam_navatashcaramam_dashatah(X, base=10),
        lambda: engine.urdhva_tiryagbhyam(X, Y),
        lambda: engine.paravartya_yojayet(X, divisor=1.0),
        lambda: engine.shunyam_samyasamuccaye(X, X + 1e-9),
        lambda: engine.anurupye_sunyamanyat(X, X, 1.0),
        lambda: engine.sankalana_vyavakalanabhyam(X, Y),
        lambda: engine.puranapuranabyham(X, complement_base=10),
        lambda: engine.chalana_kalanabyham(X, steps=2, direction=1),
        lambda: engine.yavadunam(X, deficit=0.3),
        lambda: engine.vyashtisamanstih(X + Y, [X, Y]),
        lambda: engine.shesanyankena_charamena(coeffs, int(np.floor(X.mean()))),
        lambda: engine.sopaantyadvayamantyam(X.copy()),
        lambda: engine.ekanyunena_purvena(X),
        lambda: engine.gunitasamuchyah(X, Y),
        lambda: engine.gunakasamuchyah([X, Y + 1]),
        lambda: engine.anurupyena(X, Y, ratio=1.2),
        lambda: engine.sisyate_sesasamjnah(np.floor(X), modulus=9),
        lambda: engine.adyamadyenantyamantyena(X),
        lambda: engine.kevalaih_saptakam_gunyat(X),
        lambda: engine.vestanam(V),
        lambda: engine.yavadunam_tavadunam(X, Y),
        lambda: engine.yavadunam_tavadunikritya(X, base=10),
        lambda: engine.antyayordashakepi(V10),
        lambda: engine.antyayoreva(X),
        lambda: engine.samuccayagunitah([np.array([2, 3]), np.array([5, 7])]),
        lambda: engine.lopanasthapanabhyam(X, eliminate_index=0, substitute_value=0.0),
        lambda: engine.vilokanam(X),
        lambda: engine.gunitasamuccayah_samuccayagunitah(X, Y),
    ]


class ProofTester:
    """Validation harness for the classical Vedic sutra engine and hypercube.

    Each ``verify_*`` method invokes every operator directly. Any failure
    is allowed to propagate — there is no swallowing of exceptions.
    """

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, object]] = {}

    def verify_sutra_engine(self) -> None:
        engine = VedicSutraEngine()
        data = _make_inputs()
        invocations = _sutra_invocations(engine, data)
        outputs = [call() for call in invocations]
        self.results["vedic_sutras"] = {
            "main_sutras": len(engine.sutras),
            "sub_sutras": len(engine.sub_sutras),
            "executed": len(outputs),
            "verified": True,
        }

    def verify_hypercube(self) -> None:
        hypercube = Hypercube(size=8)
        alpha_vector = np.ones(16)
        chi = 0.5
        P = hypercube.weighted_hypercube(chi)
        Lambda = hypercube.lambda_operator(alpha_vector)
        Omega = hypercube.omega_operator(chi, alpha_vector)
        Upsilon = hypercube.upsilon_operator(chi, alpha_vector)
        self.results["hypercube"] = {
            "P_shape": P.shape,
            "Lambda_shape": Lambda.shape,
            "Omega_shape": Omega.shape,
            "Upsilon_shape": Upsilon.shape,
        }
