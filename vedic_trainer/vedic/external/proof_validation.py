"""Validation harness for the classical Vedic sutra engine and hypercube.

Not a smoke test. It was described as one and behaved like less than one:
``verify_sutra_engine`` collected all 29 outputs, inspected none of them, and
recorded ``"verified": True`` as a literal — which the test then asserted, so
neither could fail. The verdict is now derived from the outputs.

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


def _make_inputs() -> Dict[str, np.ndarray]:
    """Deterministic integer-valued inputs, exact in float64.

    These were ``np.abs(rng.standard_normal(...)) + 0.5`` — synthetic random
    floats with full mantissas. Nothing this harness checks (emptiness,
    finiteness, shape) needs randomness, and full-mantissa inputs force any
    downstream comparison into a tolerance. Integers are exact, reproducible
    without a generator, and distinct and nonzero so an operator returning a
    constant or its input unchanged is still distinguishable.

    The ``seed`` parameter is gone rather than kept and ignored: a seed that
    selects nothing is a knob a caller can reasonably expect to matter.
    """
    def matrix(n: int, offset: int) -> np.ndarray:
        return np.array([[float(offset + i * n + j + 1) for j in range(n)]
                         for i in range(n)], dtype=np.float64)

    return {
        "X": matrix(8, 0),
        "Y": matrix(8, 100),
        "V": np.array([float(i + 1) for i in range(16)], dtype=np.float64),
    }


def _sutra_names(engine: VedicSutraEngine) -> List[str]:
    """Names in the same order as `_sutra_invocations`, for error messages."""
    return [getattr(m, "__name__", repr(m))
            for m in list(engine.sutras) + list(engine.sub_sutras)]


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
        # Scalar identity: np.sum(parts) totals every element of every part,
        # so the "whole" is the scalar total, not the elementwise sum. This
        # passed ``X + Y`` — a matrix — against that scalar, which under the
        # old isclose-based implementation compared each element to the grand
        # total and was almost entirely False. Nothing noticed, because this
        # harness only checks that the result is non-empty and finite.
        lambda: engine.vyashtisamanstih(float(np.sum(X) + np.sum(Y)), [X, Y]),
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

        # The verdict is computed from what the engine returned. Previously
        # this was the literal True and the outputs were discarded.
        problems: List[str] = []
        for name, value in zip(_sutra_names(engine), outputs):
            # No dtype coercion: several sutras return complex arrays, and
            # `dtype=float` would silently discard the imaginary part --
            # exactly the kind of quiet degradation this harness is checking
            # for. np.isfinite handles complex directly.
            arr = np.asarray(value)
            if arr.size == 0:
                problems.append(f"{name}: returned an empty result")
            elif not np.all(np.isfinite(arr)):
                problems.append(f"{name}: returned non-finite values")
        expected = len(engine.sutras) + len(engine.sub_sutras)
        if len(outputs) != expected:
            problems.append(
                f"ran {len(outputs)} sutras, engine declares {expected}")

        self.results["vedic_sutras"] = {
            "main_sutras": len(engine.sutras),
            "sub_sutras": len(engine.sub_sutras),
            "executed": len(outputs),
            "problems": problems,
            "verified": not problems,
        }

    def record_hypercube_shapes(self) -> None:
        """Record the output shapes of the four hypercube operators. No verdict.

        This does not verify anything, and is named so it cannot be read as
        doing so. It builds ``Hypercube(size=8)`` itself and stores the four
        result shapes; there is no reference value, no comparison, and no
        pass/fail key in what it writes to ``self.results``.

        It deliberately does not emit a ``verified`` key the way
        ``verify_sutra_engine`` does. Every verdict available here is settled
        before it runs: the shapes are (8, 8) because this method chose
        ``size=8``; Λ is symmetric because ``lambda_operator`` returns
        ``0.5 * (result + result.T)``; and the outputs are finite because
        ``lambda_operator`` already raises on a non-finite adapter result. A
        ``"verified": True`` computed from any of those would be a foregone
        conclusion dressed as a measurement — the same defect this module's
        header describes removing from ``verify_sutra_engine``, which had
        recorded that key as a literal.
        """
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
