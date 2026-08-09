"""At-inference audit-closure rate.

Given generated outputs, encode each into Ψ via the deterministic
``encode_text_to_psi`` and check the four conservation residuals (R1
trace counter is fed by the iteration index, R2/R3/R4 are algebraic).
The closure rate is the fraction of generations whose Ψ has all four
residuals exactly zero in ℚ.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

from vedic.data.audit_filter import audit_psi
from vedic.data.tesseract_encode import encode_text_to_psi


def score_audit_psi_batch(texts: Sequence[str], trace_start: int = 0) -> list[bool]:
    """Run the audit on each text. Trace counter increments by 1 per text."""
    out: list[bool] = []
    for i, text in enumerate(texts):
        psi = encode_text_to_psi(text)
        # The R1 residual closes at multiples of 435; we feed
        # ``trace_start + i`` so the auditor sees a deterministic counter.
        result = audit_psi(psi, Fraction(trace_start + i))
        out.append(result.closed)
    return out


def audit_closure_rate(texts: Sequence[str], trace_start: int = 0) -> float:
    if not texts:
        return 0.0
    flags = score_audit_psi_batch(texts, trace_start=trace_start)
    return sum(1 for f in flags if f) / len(flags)
