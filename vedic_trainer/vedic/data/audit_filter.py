"""Audit filter: closed Ψ ⇔ all four conservation residuals vanish.

Operates over exact ℚ — the residuals R2, R3, R4 are algebraic
identities on tensor-product-encoded Ψ and evaluate to exact zero. R1
closes when the trace counter is a multiple of T(29) = 435; the filter
takes the trace value as an argument so the caller can sample-and-hold
the running count from the trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from vedic.kernel.conservation_exact import all_residuals
from vedic.kernel.q import Q16


@dataclass(frozen=True)
class AuditResult:
    closed: bool
    residuals: Tuple[Fraction, Fraction, Fraction, Fraction]


def audit_psi(psi: Q16, trace_sum: Fraction) -> AuditResult:
    r1, r2, r3, r4 = all_residuals(psi, trace_sum)
    closed = (r1 == 0) and (r2 == 0) and (r3 == 0) and (r4 == 0)
    return AuditResult(closed=closed, residuals=(r1, r2, r3, r4))


def audit_closed(psi: Q16, trace_sum: Fraction = Fraction(0)) -> bool:
    """Boolean shorthand. Default trace_sum = 0 makes R1 vacuously closed."""
    return audit_psi(psi, trace_sum).closed
