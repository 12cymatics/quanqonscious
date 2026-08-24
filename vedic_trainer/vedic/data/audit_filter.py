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


def audit_closed(psi: Q16, trace_sum: Fraction) -> bool:
    """Boolean shorthand for ``audit_psi(...).closed`` — all four residuals.

    ``trace_sum`` is required. It used to default to ``Fraction(0)``, which
    made R1 vacuously closed: R1 measures the trace counter against T(29) =
    435, and 0 is a multiple of 435, so the default answered one of the four
    checks in the affirmative without ever consulting the caller's trace. A
    predicate named ``audit_closed`` that silently skips a quarter of the
    audit reports closure it did not verify.

    There is no sentinel for "no trace available". A caller that does not
    have the running trace count cannot answer R1, and so cannot answer this
    predicate; it should call :func:`audit_psi` and read the residuals it can
    actually justify.
    """
    return audit_psi(psi, trace_sum).closed
