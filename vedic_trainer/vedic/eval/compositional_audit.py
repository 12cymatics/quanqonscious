"""At-inference audit-closure rate.

Given generated outputs, encode each into Ψ via the deterministic
``encode_text_to_psi`` and check the four conservation residuals. The
closure rate is the fraction of generations whose Ψ has all four residuals
exactly zero in ℚ.

**This metric cannot measure anything about the model, and must not be
reported as if it could.**

R2, R3 and R4 are algebraic identities on tensor-product-encoded Ψ: they are
exactly zero for every input, so they never constrain anything. That leaves
R1, which closes exactly when the trace counter is a multiple of
T(29) = 435 — and the counter here is ``trace_start + i``, the position in
the list. So closure is a function of the loop index alone.

Measured directly: two corpora with nothing in common — 480 English
sentences and 480 strings of random consonants — produce *identical* closure
flags and an identical rate of 0.0042 (2 of 480). Across 960 distinct texts
at a fixed trace index, the closure value takes exactly one value.

The README's falsification criterion 2 ("audit-closure rate at inference for
``full`` minus ``no_sutra`` < 10% absolute") is therefore structurally
incapable of discriminating: both arms are guaranteed the same number, so
the criterion is met for any two models whatsoever, including two copies of
the same one. It is recorded as unmeasurable rather than reported as passed.

``test_audit_closure_is_index_dependent_only`` pins this down. If the
residuals ever become text-dependent, that test fails and this docstring —
and the criterion — need revisiting.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

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
    """Fraction of generations whose Ψ closes all four residuals.

    An empty corpus used to return 0.0 -- reporting "nothing closed" for a
    corpus in which nothing was measured.
    """
    if not texts:
        raise ValueError(
            "audit_closure_rate got no texts; there is no rate to report. "
            "An empty corpus is a loading failure, not a 0% closure rate.")
    flags = score_audit_psi_batch(texts, trace_start=trace_start)
    return sum(1 for f in flags if f) / len(flags)
