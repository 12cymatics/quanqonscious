"""SCAN / COGS evaluators.

The runners depend on ``transformers`` and ``datasets``; they are imported
lazily so the kernel-only path used in CI does not pull those heavy
dependencies in.

There is no audit-closure metric here any more. ``audit_closure_rate`` and
``score_audit_psi_batch`` scored generated *text* by encoding it to Ψ with
the synthetic text encoder and reading the four conservation residuals. The
encoder is gone, and the metric was never able to measure a model in any
case: three of the four residuals are algebraic identities and the fourth
takes no Ψ, so the verdict was a function of the loop index alone. That is
proved over all of ℚ^16 — not on a sample of encoded strings — in
``vedic/kernel/tests/test_audit_closure_degeneracy.py``.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

# Re-exported from the module that validates against it, never redeclared.
# This constant used to read ("simple", "length", "jump") while scan.py read
# ("simple", "length", "addprim_jump"), so any caller who imported the
# package-level name and passed it straight through hit
# `ValueError: unknown SCAN split: 'jump'`.
from .scan_splits import COGS_SPLITS, SCAN_SPLITS  # noqa: E402


def evaluate_scan(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    return import_module("vedic.eval.scan").evaluate_scan(*args, **kwargs)


def evaluate_cogs(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    return import_module("vedic.eval.cogs").evaluate_cogs(*args, **kwargs)


__all__ = [
    "SCAN_SPLITS",
    "COGS_SPLITS",
    "evaluate_scan",
    "evaluate_cogs",
]
