"""SCAN / COGS evaluators + at-inference audit-closure rate.

The SCAN/COGS runners depend on ``transformers`` and ``datasets``; they
are imported lazily so the kernel-only path (used in CI and by data
generators) does not pull those heavy dependencies in.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .compositional_audit import audit_closure_rate, score_audit_psi_batch


SCAN_SPLITS: tuple[str, ...] = ("simple", "length", "jump")


def evaluate_scan(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    return import_module("vedic.eval.scan").evaluate_scan(*args, **kwargs)


def evaluate_cogs(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    return import_module("vedic.eval.cogs").evaluate_cogs(*args, **kwargs)


__all__ = [
    "SCAN_SPLITS",
    "evaluate_scan",
    "evaluate_cogs",
    "audit_closure_rate",
    "score_audit_psi_batch",
]
