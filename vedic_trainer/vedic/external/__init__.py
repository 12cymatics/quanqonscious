"""Optional sidecar adapters for vedic_trainer.

These modules wrap code pulled from sibling branches of the parent
repo and surface it as an optional sub-package. They are intentionally
**not** imported by the kernel/training/data layers — the bit-exact ℚ
algebra and the LoRA training path do not depend on any of them.

What's here:

- ``vedic_engine`` — the classical (NumPy float64) interpretation of
  the 29 sutras as arithmetic recipes. Source:
  ``codex/replace-blocks-with-fixed-implementations:src/quanqonscious/vedic_sutra_engine.py``.
  Complements ``vedic.kernel.z2_primitives`` (Z₂⁴ structural algebra
  over ℚ) — the two interpretations are different lenses on the same
  29 Vedic algorithms.
- ``hypercube`` — float hypercube operators (Λ, Ω, Υ, weighted) that
  build on top of ``vedic_engine``. Source:
  ``codex/replace-blocks-with-fixed-implementations:src/quanqonscious/hypercube.py``.
- ``lean4_mirror`` — Lean 4 backed mirror that runs Bool-valued sutra
  statements through the Lean compiler. Source:
  ``codex/fix-package-exports-in-__all__-definition:src/quanqonscious/lean4_mirror.py``.
- ``lean_props`` — exposes our 30 ℚ-exact algebraic identities as
  Lean 4 ``Prop``s so the Lean mirror can prove them on demand.
- ``executor`` — serial / threads / processes orchestration adapter.
  Source: ``codex/locate-runnable-simulations-in-repos:src/quanqonscious/sutra_executor.py``.
  Adapted to run our exact-ℚ z2_primitives functions across batches.
- ``proof_validation`` — smoke test harness that asserts every sutra
  in ``vedic_engine`` is callable on representative inputs. Source:
  ``codex/replace-blocks-with-fixed-implementations:src/quanqonscious/proof_validation.py``.

All adapters are pure Python; they have soft dependencies on numpy
(present everywhere) and on the Lean 4 binary (only for the mirror).
"""
from __future__ import annotations

from .vedic_engine import VedicSutraEngine
from .hypercube import Hypercube
from .lean4_mirror import (
    VEDIC_SUTRAS,
    Lean4Mirror,
    Lean4MirrorResult,
    Lean4SessionConfig,
)
from .lean_props import build_lean_props
from .executor import ExecutionMode, SutraExecutor
from .proof_validation import ProofTester

__all__ = [
    "VedicSutraEngine",
    "Hypercube",
    "VEDIC_SUTRAS",
    "Lean4Mirror",
    "Lean4MirrorResult",
    "Lean4SessionConfig",
    "build_lean_props",
    "ExecutionMode",
    "SutraExecutor",
    "ProofTester",
]
