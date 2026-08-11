"""Synthetic data generators (contradiction pairs, axis-emphasis paraphrases) and audit filter."""
from __future__ import annotations

from .audit_filter import AuditResult, audit_closed, audit_psi
from .synthetic_contradiction import ContradictionPair, generate_contradiction_pair
from .synthetic_paraphrase import ParaphrasePair, generate_paraphrase_pair
from .tesseract_encode import decode_psi_to_axes, encode_text_to_psi

__all__ = [
    "AuditResult",
    "audit_closed",
    "audit_psi",
    "ContradictionPair",
    "ParaphrasePair",
    "generate_contradiction_pair",
    "generate_paraphrase_pair",
    "encode_text_to_psi",
    "decode_psi_to_axes",
]
