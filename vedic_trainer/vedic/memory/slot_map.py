"""Fixed 16-slot semantic feature map for the Z₂⁴ working memory.

Each vertex v ∈ {0..15} of the Boolean cube carries a four-axis label
(polarity, subject/object, tense, evidentiality). Bit k of v encodes
axis k:

    bit 0 (LSB): polarity                — 0: positive,    1: negative
    bit 1      : subject/object          — 0: subjective,  1: objective
    bit 2      : tense                   — 0: present/past 1: future/irrealis
    bit 3 (MSB): evidentiality           — 0: direct,      1: inferred

The 16 slot names below are the Cartesian product of these axes; the
Walsh-Hadamard axes (computed in ``vedic/kernel/wht.py``) line up with
these four bits, which is why the WHT-axis projection in L_dual reads
out polarity / S/O / tense / evidentiality independently.

Beyond the four primary axes, downstream sutras decompose Ψ further; the
catalogue here is the canonical first-order labeling used by the audit
filter and by the WHT-axis projection in L_dual.
"""
from __future__ import annotations

from typing import Tuple

# Axis names indexed by bit position.
AXIS_NAMES: Tuple[str, ...] = (
    "polarity",         # bit 0
    "subject_object",   # bit 1
    "tense",            # bit 2
    "evidentiality",    # bit 3
)

# Per-axis labels: AXIS_LABELS[axis][bit_value] -> human-readable token.
AXIS_LABELS: Tuple[Tuple[str, str], ...] = (
    ("pos", "neg"),
    ("subj", "obj"),
    ("now", "future"),
    ("direct", "inferred"),
)


def _name_for_vertex(v: int) -> str:
    parts = []
    for k in range(4):
        bit = (v >> k) & 1
        parts.append(AXIS_LABELS[k][bit])
    return ".".join(parts)


SLOT_NAMES: Tuple[str, ...] = tuple(_name_for_vertex(v) for v in range(16))
"""SLOT_NAMES[v] is the canonical name for vertex v."""


def slot_index_for(name: str) -> int:
    """Inverse map: given a slot name, return the vertex index 0..15.

    Names use the dotted form ``axis0.axis1.axis2.axis3`` as in SLOT_NAMES.
    Raises ValueError if the name is not in the catalogue (no fail-safe;
    typos blow up).
    """
    try:
        return SLOT_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown slot name: {name!r}") from exc
