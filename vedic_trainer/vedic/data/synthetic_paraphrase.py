"""Axis-emphasis paraphrase generator.

For each base sentence we produce two paraphrases that differ only on a
single Walsh axis (axis 0 = polarity, axis 1 = subject/object, axis 2 =
tense, axis 3 = evidentiality). The Ψ vectors are constructed directly
in ℚ via the same tensor-product encoder as ``encode_text_to_psi`` but
with the targeted-axis feature flipped, which guarantees by
construction that the difference projects onto exactly that Walsh axis.
The text labels follow a closed-form template so the pairs remain
inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from vedic.kernel.q import Q16
from vedic.kernel.tesseract import BIT_WIDTH, NUM_VERTICES

from .tesseract_encode import _axis_features


@dataclass(frozen=True)
class ParaphrasePair:
    axis: int                  # 0..3 — which Walsh axis is emphasised
    text_a: str
    text_b: str
    psi_a: Q16
    psi_b: Q16
    label: str = "paraphrase"


_AXIS_TEMPLATES = (
    # axis 0: polarity
    ("[POS] {text}", "[NEG] {text}"),
    # axis 1: subject/object
    ("[SUBJ] {text}", "[OBJ] {text}"),
    # axis 2: tense
    ("[NOW] {text}", "[FUT] {text}"),
    # axis 3: evidentiality
    ("[DIR] {text}", "[INF] {text}"),
)


def _psi_from_features(fs: Tuple[Fraction, Fraction, Fraction, Fraction]) -> Q16:
    out: list[Fraction] = []
    for v in range(NUM_VERTICES):
        component = Fraction(1)
        for k in range(BIT_WIDTH):
            bit = (v >> k) & 1
            sign = -1 if bit else 1
            component *= (Fraction(1) + Fraction(sign) * fs[k]) / Fraction(2)
        out.append(component)
    return tuple(out)


def generate_paraphrase_pair(text: str, axis: int) -> ParaphrasePair:
    if not 0 <= axis < BIT_WIDTH:
        raise ValueError(f"axis must be in 0..{BIT_WIDTH - 1}; got {axis}")
    fs_base = _axis_features(text)
    fs_a = list(fs_base)
    fs_b = list(fs_base)
    fs_a[axis] = Fraction(1)
    fs_b[axis] = Fraction(-1)
    psi_a = _psi_from_features(tuple(fs_a))   # type: ignore[arg-type]
    psi_b = _psi_from_features(tuple(fs_b))   # type: ignore[arg-type]
    tmpl_a, tmpl_b = _AXIS_TEMPLATES[axis]
    return ParaphrasePair(
        axis=axis,
        text_a=tmpl_a.format(text=text),
        text_b=tmpl_b.format(text=text),
        psi_a=psi_a,
        psi_b=psi_b,
    )


def axis_difference(psi_a: Q16, psi_b: Q16) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    """Per-axis Walsh-axis projection of (psi_a − psi_b).

    For tensor-product-encoded inputs this is non-zero on exactly the
    targeted axis, so the result is a finger-print of which axis the
    paraphrase emphasises.
    """
    diff = tuple(a - b for a, b in zip(psi_a, psi_b))
    out: list[Fraction] = []
    for k in range(BIT_WIDTH):
        accum = Fraction(0)
        for v in range(NUM_VERTICES):
            sign = -1 if ((v >> k) & 1) else 1
            accum += Fraction(sign) * diff[v]
        out.append(accum)
    return tuple(out)  # type: ignore[return-value]
