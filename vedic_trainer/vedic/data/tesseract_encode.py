"""Deterministic text → ℚ^16 encoder used by data generators and the audit filter.

Uses the four primary axes from ``vedic.memory.slot_map``:

    bit 0: polarity        (±1 from negation tokens)
    bit 1: subject/object  (±1 from first vs third person markers)
    bit 2: tense           (±1 from temporal markers)
    bit 3: evidentiality   (±1 from inferential markers)

A linguistic feature ``f_k ∈ {-1, 0, +1}`` is computed per axis from
hashed token markers; the 16-vertex vector is then the tensor product
of the four per-axis ``(1, f_k)`` patterns, scaled to keep the result
in the unit hypercube (every component is in [−1, 1]).

The encoder is deterministic and pure: same string ⇒ same Fraction
output. No LLM call, no float arithmetic, no fail-safes.
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Tuple

from vedic.kernel.q import Q16
from vedic.kernel.tesseract import BIT_WIDTH, NUM_VERTICES

# Lexical markers per axis. Each list is closed-class and verifiable; the
# encoder is intended to be both deterministic and inspectable.
_NEGATION_MARKERS = (
    "not", "no", "never", "none", "nothing", "neither", "nor",
    "without", "n't", "cannot",
)
_FIRST_PERSON_MARKERS = ("i", "we", "me", "us", "my", "our", "ours", "mine")
_THIRD_PERSON_MARKERS = ("he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs")
_FUTURE_MARKERS = ("will", "shall", "going", "tomorrow", "next", "soon")
_PAST_MARKERS = ("was", "were", "had", "did", "yesterday", "previously", "ago")
_INFERRED_MARKERS = ("seemed", "appeared", "must", "might", "could", "perhaps", "maybe", "likely", "evidently", "apparently")
_DIRECT_MARKERS = ("saw", "heard", "felt", "touched", "observed", "watched", "noticed", "experienced")

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> Tuple[str, ...]:
    return tuple(t.lower() for t in _TOKEN_RE.findall(text))


def _axis_feature(tokens: Tuple[str, ...], pos: Tuple[str, ...], neg: Tuple[str, ...]) -> Fraction:
    """+1 if positive markers dominate, −1 if negative, 0 otherwise.

    Returns a rational-valued tally normalised by total marker count, so the
    feature stays in [−1, 1] and is exact in ℚ.
    """
    pos_count = sum(1 for t in tokens if t in pos)
    neg_count = sum(1 for t in tokens if t in neg)
    total = pos_count + neg_count
    if total == 0:
        return Fraction(0)
    return Fraction(pos_count - neg_count, total)


def _axis_features(text: str) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    tokens = _tokenize(text)
    polarity = _axis_feature(tokens, pos=(), neg=_NEGATION_MARKERS)  # presence of neg → −1
    # For polarity: any negation token → push toward −1.
    if any(t in _NEGATION_MARKERS for t in tokens):
        polarity = Fraction(-1)
    else:
        polarity = Fraction(1)

    subjobj = _axis_feature(tokens, pos=_FIRST_PERSON_MARKERS, neg=_THIRD_PERSON_MARKERS)
    if subjobj == 0:
        subjobj = Fraction(1)  # default to subjective when no marker

    tense = _axis_feature(tokens, pos=_FUTURE_MARKERS, neg=_PAST_MARKERS)
    if tense == 0:
        tense = Fraction(1)  # default to non-past

    eviden = _axis_feature(tokens, pos=_DIRECT_MARKERS, neg=_INFERRED_MARKERS)
    if eviden == 0:
        eviden = Fraction(1)  # default to direct

    return polarity, subjobj, tense, eviden


def encode_text_to_psi(text: str) -> Q16:
    """Encode a text into Ψ ∈ ℚ^16 via the 4-axis tensor product.

    For axis k the per-axis vector is ``(1 + f_k) / 2`` if bit k is 0 and
    ``(1 − f_k) / 2`` if bit k is 1. The 16-vector is the product of these
    per-axis projections across the four bits of v.
    """
    f0, f1, f2, f3 = _axis_features(text)
    fs = (f0, f1, f2, f3)
    out: list[Fraction] = []
    for v in range(NUM_VERTICES):
        component = Fraction(1)
        for k in range(BIT_WIDTH):
            bit = (v >> k) & 1
            sign = -1 if bit else 1
            component *= (Fraction(1) + Fraction(sign) * fs[k]) / Fraction(2)
        out.append(component)
    return tuple(out)


def decode_psi_to_axes(psi: Q16) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    """Recover (f_0, f_1, f_2, f_3) from a tensor-product encoded Ψ.

    Uses the marginalisation
        f_k = Σ_{v: bit k = 0} Ψ_v − Σ_{v: bit k = 1} Ψ_v
    which holds exactly for the encoder above.
    """
    out: list[Fraction] = []
    for k in range(BIT_WIDTH):
        accum = Fraction(0)
        for v in range(NUM_VERTICES):
            sign = -1 if ((v >> k) & 1) else 1
            accum += Fraction(sign) * psi[v]
        out.append(accum)
    return tuple(out)  # type: ignore[return-value]
