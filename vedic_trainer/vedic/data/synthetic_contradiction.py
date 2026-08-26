"""Generate (P, ¬P) contradiction pairs by deterministic polarity flip.

For every base sentence we:

    1. Encode it via ``encode_text_to_psi``.
    2. Apply the S2 (NikhilamComplement) map followed by S5 (centering)
       to the encoded Ψ; this yields the antipodal Ψ on the polarity axis
       and re-centres the result so the conservation residuals stay
       exactly zero. Mathematically: Ψ_neg = S5(S2(Ψ)).
    3. Mutate the text by inserting a polarity-flip token chosen by a
       fixed table (``not`` after the first auxiliary, or ``no`` before
       the head noun) so the resulting string is also a contradiction
       linguistically, not just in latent space.

The (P, ¬P, axis_label) triple is deterministic in the seed and the
input text. No LLM call is made at generation time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vedic.kernel.q import Q16
from vedic.kernel.z2_primitives import s2_nikhilam, s5_shunyam_samya

from .tesseract_encode import encode_text_to_psi


@dataclass(frozen=True)
class ContradictionPair:
    base_text: str
    contradiction_text: str
    base_psi: Q16
    contradiction_psi: Q16
    label: str = "contradictory"


_AUX_VERBS = (
    "is", "are", "was", "were", "has", "have", "had",
    "does", "did", "do", "can", "could", "will", "would", "should",
    "may", "might", "must",
)


def _flip_polarity(text: str) -> str:
    """Insert a single 'not' after the first auxiliary; if none, prepend 'It is not the case that '."""
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok.lower().rstrip(".,;:") in _AUX_VERBS:
            tokens.insert(i + 1, "not")
            return " ".join(tokens)
    return "It is not the case that " + text[0].lower() + text[1:]


def _antipodal_psi(psi: Q16) -> Q16:
    """Ψ ↦ S5(S2(Ψ)). The S5 step makes R3 (mean preservation) exact zero."""
    return s5_shunyam_samya(s2_nikhilam(psi))


def generate_contradiction_pair(text: str) -> ContradictionPair:
    base_psi = encode_text_to_psi(text)
    contradiction_text = _flip_polarity(text)
    contradiction_psi = _antipodal_psi(base_psi)
    return ContradictionPair(
        base_text=text,
        contradiction_text=contradiction_text,
        base_psi=base_psi,
        contradiction_psi=contradiction_psi,
    )


def generate_corpus(texts: Iterable[str]) -> list[ContradictionPair]:
    return [generate_contradiction_pair(t) for t in texts]
