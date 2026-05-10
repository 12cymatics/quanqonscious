"""All 30 algebraic identities in INTERACTIONS hold bit-exactly over ℚ."""
from __future__ import annotations

from vedic.kernel.interaction_matrix import INTERACTIONS, verify_all
from vedic.kernel.q import Q16


def test_interactions_count() -> None:
    assert len(INTERACTIONS) == 30


def test_all_identities_hold(q16_corpus_pairs: list[tuple[Q16, Q16]]) -> None:
    failures: list[tuple[str, int]] = []
    for i, (psi, phi) in enumerate(q16_corpus_pairs):
        results = verify_all(psi, phi)
        for name, ok in results:
            if not ok:
                failures.append((name, i))
    assert not failures, f"identity failures: {failures[:10]}"
