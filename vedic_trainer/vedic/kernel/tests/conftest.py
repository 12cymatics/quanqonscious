"""Shared fixtures. The vectors come from ``psi_corpus``; nothing is random.

These fixtures used to build their corpus with ``random.Random(0xC0DEC0DE)``.
Deterministic, but still a sample: it established each property on the
hundred vectors drawn and said nothing about ℚ^16. They now serve the same
enumerated inputs the parametrised tests use, so a property proved through a
fixture and a property proved through a parameter are proved on the same set.
"""
from __future__ import annotations

import pytest

from vedic.kernel.q import Q16
from vedic.kernel.tests.psi_corpus import PSI_CASES, PSI_VECTORS, SPANNING_SET


@pytest.fixture(scope="session")
def q16_corpus() -> list[Q16]:
    """The structured corpus plus the full 137-vector spanning set.

    Deduplicated: the corpus already contains the basis vectors and several
    two-vertex sums.
    """
    seen: dict[tuple, None] = {}
    for psi in tuple(PSI_VECTORS) + tuple(SPANNING_SET):
        seen.setdefault(psi, None)
    return list(seen)


@pytest.fixture(scope="session")
def q16_corpus_pairs() -> list[tuple[Q16, Q16]]:
    """Each corpus vector against its successor, plus all 120 basis pairs."""
    from vedic.kernel.tests.psi_corpus import BASIS

    labels = [n for n, _ in PSI_CASES]
    by = dict(PSI_CASES)
    seen: dict[tuple[Q16, Q16], None] = {}
    for i, label in enumerate(labels):
        seen.setdefault((by[label], by[labels[(i + 1) % len(labels)]]), None)
    for i in range(16):
        for j in range(i + 1, 16):
            seen.setdefault((BASIS[i], BASIS[j]), None)
    return list(seen)
