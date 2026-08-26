"""Shared fixtures: deterministic random Q16 generators in unit scale."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from vedic.kernel.q import Q16


def random_q16(rng: random.Random, denom_max: int = 100) -> Q16:
    """Sample one length-16 ℚ vector with values in [-1, 1].

    Each component is num/den where |num| ≤ den ≤ denom_max. This keeps the
    inputs in unit scale so float32 precision (~1.19e-7) is the dominant
    source of round-off and the bit-exact tolerance can be set tightly.
    """
    out = []
    for _ in range(16):
        den = rng.randint(1, denom_max)
        num = rng.randint(-den, den)
        out.append(Fraction(num, den))
    return tuple(out)


@pytest.fixture(scope="session")
def rng_seeded() -> random.Random:
    return random.Random(0xC0DEC0DE)


@pytest.fixture(scope="session")
def q16_corpus(rng_seeded: random.Random) -> list[Q16]:
    """100 deterministic random Q16 inputs (unit scale)."""
    return [random_q16(rng_seeded) for _ in range(100)]


@pytest.fixture(scope="session")
def q16_corpus_pairs(rng_seeded: random.Random) -> list[tuple[Q16, Q16]]:
    """50 deterministic random (Ψ, Φ) pairs (unit scale)."""
    return [(random_q16(rng_seeded), random_q16(rng_seeded)) for _ in range(50)]
