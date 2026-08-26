"""The shared Ψ corpus every kernel test runs against.

One corpus, imported rather than re-declared, so a test cannot quietly run
against a weaker set of inputs than its neighbours. It is a plain module and
not a fixture because ``pytest.mark.parametrize`` is evaluated at collection
time, before fixtures exist.

Why the structured entries are here
-----------------------------------
The kernel tests used to share one Ψ, ``(v²+1)/7``: strictly positive,
monotone increasing, no repeated values, nonzero mean. That single vector
cannot distinguish an operator that mishandles a sign, a constant field, a
zero component, or a field whose mean is zero — and eighteen of the
twenty-nine sutras were checked against nothing else.
"""
from __future__ import annotations

import random
from fractions import Fraction

Q16 = tuple


def random_psi(seed: int, lo: int = -9, hi: int = 9, den_max: int = 7) -> tuple:
    """One deterministic random Ψ. Same seed, same vector, on every machine."""
    r = random.Random(seed)
    return tuple(Fraction(r.randint(lo, hi), r.randint(1, den_max))
                 for _ in range(16))


def _cases() -> tuple[tuple[str, tuple], ...]:
    zero = tuple(Fraction(0) for _ in range(16))
    const = tuple(Fraction(3, 7) for _ in range(16))
    alt = tuple(Fraction(1 if v % 2 == 0 else -1) for v in range(16))
    spike = tuple(Fraction(1) if v == 0 else Fraction(0) for v in range(16))
    spike_hi = tuple(Fraction(1) if v == 15 else Fraction(0) for v in range(16))
    neg = tuple(Fraction(-(v * v + 1), 7) for v in range(16))
    legacy = tuple(Fraction(v * v + 1, 7) for v in range(16))
    # Σ(2v − 15) for v = 0..15 is 2·120 − 16·15 = 0, so the mean is exactly 0.
    centred = tuple(Fraction(2 * v - 15, 6) for v in range(16))
    # Large denominators: exercises exact ℚ rather than anything float-like.
    fine = tuple(Fraction((-1) ** v * (v + 1), 100003) for v in range(16))
    cases = [
        ("zero", zero), ("constant", const), ("alternating", alt),
        ("spike_low", spike), ("spike_high", spike_hi), ("negative", neg),
        ("legacy_v2_over_7", legacy), ("mean_zero", centred),
        ("fine_denominators", fine),
    ]
    cases += [(f"random_{s}", random_psi(s)) for s in range(8)]
    return tuple(cases)


PSI_CASES: tuple[tuple[str, tuple], ...] = _cases()

#: Vectors only, for tests that do not need the labels.
PSI_VECTORS: tuple[tuple, ...] = tuple(psi for _, psi in PSI_CASES)

#: Labelled lookup, for tests that want one specific structural case.
BY_LABEL: dict[str, tuple] = dict(PSI_CASES)

#: The Ψ the pre-rewrite tests used, kept so a test that is genuinely about
#: one particular input can name it rather than indexing into the corpus.
LEGACY_PSI = BY_LABEL["legacy_v2_over_7"]

#: A second independent field, for the binary operators (S3, S17, S23).
PHI = tuple(Fraction(3 * v + 2, 5) for v in range(16))

#: Strengths spanning zero, one, the canonical 50, full 100, a fraction and a
#: value above 100. α is linear in strength and nothing caps it, so a formula
#: that only agrees at 50 must still be caught.
STRENGTHS: tuple[Fraction, ...] = (
    Fraction(0), Fraction(1), Fraction(50), Fraction(100),
    Fraction(7, 3), Fraction(250),
)
