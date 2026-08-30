"""The shared Ψ inputs every kernel test runs against. Nothing is random.

One module, imported rather than re-declared, so a test cannot quietly run
against a weaker set of inputs than its neighbours. It is a plain module and
not a fixture because ``pytest.mark.parametrize`` is evaluated at collection
time, before fixtures exist.

Why there are no random vectors
-------------------------------
This module used to hand out ``random_psi(seed)`` — deterministic, but
synthetic: a sample of ℚ^16 chosen by a PRNG, which establishes a property on
the vectors drawn and nothing about the ones that were not. Passing on a
hundred samples is evidence, not a result, and the number of samples was a
knob nobody could justify.

It is replaced by two things that are neither random nor samples:

``STRUCTURED``
    Seven vectors chosen because each is a *case*: the zero field, a constant
    field, an alternating field, an all-negative field, the positive monotone
    field the tests originally used, a field with mean exactly zero, and one
    with six-digit denominators. Every one exists to exercise a specific way
    an operator can be wrong.

``SPANNING_SET``
    ``{0} ∪ {eᵢ} ∪ {eᵢ + eⱼ}`` — 137 vectors. For any map of degree at most
    two in Ψ, these evaluations determine the map completely: writing
    F(Ψ) = c + L(Ψ) + Q(Ψ, Ψ) with L linear and Q symmetric bilinear,

        c        = F(0)
        Q(eᵢ,eⱼ) = [F(eᵢ+eⱼ) − F(eᵢ) − F(eⱼ) + F(0)] / 2
        L(eᵢ)    = F(eᵢ) − F(0) − Q(eᵢ,eᵢ)

    so two degree-≤2 maps that agree on the spanning set are the same map on
    all of ℚ^16. Agreement there is therefore a proof, not a sample — which
    is why the corpus needed no random vectors in the first place.

``TRIPLE_SET``
    ``{eᵢ + eⱼ + e_k}`` — 560 vectors, used to *establish* the degree-≤2
    premise rather than assume it: a map of degree three or higher is not
    reproduced by the reconstruction above, and fails there.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations

Q16 = tuple

N = 16
ZERO: tuple = tuple(Fraction(0) for _ in range(N))


def basis(i: int) -> tuple:
    """eᵢ — the i-th standard basis vector of ℚ^16."""
    if not 0 <= i < N:
        raise ValueError(f"basis index out of range: {i}")
    return tuple(Fraction(1) if j == i else Fraction(0) for j in range(N))


def _add(*vs: tuple) -> tuple:
    out = ZERO
    for v in vs:
        out = tuple(a + b for a, b in zip(out, v))
    return out


BASIS: tuple[tuple, ...] = tuple(basis(i) for i in range(N))

SPANNING_SET: tuple[tuple, ...] = (
    (ZERO,) + BASIS
    + tuple(_add(BASIS[i], BASIS[j]) for i, j in combinations(range(N), 2))
)

TRIPLE_SET: tuple[tuple, ...] = tuple(
    _add(BASIS[i], BASIS[j], BASIS[k])
    for i, j, k in combinations(range(N), 3)
)


def _structured() -> tuple[tuple[str, tuple], ...]:
    const = tuple(Fraction(3, 7) for _ in range(N))
    alt = tuple(Fraction(1 if v % 2 == 0 else -1) for v in range(N))
    neg = tuple(Fraction(-(v * v + 1), 7) for v in range(N))
    monotone = tuple(Fraction(v * v + 1, 7) for v in range(N))
    # Σ(2v − 15) for v = 0..15 is 2·120 − 16·15 = 0, so the mean is exactly 0.
    centred = tuple(Fraction(2 * v - 15, 6) for v in range(N))
    # Large denominators: exercises exact ℚ rather than anything float-like.
    fine = tuple(Fraction((-1) ** v * (v + 1), 100003) for v in range(N))
    # No "spike" entries: a single-vertex spike at index 0 or 15 *is* e0 or
    # e15, which BASIS already supplies. Listing them twice under different
    # names would inflate the corpus count without adding an input, and the
    # non-degeneracy guard below would have to be relaxed to permit it.
    return (
        ("zero", ZERO), ("constant", const), ("alternating", alt),
        ("negative", neg), ("monotone_square_over_7", monotone),
        ("mean_zero", centred), ("fine_denominators", fine),
    )


STRUCTURED: tuple[tuple[str, tuple], ...] = _structured()

#: The labelled corpus: the structured cases, plus the basis and a fixed
#: selection of two-vertex sums so ordinary tests exercise the spanning
#: geometry too. Labels are stable and carry no seed.
PSI_CASES: tuple[tuple[str, tuple], ...] = (
    STRUCTURED
    + tuple((f"basis_{i}", BASIS[i]) for i in range(N))
    + tuple((f"basis_{i}+{j}", _add(BASIS[i], BASIS[j]))
            for i, j in ((0, 1), (0, 15), (3, 12), (7, 8), (1, 2), (5, 10)))
)

PSI_VECTORS: tuple[tuple, ...] = tuple(psi for _, psi in PSI_CASES)
BY_LABEL: dict[str, tuple] = dict(PSI_CASES)

# Aliases for the two basis vectors that are also natural structural cases.
# They name the same tuples, so they add no entries to PSI_CASES.
BY_LABEL["spike_low"] = BASIS[0]
BY_LABEL["spike_high"] = BASIS[N - 1]

#: (v² + 1)/7 — the positive monotone field the pre-rewrite tests used, named
#: so a test genuinely about one particular input can say so rather than
#: indexing into the corpus.
MONOTONE_PSI = BY_LABEL["monotone_square_over_7"]

#: A second independent field, for the binary operators (S3, S17, S23).
PHI = tuple(Fraction(3 * v + 2, 5) for v in range(N))

#: Dyadic vectors: every component is k/2^m, so float64 represents them and
#: every sum, difference and halving of them exactly. Used where an exact
#: float comparison is wanted instead of a tolerance. Enumerated, not sampled.
DYADIC: tuple[tuple[str, tuple], ...] = tuple(
    (f"dyadic_2^-{m}", tuple(Fraction((-1) ** v * (v + 1), 2 ** m)
                             for v in range(N)))
    for m in range(0, 11)
) + (
    ("dyadic_zero", ZERO),
    ("dyadic_units", tuple(Fraction(1) for _ in range(N))),
    ("dyadic_alternating_half", tuple(Fraction((-1) ** v, 2) for v in range(N))),
    ("dyadic_powers", tuple(Fraction(1, 2 ** (v % 11)) for v in range(N))),
)

#: Strengths spanning zero, one, the canonical 50, full 100, a fraction and a
#: value above 100. α is linear in strength and nothing caps it, so a formula
#: that only agrees at 50 must still be caught.
STRENGTHS: tuple[Fraction, ...] = (
    Fraction(0), Fraction(1), Fraction(50), Fraction(100),
    Fraction(7, 3), Fraction(250),
)
