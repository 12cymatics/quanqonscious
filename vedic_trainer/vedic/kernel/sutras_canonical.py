"""The 29 α-weighted sutra ids — exact ℚ, all 29 present, **nine distinct maps**.

How many formulas this module holds
-----------------------------------
Nine, not twenty-nine. ``STRICT_SUTRA_KERNEL`` dispatches on
``SUTRA_KIND[id]`` through seven templates; two branch internally (REFL on
``id == 5``, PERM on ``axis = (id+1) & 3``), so seven become nine. Between two
ids of the same class the *only* difference is the scalar
α(n) = (n/435)·(strength/100).

That is faithful to upstream — ``test_upstream_agreement.py`` matches the real
JavaScript on 6,380 of 6,380 triples, and upstream is nine maps too. It is
recorded here because nothing said it, and "the 29 sutras, canonically
defined" (which is how this line read) invites the opposite reading.

``test_sutras_canonical.py::test_the_twenty_nine_ids_are_nine_distinct_maps``
measures the partition and fails if it changes.

**If you want 29 genuinely distinct operators, they exist elsewhere in this
repository**: ``vedic/kernel/z2_primitives.py`` is 29 distinct functions
(verified pairwise), ``vedic_v18.51.1_exact_phi.html`` implements 29 separate
cases over the C4/K2(√5) extension, and ``vedic_sutras_complete.hpp`` has 29
C++ implementations. This module is not those, and does not claim to be.

Source of truth
---------------
Ported from ``vedic_v18.24_full_kernel.html`` (tracked at the repository
root, despite several documents having said it was not):

* ``§12Z VERTEX-FIELD SUTRA OPERATORS``  (line 5544) — the seven operator types
* ``STRICT_SUTRA_KERNEL``                (line 6527) — the exact, float-free path
* ``SUTRA_KIND``                         (line 3558) — the id → kind dispatch
* ``ALPHA.computeQ``                     — the α weight
* ``§12Y STRUCTURAL SINGULARITY SUPPRESSION`` — the identity guarantee

``STRICT_SUTRA_KERNEL`` is the canonical path (line 5449:
``psi' = STRICT_SUTRA_KERNEL_i(psi, alpha_i)``). The ``SUTRAS[].evolve()``
bodies are the float-contaminated display path and are NOT the definition.

Substrate
---------
Ψ : V_d → ℚ over V_d = (ℤ/2ℤ)⁴ — the 16 vertices of the tesseract.
Vertices are adjacent iff their labels differ in one bit; complements iff the
labels XOR to 1111; hw(v) = popcount(v) ∈ {0..4}.

The α weight
------------
    α(n) = (n / 435) · (strength / 100),      435 = T(29) = 29·30/2

The triangular denominator is S29's conservation identity Σδ(1..29) = 435.

Structural guarantee (§12Y)
---------------------------
Every operator collapses to the identity when α → 0:

    MULTIPLICATIVE  Ψ'ᵢ = Ψᵢ·(1 + α·Ψ_{i⊕1})           → Ψᵢ
    REFLECTIVE      Ψ'ᵢ = blend(Ψᵢ, R(Ψ)ᵢ, α)          → Ψᵢ
    CONVOLUTIVE     Ψ'ᵢ = blend(Ψᵢ, (Ψ⊛Ψ)ᵢ/16, α)      → Ψᵢ
    DIVISIVE        Ψ'ᵢ = blend(Ψᵢ, D(Ψ)ᵢ, α)          → Ψᵢ
    DIFFUSIVE       Ψ'ᵢ = blend(Ψᵢ, edgeMean(i), α)     → Ψᵢ
    PERMUTATIVE     Ψ'ᵢ = blend(Ψᵢ, Ψ_{i⊕2^axis}, α)    → Ψᵢ
    MODULAR         Ψ'ᵢ = blend(Ψᵢ, mean(Ψ), α)         → Ψᵢ

with ``blend(c, t, w) = c + (t − c)·w``.

Exactness: every value is a ``Fraction``. No floats, no epsilons, no clamps,
no fallbacks. Out-of-domain arguments raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Sequence, Tuple

from .q import Q16
from .tesseract import NUM_VERTICES

# ----------------------------------------------------------------------
# Canonical constants
# ----------------------------------------------------------------------

N_SUTRAS: int = 29

SUTRA_SUM: int = 435
"""T(29) = 29·30/2. The triangular denominator of the α weight (§12Y)."""

FULL_MASK: int = 0b1111
"""Complement involution v ↦ v̄ on the 4-bit vertex label."""

BIT_WIDTH: int = 4

# id → kind, transcribed from SUTRA_KIND (vedic_v18.24_full_kernel.html:3558).
# Index 0 is unused so that SUTRA_KIND[id] reads naturally with 1-based ids.
SUTRA_KIND: Tuple[str, ...] = (
    "",
    "MULT", "REFL", "CONV", "DIV", "REFL", "PERM", "PERM", "DIV", "DIFF",
    "MULT", "CONV", "REFL", "DIV", "MULT", "MULT", "DIV", "DIFF", "MOD",
    "DIV", "MOD", "MOD", "REFL", "REFL", "MOD", "CONV", "PERM", "DIFF",
    "DIFF", "MOD",
)

# The seven §12Z operator types, plus CONSERVATION which the engine gives S29
# its own desc for while dispatching it as MOD.
CATEGORY: Tuple[str, ...] = (
    "",
    "MULTIPLICATIVE", "REFLECTIVE", "CONVOLUTIVE", "DIVISIVE", "REFLECTIVE",
    "PERMUTATIVE", "PERMUTATIVE", "DIVISIVE", "DIFFUSIVE", "MULTIPLICATIVE",
    "CONVOLUTIVE", "REFLECTIVE", "DIVISIVE", "MULTIPLICATIVE",
    "MULTIPLICATIVE", "DIVISIVE", "DIFFUSIVE", "MODULAR", "DIVISIVE",
    "MODULAR", "MODULAR", "REFLECTIVE", "REFLECTIVE", "MODULAR",
    "CONVOLUTIVE", "PERMUTATIVE", "DIFFUSIVE", "DIFFUSIVE", "CONSERVATION",
)

NAMES: Tuple[str, ...] = (
    "",
    "Ekādhikena Pūrveṇa", "Nikhilam", "Ūrdhva-Tiryagbhyām",
    "Parāvartya Yojayet", "Śūnyam Sāmyasamuccaye", "Ānurūpye Śūnyamanyat",
    "Saṅkalana-Vyavakalanābhyām", "Pūraṇāpūraṇābhyām", "Calana-Kalanābhyām",
    "Yāvadūnam", "Vyaṣṭisamaṣṭiḥ", "Veṣṭanaṃ", "Sopāntyadvayamantyam",
    "Ekanyūnena Pūrveṇa", "Guṇitasamuccayaḥ", "Guṇakasamuccayaḥ",
    "Ānurūpyeṇa", "Śiṣyate Śeṣasaṃjñaḥ", "Kevalaih Saptakam Guṇyāt",
    "Vilokanam", "Guṇitasamuccayah Samuccayaguṇitaḥ", "Dvandvayogaḥ",
    "Antyayordaśake'pi", "Lopanasthāpanābhyām", "Samuccayaguṇitaḥ",
    "Dhvajāṅka", "Gunitasamuccayah", "Antyayoḥ Eva", "Conservation",
)

SANSKRIT: Tuple[str, ...] = (
    "",
    "एकाधिकेन", "निखिलं", "ऊर्ध्व", "परावर्त्य", "शून्यं", "आनुरूप्ये", "संकलन",
    "पूरणा", "चलन", "यावदूनम्", "व्यष्टि", "वेष्टनम्", "सोपान्त्य", "एकन्यूनेन",
    "गुणित", "गुणक", "आनुरूप्येण", "शिष्यते", "केवलैः", "विलोकनम्",
    "गुणितसमुच्चयः", "द्वन्द्वयोगः", "अन्त्ययोः", "लोपन", "समुच्चयगुणितः",
    "ध्वजाङ्क", "गुणितसमुच्चयः", "अन्त्ययोः", "संरक्षण",
)

# Per-sutra coefficients (Vedic protocol v4.0 §4.3). Every one is an exact
# rational; no IEEE-754 representation exists in the arithmetic path.
#
# CAVEAT, per the Exact Kernel Evolution Blueprint (§Golden-ratio extension):
# these are rational *representatives*, not the transcendental/algebraic
# constants themselves. φ, √2, √3, √5, e, π, ln2 … are irrational and cannot
# live in ℚ. S1's 12586269025/7778742049 is the Fibonacci convergent F₅₀/F₄₉,
# which agrees with φ to double precision but is NOT φ. The blueprint is
# explicit that "treating those values as elements of K2 would be false" and
# that the canonical home for φ and √5 is the extension C4 = K2(√5), where
# φ = (1+√5)/2 and φ³ = 2 + √5 exactly. See ``k2_field.py``; use
# ``k2_field.phi()`` when exactness in φ is required, and these rationals only
# where a declared rational representative is what the operator wants.
COEFFICIENT: Dict[int, Fraction] = {
    1:  Fraction(12586269025, 7778742049),   # F₅₀/F₄₉ ≈ φ (see caveat; exact φ is in C4)
    2:  Fraction(2718282, 1000000),          # e
    3:  Fraction(355, 113),                  # π  (Milü)
    4:  Fraction(577, 408),                  # √2 (Pell convergent)
    5:  Fraction(5772157, 10000000),         # γ  (Euler–Mascheroni)
    6:  Fraction(2302585, 1000000),          # ln 10
    7:  Fraction(97, 56),                    # √3
    8:  Fraction(2236068, 1000000),          # √5
    9:  Fraction(2665144, 1000000),          # δ_s (silver ratio)
    10: Fraction(1202057, 1000000),          # ζ(3) (Apéry)
    11: Fraction(9159656, 10000000),         # Catalan's constant
    12: Fraction(14513692, 10000000),        # Backhouse's constant
    13: Fraction(6623490, 10000000),         # Laplace limit
    14: Fraction(13247180, 10000000),        # plastic number
    15: Fraction(25029079, 10000000),        # Feigenbaum α
    16: Fraction(46692016, 10000000),        # Feigenbaum δ
    17: Fraction(3729671, 10000000),         # K_CF
    18: Fraction(4530103, 10000000),         # ∫₀¹ xˣ dx
    19: Fraction(5671433, 10000000),         # Ω (omega constant)
    20: Fraction(6243300, 10000000),         # Li₂(½)
    21: Fraction(6931472, 10000000),         # ln 2
    22: Fraction(7651977, 10000000),         # ζ(2)/2
    23: Fraction(8241323, 10000000),         # K_π/2
    24: Fraction(8765482, 10000000),         # Dottie number
    25: Fraction(9159656, 10000000),         # Catalan's constant
    26: Fraction(9560319, 10000000),         # ∫₀¹ ln(1+x)/x dx
    27: Fraction(9829780, 10000000),         # ∏(1 − 2⁻ⁿ)
    28: Fraction(9961578, 10000000),         # ≈ 1 − 1/256
    29: Fraction(9990234, 10000000),         # ≈ 1 − 1/1024
}


@dataclass(frozen=True)
class Sutra:
    """One canonical sutra."""

    id: int              # 1-based, matching the engine
    name: str
    sanskrit: str
    kind: str            # dispatch kind: MULT/REFL/CONV/DIV/DIFF/PERM/MOD
    category: str        # §12Z operator type (S29 is CONSERVATION)
    delta: int           # δ_n = n; Σδ(1..29) = 435
    coefficient: Fraction


SUTRAS: Tuple[Sutra, ...] = tuple(
    Sutra(id=i, name=NAMES[i], sanskrit=SANSKRIT[i], kind=SUTRA_KIND[i],
          category=CATEGORY[i], delta=i, coefficient=COEFFICIENT[i])
    for i in range(1, N_SUTRAS + 1)
)

ALL: Tuple[int, ...] = tuple(range(1, N_SUTRAS + 1))


# ----------------------------------------------------------------------
# Substrate helpers (VTX in the engine)
# ----------------------------------------------------------------------


def hw(v: int) -> int:
    """Hamming weight — popcount of the 4-bit vertex label."""
    return bin(v & FULL_MASK).count("1")


def comp(v: int) -> int:
    """Complement vertex: label XOR 1111."""
    return v ^ FULL_MASK


def neighbors(v: int) -> Tuple[int, int, int, int]:
    """The four Hamming-distance-1 neighbours."""
    return (v ^ 1, v ^ 2, v ^ 4, v ^ 8)


def mean(psi: Q16) -> Fraction:
    """(1/16) Σᵢ Ψᵢ."""
    return sum(psi, Fraction(0)) / Fraction(NUM_VERTICES)


def edge_mean(psi: Q16, v: int) -> Fraction:
    """(1/4) Σ_{j ∈ neighbours(v)} Ψ_j."""
    return sum((psi[j] for j in neighbors(v)), Fraction(0)) / Fraction(4)


def norm_sq(psi: Q16) -> Fraction:
    """Q = Σᵢ Ψᵢ² — the conservation monitor quantity (§3.5)."""
    return sum((p * p for p in psi), Fraction(0))


# ----------------------------------------------------------------------
# α weight and blend
# ----------------------------------------------------------------------


def alpha(sid: int, strength: Fraction) -> Fraction:
    """α(n) = (n / 435) · (strength / 100).   ALPHA.computeQ, §12Y.

    ``strength`` is the single external control. α = 0 makes every operator
    the identity, which is the structural guarantee the whole design rests on.
    """
    _check_id(sid)
    return Fraction(sid, SUTRA_SUM) * Fraction(strength, 100)


def blend(current: Fraction, target: Fraction, w: Fraction) -> Fraction:
    """blend(c, t, w) = c + (t − c)·w.   STRICT_SUTRA_KERNEL.blend."""
    return current + (target - current) * w


def _check_id(sid: int) -> None:
    if not 1 <= sid <= N_SUTRAS:
        raise ValueError(f"sutra id out of range 1..{N_SUTRAS}: {sid}")


def _check_psi(psi: Sequence[Fraction]) -> None:
    if len(psi) != NUM_VERTICES:
        raise ValueError(f"Ψ must have {NUM_VERTICES} vertices; got {len(psi)}")


# ----------------------------------------------------------------------
# The seven operator types
# ----------------------------------------------------------------------


def _mult(psi: Q16, w: Fraction) -> Q16:
    """MULTIPLICATIVE — Ψ'ᵢ = Ψᵢ · (1 + w·Ψ_{i⊕1}).

    'By one more than the previous': the multiplier is the bit-0 neighbour.
    """
    one = Fraction(1)
    return tuple(psi[i] * (one + w * psi[i ^ 1]) for i in range(NUM_VERTICES))


def _refl(psi: Q16, w: Fraction, sid: int) -> Q16:
    """REFLECTIVE — blend toward the complement mirror.

    target = −Ψ_c            for S5 (Śūnyam Sāmyasamuccaye: zero-sum)
           = (Ψᵢ + Ψ_c)/2    otherwise (complement average)
    """
    two = Fraction(2)
    out = []
    for i in range(NUM_VERTICES):
        c = comp(i)
        target = -psi[c] if sid == 5 else (psi[i] + psi[c]) / two
        out.append(blend(psi[i], target, w))
    return tuple(out)


def _conv(psi: Q16, w: Fraction) -> Q16:
    """CONVOLUTIVE — blend toward the normalised XOR self-convolution.

        (Ψ ⊛ Ψ)ᵢ = (1/16) Σ_j Ψ_j · Ψ_{i⊕j}
    """
    sixteen = Fraction(NUM_VERTICES)
    conv = []
    for i in range(NUM_VERTICES):
        acc = Fraction(0)
        for j in range(NUM_VERTICES):
            acc += psi[j] * psi[i ^ j]
        conv.append(acc / sixteen)
    return tuple(blend(psi[i], conv[i], w) for i in range(NUM_VERTICES))


def _diff(psi: Q16, w: Fraction) -> Q16:
    """DIFFUSIVE — blend toward the edge mean (graph Laplacian smoothing)."""
    return tuple(blend(psi[i], edge_mean(psi, i), w) for i in range(NUM_VERTICES))


def _perm(psi: Q16, w: Fraction, sid: int) -> Q16:
    """PERMUTATIVE — blend toward the reflection across axis (id+1) & 3."""
    axis = (sid + 1) & 3
    step = 1 << axis
    return tuple(blend(psi[i], psi[i ^ step], w) for i in range(NUM_VERTICES))


def _div(psi: Q16, w: Fraction) -> Q16:
    """DIVISIVE — Hamming-layer interpolation between mean and edge mean.

        h = hw(i)/4;   target = mean + h·(edgeMean(i) − mean)

    hw = 0 sits at the global mean, hw = 4 at the local edge mean.
    """
    m = mean(psi)
    four = Fraction(4)
    out = []
    for i in range(NUM_VERTICES):
        h = Fraction(hw(i), 1) / four
        target = m + h * (edge_mean(psi, i) - m)
        out.append(blend(psi[i], target, w))
    return tuple(out)


def _mod(psi: Q16, w: Fraction) -> Q16:
    """MODULAR / CONSERVATION — blend toward the global mean."""
    m = mean(psi)
    return tuple(blend(psi[i], m, w) for i in range(NUM_VERTICES))


_DISPATCH = {
    "MULT": lambda psi, w, sid: _mult(psi, w),
    "REFL": lambda psi, w, sid: _refl(psi, w, sid),
    "CONV": lambda psi, w, sid: _conv(psi, w),
    "DIFF": lambda psi, w, sid: _diff(psi, w),
    "PERM": lambda psi, w, sid: _perm(psi, w, sid),
    "DIV":  lambda psi, w, sid: _div(psi, w),
    "MOD":  lambda psi, w, sid: _mod(psi, w),
}


# ----------------------------------------------------------------------
# Public application
# ----------------------------------------------------------------------


def apply_sutra(sid: int, psi: Q16, strength: Fraction) -> Q16:
    """Apply sutra ``sid`` to Ψ at the given strength.  STRICT_SUTRA_KERNEL.applyOne.

    Returns Ψ unchanged when α = 0 — the §12Y structural guarantee, checked
    explicitly rather than emerging from arithmetic.
    """
    _check_id(sid)
    _check_psi(psi)
    w = alpha(sid, strength)
    if w == 0:
        return tuple(psi)
    kind = SUTRA_KIND[sid]
    if kind not in _DISPATCH:
        raise ValueError(f"S{sid} has unknown kind {kind!r}")
    return _DISPATCH[kind](tuple(psi), w, sid)


def apply_all(psi: Q16, strength: Fraction, order: Sequence[int] = ALL) -> Q16:
    """Apply a queue of sutras in SERIES at one strength."""
    out = tuple(psi)
    for sid in order:
        out = apply_sutra(sid, out, strength)
    return out


def drift(sid: int, psi: Q16, strength: Fraction) -> Fraction:
    """D_k(Ψ) = |Q(S_k Ψ) − Q(Ψ)| — the §3.7 drift ranker, exact."""
    return abs(norm_sq(apply_sutra(sid, psi, strength)) - norm_sq(psi))


def rank_by_drift(psi: Q16, strength: Fraction,
                  order: Sequence[int] = ALL) -> Tuple[Tuple[int, Fraction], ...]:
    """All 29 ranked by individual drift contribution, ascending (§3.7).

    Used to discover and validate conservation cores.
    """
    scored = [(sid, drift(sid, psi, strength)) for sid in order]
    scored.sort(key=lambda kv: (kv[1], kv[0]))
    return tuple(scored)


# Conservation-core candidates (§3.8). Composition order as specified.
WORMHOLE_CORE: Tuple[int, ...] = (29, 26, 23, 22, 9, 5)
SYMMETRY_CORE: Tuple[int, ...] = (29, 22, 12, 7, 5)


# ----------------------------------------------------------------------
# Execution modes (Vedic protocol §4.2) over the canonical operators
# ----------------------------------------------------------------------
#
#   PARALLEL    independent contributions, no ordering
#   SERIES      each operator compounds the last, order matters
#   CONCURRENT  BSP wavefront: parallel within a wave, series across waves
#   COMPOSITE   chained sequences mixing modes
#   INVERSE     anti-sutra: negated displacement, push-pull equilibrium
#
# The deterministic wave scheduler is shared with ``composition`` so both
# paths partition a queue identically.


def _mean_of(states: Sequence[Q16]) -> Q16:
    if not states:
        raise ValueError("no branches to join")
    n = Fraction(len(states))
    return tuple(sum((s[i] for s in states), Fraction(0)) / n
                 for i in range(NUM_VERTICES))


def series(psi: Q16, strength: Fraction, order: Sequence[int] = ALL) -> Q16:
    """SERIES — strict left fold; each output feeds the next input."""
    return apply_all(psi, strength, order)


def parallel(psi: Q16, strength: Fraction, order: Sequence[int] = ALL) -> Q16:
    """PARALLEL — every branch reads Ψ; single mean join."""
    ks = list(order)
    if not ks:
        raise ValueError("empty sutra queue")
    return _mean_of([apply_sutra(sid, psi, strength) for sid in ks])


def concurrent(psi: Q16, strength: Fraction, order: Sequence[int] = ALL) -> Q16:
    """CONCURRENT — BSP wavefront, W = ⌈√N⌉ waves, deterministic schedule."""
    from .composition import concurrent_waves
    ks = list(order)
    if not ks:
        raise ValueError("empty sutra queue")
    # composition schedules 0-based indices; canonical ids are 1-based.
    waves = concurrent_waves([k - 1 for k in ks])
    out = tuple(psi)
    for wave in waves:
        out = _mean_of([apply_sutra(k + 1, out, strength) for k in wave])
    return out


def inverse(psi: Q16, strength: Fraction, order: Sequence[int] = ALL) -> Q16:
    """INVERSE — anti-sutra: negate the operator displacement (§4.2).

        Ψ' = Ψ − (S_k(Ψ) − Ψ)

    applied in reverse queue order, giving the push-pull equilibrium partner
    of SERIES rather than an exact left inverse.
    """
    ks = list(order)
    if not ks:
        raise ValueError("empty sutra queue")
    out = tuple(psi)
    for sid in reversed(ks):
        forward = apply_sutra(sid, out, strength)
        out = tuple(2 * out[i] - forward[i] for i in range(NUM_VERTICES))
    return out


MODES = {
    "SERIES": series,
    "PARALLEL": parallel,
    "CONCURRENT": concurrent,
    "INVERSE": inverse,
}


def compose(mode: str, psi: Q16, strength: Fraction,
            order: Sequence[int] = ALL) -> Q16:
    """Dispatch by mode name. Unknown modes raise — there is no default."""
    key = mode.upper()
    if key not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    return MODES[key](psi, strength, order)


# ----------------------------------------------------------------------
# Operator records (Exact Kernel Evolution Blueprint, Gate E)
# ----------------------------------------------------------------------
#
# The blueprint requires, per operator: exact domain/codomain, decomposition,
# linearity status, reversibility conditions, matrix/operator invariant, and
# a separation of extensional from intensional evidence.
#
#   EXTENSIONAL — the output equals the declared mathematical map.
#   INTENSIONAL — the implementation uses the claimed *Vedic decomposition*.
#
# The blueprint states plainly that the current 29-entry runtime is
# "executable but several entries are generic phase, permutation, or
# pair-rotation constructions ... not yet certified as the intended Vedic
# decompositions." That assessment is carried here rather than papered over:
# every operator is INTENSIONALLY uncertified, and ``INTENSIONAL_STATUS``
# says so on every record.
#
# There is deliberately no matching ``extensional`` field. One used to exist,
# holding the constant string "defined and tested against the declared map",
# stamped unconditionally onto all 29 records by ``operator_record()``. It was
# removed for two reasons. First, being identical for every operator it could
# not distinguish any one from any other, which is the only thing a
# per-operator evidence field is for. Second, "tested against the declared
# map" is a claim about the test suite, and this module cannot observe the
# test suite — so whatever it reported was an assertion, never a measurement.
#
# The evidence it gestured at is real and stays where it is checkable: the
# closed form for each kind is asserted against the implementation in
# ``tests/test_sutras_canonical.py`` — ``test_mult_formula``,
# ``test_refl_formula_general``,
# ``test_refl_formula_s5_is_the_negated_complement``, ``test_conv_formula``,
# ``test_diff_formula``, ``test_perm_axis_is_id_plus_one_mod_four``,
# ``test_div_formula_interpolates_hamming_layers`` and
# ``test_mod_formula_blends_toward_the_mean``. The record's ``decomposition``
# field names the map each of those checks, so the pointer to the evidence
# survives; only the self-awarded verdict is gone.

LINEAR_KINDS = frozenset({"REFL", "DIV", "DIFF", "PERM", "MOD"})
"""Kinds whose action is linear in Ψ. MULT and CONV are quadratic:
MULT multiplies Ψᵢ by Ψ_{i⊕1}, CONV convolves Ψ with itself."""

QUADRATIC_KINDS = frozenset({"MULT", "CONV"})

INTENSIONAL_STATUS = (
    "UNCERTIFIED — the operator is a generic blend/permutation/convolution "
    "form, not a proven Vedic arithmetic decomposition"
)


def is_linear(sid: int) -> bool:
    """Whether S_sid acts linearly on Ψ (blueprint: linearity status)."""
    _check_id(sid)
    return SUTRA_KIND[sid] in LINEAR_KINDS


def operator_matrix(sid: int, strength: Fraction) -> Tuple[Tuple[Fraction, ...], ...]:
    """The exact 16×16 matrix of a linear sutra (blueprint: matrix invariant).

    Raises for quadratic kinds, which have no matrix representation — that
    refusal is the point: MULT and CONV are not linear operators and must not
    be reported as though they were.
    """
    _check_id(sid)
    if not is_linear(sid):
        raise ValueError(
            f"S{sid} is {SUTRA_KIND[sid]}, which is quadratic in Ψ; "
            "it has no 16×16 matrix representation"
        )
    cols = []
    for j in range(NUM_VERTICES):
        basis = tuple(Fraction(1) if i == j else Fraction(0)
                      for i in range(NUM_VERTICES))
        cols.append(apply_sutra(sid, basis, strength))
    # cols[j] is the image of e_j, i.e. column j; transpose to row-major.
    return tuple(tuple(cols[j][i] for j in range(NUM_VERTICES))
                 for i in range(NUM_VERTICES))


def determinant(matrix: Sequence[Sequence[Fraction]]) -> Fraction:
    """Exact determinant by fraction-free Gaussian elimination."""
    n = len(matrix)
    a = [list(row) for row in matrix]
    det = Fraction(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r][col] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
            det = -det
        det *= a[col][col]
        inv = Fraction(1) / a[col][col]
        for r in range(col + 1, n):
            if a[r][col] != 0:
                f = a[r][col] * inv
                for c in range(col, n):
                    a[r][c] -= f * a[col][c]
    return det


def is_reversible(sid: int, strength: Fraction) -> bool:
    """Reversibility condition: a linear sutra is invertible iff det ≠ 0.

    Quadratic kinds are not decided here and raise rather than guess.
    """
    return determinant(operator_matrix(sid, strength)) != 0


@dataclass(frozen=True)
class OperatorRecord:
    """The blueprint's per-operator evidence record."""

    id: int
    name: str
    kind: str
    category: str
    domain: str
    codomain: str
    decomposition: str
    linear: bool
    intensional: str


DECOMPOSITION = {
    "MULT": "Ψᵢ ↦ Ψᵢ·(1 + α·Ψ_{i⊕1})   [bit-0 neighbour product]",
    "REFL": "Ψᵢ ↦ blend(Ψᵢ, (Ψᵢ+Ψ_c)/2, α)   [complement average]",
    "CONV": "Ψᵢ ↦ blend(Ψᵢ, (Ψ⊛Ψ)ᵢ/16, α)   [XOR self-convolution]",
    "DIV":  "Ψᵢ ↦ blend(Ψᵢ, m + hw(i)/4·(edge−m), α)   [Hamming layers]",
    "DIFF": "Ψᵢ ↦ blend(Ψᵢ, edgeMean(i), α)   [graph Laplacian]",
    "PERM": "Ψᵢ ↦ blend(Ψᵢ, Ψ_{i⊕2^((id+1)&3)}, α)   [axis reflection]",
    "MOD":  "Ψᵢ ↦ blend(Ψᵢ, mean(Ψ), α)   [residue toward mean]",
}


def operator_record(sid: int) -> OperatorRecord:
    """The full record for one sutra."""
    _check_id(sid)
    s = SUTRAS[sid - 1]
    decomp = DECOMPOSITION[s.kind]
    if sid == 5:
        decomp = "Ψᵢ ↦ blend(Ψᵢ, −Ψ_c, α)   [zero-sum complement]"
    return OperatorRecord(
        id=sid, name=s.name, kind=s.kind, category=s.category,
        domain="ℚ^16 over V4 = Z₂⁴", codomain="ℚ^16 over V4 = Z₂⁴",
        decomposition=decomp, linear=is_linear(sid),
        intensional=INTENSIONAL_STATUS,
    )


def all_operator_records() -> Tuple[OperatorRecord, ...]:
    return tuple(operator_record(i) for i in ALL)
