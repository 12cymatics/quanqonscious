"""Sutra composition algebra over exact ℚ: SERIES, PARALLEL, CONCURRENT.

Ported from the composition algebra specified in ``R4_tesseract_cymatic_v4.html``
(§8), which is the authoritative statement of these modes in this repository:

    SERIES      S′ = (T_{k_N} ∘ ⋯ ∘ T_{k_1})(S₀)          [output → input]
    PARALLEL    S′ = (1/N) Σ_k T_k(S₀)                     [fork S₀, one join]
    CONCURRENT  S′ = (Φ_W ∘ ⋯ ∘ Φ_1)(S₀),
                Φ_w(S) = (1/|A_w|) Σ_{k ∈ A_w} T_k(S)      [BSP wavefront]
    CANONICAL   S′ = Ψ_upa(Ψ_mukhya(S₀))                   [16 SERIES, then 13 PARALLEL]
    COMPOSITE   PARALLEL linear part + first-order BCH commutator coupling

CONCURRENT strictly interpolates the other two: W = N is SERIES, W = 1 is
PARALLEL.

Determinism
-----------
CONCURRENT's wave schedule is seeded from the queue itself, not from a shared
stream. A real scheduler's interleaving is arbitrary but must be *fixed* for a
given input, or the same queue would evolve differently on each run and break
the determinism invariant (CODEX 7.2). The LFSR and Fisher–Yates permutation
are ported bit-for-bit from the reference so both implementations schedule
identically.

Everything here is exact ℚ. No floats, no epsilons, no silent fallbacks:
an undefined composition raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from .q import Q16
from .tesseract import NUM_VERTICES
from . import sutras_exact as S

# ----------------------------------------------------------------------
# State arithmetic on ℚ^16
# ----------------------------------------------------------------------


def st_zero() -> Q16:
    """Additive identity — the accumulator seed, never S₀."""
    return tuple(Fraction(0) for _ in range(NUM_VERTICES))


def st_add(a: Q16, b: Q16) -> Q16:
    return tuple(x + y for x, y in zip(a, b))


def st_sub(a: Q16, b: Q16) -> Q16:
    return tuple(x - y for x, y in zip(a, b))


def st_scale(a: Q16, k: Fraction) -> Q16:
    return tuple(x * k for x in a)


def st_mean(items: Sequence[Q16]) -> Q16:
    """Arithmetic mean — the join barrier for PARALLEL and each CONCURRENT wave."""
    if not items:
        raise ValueError("st_mean: empty branch list has no mean")
    acc = st_zero()
    for it in items:
        acc = st_add(acc, it)
    return st_scale(acc, Fraction(1, len(items)))


# ----------------------------------------------------------------------
# Deterministic scheduler (LFSR + Fisher-Yates), ported bit-for-bit
# ----------------------------------------------------------------------

_MASK32 = 0xFFFFFFFF
_POLY = 0xB4BCD35C
_GOLDEN = 0x9E3779B9


class Lfsr:
    """32-bit Galois LFSR. Same polynomial and step order as the reference."""

    __slots__ = ("s",)

    def __init__(self, seed: int) -> None:
        self.s = (seed & _MASK32) or 1

    def next(self) -> int:
        lsb = self.s & 1
        self.s >>= 1
        if lsb:
            self.s ^= _POLY
        self.s &= _MASK32
        return self.s

    def below(self, n: int) -> int:
        if n <= 0:
            raise ValueError(f"Lfsr.below: bound must be positive, got {n}")
        return self.next() % n

    def permute(self, items: Sequence[int]) -> List[int]:
        a = list(items)
        for i in range(len(a) - 1, 0, -1):
            j = self.below(i + 1)
            a[i], a[j] = a[j], a[i]
        return a


def schedule_seed(indices: Sequence[int]) -> int:
    """Seed derived from the queue: h = 0x9E3779B9, h = h*33 + (k+1) mod 2³²."""
    h = _GOLDEN
    for k in indices:
        h = (h * 33 + (k + 1)) & _MASK32
    return h


# ----------------------------------------------------------------------
# Registry: uniform unary operators T_k : ℚ^16 → ℚ^16
# ----------------------------------------------------------------------
#
# Four sutras are not natively ℚ^16 → ℚ^16. Composition needs a uniform
# signature, so each gets an explicit, documented lift. The raw functions in
# sutras_exact stay untouched and keep their true return types; only the
# composed view is lifted.
#
#   S7  -> returns (sym, anti). Lift selects one part; which part is an
#          operand of the spec, defaulting to the symmetric part.
#   S18 -> returns a scalar. Lift scales the input by it: T(Ψ) = S18(Ψ)·Ψ.
#   S22 -> returns 8 pair differences. Lift embeds them at the 8 pair-lead
#          indices (v < v⊕mask), zero elsewhere.
#   S27 -> returns a scalar. Lifted like S18.
#
# Binary sutras (S3, S17, S23) take a second operand Φ. Composition needs a
# unary operator, so Φ must be bound to a function of Ψ.
#
# The obvious binding, Φ = Ψ, is wrong for S17: S17(Ψ, Ψ) = Ψ·Ψ_ref/Ψ_ref = Ψ
# for every ref, so T₁₇ becomes the identity — a no-op in SERIES and pure
# dilution in a PARALLEL mean. The default binding is therefore the Nikhilam
# complement Φ = S2(Ψ), which is the pairing involution S7/S8/S22/S23 already
# use, and which degenerates for none of the three (verified in
# test_composition.py). BINARY_BINDING can be swapped for another policy.


@dataclass(frozen=True)
class SutraSpec:
    """One sutra as a composable unit."""

    index: int          # 0-based position in the 29-sutra queue
    name: str
    delta: int          # δ_k = k+1; Σ δ over all 29 = 435 = T(29)
    arity: int          # 1 or 2 operands
    native: str         # 'vector' | 'pair' | 'scalar' | 'pairs8'


def _embed_pairs8(vals: Sequence[Fraction], mask: int) -> Q16:
    """Place 8 pair differences at their lead indices; zero elsewhere."""
    out = [Fraction(0)] * NUM_VERTICES
    from .tesseract import xor_pairs_lt
    for (v, _c), x in zip(xor_pairs_lt(mask), vals):
        out[v] = x
    return tuple(out)


def nikhilam_binding(psi: Q16) -> Q16:
    """Default Φ for the binary sutras: the Nikhilam complement S2(Ψ)."""
    return S.s2_nikhilam(psi)


BINARY_BINDING: Callable[[Q16], Q16] = nikhilam_binding
"""Policy that supplies Φ to S3/S17/S23 under composition.

Must not make any of them the identity — Φ = Ψ does exactly that to S17.
"""


def _unary_ops() -> Dict[int, Callable[[Q16], Q16]]:
    """Build the canonical T_k for k = 0..28 (sutra S(k+1))."""
    return {
        0:  lambda p: S.s1_eka_adhikena(p),
        1:  lambda p: S.s2_nikhilam(p),
        2:  lambda p: S.s3_urdhva_tiryak(p, BINARY_BINDING(p)),
        3:  lambda p: S.s4_paravartya(p),
        4:  lambda p: S.s5_shunyam_samya(p),
        5:  lambda p: S.s6_anurupya_shunyam(p),
        6:  lambda p: S.s7_sankalana_vyavakalana(p)[0],  # symmetric part
        7:  lambda p: S.s8_puranapuranabhyam_fill(p),
        8:  lambda p: S.s9_chalana_kalanabhyam(p),
        9:  lambda p: S.s10_yavadunam_tavadunikrtya(p),
        10: lambda p: S.s11_vyasti_samasti(p),
        11: lambda p: S.s12_shesanyankena_charamena(p),
        12: lambda p: S.s13_sopantyadvayamantyam_last2(p),
        13: lambda p: S.s14_ekanyunena_purvena(p),
        14: lambda p: S.s15_gunitasamucchaya_product(p),
        15: lambda p: S.s16_gunaka_samucchaya(p),
        16: lambda p: S.s17_anurupyena_proportion(p, BINARY_BINDING(p)),
        17: lambda p: st_scale(p, S.s18_adyamadyena_antyamantyena(p)),
        18: lambda p: S.s19_lopana_sthapanabhyam(p),
        19: lambda p: S.s20_vilokanam_spect(p),
        20: lambda p: S.s21_dhvajanka_flag(p),
        21: lambda p: _embed_pairs8(S.s22_parity_complement(p), S.FULL_MASK),
        22: lambda p: S.s23_dwandwa_yoga(p, BINARY_BINDING(p)),
        23: lambda p: S.s24_kevalaih_saptakam(p),
        24: lambda p: S.s25_vestana_circular(p),
        25: lambda p: S.s26_yavadunam_square(p),
        26: lambda p: st_scale(p, S.s27_samuccaya_gunitah(p)),
        27: lambda p: S.s28_lopana_restore(p),
        28: lambda p: S.s29_mean_drive(p),
    }


_NATIVE = {6: "pair", 17: "scalar", 21: "pairs8", 26: "scalar"}
_BINARY = {2, 16, 22}

SUTRAS: Tuple[SutraSpec, ...] = tuple(
    SutraSpec(
        index=i,
        name=S.SUTRA_NAMES[i],
        delta=i + 1,
        arity=2 if i in _BINARY else 1,
        native=_NATIVE.get(i, "vector"),
    )
    for i in range(len(S.SUTRA_NAMES))
)

OPS: Dict[int, Callable[[Q16], Q16]] = _unary_ops()

N_SUTRAS: int = len(SUTRAS)
DELTA_TOTAL: int = sum(s.delta for s in SUTRAS)   # 435 = 29·30/2
ALL: Tuple[int, ...] = tuple(range(N_SUTRAS))
MUKHYA: Tuple[int, ...] = tuple(range(16))        # the 16 main sutras
UPA: Tuple[int, ...] = tuple(range(16, N_SUTRAS))  # the 13 sub-sutras


def _check_queue(indices: Sequence[int]) -> List[int]:
    ks = list(indices)
    if not ks:
        raise ValueError("empty sutra queue: composition is undefined")
    for k in ks:
        if not 0 <= k < N_SUTRAS:
            raise ValueError(f"sutra index out of range 0..{N_SUTRAS - 1}: {k}")
    return ks


def apply_one(k: int, psi: Q16) -> Q16:
    """Apply the single unary operator T_k."""
    if not 0 <= k < N_SUTRAS:
        raise ValueError(f"sutra index out of range: {k}")
    return OPS[k](psi)


# ----------------------------------------------------------------------
# Composition modes
# ----------------------------------------------------------------------


def series(psi: Q16, indices: Sequence[int]) -> Q16:
    """SERIES — strict left fold; each output feeds the next input."""
    ks = _check_queue(indices)
    s = psi
    for k in ks:
        s = OPS[k](s)
    return s


def parallel(psi: Q16, indices: Sequence[int]) -> Q16:
    """PARALLEL — every branch reads S₀; single mean-join barrier."""
    ks = _check_queue(indices)
    return st_mean([OPS[k](psi) for k in ks])


def wave_count(n: int) -> int:
    """W = ⌈√N⌉ — an integer wave count, never part of the state."""
    if n <= 0:
        raise ValueError(f"wave_count: N must be positive, got {n}")
    r = isqrt(n)
    return max(1, r if r * r == n else r + 1)


def concurrent_waves(indices: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """The deterministic wave partition A_1..A_W for a queue (schedule only)."""
    ks = _check_queue(indices)
    w = wave_count(len(ks))
    order = Lfsr(schedule_seed(ks)).permute(ks)
    waves: List[List[int]] = [[] for _ in range(w)]
    for i, k in enumerate(order):
        waves[i % w].append(k)
    return tuple(tuple(a) for a in waves if a)


def concurrent(psi: Q16, indices: Sequence[int]) -> Q16:
    """CONCURRENT — BSP wavefront: parallel inside a wave, series across waves.

    W = ⌈√N⌉. Every sutra in a wave reads the same incoming state; the wave
    joins on the mean; the next wave consumes that join. W = N degenerates to
    SERIES, W = 1 to PARALLEL.
    """
    s = psi
    for wave in concurrent_waves(indices):
        s = st_mean([OPS[k](s) for k in wave])
    return s


def canonical(psi: Q16, indices: Sequence[int] = ALL) -> Q16:
    """CANONICAL — the 16 mukhya in SERIES, then the 13 upasutras in PARALLEL.

    The queue is partitioned by index rather than ignored, so a mukhya-only
    queue degrades to SERIES and an upa-only queue to PARALLEL.
    """
    ks = _check_queue(indices)
    mukhya = [k for k in ks if k < 16]
    upa = [k for k in ks if k >= 16]
    mid = series(psi, mukhya) if mukhya else psi
    return parallel(mid, upa) if upa else mid


def composite(psi: Q16, indices: Sequence[int]) -> Q16:
    """COMPOSITE — δ-weighted linear part plus first-order BCH commutator.

        L = Σ_k w_k T_k(S₀),                w_k = δ_k / Σ δ
        C = Σ_{i<j} w_i w_j ([T_i, T_j])(S₀)
        S′ = L + Γ·C,                       Γ = 1/N

    The commutator uses the cached images T_k(S₀), matching the reference.
    """
    ks = _check_queue(indices)
    n = len(ks)
    tot = sum(SUTRAS[k].delta for k in ks)
    w = [Fraction(SUTRAS[k].delta, tot) for k in ks]
    img = [OPS[k](psi) for k in ks]

    lin = st_zero()
    for wi, im in zip(w, img):
        lin = st_add(lin, st_scale(im, wi))

    comm_acc = st_zero()
    for i in range(n):
        for j in range(i + 1, n):
            comm = st_sub(OPS[ks[i]](img[j]), OPS[ks[j]](img[i]))
            comm_acc = st_add(comm_acc, st_scale(comm, w[i] * w[j]))

    gamma = Fraction(1, n)
    return st_add(lin, st_scale(comm_acc, gamma))


MODES: Dict[str, Callable[[Q16, Sequence[int]], Q16]] = {
    "SERIES": series,
    "PARALLEL": parallel,
    "CONCURRENT": concurrent,
    "CANONICAL": canonical,
    "COMPOSITE": composite,
}


def compose(mode: str, psi: Q16, indices: Sequence[int] = ALL) -> Q16:
    """Dispatch by mode name. Unknown modes raise — there is no default."""
    key = mode.upper()
    if key not in MODES:
        raise ValueError(f"unknown composition mode {mode!r}; "
                         f"expected one of {sorted(MODES)}")
    return MODES[key](psi, indices)


def annihilating_runs(indices: Sequence[int] = ALL) -> Tuple[Tuple[int, ...], ...]:
    """Ordered sub-runs of the queue that are the zero map on every input.

    SERIES silently returning a zero vector is a valid computation but a
    useless result a caller could mistake for signal. This reports *why*.

    The known structural run is S20 → S21 → S22: S20 projects onto one Walsh
    row so its image is c·h_axis; S21 takes absolute values, turning the
    alternating vector into the constant |c|; S22 differences (v, v⊕mask)
    pairs, which is exactly zero on a constant. It is a property of the
    specified operators, not of any particular Ψ, so it cannot be fixed by a
    different binding or lift — only by changing S20/S21/S22 themselves.
    """
    ks = _check_queue(indices)
    found: List[Tuple[int, ...]] = []
    for run in ((19, 20, 21),):
        pos = [i for i, k in enumerate(ks) if k in run]
        ordered = [ks[i] for i in pos]
        if ordered == list(run):
            found.append(run)
    return tuple(found)


def is_degenerate_series(indices: Sequence[int] = ALL) -> bool:
    """True when SERIES over this queue is the zero map for every input."""
    return bool(annihilating_runs(indices))
