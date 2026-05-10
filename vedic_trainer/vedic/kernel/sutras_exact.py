"""All 29 sutras in exact ℚ arithmetic.

Every function is `Q16 -> Q16` (sometimes scalar / `Q16 × Q16 -> Q16`). The
formulas follow the spec table verbatim where it is unambiguous; for the
three entries that the spec marks as authoritative-from-simulator but does
not fully spell out (S6 e₀, S19 LopanaSthapanabhyam, S21 DhvajankaFlag),
the chosen interpretation is documented in ``docs/SUTRA_CATALOGUE.md``
and at the top of the function.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Tuple

from .q import Q16
from .tesseract import (
    BIT_WIDTH,
    COMPLEMENT,
    NUM_VERTICES,
    POPCOUNT,
    SHELLS,
    pairs_v_lt_complement,
    rotate_left_1,
)

# ----------------------------------------------------------------------
# 16 main sutras
# ----------------------------------------------------------------------


def s1_eka_adhikena(psi: Q16) -> Q16:
    """(S1 Ψ)_v = Ψ_{v ⊕ 0001}."""
    return tuple(psi[v ^ 0b0001] for v in range(NUM_VERTICES))


def s2_nikhilam(psi: Q16) -> Q16:
    """(S2 Ψ)_v = Ψ_{v̄}."""
    return tuple(psi[COMPLEMENT[v]] for v in range(NUM_VERTICES))


def s3_urdhva_tiryak(psi: Q16, phi: Q16) -> Q16:
    """(S3(Ψ, Φ))_v = Σ_{a ⊕ b = v} Ψ_a · Φ_b  (XOR convolution)."""
    out: list[Fraction] = [Fraction(0)] * NUM_VERTICES
    for a in range(NUM_VERTICES):
        psi_a = psi[a]
        if psi_a == 0:
            continue
        for b in range(NUM_VERTICES):
            phi_b = phi[b]
            if phi_b == 0:
                continue
            out[a ^ b] += psi_a * phi_b
    return tuple(out)


def s4_paravartya(psi: Q16) -> Q16:
    """(S4 Ψ)_v = Ψ_v − Ψ_{v ⊕ 0001}."""
    return tuple(psi[v] - psi[v ^ 0b0001] for v in range(NUM_VERTICES))


def s5_shunyam_samya(psi: Q16) -> Q16:
    """(S5 Ψ)_v = Ψ_v − (1/16) Σ_u Ψ_u  (centering / mean-subtraction)."""
    mean = sum(psi, Fraction(0)) / Fraction(NUM_VERTICES)
    return tuple(x - mean for x in psi)


def s6_anurupya_shunyam(psi: Q16) -> Q16:
    """(S6 Ψ)_v = Ψ_v − (⟨Ψ, e₀⟩ / ⟨e₀, e₀⟩) · e₀,v.

    Interpretation: ``e₀`` is the standard basis vector at vertex 0, so
    ⟨Ψ, e₀⟩ = Ψ_0 and ⟨e₀, e₀⟩ = 1. The result subtracts Ψ_0 from index 0
    only and leaves every other index unchanged.
    """
    psi0 = psi[0]
    return tuple(psi[v] - psi0 if v == 0 else psi[v] for v in range(NUM_VERTICES))


def s7_sankalana_vyavakalana(psi: Q16) -> Tuple[Q16, Q16]:
    """Return (S(Ψ), A(Ψ)) where
        S_v = (Ψ_v + Ψ_{v̄}) / 2   (symmetric part)
        A_v = (Ψ_v − Ψ_{v̄}) / 2   (antisymmetric part)
    """
    half = Fraction(1, 2)
    sym = tuple((psi[v] + psi[COMPLEMENT[v]]) * half for v in range(NUM_VERTICES))
    anti = tuple((psi[v] - psi[COMPLEMENT[v]]) * half for v in range(NUM_VERTICES))
    return sym, anti


def s8_puranapuranabhyam_fill(psi: Q16) -> Q16:
    """(S8 Ψ)_v = Ψ_v + Ψ_{v̄} if v < v̄ else 0."""
    out: list[Fraction] = [Fraction(0)] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        c = COMPLEMENT[v]
        if v < c:
            out[v] = psi[v] + psi[c]
    return tuple(out)


def s9_chalana_kalanabhyam(psi: Q16) -> Q16:
    """(S9 Ψ)_v = Σ_{k=0..3} (Ψ_{v ⊕ (1<<k)} − Ψ_v).

    This is the discrete Laplacian on the 4-cube (degree-4 graph).
    """
    out: list[Fraction] = [Fraction(0)] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        acc = Fraction(0)
        for k in range(BIT_WIDTH):
            acc += psi[v ^ (1 << k)] - psi[v]
        out[v] = acc
    return tuple(out)


def s10_yavadunam_tavadunikrtya(psi: Q16) -> Q16:
    """(S10 Ψ)_v = Ψ_v² − 2Ψ_v + 1 = (Ψ_v − 1)²."""
    return tuple((x - 1) * (x - 1) for x in psi)


def s11_vyasti_samasti(psi: Q16) -> Q16:
    """(S11 Ψ)_v = Ψ_v − (1/4) · shell_mean(v).

    ``shell_mean(v)`` averages Ψ over all u with ``popcount(u) = popcount(v)``.
    The constant 1/4 follows the spec verbatim and is independent of shell
    cardinality.
    """
    shell_means: list[Fraction] = []
    for shell in SHELLS:
        if not shell:
            shell_means.append(Fraction(0))
        else:
            total = sum((psi[u] for u in shell), Fraction(0))
            shell_means.append(total / Fraction(len(shell)))
    quarter = Fraction(1, 4)
    return tuple(psi[v] - quarter * shell_means[POPCOUNT[v]] for v in range(NUM_VERTICES))


def s12_shesanyankena_charamena(psi: Q16) -> Q16:
    """(S12 Ψ)_v = Ψ_v if (v & 0b1000) else 0."""
    return tuple(psi[v] if (v & 0b1000) else Fraction(0) for v in range(NUM_VERTICES))


def s13_sopantyadvayamantyam_last2(psi: Q16) -> Q16:
    """(S13 Ψ)_v = Ψ_v if (v & 0b1100) == 0b1100 else 0."""
    mask = 0b1100
    return tuple(psi[v] if (v & mask) == mask else Fraction(0) for v in range(NUM_VERTICES))


def s14_ekanyunena_purvena(psi: Q16) -> Q16:
    """(S14 Ψ)_v = Ψ_{(v − 1) mod 16}."""
    return tuple(psi[(v - 1) % NUM_VERTICES] for v in range(NUM_VERTICES))


def s15_gunitasamucchaya_product(psi: Q16) -> Q16:
    """(S15 Ψ)_v = 2^popcount(v) · Ψ_v."""
    return tuple(Fraction(1 << POPCOUNT[v]) * psi[v] for v in range(NUM_VERTICES))


def s16_gunaka_samucchaya(psi: Q16) -> Q16:
    """(S16 Ψ)_v = Ψ_v / 2^popcount(v)."""
    return tuple(psi[v] / Fraction(1 << POPCOUNT[v]) for v in range(NUM_VERTICES))


# ----------------------------------------------------------------------
# 13 sub-sutras (17 .. 29)
# ----------------------------------------------------------------------


def s17_anurupyena_proportion(psi: Q16, phi: Q16) -> Q16:
    """(S17(Ψ, Φ))_v = Ψ_v · Φ_0 / Ψ_0   (precondition: Ψ_0 ≠ 0).

    The kernel does not silently fall back; the precondition is asserted.
    """
    if psi[0] == 0:
        raise ValueError("S17 precondition violated: Ψ_0 must be non-zero")
    factor = phi[0] / psi[0]
    return tuple(x * factor for x in psi)


def s18_adyamadyena_antyamantyena(psi: Q16) -> Fraction:
    """S18(Ψ) = Ψ_0 · Ψ_15  (returns a scalar)."""
    return psi[0] * psi[NUM_VERTICES - 1]


def s19_lopana_sthapanabhyam(psi: Q16) -> Q16:
    """(S19 Ψ)_v = Ψ_v − Ψ_{v & 0b1110} + Ψ_{v | 0b0001}.

    Interpretation: the spec table is truncated at "Ψ_v − Ψ_{v & 0b1110}
    + Ψ_{v". The simulator authority is invoked to choose the third term.
    The natural completion (and the one that admits an explicit inverse on
    its image, required by S28) is to add the bit-0-set partner. This is
    documented in ``docs/SUTRA_CATALOGUE.md`` and verified by
    ``test_interaction_matrix.py``.

    Note: ``v & 0b1110`` clears bit 0; ``v | 0b0001`` sets bit 0. So the
    operator pairs each vertex with its bit-0-clear and bit-0-set siblings.
    """
    out: list[Fraction] = []
    for v in range(NUM_VERTICES):
        cleared = v & 0b1110
        forced = v | 0b0001
        out.append(psi[v] - psi[cleared] + psi[forced])
    return tuple(out)


def s20_vilokanam_spect(psi: Q16) -> Q16:
    """S20(Ψ) = ⟨Ψ, h₁⟩ · h₁  where h₁ is the first non-constant Walsh row.

    Convention: ``h₁[v] = (−1)^{(v >> 0) & 1}`` (sign by the lowest bit).
    The result is a Q16 vector (the rank-one projection of Ψ onto h₁).
    """
    h1 = tuple(Fraction(1 if not (v & 1) else -1) for v in range(NUM_VERTICES))
    inner = sum((psi[v] * h1[v] for v in range(NUM_VERTICES)), Fraction(0))
    norm_sq = Fraction(NUM_VERTICES)  # ⟨h₁, h₁⟩ = 16
    coeff = inner / norm_sq
    return tuple(coeff * h1[v] for v in range(NUM_VERTICES))


def s21_dhvajanka_flag(psi: Q16) -> Q16:
    """(S21 Ψ)_v = sgn(Ψ_v) · Ψ_v = |Ψ_v|  (absolute-value flag).

    The truncated spec line says ``sgn(Ψ_v) · Ψ_v (i.e. ``; we resolve to
    the absolute-value reading.
    """
    return tuple(abs(x) for x in psi)


def s22_parity_complement(psi: Q16) -> Tuple[Fraction, ...]:
    """S22(Ψ)_i = Ψ_{v_i} − Ψ_{v̄_i}   for i = 0..7.

    Output is length 8: one entry per (v, v̄) pair with v < v̄, in order of
    ascending v (i.e. v in (0, 1, 2, 3, 4, 5, 6, 7)).
    """
    return tuple(psi[v] - psi[c] for v, c in pairs_v_lt_complement())


def s23_dwandwa_yoga(psi: Q16, phi: Q16) -> Q16:
    """(S23(Ψ, Φ))_v = Ψ_v · Φ_{v̄} + Ψ_{v̄} · Φ_v."""
    return tuple(psi[v] * phi[COMPLEMENT[v]] + psi[COMPLEMENT[v]] * phi[v]
                 for v in range(NUM_VERTICES))


def s24_kevalaih_saptakam(psi: Q16) -> Q16:
    """(S24 Ψ)_v = Ψ_v if (v % 7 != 0) else 0.

    Zeros out indices v ∈ {0, 7, 14}; keeps the rest.
    """
    return tuple(psi[v] if (v % 7 != 0) else Fraction(0) for v in range(NUM_VERTICES))


def s25_vestana_circular(psi: Q16) -> Q16:
    """(S25 Ψ)_v = Ψ_{σ(v)}  where σ is bit-rotate-left-1 on the 4-bit field."""
    return tuple(psi[rotate_left_1(v)] for v in range(NUM_VERTICES))


def s26_yavadunam_square(psi: Q16) -> Q16:
    """(S26 Ψ)_v = Ψ_v²."""
    return tuple(x * x for x in psi)


def s27_samuccaya_gunitah(psi: Q16) -> Fraction:
    """S27(Ψ) = Π_{popcount(v) even} Ψ_v − Π_{popcount(v) odd} Ψ_v   (scalar)."""
    prod_even = Fraction(1)
    prod_odd = Fraction(1)
    for v in range(NUM_VERTICES):
        if POPCOUNT[v] % 2 == 0:
            prod_even *= psi[v]
        else:
            prod_odd *= psi[v]
    return prod_even - prod_odd


def s28_lopana_restore(psi: Q16) -> Q16:
    """S28 = inverse of S19 on im(S19).

    With S19 defined as
        (S19 Ψ)_v = Ψ_v − Ψ_{v & 0b1110} + Ψ_{v | 0b0001}

    the operator factors block-diagonally on the eight pairs (2k, 2k+1)
    for k = 0..7. On each pair the linear map is

        [ y_{2k}   ]   [  0   1 ] [ x_{2k}   ]
        [ y_{2k+1} ] = [  0   2 ] [ x_{2k+1} ]

    which is singular: the bit-0-clear column is zero. The image is the
    subspace where ``y_{2k} = (1/2) y_{2k+1}`` for every k. On that image
    the inverse picks the unique pre-image with ``x_{2k} = 0``, i.e.

        x_{2k}   = 0
        x_{2k+1} = y_{2k+1} / 2 = y_{2k}

    This is implemented exactly below; it is bit-exact-invertible on the
    image, and ``test_interaction_matrix.py`` verifies S28 ∘ S19 = identity
    on the projected pre-image (the subspace with bit-0-clear coordinate
    zero), which is the canonical right-inverse choice.
    """
    out: list[Fraction] = []
    for v in range(NUM_VERTICES):
        if v & 1:
            # odd index: x_{2k+1} = (y_{2k+1}) / 2
            out.append(psi[v] / Fraction(2))
        else:
            # even index: x_{2k} = 0  (canonical right-inverse choice)
            out.append(Fraction(0))
    return tuple(out)


def s29_mean_drive(psi: Q16) -> Q16:
    """(S29 Ψ)_v = (Ψ_v + mean(Ψ)) / 2.

    This is the simplified form noted at the bottom of the spec table; it
    is verified by ``test_conservation_laws.py`` to satisfy the
    mean-preservation residual R3 = 0 exactly.
    """
    mean = sum(psi, Fraction(0)) / Fraction(NUM_VERTICES)
    half = Fraction(1, 2)
    return tuple((x + mean) * half for x in psi)


# ----------------------------------------------------------------------
# Composition helpers used by losses + audit chain
# ----------------------------------------------------------------------


def s5_then_s11(psi: Q16) -> Q16:
    """(S11 ∘ S5)(Ψ) — used by L_dual and several conservation tests."""
    return s11_vyasti_samasti(s5_shunyam_samya(psi))


SUTRA_NAMES: Tuple[str, ...] = (
    "EkAdhikena",
    "NikhilamComplement",
    "UrdhvaTiryak",
    "ParavartyaYojayet",
    "ShunyamSamya",
    "AnurupyaShunyam",
    "SankalanaVyavakalana",
    "PuranapuranabhyamFill",
    "ChalanaKalanabhyam",
    "YavadunamTavadunikrtya",
    "VyastiSamasti",
    "ShesanyankenaCharamena",
    "SopantyadvayamantyamLast2",
    "EkanyunenaPurvena",
    "GunitasamucchayaProduct",
    "GunakaSamucchaya",
    "AnurupyenaProportion",
    "AdyamadyenaAntyamantyena",
    "LopanaSthapanabhyam",
    "VilokanamSpect",
    "DhvajankaFlag",
    "ParityComplement",
    "DwandwaYoga",
    "KevalaihSaptakam",
    "VestanaCircular",
    "YavadunamSquare",
    "SamuccayaGunitah",
    "LopanaRestore",
    "MeanDrive",
)
