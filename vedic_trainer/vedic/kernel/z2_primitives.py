"""Primitive exact-ℚ maps on Z₂⁴ — NOT the 29 sutras.

**This module is not an implementation of the Vedic sutras.** The authority
for those is ``sutras_canonical``, which ports the engine's
``STRICT_SUTRA_KERNEL``. This file is kept because the conservation
residuals, the interaction-matrix catalogue, the synthetic-data encoder and
the committed fixtures are all built on these *unweighted* primitives, and
they are a different mathematical layer from the α-weighted sutra operators.

The two disagree, concretely:

* these maps carry no α weight, so none of them reduces to the identity as
  strength → 0, which is the §12Y guarantee the real sutras satisfy;
* the numbering conflicts — here S19 is LopanaSthapanabhyam and S24 is
  KevalaihSaptakam, while the engine has S19 = Kevalaih Saptakam Guṇyāt and
  S24 = Lopanasthāpanābhyām;
* canonical S7 is PERMUTATIVE (an axis reflection), whereas the primitive
  here is the symmetric/antisymmetric split that R4 needs.

The historical Sanskrit function names are retained only so the fixtures and
the interaction catalogue keep resolving. Do not read them as sutra
definitions. Anything that means "the 29 sutras" must import
``sutras_canonical``.

Every operand is an explicit parameter with a named canonical default; the
values reproduce the committed fixtures bit-for-bit. No epsilons, no clamps,
no silent fallbacks: preconditions raise.
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
    rotate_left_k,
    xor_pairs_lt,
)

# ----------------------------------------------------------------------
# Canonical operands
#
# The values here are the canonical spec settings the committed fixtures
# were built from. They are named constants, never inline literals, and
# every one can be overridden per call.
# ----------------------------------------------------------------------

FULL_MASK: int = 0b1111
"""Complement involution v ↦ v̄ — the pairing operand of S2/S7/S8/S22/S23."""

S1_MASK: int = 0b0001
"""EkAdhikena: 'by one more than the previous' — the successor XOR step."""

S4_MASK: int = 0b0001
"""ParavartyaYojayet: the difference partner offset."""

S6_REF: int = 0
"""AnurupyaShunyam: index of the basis vector e_ref projected out."""

S9_AXES: Tuple[int, ...] = tuple(range(BIT_WIDTH))
"""ChalanaKalanabhyam: axes summed by the discrete Laplacian."""

S10_BASE: Fraction = Fraction(1)
"""YavadunamTavadunikrtya: 'by the deficiency from the base' — the base."""

S11_WEIGHT: Fraction = Fraction(1, 4)
"""VyastiSamasti: shell-mean subtraction weight (spec verbatim)."""

S12_MASK: int = 0b1000
"""ShesanyankenaCharamena: retained-bit selector."""

S13_MASK: int = 0b1100
"""SopantyadvayamantyamLast2: 'last two' — every bit in the mask must be set."""

S14_SHIFT: int = -1
"""EkanyunenaPurvena: 'by one less than the previous' — the index shift."""

S15_BASE: Fraction = Fraction(2)
"""GunitasamucchayaProduct: per-set-bit multiplier."""

S16_BASE: Fraction = Fraction(2)
"""GunakaSamucchaya: per-set-bit divisor."""

S17_REF: int = 0
"""AnurupyenaProportion: index the proportion is taken at."""

S18_I: int = 0
"""AdyamadyenaAntyamantyena: 'the first' index."""

S18_J: int = NUM_VERTICES - 1
"""AdyamadyenaAntyamantyena: 'the last' index."""

S19_MASK: int = 0b0001
"""LopanaSthapanabhyam: elimination/retention bit (S28 inverts on this mask)."""

S20_AXIS: int = 0
"""VilokanamSpect: which Walsh axis row h_axis is projected onto."""

S24_MODULUS: int = 7
"""KevalaihSaptakam: 'only the sevens' — the annihilated residue class."""

S25_ROT: int = 1
"""VestanaCircular: bit-rotation amount."""

S29_WEIGHT: Fraction = Fraction(1, 2)
"""MeanDrive: blend weight toward the mean."""


def _check_mask(mask: int, what: str) -> None:
    if not 0 <= mask < NUM_VERTICES:
        raise ValueError(f"{what} out of range for Z₂⁴: {mask}")


def _check_index(idx: int, what: str) -> None:
    if not 0 <= idx < NUM_VERTICES:
        raise ValueError(f"{what} out of range: {idx}")


def _mean(psi: Q16) -> Fraction:
    return sum(psi, Fraction(0)) / Fraction(NUM_VERTICES)


# ----------------------------------------------------------------------
# 16 main sutras
# ----------------------------------------------------------------------


def s1_eka_adhikena(psi: Q16, mask: int = S1_MASK) -> Q16:
    """(S1 Ψ)_v = Ψ_{v ⊕ mask}.  Canonical mask = 0b0001 ('one more')."""
    _check_mask(mask, "S1 mask")
    return tuple(psi[v ^ mask] for v in range(NUM_VERTICES))


def s2_nikhilam(psi: Q16, mask: int = FULL_MASK) -> Q16:
    """(S2 Ψ)_v = Ψ_{v ⊕ mask}.  Canonical mask = 0b1111 (full complement v̄)."""
    _check_mask(mask, "S2 mask")
    return tuple(psi[v ^ mask] for v in range(NUM_VERTICES))


def s3_urdhva_tiryak(psi: Q16, phi: Q16) -> Q16:
    """(S3(Ψ, Φ))_v = Σ_{a ⊕ b = v} Ψ_a · Φ_b  (XOR convolution).

    Genuinely binary: 'vertically and crosswise' multiplies two operands.
    """
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


def s4_paravartya(psi: Q16, mask: int = S4_MASK) -> Q16:
    """(S4 Ψ)_v = Ψ_v − Ψ_{v ⊕ mask}.  'Transpose and apply.'"""
    _check_mask(mask, "S4 mask")
    return tuple(psi[v] - psi[v ^ mask] for v in range(NUM_VERTICES))


def s5_shunyam_samya(psi: Q16) -> Q16:
    """(S5 Ψ)_v = Ψ_v − (1/16) Σ_u Ψ_u  (centering / mean-subtraction).

    'When the samuccaya is the same, that samuccaya is zero': the operand is
    the whole vector, so there is no free index to expose.
    """
    mean = _mean(psi)
    return tuple(x - mean for x in psi)


def s6_anurupya_shunyam(psi: Q16, ref: int = S6_REF) -> Q16:
    """(S6 Ψ)_v = Ψ_v − (⟨Ψ, e_ref⟩ / ⟨e_ref, e_ref⟩) · e_ref,v.

    Interpretation: ``e_ref`` is the standard basis vector at ``ref``, so
    ⟨Ψ, e_ref⟩ = Ψ_ref and ⟨e_ref, e_ref⟩ = 1. The result subtracts Ψ_ref at
    index ``ref`` only and leaves every other index unchanged. Canonical
    ref = 0 reproduces the original e₀ reading.
    """
    _check_index(ref, "S6 ref")
    pref = psi[ref]
    return tuple(psi[v] - pref if v == ref else psi[v] for v in range(NUM_VERTICES))


def s7_sankalana_vyavakalana(psi: Q16, mask: int = FULL_MASK) -> Tuple[Q16, Q16]:
    """Return (S(Ψ), A(Ψ)) about the involution v ↦ v ⊕ mask:
        S_v = (Ψ_v + Ψ_{v⊕mask}) / 2   (symmetric part)
        A_v = (Ψ_v − Ψ_{v⊕mask}) / 2   (antisymmetric part)
    """
    _check_mask(mask, "S7 mask")
    half = Fraction(1, 2)
    sym = tuple((psi[v] + psi[v ^ mask]) * half for v in range(NUM_VERTICES))
    anti = tuple((psi[v] - psi[v ^ mask]) * half for v in range(NUM_VERTICES))
    return sym, anti


def s8_puranapuranabhyam_fill(psi: Q16, mask: int = FULL_MASK) -> Q16:
    """(S8 Ψ)_v = Ψ_v + Ψ_{v⊕mask} if v < v⊕mask else 0.  'By completion.'"""
    _check_mask(mask, "S8 mask")
    out: list[Fraction] = [Fraction(0)] * NUM_VERTICES
    for v, c in xor_pairs_lt(mask):
        out[v] = psi[v] + psi[c]
    return tuple(out)


def s9_chalana_kalanabhyam(psi: Q16, axes: Tuple[int, ...] = S9_AXES) -> Q16:
    """(S9 Ψ)_v = Σ_{k ∈ axes} (Ψ_{v ⊕ (1<<k)} − Ψ_v).

    The discrete Laplacian on the sub-cube spanned by ``axes``. Canonical
    axes = (0,1,2,3) gives the full degree-4 4-cube Laplacian.
    """
    for k in axes:
        if not 0 <= k < BIT_WIDTH:
            raise ValueError(f"S9 axis out of range: {k}")
    out: list[Fraction] = [Fraction(0)] * NUM_VERTICES
    for v in range(NUM_VERTICES):
        acc = Fraction(0)
        for k in axes:
            acc += psi[v ^ (1 << k)] - psi[v]
        out[v] = acc
    return tuple(out)


def s10_yavadunam_tavadunikrtya(psi: Q16, base: Fraction = S10_BASE) -> Q16:
    """(S10 Ψ)_v = (Ψ_v − base)².  'Whatever the deficiency from the base.'"""
    return tuple((x - base) * (x - base) for x in psi)


def s11_vyasti_samasti(psi: Q16, weight: Fraction = S11_WEIGHT) -> Q16:
    """(S11 Ψ)_v = Ψ_v − weight · shell_mean(v).

    ``shell_mean(v)`` averages Ψ over all u with ``popcount(u) = popcount(v)``.
    The canonical weight 1/4 follows the spec verbatim and is independent of
    shell cardinality. 'Part and whole.'
    """
    shell_means: list[Fraction] = []
    for shell in SHELLS:
        if not shell:
            shell_means.append(Fraction(0))
        else:
            total = sum((psi[u] for u in shell), Fraction(0))
            shell_means.append(total / Fraction(len(shell)))
    return tuple(psi[v] - weight * shell_means[POPCOUNT[v]] for v in range(NUM_VERTICES))


def s12_shesanyankena_charamena(psi: Q16, mask: int = S12_MASK) -> Q16:
    """(S12 Ψ)_v = Ψ_v if (v & mask) else 0.  'The remainders by the last.'"""
    _check_mask(mask, "S12 mask")
    return tuple(psi[v] if (v & mask) else Fraction(0) for v in range(NUM_VERTICES))


def s13_sopantyadvayamantyam_last2(psi: Q16, mask: int = S13_MASK) -> Q16:
    """(S13 Ψ)_v = Ψ_v if (v & mask) == mask else 0.  'The ultimate and twice
    the penultimate' — every bit of ``mask`` must be present."""
    _check_mask(mask, "S13 mask")
    return tuple(psi[v] if (v & mask) == mask else Fraction(0)
                 for v in range(NUM_VERTICES))


def s14_ekanyunena_purvena(psi: Q16, shift: int = S14_SHIFT) -> Q16:
    """(S14 Ψ)_v = Ψ_{(v + shift) mod 16}.  Canonical shift = −1 ('one less')."""
    return tuple(psi[(v + shift) % NUM_VERTICES] for v in range(NUM_VERTICES))


def s15_gunitasamucchaya_product(psi: Q16, base: Fraction = S15_BASE) -> Q16:
    """(S15 Ψ)_v = base^popcount(v) · Ψ_v.  'The product of the sum.'"""
    return tuple(base ** POPCOUNT[v] * psi[v] for v in range(NUM_VERTICES))


def s16_gunaka_samucchaya(psi: Q16, base: Fraction = S16_BASE) -> Q16:
    """(S16 Ψ)_v = Ψ_v / base^popcount(v).  'The sum of the product.'"""
    if base == 0:
        raise ValueError("S16 base must be non-zero")
    return tuple(psi[v] / base ** POPCOUNT[v] for v in range(NUM_VERTICES))


# ----------------------------------------------------------------------
# 13 sub-sutras (17 .. 29)
# ----------------------------------------------------------------------


def s17_anurupyena_proportion(psi: Q16, phi: Q16, ref: int = S17_REF) -> Q16:
    """(S17(Ψ, Φ))_v = Ψ_v · Φ_ref / Ψ_ref   (precondition: Ψ_ref ≠ 0).

    'Proportionately.' The kernel does not silently fall back; the
    precondition is asserted.
    """
    _check_index(ref, "S17 ref")
    if psi[ref] == 0:
        raise ValueError(f"S17 precondition violated: Ψ_{ref} must be non-zero")
    factor = phi[ref] / psi[ref]
    return tuple(x * factor for x in psi)


def s18_adyamadyena_antyamantyena(psi: Q16, i: int = S18_I, j: int = S18_J) -> Fraction:
    """S18(Ψ) = Ψ_i · Ψ_j  (scalar).  'The first by the first, the last by
    the last' — canonical (i, j) = (0, 15)."""
    _check_index(i, "S18 i")
    _check_index(j, "S18 j")
    return psi[i] * psi[j]


def s19_lopana_sthapanabhyam(psi: Q16, mask: int = S19_MASK) -> Q16:
    """(S19 Ψ)_v = Ψ_v − Ψ_{v & ~mask} + Ψ_{v | mask}.

    Interpretation: the spec table is truncated at "Ψ_v − Ψ_{v & 0b1110}
    + Ψ_{v". The simulator authority is invoked to choose the third term.
    The natural completion (and the one that admits an explicit inverse on
    its image, required by S28) is to add the mask-set partner. Documented
    in ``docs/SUTRA_CATALOGUE.md`` and verified by
    ``test_interaction_matrix.py``.

    ``v & ~mask`` clears the mask bits; ``v | mask`` sets them. So the
    operator pairs each vertex with its mask-clear and mask-set siblings.
    Canonical mask = 0b0001 reproduces the original 0b1110 / 0b0001 reading.
    """
    _check_mask(mask, "S19 mask")
    inv = (~mask) & FULL_MASK
    out: list[Fraction] = []
    for v in range(NUM_VERTICES):
        out.append(psi[v] - psi[v & inv] + psi[v | mask])
    return tuple(out)


def s20_vilokanam_spect(psi: Q16, axis: int = S20_AXIS) -> Q16:
    """S20(Ψ) = ⟨Ψ, h_axis⟩ / ⟨h_axis, h_axis⟩ · h_axis.

    ``h_axis[v] = (−1)^{(v >> axis) & 1}`` is the Walsh row for that axis.
    'By mere observation' — the rank-one projection of Ψ onto h_axis.
    Canonical axis = 0 reproduces the original h₁ (lowest-bit) reading.
    """
    if not 0 <= axis < BIT_WIDTH:
        raise ValueError(f"S20 axis out of range: {axis}")
    h = tuple(Fraction(1 if not ((v >> axis) & 1) else -1) for v in range(NUM_VERTICES))
    inner = sum((psi[v] * h[v] for v in range(NUM_VERTICES)), Fraction(0))
    norm_sq = Fraction(NUM_VERTICES)  # ⟨h, h⟩ = 16 for every axis
    coeff = inner / norm_sq
    return tuple(coeff * h[v] for v in range(NUM_VERTICES))


def s21_dhvajanka_flag(psi: Q16) -> Q16:
    """(S21 Ψ)_v = sgn(Ψ_v) · Ψ_v = |Ψ_v|  (absolute-value flag).

    The truncated spec line says ``sgn(Ψ_v) · Ψ_v (i.e. ``; we resolve to
    the absolute-value reading. 'On the flag' — no free operand.
    """
    return tuple(abs(x) for x in psi)


def s22_parity_complement(psi: Q16, mask: int = FULL_MASK) -> Tuple[Fraction, ...]:
    """S22(Ψ)_i = Ψ_{v_i} − Ψ_{v_i ⊕ mask}  over the 8 pairs with v < v⊕mask,
    ascending in v. Output length 8."""
    _check_mask(mask, "S22 mask")
    return tuple(psi[v] - psi[c] for v, c in xor_pairs_lt(mask))


def s23_dwandwa_yoga(psi: Q16, phi: Q16, mask: int = FULL_MASK) -> Q16:
    """(S23(Ψ, Φ))_v = Ψ_v · Φ_{v⊕mask} + Ψ_{v⊕mask} · Φ_v.  'Duplex.'

    Genuinely binary; ``mask`` selects the duplex partner involution.
    """
    _check_mask(mask, "S23 mask")
    return tuple(psi[v] * phi[v ^ mask] + psi[v ^ mask] * phi[v]
                 for v in range(NUM_VERTICES))


def s24_kevalaih_saptakam(psi: Q16, modulus: int = S24_MODULUS) -> Q16:
    """(S24 Ψ)_v = Ψ_v if (v % modulus != 0) else 0.

    'Only the sevens.' Canonical modulus 7 zeroes v ∈ {0, 7, 14}.
    """
    if modulus <= 0:
        raise ValueError(f"S24 modulus must be positive: {modulus}")
    return tuple(psi[v] if (v % modulus != 0) else Fraction(0)
                 for v in range(NUM_VERTICES))


def s25_vestana_circular(psi: Q16, k: int = S25_ROT) -> Q16:
    """(S25 Ψ)_v = Ψ_{σ_k(v)} where σ_k is bit-rotate-left-k on the 4-bit
    field. 'Osculation' — canonical k = 1."""
    return tuple(psi[rotate_left_k(v, k)] for v in range(NUM_VERTICES))


def s26_yavadunam_square(psi: Q16) -> Q16:
    """(S26 Ψ)_v = Ψ_v².  Squaring — no free operand."""
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


def s28_lopana_restore(psi: Q16, mask: int = S19_MASK) -> Q16:
    """S28 = inverse of S19 on im(S19), for the same ``mask``.

    With S19 defined as
        (S19 Ψ)_v = Ψ_v − Ψ_{v & ~mask} + Ψ_{v | mask}

    the operator factors block-diagonally on the eight (mask-clear,
    mask-set) partner pairs. On each pair the linear map is

        [ y_clear ]   [  0   1 ] [ x_clear ]
        [ y_set   ] = [  0   2 ] [ x_set   ]

    which is singular: the mask-clear column is zero. The image is the
    subspace where ``y_clear = (1/2) y_set``. On that image the inverse
    picks the unique pre-image with ``x_clear = 0``, i.e.

        x_clear = 0
        x_set   = y_set / 2 = y_clear

    Bit-exactly invertible on the image; ``test_interaction_matrix.py``
    verifies S28 ∘ S19 = identity on the projected pre-image.
    """
    _check_mask(mask, "S28 mask")
    out: list[Fraction] = []
    for v in range(NUM_VERTICES):
        if v & mask:
            out.append(psi[v] / Fraction(2))
        else:
            out.append(Fraction(0))
    return tuple(out)


def s29_mean_drive(psi: Q16, weight: Fraction = S29_WEIGHT) -> Q16:
    """(S29 Ψ)_v = (1 − weight) · Ψ_v + weight · mean(Ψ).

    Canonical weight = 1/2 gives the spec's (Ψ_v + mean)/2. Verified by
    ``test_conservation_laws.py`` to satisfy R3 = 0 exactly.
    """
    mean = _mean(psi)
    one = Fraction(1)
    return tuple((one - weight) * x + weight * mean for x in psi)


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
