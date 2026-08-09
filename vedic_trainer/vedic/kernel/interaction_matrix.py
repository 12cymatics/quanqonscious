"""30 verified sutra interaction pairs/triples.

Each entry encodes an algebraic identity that must hold bit-exactly over ℚ
on every input Ψ (or pair (Ψ, Φ)). The runtime check ``verify_identity``
evaluates both sides over Fraction and asserts equality. The identities
double as the catalogue of "things L_curv-style derived losses rely on
being exactly true", i.e. the structural backbone of the algebra.

Each identity is a pure function of inputs — there are no fail-safes; if
an identity fails the kernel formula is wrong and the tests fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Sequence, Tuple

from .q import Q16, q_eq
from .sutras_exact import (
    s1_eka_adhikena,
    s2_nikhilam,
    s3_urdhva_tiryak,
    s4_paravartya,
    s5_shunyam_samya,
    s6_anurupya_shunyam,
    s7_sankalana_vyavakalana,
    s8_puranapuranabhyam_fill,
    s9_chalana_kalanabhyam,
    s10_yavadunam_tavadunikrtya,
    s11_vyasti_samasti,
    s12_shesanyankena_charamena,
    s13_sopantyadvayamantyam_last2,
    s14_ekanyunena_purvena,
    s15_gunitasamucchaya_product,
    s16_gunaka_samucchaya,
    s19_lopana_sthapanabhyam,
    s20_vilokanam_spect,
    s21_dhvajanka_flag,
    s23_dwandwa_yoga,
    s24_kevalaih_saptakam,
    s25_vestana_circular,
    s26_yavadunam_square,
    s28_lopana_restore,
    s29_mean_drive,
)
from .tesseract import COMPLEMENT, NUM_VERTICES, POPCOUNT


@dataclass(frozen=True)
class Identity:
    name: str
    arity: int  # 1 = unary on Ψ, 2 = binary on (Ψ, Φ)
    check: Callable[..., bool]  # returns True iff the identity holds for the inputs


def _q16_zero() -> Q16:
    return tuple(Fraction(0) for _ in range(NUM_VERTICES))


# ----- 30 identities. The list is exposed; each `check` is a closure. -----


def _id_s1_period() -> Identity:
    return Identity("S1∘S1 = id (bit-0 toggle is involution)",
                    1,
                    lambda psi: q_eq(s1_eka_adhikena(s1_eka_adhikena(psi)), psi))


def _id_s2_involution() -> Identity:
    return Identity("S2∘S2 = id (complement is involution)",
                    1,
                    lambda psi: q_eq(s2_nikhilam(s2_nikhilam(psi)), psi))


def _id_s4_telescope() -> Identity:
    return Identity("S4 = (I − S1)",
                    1,
                    lambda psi: q_eq(s4_paravartya(psi),
                                      tuple(psi[v] - s1_eka_adhikena(psi)[v]
                                            for v in range(NUM_VERTICES))))


def _id_s5_idempotent() -> Identity:
    return Identity("S5∘S5 = S5 (centering is idempotent)",
                    1,
                    lambda psi: q_eq(s5_shunyam_samya(s5_shunyam_samya(psi)),
                                      s5_shunyam_samya(psi)))


def _id_s6_kills_index_zero() -> Identity:
    return Identity("(S6 Ψ)_0 = 0",
                    1,
                    lambda psi: s6_anurupya_shunyam(psi)[0] == Fraction(0))


def _id_s7_recovers() -> Identity:
    def chk(psi: Q16) -> bool:
        sym, anti = s7_sankalana_vyavakalana(psi)
        return all(sym[v] + anti[v] == psi[v] for v in range(NUM_VERTICES))
    return Identity("S(Ψ) + A(Ψ) = Ψ", 1, chk)


def _id_s8_v_complement_zero() -> Identity:
    return Identity("S8 zero on v > v̄ half",
                    1,
                    lambda psi: all(
                        s8_puranapuranabhyam_fill(psi)[v] == Fraction(0)
                        for v in range(NUM_VERTICES)
                        if v >= COMPLEMENT[v]
                    ))


def _id_s9_self_adjoint_zero_sum() -> Identity:
    return Identity("S9 has zero column sum (graph Laplacian property)",
                    1,
                    lambda psi: sum(s9_chalana_kalanabhyam(psi), Fraction(0)) == Fraction(0))


def _id_s10_nonneg() -> Identity:
    return Identity("S10 Ψ ≥ 0 elementwise",
                    1,
                    lambda psi: all(x >= 0 for x in s10_yavadunam_tavadunikrtya(psi)))


def _id_s11_constant_scales_by_three_quarters() -> Identity:
    """S11 of a constant Ψ = c·1 is (3c/4)·1.

    Reason: S11 Ψ_v = c − (1/4)·shell_mean(v) = c − c/4 = 3c/4 for every v.
    """
    def chk(_psi: Q16) -> bool:
        c = Fraction(7, 3)
        psi_const: Q16 = tuple(c for _ in range(NUM_VERTICES))
        out = s11_vyasti_samasti(psi_const)
        target = tuple(Fraction(3) * c / Fraction(4) for _ in range(NUM_VERTICES))
        return q_eq(out, target)
    return Identity("S11(c·1) = (3c/4)·1", 1, chk)


def _id_s12_keeps_high_half() -> Identity:
    return Identity("S12 zeros bit-3-clear half",
                    1,
                    lambda psi: all(
                        s12_shesanyankena_charamena(psi)[v] == Fraction(0)
                        for v in range(NUM_VERTICES)
                        if not (v & 0b1000)
                    ))


def _id_s13_keeps_top_quad() -> Identity:
    return Identity("S13 zeros outside (v & 0b1100)==0b1100",
                    1,
                    lambda psi: all(
                        s13_sopantyadvayamantyam_last2(psi)[v] == Fraction(0)
                        for v in range(NUM_VERTICES)
                        if (v & 0b1100) != 0b1100
                    ))


def _id_s14_period_16() -> Identity:
    def chk(psi: Q16) -> bool:
        x = psi
        for _ in range(NUM_VERTICES):
            x = s14_ekanyunena_purvena(x)
        return q_eq(x, psi)
    return Identity("S14^16 = id", 1, chk)


def _id_s15_s16_inverse() -> Identity:
    return Identity("S15 ∘ S16 = id (popcount scale is invertible)",
                    1,
                    lambda psi: q_eq(s15_gunitasamucchaya_product(s16_gunaka_samucchaya(psi)),
                                      psi))


def _id_s16_s15_inverse() -> Identity:
    return Identity("S16 ∘ S15 = id",
                    1,
                    lambda psi: q_eq(s16_gunaka_samucchaya(s15_gunitasamucchaya_product(psi)),
                                      psi))


def _id_s19_s28_left_inverse() -> Identity:
    """S28 is a left inverse of S19 on the canonical pre-image (bit-0-clear = 0).

    Choose x with x[2k] = 0 for all k. Then S19(x)[2k] = x[2k+1], and
    S19(x)[2k+1] = 2 x[2k+1]. Applying S28 yields x' with x'[2k] = 0 (by
    the canonical-right-inverse choice) and x'[2k+1] = (2 x[2k+1]) / 2 =
    x[2k+1], so x' = x bit-exactly.
    """
    def chk(psi: Q16) -> bool:
        # Project Ψ onto bit-0-clear-zero subspace.
        proj = tuple(Fraction(0) if not (v & 1) else psi[v] for v in range(NUM_VERTICES))
        return q_eq(s28_lopana_restore(s19_lopana_sthapanabhyam(proj)), proj)
    return Identity("S28 ∘ S19 = id on im(S19)", 1, chk)


def _id_s20_in_h1_span() -> Identity:
    """S20 maps into the rank-1 subspace spanned by h₁."""
    def chk(psi: Q16) -> bool:
        out = s20_vilokanam_spect(psi)
        # h₁[v] = (-1)^{v & 1}
        h1 = tuple(Fraction(1 if not (v & 1) else -1) for v in range(NUM_VERTICES))
        coeff = out[0] / h1[0]
        return all(out[v] == coeff * h1[v] for v in range(NUM_VERTICES))
    return Identity("S20 ⊂ span(h₁)", 1, chk)


def _id_s21_nonneg_idempotent() -> Identity:
    return Identity("S21 ∘ S21 = S21 and ≥ 0",
                    1,
                    lambda psi: q_eq(s21_dhvajanka_flag(s21_dhvajanka_flag(psi)),
                                      s21_dhvajanka_flag(psi))
                                 and all(x >= 0 for x in s21_dhvajanka_flag(psi)))


def _id_s23_symmetric() -> Identity:
    return Identity("S23(Ψ, Φ) = S23(Φ, Ψ)",
                    2,
                    lambda psi, phi: q_eq(s23_dwandwa_yoga(psi, phi), s23_dwandwa_yoga(phi, psi)))


def _id_s24_zeros_on_multiples_of_7() -> Identity:
    return Identity("S24 zeros indices 0, 7, 14",
                    1,
                    lambda psi: all(s24_kevalaih_saptakam(psi)[v] == Fraction(0)
                                     for v in (0, 7, 14)))


def _id_s25_period_4() -> Identity:
    def chk(psi: Q16) -> bool:
        x = psi
        for _ in range(4):  # rotate-left-1 has period 4 on 4 bits.
            x = s25_vestana_circular(x)
        return q_eq(x, psi)
    return Identity("S25^4 = id", 1, chk)


def _id_s26_nonneg() -> Identity:
    return Identity("S26 Ψ ≥ 0 elementwise",
                    1,
                    lambda psi: all(x >= 0 for x in s26_yavadunam_square(psi)))


def _id_s29_mean_preserved() -> Identity:
    def chk(psi: Q16) -> bool:
        m1 = sum(psi, Fraction(0)) / Fraction(NUM_VERTICES)
        post = s29_mean_drive(psi)
        m2 = sum(post, Fraction(0)) / Fraction(NUM_VERTICES)
        return m1 == m2
    return Identity("mean(S29 Ψ) = mean(Ψ)", 1, chk)


def _id_s5_orthogonal_to_constant() -> Identity:
    return Identity("Σ_v (S5 Ψ)_v = 0",
                    1,
                    lambda psi: sum(s5_shunyam_samya(psi), Fraction(0)) == Fraction(0))


def _id_s11_shell_sum_relation() -> Identity:
    """For every shell, sum_{u in shell}(S11 Ψ)_u = (3/4) · sum_{u in shell} Ψ_u.

    Reason: in shell K of size N, shell_mean = (Σ_{u∈K} Ψ_u)/N. So
        Σ_{u∈K}(S11 Ψ)_u = Σ Ψ_u − (1/4)·N·shell_mean = (3/4)·Σ Ψ_u.
    """
    def chk(psi: Q16) -> bool:
        from .tesseract import SHELLS
        post = s11_vyasti_samasti(psi)
        three_quarters = Fraction(3, 4)
        for shell in SHELLS:
            if not shell:
                continue
            sum_after = sum((post[u] for u in shell), Fraction(0))
            sum_before = sum((psi[u] for u in shell), Fraction(0))
            if sum_after != three_quarters * sum_before:
                return False
        return True
    return Identity("Σ_shell(S11 Ψ) = (3/4)·Σ_shell Ψ", 1, chk)


def _id_s7_orthogonality() -> Identity:
    def chk(psi: Q16) -> bool:
        sym, anti = s7_sankalana_vyavakalana(psi)
        ip = sum((sym[v] * anti[v] for v in range(NUM_VERTICES)), Fraction(0))
        return ip == Fraction(0)
    return Identity("⟨S(Ψ), A(Ψ)⟩ = 0", 1, chk)


def _id_s2_diagonalizes_complement() -> Identity:
    """S2 has eigenspaces +1 and -1 spanned by S(Ψ) and A(Ψ)."""
    def chk(psi: Q16) -> bool:
        sym, anti = s7_sankalana_vyavakalana(psi)
        return q_eq(s2_nikhilam(sym), sym) and q_eq(s2_nikhilam(anti),
                                                      tuple(-x for x in anti))
    return Identity("S2 acts as ±1 on S/A subspaces", 1, chk)


def _id_s9_kills_constant() -> Identity:
    return Identity("S9 of constant vector is zero",
                    1,
                    lambda _psi: q_eq(
                        s9_chalana_kalanabhyam(tuple(Fraction(5) for _ in range(NUM_VERTICES))),
                        _q16_zero(),
                    ))


def _id_s3_distributive() -> Identity:
    """S3 distributes over addition in the second argument."""
    def chk(psi: Q16, phi: Q16) -> bool:
        # Check S3(Ψ, Φ + Ψ) = S3(Ψ, Φ) + S3(Ψ, Ψ)
        sum_phi = tuple(phi[v] + psi[v] for v in range(NUM_VERTICES))
        lhs = s3_urdhva_tiryak(psi, sum_phi)
        rhs = tuple(s3_urdhva_tiryak(psi, phi)[v] + s3_urdhva_tiryak(psi, psi)[v]
                    for v in range(NUM_VERTICES))
        return q_eq(lhs, rhs)
    return Identity("S3 distributes over +", 2, chk)


def _id_s3_commutes() -> Identity:
    return Identity("S3 is commutative",
                    2,
                    lambda psi, phi: q_eq(s3_urdhva_tiryak(psi, phi),
                                           s3_urdhva_tiryak(phi, psi)))


# Public catalogue (30 identities).
INTERACTIONS: Tuple[Identity, ...] = (
    _id_s1_period(),
    _id_s2_involution(),
    _id_s4_telescope(),
    _id_s5_idempotent(),
    _id_s6_kills_index_zero(),
    _id_s7_recovers(),
    _id_s8_v_complement_zero(),
    _id_s9_self_adjoint_zero_sum(),
    _id_s10_nonneg(),
    _id_s11_constant_scales_by_three_quarters(),
    _id_s12_keeps_high_half(),
    _id_s13_keeps_top_quad(),
    _id_s14_period_16(),
    _id_s15_s16_inverse(),
    _id_s16_s15_inverse(),
    _id_s19_s28_left_inverse(),
    _id_s20_in_h1_span(),
    _id_s21_nonneg_idempotent(),
    _id_s23_symmetric(),
    _id_s24_zeros_on_multiples_of_7(),
    _id_s25_period_4(),
    _id_s26_nonneg(),
    _id_s29_mean_preserved(),
    _id_s5_orthogonal_to_constant(),
    _id_s11_shell_sum_relation(),
    _id_s7_orthogonality(),
    _id_s2_diagonalizes_complement(),
    _id_s9_kills_constant(),
    _id_s3_distributive(),
    _id_s3_commutes(),
)


def verify_all(psi: Q16, phi: Q16) -> Tuple[Tuple[str, bool], ...]:
    """Run every identity in INTERACTIONS against (Ψ, Φ) and return per-name status."""
    out: list[Tuple[str, bool]] = []
    for ident in INTERACTIONS:
        ok = ident.check(psi) if ident.arity == 1 else ident.check(psi, phi)
        out.append((ident.name, bool(ok)))
    return tuple(out)
