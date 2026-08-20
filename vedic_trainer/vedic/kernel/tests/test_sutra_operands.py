"""Every sutra operand is explicit, live, and validated.

Two properties are asserted for each parameterised sutra:

1. **Canonical fidelity** — calling with the canonical operand reproduces the
   pre-parameterisation behaviour, so the committed fixtures stay bit-exact.
2. **Liveness** — a non-canonical operand changes the result. An operand that
   cannot change anything is decorative, and this test fails on it.

Out-of-domain operands must raise rather than clamp or fall back.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

import vedic.kernel.z2_primitives as S

PSI = tuple(Fraction(v * v + 1, 3) for v in range(16))
# A second, independent operand. S17/S23 are binary; exercising their operands
# requires Phi != Psi (see test_s17_with_phi_equal_psi_is_the_identity).
PHI = tuple(Fraction(3 * v + 2, 5) for v in range(16))

# (label, canonical call, non-canonical call)
LIVE_CASES = [
    ("s1.mask",     lambda: S.s1_eka_adhikena(PSI),                 lambda: S.s1_eka_adhikena(PSI, 0b0100)),
    ("s2.mask",     lambda: S.s2_nikhilam(PSI),                     lambda: S.s2_nikhilam(PSI, 0b0011)),
    ("s4.mask",     lambda: S.s4_paravartya(PSI),                   lambda: S.s4_paravartya(PSI, 0b1000)),
    ("s6.ref",      lambda: S.s6_anurupya_shunyam(PSI),             lambda: S.s6_anurupya_shunyam(PSI, 9)),
    ("s7.mask",     lambda: S.s7_sankalana_vyavakalana(PSI),        lambda: S.s7_sankalana_vyavakalana(PSI, 0b0001)),
    ("s8.mask",     lambda: S.s8_puranapuranabhyam_fill(PSI),       lambda: S.s8_puranapuranabhyam_fill(PSI, 0b0001)),
    ("s9.axes",     lambda: S.s9_chalana_kalanabhyam(PSI),          lambda: S.s9_chalana_kalanabhyam(PSI, (0,))),
    ("s10.base",    lambda: S.s10_yavadunam_tavadunikrtya(PSI),     lambda: S.s10_yavadunam_tavadunikrtya(PSI, Fraction(5))),
    ("s11.weight",  lambda: S.s11_vyasti_samasti(PSI),              lambda: S.s11_vyasti_samasti(PSI, Fraction(1, 2))),
    ("s12.mask",    lambda: S.s12_shesanyankena_charamena(PSI),     lambda: S.s12_shesanyankena_charamena(PSI, 0b0001)),
    ("s13.mask",    lambda: S.s13_sopantyadvayamantyam_last2(PSI),  lambda: S.s13_sopantyadvayamantyam_last2(PSI, 0b0011)),
    ("s14.shift",   lambda: S.s14_ekanyunena_purvena(PSI),          lambda: S.s14_ekanyunena_purvena(PSI, 3)),
    ("s15.base",    lambda: S.s15_gunitasamucchaya_product(PSI),    lambda: S.s15_gunitasamucchaya_product(PSI, Fraction(3))),
    ("s16.base",    lambda: S.s16_gunaka_samucchaya(PSI),           lambda: S.s16_gunaka_samucchaya(PSI, Fraction(4))),
    ("s17.ref",     lambda: S.s17_anurupyena_proportion(PSI, PHI),  lambda: S.s17_anurupyena_proportion(PSI, PHI, 9)),
    ("s18.indices", lambda: S.s18_adyamadyena_antyamantyena(PSI),   lambda: S.s18_adyamadyena_antyamantyena(PSI, 2, 5)),
    ("s19.mask",    lambda: S.s19_lopana_sthapanabhyam(PSI),        lambda: S.s19_lopana_sthapanabhyam(PSI, 0b0010)),
    ("s20.axis",    lambda: S.s20_vilokanam_spect(PSI),             lambda: S.s20_vilokanam_spect(PSI, 2)),
    ("s22.mask",    lambda: S.s22_parity_complement(PSI),           lambda: S.s22_parity_complement(PSI, 0b0001)),
    ("s23.mask",    lambda: S.s23_dwandwa_yoga(PSI, PHI),           lambda: S.s23_dwandwa_yoga(PSI, PHI, 0b0001)),
    ("s24.modulus", lambda: S.s24_kevalaih_saptakam(PSI),           lambda: S.s24_kevalaih_saptakam(PSI, 3)),
    ("s25.rot",     lambda: S.s25_vestana_circular(PSI),            lambda: S.s25_vestana_circular(PSI, 2)),
    ("s28.mask",    lambda: S.s28_lopana_restore(PSI),              lambda: S.s28_lopana_restore(PSI, 0b0010)),
    ("s29.weight",  lambda: S.s29_mean_drive(PSI),                  lambda: S.s29_mean_drive(PSI, Fraction(1, 5))),
]


@pytest.mark.parametrize("label,canonical,other", LIVE_CASES,
                         ids=[c[0] for c in LIVE_CASES])
def test_operand_is_live(label, canonical, other):
    """A non-canonical operand must change the result."""
    assert canonical() != other(), f"{label} is decorative: it changes nothing"


def test_canonical_defaults_match_the_named_constants():
    """The defaults are the named spec constants, not stray literals."""
    assert S.FULL_MASK == 0b1111
    assert S.S1_MASK == 0b0001
    assert S.S4_MASK == 0b0001
    assert S.S6_REF == 0
    assert S.S9_AXES == (0, 1, 2, 3)
    assert S.S10_BASE == Fraction(1)
    assert S.S11_WEIGHT == Fraction(1, 4)
    assert S.S12_MASK == 0b1000
    assert S.S13_MASK == 0b1100
    assert S.S14_SHIFT == -1
    assert S.S15_BASE == Fraction(2)
    assert S.S16_BASE == Fraction(2)
    assert S.S17_REF == 0
    assert (S.S18_I, S.S18_J) == (0, 15)
    assert S.S19_MASK == 0b0001
    assert S.S20_AXIS == 0
    assert S.S24_MODULUS == 7
    assert S.S25_ROT == 1
    assert S.S29_WEIGHT == Fraction(1, 2)


def test_explicit_canonical_equals_default():
    """Passing the canonical operand explicitly is identical to omitting it."""
    assert S.s1_eka_adhikena(PSI) == S.s1_eka_adhikena(PSI, S.S1_MASK)
    assert S.s11_vyasti_samasti(PSI) == S.s11_vyasti_samasti(PSI, S.S11_WEIGHT)
    assert S.s24_kevalaih_saptakam(PSI) == S.s24_kevalaih_saptakam(PSI, S.S24_MODULUS)
    assert S.s25_vestana_circular(PSI) == S.s25_vestana_circular(PSI, S.S25_ROT)
    assert S.s20_vilokanam_spect(PSI) == S.s20_vilokanam_spect(PSI, S.S20_AXIS)


# ------------------------------------------------------------ domain checks


@pytest.mark.parametrize("call", [
    lambda: S.s1_eka_adhikena(PSI, 16),
    lambda: S.s2_nikhilam(PSI, -1),
    lambda: S.s4_paravartya(PSI, 99),
    lambda: S.s6_anurupya_shunyam(PSI, 16),
    lambda: S.s12_shesanyankena_charamena(PSI, 16),
    lambda: S.s13_sopantyadvayamantyam_last2(PSI, 16),
    lambda: S.s19_lopana_sthapanabhyam(PSI, 16),
    lambda: S.s28_lopana_restore(PSI, 16),
])
def test_out_of_range_mask_raises(call):
    with pytest.raises(ValueError):
        call()


def test_out_of_range_axis_raises():
    with pytest.raises(ValueError):
        S.s20_vilokanam_spect(PSI, 4)
    with pytest.raises(ValueError):
        S.s9_chalana_kalanabhyam(PSI, (0, 7))


def test_degenerate_scalar_operands_raise():
    with pytest.raises(ValueError):
        S.s16_gunaka_samucchaya(PSI, Fraction(0))
    with pytest.raises(ValueError):
        S.s24_kevalaih_saptakam(PSI, 0)


def test_s18_index_bounds_are_checked():
    with pytest.raises(ValueError):
        S.s18_adyamadyena_antyamantyena(PSI, 0, 16)


# ------------------------------------------------------- structural identities


def test_s1_and_s2_are_the_same_rule_at_different_masks():
    """Both are XOR translations; only the operand distinguishes them."""
    assert S.s1_eka_adhikena(PSI, S.FULL_MASK) == S.s2_nikhilam(PSI, S.FULL_MASK)
    assert S.s2_nikhilam(PSI, S.S1_MASK) == S.s1_eka_adhikena(PSI, S.S1_MASK)


def test_s25_rotation_is_cyclic_of_order_four():
    out = PSI
    for _ in range(4):
        out = S.s25_vestana_circular(out, 1)
    assert out == PSI


def test_s25_zero_rotation_is_identity():
    assert S.s25_vestana_circular(PSI, 0) == PSI


def test_s7_parts_reconstruct_the_input_for_any_mask():
    for mask in (0b1111, 0b0001, 0b0110):
        sym, anti = S.s7_sankalana_vyavakalana(PSI, mask)
        assert tuple(a + b for a, b in zip(sym, anti)) == PSI


def test_s29_weight_zero_is_identity_and_one_is_the_mean():
    assert S.s29_mean_drive(PSI, Fraction(0)) == PSI
    out = S.s29_mean_drive(PSI, Fraction(1))
    mean = sum(PSI, Fraction(0)) / Fraction(16)
    assert all(x == mean for x in out)


def test_s17_with_phi_equal_psi_is_the_identity():
    """S17(Ψ, Ψ) = Ψ · Ψ_ref/Ψ_ref = Ψ, for every ref.

    This matters for composition: the registry binds Φ = Ψ to give S17 a
    unary form, and that binding makes S17 the identity operator. It
    contributes nothing to a SERIES chain and only dilutes a PARALLEL mean.
    """
    for ref in (0, 1, 7, 15):
        assert S.s17_anurupyena_proportion(PSI, PSI, ref) == PSI


def test_s17_with_distinct_phi_is_not_the_identity():
    assert S.s17_anurupyena_proportion(PSI, PHI) != PSI
