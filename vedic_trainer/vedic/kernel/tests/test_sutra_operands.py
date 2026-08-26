"""Every sutra operand is explicit, live, validated, and exercised in full.

Four properties are asserted for **every one of the 25 optional operands**,
over the whole Ψ corpus:

1. **Canonical fidelity** — passing the canonical operand explicitly is
   identical to omitting it, so the committed fixtures stay bit-exact.
2. **Liveness** — a non-canonical operand changes the result. An operand
   that cannot change anything is decorative.
3. **Domain** — out-of-domain operands raise rather than clamp or fall back.
4. **Completeness** — the operand table below is checked against runtime
   introspection of the module, so an operand added later cannot go
   untested by being forgotten here.

What this replaced
------------------
* Liveness ran on **one** Ψ, so an operand that is inert on every field
  except that one would have passed.
* Canonical fidelity was checked for **5 of the 25** operands (S1, S11,
  S20, S24, S25). The other twenty were never compared against their own
  default.
* S18 takes two operands, ``i`` and ``j``. They were varied *together* in a
  single case, so an operand that was dead would still have passed as long
  as the other one was live.
* ``test_s7_parts_reconstruct_the_input_for_any_mask`` said "for any mask"
  and tested three of the sixteen; ``test_s17_with_phi_equal_psi_is_the_
  identity`` said "for every ref" and tested four of the sixteen.
"""
from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

import vedic.kernel.z2_primitives as S
from vedic.kernel.tests.psi_corpus import BY_LABEL, PHI, PSI_CASES

# The Ψ the previous version of this file used, so the structural identities
# below can still be read against a familiar vector.
PSI = tuple(Fraction(v * v + 1, 3) for v in range(16))

ALL_MASKS = tuple(range(16))
NONZERO_MASKS = tuple(range(1, 16))
ALL_REFS = tuple(range(16))


# ─────────────────────────────────────────────────────── the operand table
# (function, parameter, canonical value, in-domain alternatives, extra args)
#
# Alternatives are exhaustive where the domain is small enough to enumerate
# (all 16 masks, all 16 refs) and representative where it is not (scalars).
OPERANDS: tuple[tuple, ...] = (
    (S.s1_eka_adhikena, "mask", S.S1_MASK, ALL_MASKS, ()),
    (S.s2_nikhilam, "mask", S.FULL_MASK, ALL_MASKS, ()),
    (S.s4_paravartya, "mask", S.S4_MASK, ALL_MASKS, ()),
    (S.s6_anurupya_shunyam, "ref", S.S6_REF, ALL_REFS, ()),
    (S.s7_sankalana_vyavakalana, "mask", S.FULL_MASK, ALL_MASKS, ()),
    (S.s8_puranapuranabhyam_fill, "mask", S.FULL_MASK, ALL_MASKS, ()),
    (S.s9_chalana_kalanabhyam, "axes", S.S9_AXES,
     ((0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 3), (0, 1, 2)), ()),
    (S.s10_yavadunam_tavadunikrtya, "base", S.S10_BASE,
     (Fraction(0), Fraction(2), Fraction(5), Fraction(-3), Fraction(1, 4)), ()),
    (S.s11_vyasti_samasti, "weight", S.S11_WEIGHT,
     (Fraction(0), Fraction(1), Fraction(1, 2), Fraction(-1, 3), Fraction(7, 5)), ()),
    (S.s12_shesanyankena_charamena, "mask", S.S12_MASK, ALL_MASKS, ()),
    (S.s13_sopantyadvayamantyam_last2, "mask", S.S13_MASK, ALL_MASKS, ()),
    (S.s14_ekanyunena_purvena, "shift", S.S14_SHIFT,
     (-3, -2, 1, 2, 3, 7, 15), ()),
    (S.s15_gunitasamucchaya_product, "base", S.S15_BASE,
     (Fraction(1), Fraction(3), Fraction(1, 2), Fraction(-2)), ()),
    (S.s16_gunaka_samucchaya, "base", S.S16_BASE,
     (Fraction(1), Fraction(4), Fraction(1, 3), Fraction(-2)), ()),
    (S.s17_anurupyena_proportion, "ref", S.S17_REF, ALL_REFS, (PHI,)),
    (S.s18_adyamadyena_antyamantyena, "i", S.S18_I, tuple(range(16)), ()),
    (S.s18_adyamadyena_antyamantyena, "j", S.S18_J, tuple(range(16)), ()),
    (S.s19_lopana_sthapanabhyam, "mask", S.S19_MASK, ALL_MASKS, ()),
    (S.s20_vilokanam_spect, "axis", S.S20_AXIS, (0, 1, 2, 3), ()),
    # mask 0 is out of domain for S22: it yields no pairs (see below).
    (S.s22_parity_complement, "mask", S.FULL_MASK, NONZERO_MASKS, ()),
    (S.s23_dwandwa_yoga, "mask", S.FULL_MASK, ALL_MASKS, (PHI,)),
    (S.s24_kevalaih_saptakam, "modulus", S.S24_MODULUS,
     (1, 2, 3, 5, 11, 16, 435), ()),
    (S.s25_vestana_circular, "k", S.S25_ROT, (0, 1, 2, 3, 4, 5), ()),
    (S.s28_lopana_restore, "mask", S.S19_MASK, ALL_MASKS, ()),
    (S.s29_mean_drive, "weight", S.S29_WEIGHT,
     (Fraction(0), Fraction(1), Fraction(1, 5), Fraction(-1, 2), Fraction(3)), ()),
)

OPERAND_IDS = [f"{fn.__name__.split('_')[0]}.{param}"
               for fn, param, _, _, _ in OPERANDS]


#: Per-operand applicability. Some operands have a genuine precondition on
#: the input as well as on their own value — S17 divides by Ψ_ref, so
#: (ref, Ψ) with Ψ_ref = 0 is out of domain and the kernel raises rather
#: than substituting anything. Encoded here so the corpus-wide tests skip
#: exactly the inapplicable combinations and nothing else; a test that
#: quietly swallowed the ValueError instead would also swallow a real one.
APPLIES = {
    (S.s17_anurupyena_proportion.__name__, "ref"):
        lambda psi, ref: psi[ref] != 0,
}


def _applies(fn, param, psi, value) -> bool:
    pred = APPLIES.get((fn.__name__, param))
    return True if pred is None else pred(psi, value)


def _call(fn, psi, extra, param=None, value=None):
    kwargs = {param: value} if param is not None else {}
    return fn(psi, *extra, **kwargs)


# ───────────────────────────────────────────────────────────── completeness

def test_the_operand_table_covers_every_optional_operand() -> None:
    """Introspect the module; the table must account for what is found.

    Without this, adding an operand to ``z2_primitives`` and forgetting to
    add it here leaves it with no liveness, fidelity or domain check — and
    nothing reports the omission.
    """
    found: set[tuple[str, str]] = set()
    for name in dir(S):
        if not (name.startswith("s") and name[1:2].isdigit()):
            continue
        fn = getattr(S, name)
        if not callable(fn):
            continue
        for p in inspect.signature(fn).parameters.values():
            if p.default is not inspect.Parameter.empty:
                found.add((name, p.name))
    tabled = {(fn.__name__, param) for fn, param, _, _, _ in OPERANDS}
    assert tabled == found, (
        f"operand table is out of step with the module.\n"
        f"  untested operands: {sorted(found - tabled)}\n"
        f"  table entries with no such operand: {sorted(tabled - found)}")
    assert len(OPERANDS) == 25, f"expected 25 operands, table has {len(OPERANDS)}"


def test_every_table_entry_declares_its_canonical_as_the_module_default() -> None:
    """The canonical column must be the actual default, not a copy that drifted."""
    for fn, param, canonical, _, _ in OPERANDS:
        default = inspect.signature(fn).parameters[param].default
        assert canonical == default, (
            f"{fn.__name__}({param}=): table says {canonical!r}, "
            f"module default is {default!r}")


def test_the_corpus_is_available_and_plural() -> None:
    assert len(PSI_CASES) >= 15
    assert len({psi for _, psi in PSI_CASES}) == len(PSI_CASES)


# ─────────────────────────────────────────────────── 1. canonical fidelity

@pytest.mark.parametrize("fn,param,canonical,alternatives,extra", OPERANDS,
                         ids=OPERAND_IDS)
def test_explicit_canonical_equals_default(fn, param, canonical,
                                           alternatives, extra) -> None:
    """Passing the canonical operand explicitly is identical to omitting it.

    Every operand, every corpus input. Checked for five of twenty-five
    before, on one Ψ.
    """
    mismatches = []
    checked = 0
    for label, psi in PSI_CASES:
        if not _applies(fn, param, psi, canonical):
            continue
        checked += 1
        implicit = _call(fn, psi, extra)
        explicit = _call(fn, psi, extra, param, canonical)
        if implicit != explicit:
            mismatches.append(label)
    assert checked > 0, (
        f"{fn.__name__}({param}=) was applicable to no corpus input; the "
        f"fidelity check would pass without comparing anything")
    assert not mismatches, (
        f"{fn.__name__}({param}={canonical!r}) differs from the default on: "
        f"{mismatches}")


# ────────────────────────────────────────────────────────────── 2. liveness

@pytest.mark.parametrize("fn,param,canonical,alternatives,extra", OPERANDS,
                         ids=OPERAND_IDS)
def test_operand_is_live(fn, param, canonical, alternatives, extra) -> None:
    """Some non-canonical value must change the result on some input.

    An operand that changes nothing anywhere is decorative, and a caller
    setting it would be silently ignored.
    """
    for value in alternatives:
        if value == canonical:
            continue
        for _, psi in PSI_CASES:
            if not (_applies(fn, param, psi, value)
                    and _applies(fn, param, psi, canonical)):
                continue
            if _call(fn, psi, extra, param, value) != _call(fn, psi, extra):
                return
    assert False, (
        f"{fn.__name__}({param}=) is decorative: none of "
        f"{len(alternatives)} in-domain values changed the result on any of "
        f"{len(PSI_CASES)} corpus inputs")


@pytest.mark.parametrize("fn,param,canonical,alternatives,extra", OPERANDS,
                         ids=OPERAND_IDS)
def test_every_in_domain_operand_value_is_accepted(fn, param, canonical,
                                                   alternatives, extra) -> None:
    """The declared domain is really the domain.

    Each alternative in the table must be accepted on every corpus input. A
    value that raises here is either out of domain (so the table is wrong)
    or wrongly rejected (so the guard is).
    """
    for value in alternatives:
        for label, psi in PSI_CASES:
            if not _applies(fn, param, psi, value):
                continue
            try:
                _call(fn, psi, extra, param, value)
            except Exception as e:                       # noqa: BLE001
                raise AssertionError(
                    f"{fn.__name__}({param}={value!r}) rejected {label}: "
                    f"{type(e).__name__}: {e}") from e


def test_s18_operands_are_independently_live() -> None:
    """i and j must each matter on their own.

    They used to be varied together in one case, so a dead operand was
    invisible as long as the other one moved the result.
    """
    for label, psi in PSI_CASES:
        base = S.s18_adyamadyena_antyamantyena(psi)
        moved_i = any(S.s18_adyamadyena_antyamantyena(psi, i=k) != base
                      for k in range(16))
        moved_j = any(S.s18_adyamadyena_antyamantyena(psi, j=k) != base
                      for k in range(16))
        if moved_i and moved_j:
            return
    assert False, "neither S18 operand moved the result independently"


# ─────────────────────────────────────────────────────────────── 3. domain

MASK_TAKERS = tuple(
    (fn, param) for fn, param, _, _, _ in OPERANDS if param == "mask")
REF_TAKERS = tuple(
    (fn, param) for fn, param, _, _, _ in OPERANDS if param in ("ref", "axis"))


@pytest.mark.parametrize("fn,param", MASK_TAKERS,
                         ids=[f.__name__.split('_')[0] for f, _ in MASK_TAKERS])
def test_out_of_range_mask_raises(fn, param) -> None:
    """Every mask-taking operand, not the eight that were listed by hand."""
    extra = next(e for f, p, _, _, e in OPERANDS if f is fn and p == param)
    for bad in (-1, 16, 17, 255, 1 << 20):
        with pytest.raises(ValueError):
            _call(fn, PSI, extra, param, bad)


@pytest.mark.parametrize("fn,param", REF_TAKERS,
                         ids=[f"{f.__name__.split('_')[0]}.{p}"
                              for f, p in REF_TAKERS])
def test_out_of_range_vertex_or_axis_raises(fn, param) -> None:
    extra = next(e for f, p, _, _, e in OPERANDS if f is fn and p == param)
    limit = 4 if param == "axis" else 16
    for bad in (-1, limit, limit + 1, 1000):
        with pytest.raises(ValueError):
            _call(fn, PSI, extra, param, bad)


def test_out_of_range_axis_tuple_raises() -> None:
    for bad in ((0, 7), (-1,), (4,), (0, 1, 2, 3, 4)):
        with pytest.raises(ValueError):
            S.s9_chalana_kalanabhyam(PSI, bad)


def test_degenerate_scalar_operands_raise() -> None:
    with pytest.raises(ValueError):
        S.s16_gunaka_samucchaya(PSI, Fraction(0))
    with pytest.raises(ValueError):
        S.s24_kevalaih_saptakam(PSI, 0)
    with pytest.raises(ValueError):
        S.s24_kevalaih_saptakam(PSI, -1)


def test_s18_index_bounds_are_checked() -> None:
    for bad in (-1, 16, 100):
        with pytest.raises(ValueError):
            S.s18_adyamadyena_antyamantyena(PSI, bad, 0)
        with pytest.raises(ValueError):
            S.s18_adyamadyena_antyamantyena(PSI, 0, bad)


def test_canonical_defaults_match_the_named_constants() -> None:
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


# ─────────────────────────────────────────────────── structural identities

def test_s1_and_s2_are_the_same_rule_at_every_mask() -> None:
    """Both are XOR translations; only the operand distinguishes them.

    All sixteen masks over the whole corpus, rather than two hand-picked
    masks on one Ψ.
    """
    for label, psi in PSI_CASES:
        for mask in ALL_MASKS:
            assert S.s1_eka_adhikena(psi, mask) == S.s2_nikhilam(psi, mask), \
                f"{label} mask {mask:04b}"


@pytest.mark.parametrize("k", range(1, 5))
def test_s25_rotation_has_order_four(k: int) -> None:
    """Four applications of a k-step rotation return the input, for every
    corpus vector — and fewer than four do not, unless k is a multiple of 4."""
    for label, psi in PSI_CASES:
        out = psi
        for _ in range(4):
            out = S.s25_vestana_circular(out, k)
        assert out == psi, f"{label}: rotation by {k} is not order 4"


def test_s25_zero_rotation_is_the_identity_everywhere() -> None:
    for label, psi in PSI_CASES:
        assert S.s25_vestana_circular(psi, 0) == psi, label


@pytest.mark.parametrize("mask", ALL_MASKS)
def test_s7_parts_reconstruct_the_input_for_every_mask(mask: int) -> None:
    """sym + anti = Ψ. All sixteen masks, whole corpus.

    The old test's name said "for any mask" and it checked three.
    """
    for label, psi in PSI_CASES:
        sym, anti = S.s7_sankalana_vyavakalana(psi, mask)
        assert tuple(a + b for a, b in zip(sym, anti)) == psi, \
            f"{label} mask {mask:04b}"


def test_s29_weight_zero_is_identity_and_one_is_the_mean() -> None:
    for label, psi in PSI_CASES:
        assert S.s29_mean_drive(psi, Fraction(0)) == psi, label
        out = S.s29_mean_drive(psi, Fraction(1))
        mean = sum(psi, Fraction(0)) / Fraction(16)
        assert all(x == mean for x in out), label


@pytest.mark.parametrize("ref", ALL_REFS)
def test_s17_with_phi_equal_psi_is_the_identity(ref: int) -> None:
    """S17(Ψ, Ψ) = Ψ · Ψ_ref/Ψ_ref = Ψ, for every ref and every Ψ.

    This matters for composition: the registry binds Φ = Ψ to give S17 a
    unary form, and that binding makes S17 the identity operator. It
    contributes nothing to a SERIES chain and only dilutes a PARALLEL mean.

    Inputs with Ψ_ref = 0 are excluded and counted, because Ψ_ref/Ψ_ref is
    undefined there — stated rather than avoided by choosing inputs where it
    does not arise.
    """
    checked = 0
    for label, psi in PSI_CASES:
        if psi[ref] == 0:
            continue
        assert S.s17_anurupyena_proportion(psi, psi, ref) == psi, \
            f"{label} ref {ref}"
        checked += 1
    assert checked >= 10, \
        f"only {checked} corpus inputs had a nonzero component at ref {ref}"


def test_s17_with_distinct_phi_is_not_the_identity() -> None:
    witnesses = [label for label, psi in PSI_CASES
                 if psi[S.S17_REF] != 0
                 and S.s17_anurupyena_proportion(psi, PHI) != psi]
    assert witnesses, "S17 is the identity even with a distinct Φ"


@pytest.mark.parametrize("ref", ALL_REFS)
def test_s17_raises_rather_than_substituting_when_the_reference_is_zero(ref):
    """The precondition Ψ_ref ≠ 0 is asserted, not worked around.

    Dividing by Ψ_ref with a substituted denominator would return a
    plausible vector for an input the operator is not defined on.
    """
    zero = BY_LABEL["zero"]
    with pytest.raises(ValueError, match="non-zero"):
        S.s17_anurupyena_proportion(zero, PHI, ref)
    spike = BY_LABEL["spike_low"]        # only component 0 is nonzero
    if ref != 0:
        with pytest.raises(ValueError, match="non-zero"):
            S.s17_anurupyena_proportion(spike, PHI, ref)


@pytest.mark.parametrize("mask", NONZERO_MASKS)
def test_s22_returns_eight_pairs_for_every_nonzero_mask(mask: int) -> None:
    """S22's return shape is part of its contract, checked at every mask.

    Its docstring promised "Output length 8" unconditionally while mask 0
    returned an empty tuple — an output no caller could use and none of the
    three hand-picked masks the old test tried would have revealed.
    """
    for label, psi in PSI_CASES:
        out = S.s22_parity_complement(psi, mask)
        assert len(out) == 8, f"{label} mask {mask:04b}: length {len(out)}"
        assert all(isinstance(x, Fraction) for x in out)


def test_s22_rejects_the_empty_mask() -> None:
    """mask 0 has no (v, v⊕mask) pairs, so there is no S22 result to give."""
    for label, psi in PSI_CASES:
        with pytest.raises(ValueError, match="non-zero"):
            S.s22_parity_complement(psi, 0)
