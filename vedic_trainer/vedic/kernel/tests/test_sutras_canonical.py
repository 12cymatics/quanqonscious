"""The 29 canonical sutras — completeness, structure, and the §12Y guarantee.

Every assertion here traces to the Vedic protocol v4.0 or to
``vedic_v18.24_full_kernel.html`` (§12Z, STRICT_SUTRA_KERNEL, SUTRA_KIND,
ALPHA.computeQ, §12Y).
"""
from __future__ import annotations

import random
from collections import Counter
from fractions import Fraction

import pytest

from vedic.kernel import sutras_canonical as K


def _rnd(seed: int) -> tuple:
    r = random.Random(seed)
    return tuple(Fraction(r.randint(-9, 9), r.randint(1, 7)) for _ in range(16))


PSI = tuple(Fraction(v * v + 1, 7) for v in range(16))
STRENGTH = Fraction(50)


# ------------------------------------------------------------ completeness


def test_all_29_are_present():
    """All 29. Not 28, not 'the main 16' — the complete set."""
    assert K.N_SUTRAS == 29
    assert len(K.SUTRAS) == 29
    assert K.ALL == tuple(range(1, 30))
    assert [s.id for s in K.SUTRAS] == list(range(1, 30))


def test_every_sutra_has_name_sanskrit_kind_category_coefficient():
    for s in K.SUTRAS:
        assert s.name and s.sanskrit and s.kind and s.category
        assert isinstance(s.coefficient, Fraction)


def test_category_census_matches_protocol_4_1():
    """§4.1 seven categories; the engine gives S29 its own CONSERVATION desc."""
    census = Counter(s.category for s in K.SUTRAS)
    assert census == {
        "MULTIPLICATIVE": 4, "REFLECTIVE": 5, "CONVOLUTIVE": 3,
        "DIVISIVE": 5, "PERMUTATIVE": 3, "DIFFUSIVE": 4,
        "MODULAR": 4, "CONSERVATION": 1,
    }


def test_category_membership_matches_protocol_4_1():
    def ids(cat):
        return tuple(s.id for s in K.SUTRAS if s.category == cat)
    assert ids("MULTIPLICATIVE") == (1, 10, 14, 15)
    assert ids("REFLECTIVE") == (2, 5, 12, 22, 23)
    assert ids("CONVOLUTIVE") == (3, 11, 25)
    assert ids("DIVISIVE") == (4, 8, 13, 16, 19)
    assert ids("DIFFUSIVE") == (9, 17, 27, 28)
    assert ids("PERMUTATIVE") == (6, 7, 26)
    assert ids("MODULAR") + ids("CONSERVATION") == (18, 20, 21, 24, 29)


def test_sutra_kind_matches_engine_table():
    """SUTRA_KIND transcribed from vedic_v18.24_full_kernel.html:3558."""
    assert K.SUTRA_KIND[1:] == (
        "MULT", "REFL", "CONV", "DIV", "REFL", "PERM", "PERM", "DIV", "DIFF",
        "MULT", "CONV", "REFL", "DIV", "MULT", "MULT", "DIV", "DIFF", "MOD",
        "DIV", "MOD", "MOD", "REFL", "REFL", "MOD", "CONV", "PERM", "DIFF",
        "DIFF", "MOD",
    )


def test_all_29_coefficients_are_exact_rationals():
    assert len(K.COEFFICIENT) == 29
    assert all(isinstance(v, Fraction) for v in K.COEFFICIENT.values())


def test_named_coefficients_match_protocol_4_3():
    assert K.COEFFICIENT[1] == Fraction(12586269025, 7778742049)   # φ
    assert K.COEFFICIENT[3] == Fraction(355, 113)                  # π Milü
    assert K.COEFFICIENT[4] == Fraction(577, 408)                  # √2 Pell
    assert K.COEFFICIENT[7] == Fraction(97, 56)                    # √3
    assert K.COEFFICIENT[21] == Fraction(6931472, 10000000)        # ln 2


def test_phi_coefficient_is_the_fibonacci_convergent():
    """F₅₀/F₄₉ agrees with φ to double precision."""
    assert abs(float(K.COEFFICIENT[1]) - 1.6180339887498949) == 0.0


# ------------------------------------------------------ triangular identity


def test_delta_sum_is_t29():
    """§8 invariants 1 and 2: Σδ(1..29) = T(29) = 435."""
    assert sum(s.delta for s in K.SUTRAS) == 435
    assert K.SUTRA_SUM == 435 == 29 * 30 // 2
    assert [s.delta for s in K.SUTRAS] == list(range(1, 30))


def test_alpha_is_the_triangular_weight():
    """α(n) = (n/435)·(strength/100), exactly."""
    assert K.alpha(29, Fraction(100)) == Fraction(29, 435)
    assert K.alpha(1, Fraction(50)) == Fraction(1, 435) * Fraction(1, 2)
    assert K.alpha(15, Fraction(50)) == Fraction(1, 58)


def test_alpha_is_monotone_in_id():
    a = [K.alpha(i, STRENGTH) for i in range(1, 30)]
    assert a == sorted(a) and a[0] < a[-1]


def test_alpha_rejects_out_of_range_ids():
    for bad in (0, 30, -1):
        with pytest.raises(ValueError):
            K.alpha(bad, STRENGTH)


# ------------------------------------------- §12Y structural guarantee


def test_alpha_zero_is_the_identity_for_all_29():
    """The guarantee the whole design rests on: α → 0 ⇒ every operator = id.

    Exact equality, not approximate.
    """
    for s in K.SUTRAS:
        assert K.apply_sutra(s.id, PSI, Fraction(0)) == PSI, f"S{s.id} moved at α=0"


def test_full_cascade_at_zero_strength_is_the_identity():
    assert K.apply_all(PSI, Fraction(0)) == PSI


def test_no_sutra_is_dead_at_nonzero_strength():
    """Every one of the 29 must actually move the field."""
    dead = [s.id for s in K.SUTRAS
            if K.apply_sutra(s.id, PSI, STRENGTH) == PSI]
    assert dead == [], f"dead operators: {dead}"


def test_every_sutra_is_a_total_exact_endomorphism():
    for s in K.SUTRAS:
        out = K.apply_sutra(s.id, PSI, STRENGTH)
        assert len(out) == 16
        assert all(isinstance(x, Fraction) for x in out), f"S{s.id} left ℚ"


# ------------------------------------------------- per-kind formula checks


def test_mult_formula():
    """MULTIPLICATIVE: Ψ'ᵢ = Ψᵢ·(1 + α·Ψ_{i⊕1})."""
    w = K.alpha(1, STRENGTH)
    out = K.apply_sutra(1, PSI, STRENGTH)
    assert out == tuple(PSI[i] * (1 + w * PSI[i ^ 1]) for i in range(16))


def test_refl_formula_general():
    """REFLECTIVE (not S5): target = (Ψᵢ + Ψ_c)/2."""
    w = K.alpha(2, STRENGTH)
    out = K.apply_sutra(2, PSI, STRENGTH)
    exp = tuple(K.blend(PSI[i], (PSI[i] + PSI[i ^ 15]) / 2, w) for i in range(16))
    assert out == exp


def test_refl_formula_s5_is_the_negated_complement():
    """S5 Śūnyam Sāmyasamuccaye is the zero-sum special case: target = −Ψ_c."""
    w = K.alpha(5, STRENGTH)
    out = K.apply_sutra(5, PSI, STRENGTH)
    exp = tuple(K.blend(PSI[i], -PSI[i ^ 15], w) for i in range(16))
    assert out == exp


def test_conv_formula():
    """CONVOLUTIVE: target = (1/16) Σ_j Ψ_j·Ψ_{i⊕j}."""
    w = K.alpha(3, STRENGTH)
    conv = [sum((PSI[j] * PSI[i ^ j] for j in range(16)), Fraction(0)) / 16
            for i in range(16)]
    exp = tuple(K.blend(PSI[i], conv[i], w) for i in range(16))
    assert K.apply_sutra(3, PSI, STRENGTH) == exp


def test_diff_formula():
    """DIFFUSIVE: target = edge mean over the 4 Hamming-1 neighbours."""
    w = K.alpha(9, STRENGTH)
    exp = tuple(K.blend(PSI[i], K.edge_mean(PSI, i), w) for i in range(16))
    assert K.apply_sutra(9, PSI, STRENGTH) == exp


def test_perm_axis_is_id_plus_one_mod_four():
    """PERMUTATIVE: reflection across axis (id+1) & 3."""
    for sid in (6, 7, 26):
        w = K.alpha(sid, STRENGTH)
        step = 1 << ((sid + 1) & 3)
        exp = tuple(K.blend(PSI[i], PSI[i ^ step], w) for i in range(16))
        assert K.apply_sutra(sid, PSI, STRENGTH) == exp, f"S{sid}"


def test_div_formula_interpolates_hamming_layers():
    """DIVISIVE: target = mean + (hw(i)/4)·(edgeMean(i) − mean)."""
    w = K.alpha(4, STRENGTH)
    m = K.mean(PSI)
    exp = tuple(
        K.blend(PSI[i], m + Fraction(K.hw(i), 4) * (K.edge_mean(PSI, i) - m), w)
        for i in range(16))
    assert K.apply_sutra(4, PSI, STRENGTH) == exp


def test_mod_formula_blends_toward_the_mean():
    """MODULAR / CONSERVATION: target = mean(Ψ)."""
    for sid in (18, 29):
        w = K.alpha(sid, STRENGTH)
        exp = tuple(K.blend(PSI[i], K.mean(PSI), w) for i in range(16))
        assert K.apply_sutra(sid, PSI, STRENGTH) == exp, f"S{sid}"


def test_each_category_acts_differently():
    """Seven structurally different couplings, not seven names for one map."""
    reps = {}
    for s in K.SUTRAS:
        reps.setdefault(s.kind, s.id)
    outs = {k: K.apply_sutra(sid, PSI, STRENGTH) for k, sid in reps.items()}
    assert len(set(outs.values())) == len(outs), "two kinds produced identical output"


# ------------------------------------------------------------- substrate


def test_hamming_weight_and_complement():
    assert [K.hw(v) for v in range(16)] == [bin(v).count("1") for v in range(16)]
    assert all(K.comp(K.comp(v)) == v for v in range(16))
    assert all(K.hw(v) + K.hw(K.comp(v)) == 4 for v in range(16))


def test_every_vertex_has_four_neighbours_at_hamming_distance_one():
    for v in range(16):
        nb = K.neighbors(v)
        assert len(set(nb)) == 4
        assert all(bin(v ^ j).count("1") == 1 for j in nb)


# ------------------------------------------- cascade is NOT degenerate


def test_full_series_cascade_does_not_annihilate():
    """The canonical operators do not collapse the field.

    An earlier non-canonical implementation had S20→S21→S22 acting as a rank-1
    projection, an absolute value and a pair difference, which annihilated
    every input. That was an artefact of those wrong formulas, not a property
    of the sutras.
    """
    for seed in range(6):
        out = K.apply_all(_rnd(seed), STRENGTH)
        assert any(v != 0 for v in out), "canonical cascade should not annihilate"


def test_cascade_preserves_finiteness_and_exactness():
    out = K.apply_all(_rnd(0), STRENGTH)
    assert all(isinstance(x, Fraction) for x in out)


# ------------------------------------------------------- drift and cores


def test_drift_is_zero_at_zero_strength():
    for s in K.SUTRAS:
        assert K.drift(s.id, PSI, Fraction(0)) == 0


def test_drift_ranker_covers_all_29_and_is_sorted():
    ranked = K.rank_by_drift(PSI, STRENGTH)
    assert len(ranked) == 29
    assert sorted(sid for sid, _ in ranked) == list(range(1, 30))
    assert [d for _, d in ranked] == sorted(d for _, d in ranked)


def test_conservation_cores_drift_less_than_the_full_cascade():
    """§3.8: the wormhole and symmetry cores are conservation cores."""
    x = _rnd(0)
    q0 = K.norm_sq(x)

    def rel(order):
        return abs(K.norm_sq(K.apply_all(x, STRENGTH, order)) - q0) / q0

    full = rel(K.ALL)
    assert rel(K.WORMHOLE_CORE) < full
    assert rel(K.SYMMETRY_CORE) < full


def test_conservation_cores_are_subsets_of_the_29():
    for core in (K.WORMHOLE_CORE, K.SYMMETRY_CORE):
        assert set(core) <= set(K.ALL)


# ------------------------------------------------------------- contracts


def test_out_of_range_sutra_id_raises():
    for bad in (0, 30, -1):
        with pytest.raises(ValueError):
            K.apply_sutra(bad, PSI, STRENGTH)


def test_wrong_vertex_count_raises():
    with pytest.raises(ValueError):
        K.apply_sutra(1, tuple(Fraction(0) for _ in range(8)), STRENGTH)


# ═══════════════════════════════ Gate E — operator records (blueprint)


def test_every_sutra_has_a_complete_operator_record():
    recs = K.all_operator_records()
    assert len(recs) == 29
    for r in recs:
        assert r.domain == "ℚ^16 over V4 = Z₂⁴"
        assert r.codomain == "ℚ^16 over V4 = Z₂⁴"
        assert r.decomposition and r.extensional and r.intensional


def test_linearity_status_is_declared_and_correct():
    """MULT multiplies Ψ by Ψ and CONV convolves Ψ with itself, so both are
    quadratic. The other five kinds are linear."""
    lin = {s.id for s in K.SUTRAS if K.is_linear(s.id)}
    quad = {s.id for s in K.SUTRAS if not K.is_linear(s.id)}
    assert quad == {s.id for s in K.SUTRAS if s.kind in ("MULT", "CONV")}
    assert lin | quad == set(K.ALL)


def test_linear_operators_really_are_linear():
    """Verified by superposition, not asserted."""
    import random
    r = random.Random(0)
    f = tuple(Fraction(r.randint(-9, 9), r.randint(1, 5)) for _ in range(16))
    g = tuple(Fraction(r.randint(-9, 9), r.randint(1, 5)) for _ in range(16))
    fg = tuple(a + b for a, b in zip(f, g))
    for s in K.SUTRAS:
        if not K.is_linear(s.id):
            continue
        lhs = K.apply_sutra(s.id, fg, STRENGTH)
        rhs = tuple(a + b for a, b in zip(K.apply_sutra(s.id, f, STRENGTH),
                                          K.apply_sutra(s.id, g, STRENGTH)))
        assert lhs == rhs, f"S{s.id} declared linear but fails superposition"


def test_quadratic_operators_really_are_not_linear():
    import random
    r = random.Random(1)
    f = tuple(Fraction(r.randint(1, 9)) for _ in range(16))
    g = tuple(Fraction(r.randint(1, 9)) for _ in range(16))
    fg = tuple(a + b for a, b in zip(f, g))
    for s in K.SUTRAS:
        if K.is_linear(s.id):
            continue
        lhs = K.apply_sutra(s.id, fg, STRENGTH)
        rhs = tuple(a + b for a, b in zip(K.apply_sutra(s.id, f, STRENGTH),
                                          K.apply_sutra(s.id, g, STRENGTH)))
        assert lhs != rhs, f"S{s.id} declared quadratic but behaves linearly"


def test_quadratic_operators_refuse_a_matrix_representation():
    for s in K.SUTRAS:
        if not K.is_linear(s.id):
            with pytest.raises(ValueError, match="quadratic"):
                K.operator_matrix(s.id, STRENGTH)


def test_linear_operator_matrices_reproduce_the_action():
    """Matrix invariant: M·Ψ must equal the operator applied to Ψ."""
    for s in K.SUTRAS:
        if not K.is_linear(s.id):
            continue
        M = K.operator_matrix(s.id, STRENGTH)
        got = tuple(sum((M[i][j] * PSI[j] for j in range(16)), Fraction(0))
                    for i in range(16))
        assert got == K.apply_sutra(s.id, PSI, STRENGTH), f"S{s.id} matrix mismatch"


def test_linear_operators_are_reversible_at_this_strength():
    """Reversibility condition: det ≠ 0."""
    for s in K.SUTRAS:
        if K.is_linear(s.id):
            assert K.is_reversible(s.id, STRENGTH), f"S{s.id} is singular"


def test_matrix_at_zero_strength_is_the_identity():
    for s in K.SUTRAS:
        if not K.is_linear(s.id):
            continue
        M = K.operator_matrix(s.id, Fraction(0))
        for i in range(16):
            for j in range(16):
                assert M[i][j] == (1 if i == j else 0)


def test_determinant_is_exact_and_matches_a_known_case():
    ident = tuple(tuple(Fraction(1 if i == j else 0) for j in range(4))
                  for i in range(4))
    assert K.determinant(ident) == 1
    swap = (ident[1], ident[0], ident[2], ident[3])
    assert K.determinant(swap) == -1


def test_intensional_evidence_is_labelled_uncertified():
    """The blueprint says these are generic constructions, not proven Vedic
    decompositions. That label must not be quietly upgraded."""
    for r in K.all_operator_records():
        assert "UNCERTIFIED" in r.intensional
