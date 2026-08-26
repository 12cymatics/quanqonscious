"""SERIES / PARALLEL / CONCURRENT composition algebra over exact ℚ.

Covers the defining identities of the algebra, the determinism of the
CONCURRENT scheduler, and the structural degeneracy of the full SERIES
cascade.
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from vedic.kernel import composition as C


def _rnd(seed: int) -> tuple:
    """Random Ψ with Ψ_0 != 0 **by construction**.

    S17 divides by Ψ_ref, so queues containing it are only defined where that
    is non-zero. Index 0 is drawn from a range that excludes zero rather than
    being patched after the fact: a fixture that silently repairs its own
    input is a fallback, and hides the case it was meant to exercise. The
    precondition itself is asserted in
    ``test_s17_precondition_raises_on_zero_reference``.
    """
    r = random.Random(seed)
    vals = [Fraction(r.choice([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]), r.randint(1, 7))]
    vals += [Fraction(r.randint(-9, 9), r.randint(1, 7)) for _ in range(15)]
    return tuple(vals)


PSI = _rnd(0)


# ---------------------------------------------------------------- registry


def test_registry_is_the_full_29():
    assert C.N_SUTRAS == 29
    assert len(C.SUTRAS) == 29
    assert len(C.OPS) == 29
    assert C.MUKHYA == tuple(range(16))
    assert C.UPA == tuple(range(16, 29))


def test_delta_total_is_the_triangular_number():
    """Σ δ_k = T(29) = 435 — the denominator the audit chain closes on."""
    assert C.DELTA_TOTAL == 435
    assert [s.delta for s in C.SUTRAS] == list(range(1, 30))


def test_every_sutra_is_a_total_exact_endomorphism():
    """Each T_k maps ℚ^16 → ℚ^16 with no float contamination."""
    for k in range(C.N_SUTRAS):
        out = C.apply_one(k, PSI)
        assert isinstance(out, tuple) and len(out) == 16
        assert all(isinstance(x, Fraction) for x in out), f"S{k+1} left ℚ"


def test_binary_sutras_are_declared():
    """S3, S17, S23 are genuinely binary; composition binds Φ = Ψ."""
    binary = {s.index for s in C.SUTRAS if s.arity == 2}
    assert binary == {2, 16, 22}


def test_non_vector_sutras_are_declared():
    native = {s.index: s.native for s in C.SUTRAS if s.native != "vector"}
    assert native == {6: "pair", 17: "scalar", 21: "pairs8", 26: "scalar"}


# ---------------------------------------------------------------- modes


@pytest.mark.parametrize("mode", ["SERIES", "PARALLEL", "CONCURRENT", "CANONICAL"])
def test_modes_close_over_exact_q(mode):
    out = C.compose(mode, PSI)
    assert len(out) == 16
    assert all(isinstance(x, Fraction) for x in out)


def test_series_is_a_strict_left_fold():
    ks = [0, 3, 8, 13]
    expected = PSI
    for k in ks:
        expected = C.apply_one(k, expected)
    assert C.series(PSI, ks) == expected


def test_parallel_forks_on_s0_and_mean_joins():
    ks = [0, 3, 8, 13]
    branches = [C.apply_one(k, PSI) for k in ks]
    expected = tuple(sum(b[v] for b in branches) / Fraction(len(ks)) for v in range(16))
    assert C.parallel(PSI, ks) == expected


def test_parallel_is_order_invariant():
    """Every branch reads S₀, so the queue order cannot matter."""
    ks = [0, 3, 8, 13]
    assert C.parallel(PSI, ks) == C.parallel(PSI, list(reversed(ks)))


def test_series_is_order_sensitive():
    """SERIES depends on order for operators that do not commute.

    Not every pair does: S1 and S9 are both XOR-translation-invariant on
    ℚ[Z₂⁴] and therefore commute. S1 and S3 do not.
    """
    ks = [0, 2]
    assert C.series(PSI, ks) != C.series(PSI, list(reversed(ks)))


def test_translation_invariant_sutras_commute():
    """S1 (XOR shift) and S9 (Laplacian) are both translation-invariant."""
    assert C.series(PSI, [0, 8]) == C.series(PSI, [8, 0])


def test_s17_precondition_raises_on_zero_reference():
    zero_at_0 = (Fraction(0),) + tuple(Fraction(v) for v in range(1, 16))
    with pytest.raises(ValueError, match="S17 precondition"):
        C.apply_one(16, zero_at_0)


def test_singleton_queue_collapses_all_modes():
    """With one sutra there is nothing to interleave: all modes agree."""
    for k in (0, 4, 13, 24):
        one = [k]
        base = C.apply_one(k, PSI)
        assert C.series(PSI, one) == base
        assert C.parallel(PSI, one) == base
        assert C.concurrent(PSI, one) == base


# ------------------------------------------------- CONCURRENT interpolation


def test_wave_count_is_ceil_sqrt():
    assert [C.wave_count(n) for n in range(1, 11)] == [1, 2, 2, 2, 3, 3, 3, 3, 3, 4]


def test_wave_count_rejects_empty():
    with pytest.raises(ValueError):
        C.wave_count(0)


def test_concurrent_waves_partition_the_queue_exactly():
    ks = list(range(29))
    waves = C.concurrent_waves(ks)
    flat = [k for w in waves for k in w]
    assert sorted(flat) == sorted(ks), "wave partition lost or duplicated a sutra"
    assert len(waves) == C.wave_count(len(ks))


def test_concurrent_schedule_is_deterministic():
    """Same queue → same waves, every time. CODEX 7.2 determinism."""
    ks = list(range(29))
    assert C.concurrent_waves(ks) == C.concurrent_waves(ks)
    assert C.concurrent(PSI, ks) == C.concurrent(PSI, ks)


def test_concurrent_schedule_depends_on_the_queue():
    """The seed is derived from the queue, not a shared stream."""
    assert C.concurrent_waves([0, 1, 2, 3, 4]) != C.concurrent_waves([5, 6, 7, 8, 9])


# ---------------------------------------------------------------- CANONICAL


def test_canonical_is_mukhya_series_then_upa_parallel():
    mid = C.series(PSI, list(C.MUKHYA))
    expected = C.parallel(mid, list(C.UPA))
    assert C.canonical(PSI, C.ALL) == expected


def test_canonical_degrades_to_series_on_a_mukhya_only_queue():
    ks = [0, 3, 8]
    assert C.canonical(PSI, ks) == C.series(PSI, ks)


def test_canonical_degrades_to_parallel_on_an_upa_only_queue():
    ks = [17, 20, 24]
    assert C.canonical(PSI, ks) == C.parallel(PSI, ks)


# ------------------------------------------------------- structural finding


def test_s20_image_is_rank_one():
    """S20 projects onto a single Walsh row, so its image is 1-dimensional."""
    h0 = tuple(Fraction(1 if not (v & 1) else -1) for v in range(16))
    for seed in range(5):
        out = C.apply_one(19, _rnd(seed + 100))
        ratio = out[0] / h0[0]
        assert all(out[v] == ratio * h0[v] for v in range(16))


def test_s20_s21_s22_run_is_the_zero_map():
    """S20 → S21 → S22 annihilates every input, independent of its value.

    S20's image is c·h₀ with h₀ = (+1,−1,+1,−1,…); S21 takes |·|, giving the
    constant vector |c|; S22 takes differences over (v, v̄) pairs, which is
    exactly zero on a constant. Any SERIES queue containing that ordered run
    is therefore the zero map.
    """
    for seed in range(8):
        x = _rnd(seed + 200)
        out = C.series(x, [19, 20, 21])
        assert all(v == 0 for v in out), "S20→S21→S22 should annihilate"


def test_full_series_cascade_is_degenerate():
    """The canonical 29-sutra SERIES cascade contains S20,S21,S22 in order,
    so it annihilates every input. This is a property of the operator set,
    not of any particular Ψ."""
    for seed in range(8):
        out = C.series(_rnd(seed + 300), C.ALL)
        assert all(v == 0 for v in out)


def test_parallel_and_concurrent_are_not_degenerate():
    """The other modes never feed S22 a constant, so they retain signal."""
    for seed in range(5):
        x = _rnd(seed + 400)
        assert any(v != 0 for v in C.parallel(x, C.ALL))
        assert any(v != 0 for v in C.concurrent(x, C.ALL))
        assert any(v != 0 for v in C.canonical(x, C.ALL))


# ---------------------------------------------------------------- contracts


def test_empty_queue_raises():
    for fn in (C.series, C.parallel, C.concurrent):
        with pytest.raises(ValueError):
            fn(PSI, [])


def test_out_of_range_index_raises():
    with pytest.raises(ValueError):
        C.series(PSI, [29])
    with pytest.raises(ValueError):
        C.apply_one(-1, PSI)


def test_unknown_mode_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        C.compose("SEQUENTIAL", PSI, C.ALL)


def test_composite_propagates_s17_precondition():
    """COMPOSITE evaluates T_i(T_j(S₀)); S17 requires Ψ_ref ≠ 0 and raises
    rather than silently falling back when an intermediate image is zero
    at that index."""
    with pytest.raises(ValueError, match="S17 precondition"):
        C.composite(PSI, C.ALL)


def test_composite_works_where_preconditions_hold():
    ks = [0, 1, 3]
    out = C.composite(PSI, ks)
    assert len(out) == 16 and all(isinstance(x, Fraction) for x in out)


def test_binary_binding_never_degenerates_to_the_identity():
    """No binary sutra may become a no-op under composition.

    Φ = Ψ makes S17 the identity (Ψ·Ψ_ref/Ψ_ref = Ψ) — a no-op in SERIES and
    pure dilution in a PARALLEL mean. The default binding is the Nikhilam
    complement Φ = S2(Ψ), which degenerates for none of S3/S17/S23.
    """
    for k in (2, 16, 22):
        identical = all(C.apply_one(k, _rnd(seed + 500)) == _rnd(seed + 500)
                        for seed in range(4))
        assert not identical, f"T{k+1} is the identity under the current binding"


def test_s17_would_be_the_identity_under_self_binding():
    """Documents the defect the binding policy exists to avoid."""
    import vedic.kernel.z2_primitives as SX
    for seed in range(4):
        x = _rnd(seed + 510)
        assert SX.s17_anurupyena_proportion(x, x) == x


def test_known_annihilating_run_is_detected():
    """The one known SERIES degeneracy is reported, not silently returned as zeros.

    The negative cases assert only that the known run is absent from those
    queues. They are not evidence that SERIES over [0, 1, 2, 3] is non-zero --
    the lookup is against a fixed list and cannot make that claim.
    """
    assert C.has_known_annihilating_run(C.ALL)
    assert C.known_annihilating_runs(C.ALL) == ((19, 20, 21),)
    assert not C.has_known_annihilating_run([0, 1, 2, 3])
    assert C.known_annihilating_runs([0, 1, 2]) == ()


def test_t18_and_t27_are_scalar_rescalings():
    """S18 and S27 are natively scalar; the registry lifts them to T(Ψ)=s·Ψ,
    which is a rescaling, never a change of direction."""
    for k in (17, 26):
        x = _rnd(k + 600)
        out = C.apply_one(k, x)
        nz = [(o, v) for o, v in zip(out, x) if v != 0]
        ratios = {o / v for o, v in nz}
        assert len(ratios) == 1, f"T{k+1} should be a uniform rescaling"
