"""
Test Suite for GRVQ/MSTVQ/TGCR Cymatic Simulation Core (CODEX 7.2)

Tests:
- Toroidal index closure invariant
- Determinism invariant
- Boundedness gate invariant
- Trace replay invariant
- Sutra operator closure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction
from core.lattice import create_3d_lattice, create_4d_hypercube, LatticePoint
from core.state import create_zero_field, create_gaussian_field, RationalComplex, state_digest
from core.operators.base import OperatorContext, OperatorTrace, IdentityOperator
from core.operators.grvq_ansatz import GRVQAnsatzOperator, create_cymatic_ansatz
from core.operators.mstvq import MSTVQCompositeOperator, MSTVQConfig
from core.operators.r4_coupling import R4CompositeOperator
from core.operators.sutra_ops import get_all_sutras, get_sutra_by_number, create_sutra_pipeline
from core.observables import create_standard_invariants, create_standard_observables
from core.trace import DeterminismVerifier, StateCheckpoint, EvolutionTrace, TraceReplayer


def test_toroidal_closure():
    """Test CODEX 7.2: Toroidal index closure - all accesses wrap."""
    print("Testing toroidal closure invariant...")

    lattice = create_3d_lattice(8, 8, 8)

    # Test that out-of-bounds indices wrap correctly
    p1 = lattice.point(10, -1, 8)  # Should wrap to (2, 7, 0)
    assert p1.coords == (2, 7, 0), f"Wrap failed: {p1.coords}"
    assert lattice.validate_closure(p1), "Closure validation failed"

    # Test all points in lattice
    for point in lattice.iterate_all():
        assert lattice.validate_closure(point), f"Invalid point: {point}"

    # Test neighbor access wraps correctly
    corner = lattice.point(0, 0, 0)
    neighbors = lattice.nearest_neighbors(corner)
    for n in neighbors:
        assert lattice.validate_closure(n), f"Invalid neighbor: {n}"

    print("  ✓ Toroidal closure: PASSED")


def test_determinism():
    """Test CODEX 7.2: Determinism - same seed/config -> identical outputs."""
    print("Testing determinism invariant...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)

    # Create operators
    operators = [
        IdentityOperator(),
        create_sutra_pipeline([1, 3, 5]),
    ]

    # Verify determinism
    verifier = DeterminismVerifier()
    is_deterministic, msg = verifier.verify(state, operators, num_steps=3, seed=42)

    assert is_deterministic, f"Determinism failed: {msg}"
    print(f"  ✓ Determinism: PASSED ({msg})")


def test_boundedness():
    """Test CODEX 7.2: Boundedness gate - envelope keeps Ψ within bounds."""
    print("Testing boundedness invariant...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    # Apply operators
    grvq = GRVQAnsatzOperator()
    mstvq = MSTVQCompositeOperator()

    state = grvq(state, context)
    state = mstvq(state, context)

    # Check boundedness
    max_bound = Fraction(1000)
    assert state.validate_bounded(max_bound), f"Field exceeds bound {max_bound}"

    # Use invariant checker
    checker = create_standard_invariants()
    context.set_param('initial_norm_sq', 1.0)
    all_passed, results = checker.verify_all(state, context)

    for name, (passed, msg) in results.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {name}: {msg}")

    assert all_passed, "Boundedness invariants failed"
    print("  ✓ Boundedness: PASSED")


def test_trace_replay():
    """Test CODEX 7.2: Trace replay - operator trace replays to identical state."""
    print("Testing trace replay invariant...")

    lattice = create_3d_lattice(4, 4, 4)
    state = create_gaussian_field(lattice, (2, 2, 2), sigma=1.0, amplitude=1.0)

    # Run evolution with trace
    trace = EvolutionTrace(checkpoint_interval=1)
    trace.start(state)

    context = OperatorContext()
    context.trace = trace.operator_trace

    # Apply operators and record
    operators = [IdentityOperator(), create_sutra_pipeline([1])]

    current = state.copy()
    for t in range(3):
        context = context.with_timestep(t)
        for op in operators:
            current = op(current, context)
        trace.record_step(t, current)

    trace.finish(current)

    # Verify checkpoints
    assert trace.initial_checkpoint is not None
    assert trace.final_checkpoint is not None
    assert len(trace.checkpoints) >= 1

    # Replay the recorded evolution from the initial state and require it to
    # reproduce the run exactly.
    #
    # This test used to assert `StateCheckpoint._compute_hash(state) ==
    # trace.initial_checkpoint.state_hash` and stop -- that is hashing `state`
    # and comparing it to the hash of `state` taken by `trace.start(state)` one
    # line above, which is true however broken replay is. It never called
    # TraceReplayer at all, and replay was in fact broken four separate ways
    # while this test was green.
    replayer = TraceReplayer()
    replayer.register_operators(operators)
    replayed, verified, errors = replayer.replay(state, trace)

    assert verified, f"trace did not replay: {errors}"
    assert replayed.snapshot() == current.snapshot(), \
        "replayed final state differs from the evolved state"

    # And the replay must be able to notice a wrong starting point, otherwise
    # `verified` above says nothing.
    perturbed = state.copy()
    origin = perturbed.lattice.point(0, 0, 0)
    perturbed.set(origin, perturbed.get(origin) + RationalComplex(Fraction(1), Fraction(0)))
    _, verified_bad, errors_bad = replayer.replay(perturbed, trace)
    assert not verified_bad, "replay accepted a state that differs from the recorded initial state"
    assert errors_bad, "replay reported failure without saying why"

    print(f"  ✓ Trace replay: PASSED (replayed {len(trace.operator_trace.entries)} entries, "
          f"{len(trace.checkpoints)} checkpoints)")


def test_sutra_closure():
    """Test all 29 sutras produce valid outputs."""
    print("Testing sutra operator closure...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    all_sutras = get_all_sutras()
    assert len(all_sutras) == 29, f"Expected 29 sutras, got {len(all_sutras)}"

    passed = 0
    failed = 0

    for sutra in all_sutras:
        try:
            result = sutra(state.copy(), context)
            if result.validate_bounded(Fraction(10000)):
                passed += 1
            else:
                print(f"    ✗ {sutra.name}: unbounded output")
                failed += 1
        except Exception as e:
            print(f"    ✗ {sutra.name}: {e}")
            failed += 1

    print(f"  ✓ Sutra closure: {passed}/29 passed, {failed} failed")
    assert failed == 0, f"{failed} sutras failed"


def test_sutra_golden_values():
    """Every sutra reproduces an exact, recorded output (CODEX 7.2).

    `test_sutra_closure` above only asks that each sutra runs and stays under
    `Fraction(10000)` -- some 3000x the scale the field actually reaches. That
    is a liveness check: inverting Sutra 13's boundary rule from half-damping
    to 9x amplification leaves it reporting "29/29 passed". This pins values.

    The fixture is a zero field with three exact rationals set by hand rather
    than a Gaussian, so the goldens are readable and carry no float provenance.
    Each sutra is pinned two ways: the digest of the whole field, which
    notices a change at any of the 64 sites, and the value at one probe site,
    which says what changed when the digest fails.

    Two measured properties of this fixture, recorded because a change to
    either is worth a failing test:
      * 6 of the 29 leave the field completely untouched here (6, 8, 16, 17,
        19, 25). That is a fact about this input, not about those operators in
        general -- several read context parameters this fixture does not set.
      * only 24 of the 29 produce distinct fields.
    """
    print("Testing sutra golden values...")

    lattice = create_3d_lattice(4, 4, 4)

    def fixture():
        st = create_zero_field(lattice)
        st.set_by_coords((2, 2, 2), RationalComplex(Fraction(1), Fraction(0)))
        st.set_by_coords((1, 3, 0), RationalComplex(Fraction(3, 4), Fraction(-1, 2)))
        st.set_by_coords((0, 0, 1), RationalComplex(Fraction(-2, 5), Fraction(1, 8)))
        # A large-denominator site. The three above are too clean to exercise
        # Sutra 12's quantiser: their quotients land on exact integers, so a
        # float round trip and a limit_denominator(10000) repair both leave
        # them alone and the pin cannot see the difference. Here the exact
        # quantised imaginary part is 21/79784, which limit_denominator(10000)
        # rewrites to 1/3799 -- so reintroducing that approximation moves the
        # field digest and this test fails, which is the point of having it.
        st.set_by_coords((3, 1, 2), RationalComplex(Fraction(7, 9973), Fraction(3, 9973)))
        return st

    # sutra number -> (whole-field digest prefix, probe real, probe imag)
    GOLDEN = {
         1: ("15cd47eb09d11e9d", Fraction(19, 25), Fraction(-1, 2)),   # EkadhikenaPurvena
         2: ("d7e251d2a6f2772d", Fraction(25, 32), Fraction(-2, 5)),   # NikhilamNavatashcaramam
         3: ("e9a56d62251ce43b", Fraction(3, 4), Fraction(-1, 2)),   # UrdhvaTiryagbhyam
         4: ("ff1f10de2bf2c84f", Fraction(9, 16), Fraction(-3, 8)),   # ParavartyaYojayet
         5: ("ef74214bbc1199bc", Fraction(3, 4), Fraction(-1, 2)),   # ShunyamSamuccaye
         6: ("8115ee1ab36e006a", Fraction(3, 4), Fraction(-1, 2)),   # Anurupyena
         7: ("055df00577532dce", Fraction(3, 4), Fraction(-1, 2)),   # SankalanaVyavakalanabhyam
         8: ("8115ee1ab36e006a", Fraction(3, 4), Fraction(-1, 2)),   # Puranapuranabhyam
         9: ("c69be2f56f6f0f61", Fraction(3, 4), Fraction(-1, 2)),   # CalanaKalanabhyam
        10: ("92e78d221b648b6f", Fraction(33, 56), Fraction(-11, 28)),   # Yavadunam
        11: ("9a1b70b0e4618f7b", Fraction(111, 160), Fraction(-37, 80)),   # VyashtiSamanstih
        12: ("b9ac5151a777acd9", Fraction(3, 4), Fraction(-63, 128)),   # ShesanyankenaCharmona
        13: ("93e7ca2c6841b312", Fraction(3, 8), Fraction(-1, 4)),   # Sopantyadvayamantyam
        14: ("a543bfd8acf6ffcb", Fraction(37, 50), Fraction(-1, 2)),   # EkanyunenaPurvena
        15: ("c38c17bbeb445454", Fraction(57, 80), Fraction(-19, 40)),   # Gunitasamuccayah
        16: ("8115ee1ab36e006a", Fraction(3, 4), Fraction(-1, 2)),   # Gunakasamuccayah
        17: ("8115ee1ab36e006a", Fraction(3, 4), Fraction(-1, 2)),   # AnurupyenaSunyamanyat
        18: ("ebb1c96b2f1c719b", Fraction(87, 140), Fraction(-29, 70)),   # YavadunamTavadunikritya
        19: ("8115ee1ab36e006a", Fraction(3, 4), Fraction(-1, 2)),   # Adyamadyenantyamantyena
        20: ("063e65770a43d1fd", Fraction(111, 140), Fraction(-37, 70)),   # KevalaiSaptakamGunyat
        21: ("616099e3cb834381", Fraction(1, 2), Fraction(-1, 3)),   # Veshtanam
        22: ("dbe75b6f37cd0f22", Fraction(441, 640), Fraction(-147, 320)),   # YavadumamTavadumVilokanam
        23: ("99b45adf109aac19", Fraction(3, 5), Fraction(-3, 20)),   # AntyayorDashakepi
        24: ("60ac787f516af844", Fraction(9, 16), Fraction(-3, 8)),   # AntyayorEva
        25: ("8115ee1ab36e006a", Fraction(3, 4), Fraction(-1, 2)),   # Samuccayagunitah
        26: ("4a557425066661dc", Fraction(3, 4), Fraction(-1, 2)),   # LopanaSthapanabhyam
        27: ("980d9bdd219a4ef2", Fraction(33, 40), Fraction(-11, 20)),   # Vilokanam
        28: ("349500e3a38a1f49", Fraction(29, 40), Fraction(-9, 20)),   # GunitasamuccayahSamuccayagunitah
        29: ("99efbb9804ca1957", Fraction(9, 16), Fraction(-3, 8)),   # DwandwaYoga
    }

    probe = lattice.point(1, 3, 0)
    context = OperatorContext()
    sutras = get_all_sutras()
    assert len(sutras) == 29, f"expected 29 sutras, got {len(sutras)}"

    for number, (digest_prefix, want_real, want_imag) in GOLDEN.items():
        op = sutras[number - 1]
        out = op.apply(fixture(), context)

        got = out.get(probe)
        assert got.real == want_real and got.imag == want_imag, (
            f"sutra {number} ({op.name}) at {probe.coords}: "
            f"expected {want_real}{want_imag:+}i, got {got.real}{got.imag:+}i"
        )
        got_digest = state_digest(out)[:16]
        assert got_digest == digest_prefix, (
            f"sutra {number} ({op.name}): field digest {got_digest} != {digest_prefix} "
            f"-- the probe site is unchanged, so some other site moved"
        )

    digests = {state_digest(sutras[n - 1].apply(fixture(), context)) for n in GOLDEN}
    assert len(digests) == 24, f"expected 24 distinct fields across the 29 sutras, got {len(digests)}"

    base = state_digest(fixture())
    untouched = [n for n in GOLDEN
                 if state_digest(sutras[n - 1].apply(fixture(), context)) == base]
    assert untouched == [6, 8, 16, 17, 19, 25], \
        f"the set of sutras that leave this fixture untouched changed: {untouched}"

    print(f"  \u2713 Sutra golden values: 29/29 pinned "
          f"({len(digests)} distinct fields, {len(untouched)} identity on this fixture)")


def test_r4_coupling():
    """Test R4 adjacency kernel and coupling."""
    print("Testing R4 coupling...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    r4 = R4CompositeOperator()
    result = r4(state, context)

    # Check energy was computed
    energy = context.get_param('r4_energy')
    assert energy is not None, "R4 energy not computed"
    assert energy >= 0, f"R4 energy negative: {energy}"

    # Check result is bounded
    assert result.validate_bounded(Fraction(10000))

    print(f"  ✓ R4 coupling: PASSED (energy={float(energy):.4f})")


def test_observables():
    """Test observable computation."""
    print("Testing observables...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    obs = create_standard_observables()
    results = obs.compute_all(state, context)

    required = ['TotalNormSquared', 'MeanAmplitude', 'MaxAmplitude', 'TotalR4Energy']
    for name in required:
        assert name in results, f"Missing observable: {name}"

    print(f"  ✓ Observables: PASSED ({len(results)} computed)")


def run_all_tests():
    """Run all invariant tests."""
    print("=" * 60)
    print("GRVQ/MSTVQ/TGCR Cymatic Simulation - Test Suite")
    print("CODEX 7.2 Invariant Verification")
    print("=" * 60)
    print()

    tests = [
        test_toroidal_closure,
        test_determinism,
        test_boundedness,
        test_trace_replay,
        test_sutra_closure,
        test_sutra_golden_values,
        test_r4_coupling,
        test_observables,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
