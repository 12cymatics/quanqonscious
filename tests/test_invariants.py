"""
Test Suite for GRVQ/MSTVQ/TGCR Cymatic Simulation Core (CODEX 7.2)

Tests:
- Toroidal index closure invariant
- Determinism invariant
- Boundedness gate invariant
- Trace replay invariant
- Sutra operator closure
"""

import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractions import Fraction
from core.lattice import create_3d_lattice, create_4d_hypercube, LatticePoint
from core.state import create_zero_field, create_gaussian_field, RationalComplex, state_digest
from core.operators.base import OperatorContext, OperatorTrace, IdentityOperator
from core.operators.grvq_ansatz import GRVQAnsatzOperator, RadialSuppression, create_cymatic_ansatz
from core.operators.mstvq import MSTVQCompositeOperator, MSTVQConfig
from core.operators.r4_coupling import R4CompositeOperator
from core.operators.sutra_ops import get_all_sutras, get_sutra_by_number, create_sutra_pipeline
from core.observables import (create_standard_invariants, create_standard_observables,
                             EnergyConservationInvariant)
from core.trace import DeterminismVerifier, StateCheckpoint, EvolutionTrace, TraceReplayer


def _exact_hash(value: Fraction) -> str:
    """SHA-256 over a rational's byte form, truncated to 16 hex characters.

    Not `str(value)`: Python caps int-to-str conversion at 4300 digits by
    default, and some exact quantities here run to five figures of digits
    (R4's energy is a 12882-digit rational). Bytes skip the decimal
    conversion, so this works at any size and is much faster.
    """
    def _b(n: int) -> bytes:
        return n.to_bytes((n.bit_length() + 8) // 8, "big", signed=True)

    digest = hashlib.sha256()
    digest.update(_b(value.numerator))
    digest.update(b"/")
    digest.update(_b(value.denominator))
    return digest.hexdigest()[:16]


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

    # `validate_closure` only range-checks coordinates, so the loop above is
    # satisfied by any subset of the real neighbourhood -- deleting half the
    # adjacency kernel leaves every surviving neighbour perfectly valid and
    # this test green. (Measured: it does, while R4's energy moves 37.047 ->
    # 29.434.) So check the shape of the neighbourhood, not just that its
    # members are in range.
    dims = len(lattice.shape)
    assert len(neighbors) == 2 * dims, (
        f"von Neumann neighbourhood has {len(neighbors)} members, expected "
        f"{2 * dims} for a {dims}-dimensional lattice"
    )
    assert len(set(n.coords for n in neighbors)) == 2 * dims, \
        "neighbourhood contains duplicates"

    # It must also be closed under negation: for every offset there is an
    # opposite one. Dropping the -1 offsets halves the kernel while leaving
    # every remaining offset valid, and only this catches that.
    interior = lattice.point(4, 4, 4)
    offsets = {tuple(a - b for a, b in zip(n.coords, interior.coords))
               for n in lattice.nearest_neighbors(interior)}
    assert offsets == {tuple(-c for c in o) for o in offsets}, (
        f"neighbour offsets are not closed under negation: {sorted(offsets)}"
    )

    print(f"  \u2713 Toroidal closure: PASSED ({len(neighbors)} neighbours, "
          f"offsets symmetric)")


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
    """Test CODEX 7.2: Boundedness gate - envelope keeps Psi within bounds.

    The gate is placed after GRVQ, because that is where the envelope acts,
    and at a bound near the field's actual scale. The version this replaces
    did neither, and could not fail for the reason it names. Three compounding
    causes, all measured:

    1. The bound was `Fraction(1000)` against a post-pipeline scale of 0.452.
    2. `validate_bounded(b)` compares against b**2 while `max_amplitude()`
       already returns max |Psi|**2 (it is deprecated and misnamed -- exact
       amplitude would need a square root). So `validate_bounded(1000)`
       permitted |Psi|**2 <= 1_000_000, and the real headroom was 2.2 million
       times the field, not the 2212x the raw numbers suggest.
    3. MSTVQ's envelope R = 1/(1 + S + |T|) is unconditionally <= 1 and runs
       *after* GRVQ, so it renormalises whatever GRVQ did away before the
       assertion is reached. Disabling GRVQ's suppression entirely moves the
       post-GRVQ maximum from 29.086 to 38.601 -- but the post-MSTVQ maximum
       only from 0.452 to 0.543, which passes any bound the old test would
       plausibly have used. The masking is the reason the gate has to sit at
       the intermediate, not at the end.

    So the test asserts the property the envelope exists for, directly, and
    then proves its own gate discriminates rather than assuming it.
    """
    print("Testing boundedness invariant...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)

    class _NoEnvelope(RadialSuppression):
        """GRVQ with the radial suppression removed -- R(x) = 1 everywhere."""

        def evaluate(self, coords, lattice, context):
            return Fraction(1)

    after_grvq = GRVQAnsatzOperator()(state, OperatorContext())
    unsuppressed = GRVQAnsatzOperator(radial_suppression=_NoEnvelope())(
        state, OperatorContext())
    after_mstvq = MSTVQCompositeOperator()(after_grvq, OperatorContext())

    # The envelope must actually suppress. This is the property, stated.
    assert after_grvq.max_amplitude() < unsuppressed.max_amplitude(), (
        f"radial suppression did not reduce the field: "
        f"{float(after_grvq.max_amplitude())} vs "
        f"{float(unsuppressed.max_amplitude())} without it"
    )

    # The gate, where the envelope acts and at the scale it acts on.
    grvq_bound = Fraction(6)   # permits |Psi|^2 <= 36; the field reaches 29.086
    assert after_grvq.validate_bounded(grvq_bound), (
        f"field exceeds bound {grvq_bound} after GRVQ: "
        f"max |Psi|^2 = {float(after_grvq.max_amplitude())}"
    )

    # ... and the gate must be able to fail. Without the envelope the same
    # field reaches 38.601, over the 36 the bound permits. A gate never seen
    # to reject anything is not known to be a gate.
    assert not unsuppressed.validate_bounded(grvq_bound), (
        f"bound {grvq_bound} does not discriminate: it accepts the field "
        f"with the envelope disabled (max |Psi|^2 = "
        f"{float(unsuppressed.max_amplitude())}), so passing it establishes nothing"
    )

    # Whole-field pins. The exact maxima are 270- and 2124-digit rationals,
    # too large to write down; the digest covers every site of both states.
    assert state_digest(after_grvq)[:16] == "46d0149f158ad7dc", \
        "post-GRVQ field changed"
    assert state_digest(after_mstvq)[:16] == "b84f2d63f8e7b2f6", \
        "post-MSTVQ field changed"

    # The standard invariant set, seeded with the TRUE initial norm. The old
    # test passed the float literal `1.0`, which tripped a branch in
    # EnergyConservationInvariant reading
    #     if initial_norm <= 1.0 and current_norm > 10.0: return True
    # so that check returned True on every run without examining the field.
    # That branch is gone; the norm is now compared exactly over Q.
    context = OperatorContext()
    checker = create_standard_invariants()
    initial_norm_sq = state.total_norm_squared()
    context.set_param('initial_norm_sq', initial_norm_sq)
    _, results = checker.verify_all(after_mstvq, context)

    for name, (passed, msg) in results.items():
        status = "\u2713" if passed else "\u2717"
        print(f"    {status} {name}: {msg}")

    assert results['ToroidalClosure'][0], results['ToroidalClosure'][1]
    assert results['Boundedness'][0], results['Boundedness'][1]

    # EnergyConservation does NOT hold here, and asserting `all_passed` would
    # be asserting something false. GRVQ multiplies the field by shape
    # functions and a Vedic carrier -- it is an ansatz composition, not a
    # unitary evolution -- so it has no reason to preserve the norm, and
    # measurably does not: x62.86 through GRVQ, x2.04 net after MSTVQ.
    #
    # Whether the standard invariant set should carry an energy check at all
    # for non-unitary operators is a question about the physics, left to the
    # maintainer rather than settled by quietly deleting the check. What the
    # test can do is pin the measured behaviour, so that a change in either
    # direction shows up: the operators becoming conservative, or the check
    # regressing to passing by default the way it used to.
    assert not results['EnergyConservation'][0], (
        "EnergyConservation now passes on this pipeline. Either an operator "
        "changed or the check regressed -- it should report ~104% against a "
        "50% tolerance. Do not 'fix' this by loosening the tolerance."
    )
    relative_change = abs(after_mstvq.total_norm_squared() - initial_norm_sq) / initial_norm_sq
    assert Fraction(104, 100) < relative_change < Fraction(105, 100), (
        f"norm change moved to {float(relative_change) * 100:.2f}%; it was 104.1%"
    )
    print(f"  \u2713 Boundedness: PASSED (GRVQ max |Psi|^2 "
          f"{float(after_grvq.max_amplitude()):.3f} with the envelope, "
          f"{float(unsuppressed.max_amplitude()):.3f} without)")


def test_energy_invariant_cannot_pass_by_default():
    """EnergyConservationInvariant must answer from the field, never from the shape of its input.

    Gated separately because `test_boundedness` cannot reach this. That test
    now seeds the true initial norm, so it no longer supplies the input the
    old pass-by-default branch keyed on --

        if initial_norm <= 1.0 and current_norm > 10.0:
            return True, "Initial norm placeholder detected; ..."

    -- and restoring that branch leaves `test_boundedness` green. This feeds
    the check exactly that shape and requires a real answer.
    """
    print("Testing energy invariant cannot pass by default...")

    lattice = create_3d_lattice(4, 4, 4)
    state = create_zero_field(lattice)
    for i in range(4):
        state.set_by_coords((i, 0, 0), RationalComplex(Fraction(3), Fraction(4)))
    assert state.total_norm_squared() == Fraction(100), state.total_norm_squared()

    check = EnergyConservationInvariant(tolerance=Fraction(1, 10))

    # The placeholder shape: a nominal initial norm of 1 against a current 100.
    context = OperatorContext()
    context.set_param('initial_norm_sq', Fraction(1))
    passed, message = check.check(state, context)
    assert not passed, f"check passed on a 9900% change: {message}"
    assert "9900" in message, f"message does not report the real change: {message}"

    # Genuine conservation is reported as such, exactly.
    context = OperatorContext()
    context.set_param('initial_norm_sq', Fraction(100))
    passed, message = check.check(state, context)
    assert passed, message

    # A change just inside and just outside the tolerance, with no float
    # anywhere: 110 is exactly 10% away, 111 is more.
    for initial, want in ((Fraction(1000, 11), True), (Fraction(900, 10), False)):
        context = OperatorContext()
        context.set_param('initial_norm_sq', initial)
        got, message = check.check(state, context)
        assert got is want, f"initial={initial}: expected {want}, got {got} ({message})"

    # No initial norm recorded is not the same as conserved, but it is the
    # documented contract, so pin it rather than leave it to drift.
    passed, message = check.check(state, OperatorContext())
    assert passed and "No initial norm" in message, message

    print("  \u2713 Energy invariant: answers from the field, not from its input's shape")

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
    """Test R4 adjacency kernel and coupling.

    Pins the value. What this replaces asserted only that a key existed, that
    a sum of squares was non-negative, and that the field stayed under
    `validate_bounded(10000)` -- which permits |Psi|^2 <= 100_000_000 against
    a field reaching 1.119. Deleting half the adjacency kernel changes the
    energy 37.047 -> 29.428 and still reports PASSED under all three.
    """
    print("Testing R4 coupling...")

    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    result = R4CompositeOperator()(state, context)

    energy = context.get_param('r4_energy')
    assert energy is not None, "R4 energy not computed"
    assert isinstance(energy, Fraction), f"R4 energy is {type(energy).__name__}, not exact"
    # Strictly positive, not `>= 0`: a kernel that computed nothing at all
    # would yield 0 and satisfy the old assertion.
    assert energy > 0, f"R4 energy not positive: {float(energy)}"

    # The exact energy is a 12882-digit rational, so it is pinned by hash
    # rather than written out. Any change to the adjacency kernel moves it.
    energy_hash = _exact_hash(energy)
    assert energy_hash == "19f4d2712e8f0982", (
        f"R4 energy changed (now {float(energy)}, was 37.04723356109884). "
        f"If this is intended, re-pin; if not, the adjacency kernel moved."
    )

    # The operator must actually act, and the whole field is pinned.
    assert state_digest(result) != state_digest(state), "R4 left the field unchanged"
    assert state_digest(result)[:16] == "e9a76dcd59b98599", "post-R4 field changed"

    # A bound at the scale the field reaches (1.119), not 10000.
    assert result.validate_bounded(Fraction(2)), (
        f"field exceeds bound 2 after R4: max |Psi|^2 = {float(result.max_amplitude())}"
    )

    print(f"  \u2713 R4 coupling: PASSED (energy={float(energy):.4f}, "
          f"max |Psi|^2={float(result.max_amplitude()):.4f})")


def test_observables():
    """Test observable computation.

    Pins the values. What this replaces checked only that four names were
    present as keys and never looked at a single one, so `TotalNormSquared`
    hardwired to `Fraction(0)` still reported "8 computed" and passed.
    """
    print("Testing observables...")

    lattice = create_3d_lattice(8, 8, 8)
    sites = len(list(lattice.iterate_all()))
    assert sites == 512, f"fixture lattice has {sites} sites, not 512"

    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)
    results = create_standard_observables().compute_all(state, OperatorContext())

    assert sorted(results) == [
        'CymaticNodeCount', 'MaxAmplitude', 'MeanAmplitude', 'PhaseCoherence',
        'TotalNormSquared', 'TotalR4Energy', 'TotalStress', 'TotalTension',
    ], sorted(results)

    exact = {k: Fraction(v['value']) for k, v in results.items()}

    # The values small enough to read, written out.
    assert exact['CymaticNodeCount'] == 455
    assert exact['MaxAmplitude'] == 1
    assert exact['PhaseCoherence'] == Fraction(239, 256)

    # Stress and tension are exactly zero here, and that is a property of the
    # fixture rather than of the observables: nothing has induced any. They
    # are pinned so that a change -- an observable that starts reporting a
    # value on an unstressed field -- shows up rather than passing silently.
    assert exact['TotalStress'] == 0
    assert exact['TotalTension'] == 0

    # A relation between two of them, which no single hardwired value can
    # satisfy by accident.
    assert exact['MeanAmplitude'] * sites == exact['TotalNormSquared'], (
        "MeanAmplitude is no longer TotalNormSquared / sites"
    )

    # The remaining values run to thousands of digits, so all eight are
    # pinned jointly by hash over their exact string forms.
    canonical = "|".join(f"{k}={results[k]['value']}" for k in sorted(results))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    assert digest == "a5a59ff1c612b85e", (
        "an observable's exact value changed; the readable assertions above "
        "narrow down which, and MeanAmplitude/TotalNormSquared/TotalR4Energy "
        "are the ones not written out"
    )

    print(f"  \u2713 Observables: PASSED ({len(results)} pinned exactly)")


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
        test_energy_invariant_cannot_pass_by_default,
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
