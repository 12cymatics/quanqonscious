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
    assert state_digest(after_grvq)[:16] == "1d499cb0dddacf34", \
        "post-GRVQ field changed"
    assert state_digest(after_mstvq)[:16] == "a64fe1cc77f1fa16", \
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
    print("Testing sutra golden values...")

    lattice = create_3d_lattice(4, 4, 4)

    def fixture():
        st = create_zero_field(lattice)
        st.set_by_coords((2, 2, 2), RationalComplex(Fraction(1), Fraction(0)))
        st.set_by_coords((1, 3, 0), RationalComplex(Fraction(3, 4), Fraction(-1, 2)))
        st.set_by_coords((0, 0, 1), RationalComplex(Fraction(-2, 5), Fraction(1, 8)))
        st.set_by_coords((3, 1, 2), RationalComplex(Fraction(7, 9973), Fraction(3, 9973)))
        return st

    GOLDEN = {
         1: ("f1f320b0d34bb3a4", "a226912cb85c8134", "998be04a87c60860"),
         2: ("b5764eea954ee7e9", "da4020279d4a74b4", "acc81230d2ad6b3b"),
         3: ("9769b6a537aeb0e1", "88754ec27ff2b7de", "f5e0a9a91382f067"),
         4: ("c9af77f7bd55c320", "866b54c1b17b5930", "6ec97f50432fe0da"),
         5: ("9b512cb5031401bf", "d6cc4a7db1ccc8b8", "f45d63f8f482050a"),
         6: ("20ccd90aba09b37d", "810411ce0bd033a1", "848648275a1d623a"),
         7: ("45e55f5e352d4b30", "bb092cc9b4d9ea51", "a974dc37d33fca5a"),
         8: ("4b31067501c124e9", "cc11d745ffa0f091", "4e5d4dfb4ba1fa35"),
         9: ("5c17254568af79a4", "157cca6e9a2f7eb5", "469a8513b4d53b0a"),
        10: ("7133cb822a88c37e", "61812ff6538696ea", "72d0dab06df5b7d4"),
        11: ("07cd6510c0687c6c", "df1f2793f1046d34", "66cb853a6be77dd1"),
        12: ("516f76b95f640a95", "d8faf13f50ded6e7", "8d1d2c088a6fc33c"),
        13: ("31c8d6c78cc7cf33", "57fc6937ae4379ca", "57f01bedc7d17567"),
        14: ("89ea1e3146c48ffb", "e496215b4a71d14e", "c5abbf396a0c44c8"),
        15: ("8c365e16420fbf87", "cd19efa564645b75", "5185606217ca82e6"),
        16: ("49dd9791ea10c129", "95ffb43ea5c6ea8f", "560892c1a852323b"),
        17: ("d1b064fb2b5a086a", "26ad1cd2b7937279", "8ed1165682df1882"),
        18: ("38333e2df5403f2d", "de9f09c5dfbc6c5e", "321c9237b6c966a2"),
        19: ("fb9bd0cf1efe7ab8", "db0a21461a667828", "3fd26ff57380d559"),
        20: ("516f76b95f640a95", "d8faf13f50ded6e7", "8d1d2c088a6fc33c"),
        21: ("353db97d6ff5a71f", "cc3d8ecbf480b099", "eacd2984ade007a4"),
        22: ("1cca1771928ee2b3", "31a7ead73f82c72d", "e4d76c49f16ce026"),
        23: ("e78c1cf56ecf4f6e", "8c5705f501bd698e", "f3c94d8550e789f8"),
        24: ("bdcd7429b08c459c", "bdad0d93e1e92156", "c4f4eeacccc8f001"),
        25: ("fbd4f5914172671d", "bc2d7a9c51f72f4e", "2fa1d96c0a20e890"),
        26: ("8c365e16420fbf87", "cd19efa564645b75", "5185606217ca82e6"),
        27: ("e7fee7d9da5dfefd", "09500060ef5ccc6a", "532b085872b41839"),
        28: ("4aeadfd6e07aa7c8", "4f0cf41b9a44f69a", "089ff8fc25847527"),
        29: ("72c9e35e2524a6a5", "47802d3b70cd8346", "699b2f248a3da762"),
    }

    probe = lattice.point(1, 3, 0)
    context = OperatorContext()
    sutras = get_all_sutras()
    assert len(sutras) == 29, f"expected 29 sutras, got {len(sutras)}"
    norm = fixture().total_norm_squared()

    outputs = {}
    for number, (digest_prefix, want_real, want_imag) in GOLDEN.items():
        op = sutras[number - 1]
        out = op.apply(fixture(), context)
        outputs[number] = state_digest(out)
        got = out.get(probe)
        assert (_exact_hash(got.real), _exact_hash(got.imag)) == (want_real, want_imag), (
            f"sutra {number} ({op.name}) moved the probe site {probe.coords}")
        assert outputs[number][:16] == digest_prefix, (
            f"sutra {number} ({op.name}): field digest {outputs[number][:16]} != {digest_prefix}")
        assert out.total_norm_squared() == norm, f"sutra {number} ({op.name}) changed the exact norm"

    assert len(set(outputs.values())) == 27
    assert outputs[12] == outputs[20] and outputs[15] == outputs[26]
    assert state_digest(fixture()) not in outputs.values()

    print("  \u2713 Sutra golden values: 29/29 pinned, exact norm preserved by all 29")


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
    assert state_digest(result)[:16] == "3ebab339a457b1c5", "post-R4 field changed"

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
