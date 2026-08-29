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
from core.state import create_zero_field, create_gaussian_field, RationalComplex
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
