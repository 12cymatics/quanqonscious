"""`grvq_toroidal_hypercube.VedicSutraEngine` is the canonical 29, exactly.

Why this exists
---------------
Nothing tested this module. It was imported by `hybrid_grvq_toroidal_simulator`
and gated by no suite at all, which is how the following survived in it:

  * 29 functions named for the sutras that did not implement them --
    `sutra04_paravartya` was `p * exp(0.0005 * p)`, and
    `subsutra05_stabilization`, docstringed "Veṣṭanam - Osculation", was
    `np.clip(params, 0.0, 1.0)`. Measured before replacement, that clip turned
    [7.83, 432.0] into [1.0, 1.0] and shifted the 13-sub-sutra mean by -1.88
    and -31.64;
  * four `np.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)` and three
    `np.clip(-1e6, 1e6)`, which were load-bearing rather than defensive:
    `sutra04`'s unbounded exponential diverges past |p| ~ 1e5, and at 1e5 the
    real value 2.06e187 was handed back as exactly 1000000.0;
  * `SutraExecutionPlan.effective_context`, which downgraded QUANTUM and
    HYBRID to CLASSICAL behind a `logger.warning` and returned the classical
    answer under the requested mode's name.

The expectations below do not come from the code under test. The pinned values
are derived by hand from the published formulas -- alpha(n) = (n/435)*(s/100)
and blend(c, t, w) = c + (t - c)*w -- and the identity case is the canonical
kernel's own structural guarantee, not an observation of its output.

    python -m pytest tests/test_canonical_sutra_engine.py
    python tests/test_canonical_sutra_engine.py
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grvq_toroidal_hypercube as G  # noqa: E402
from grvq_toroidal_hypercube import (  # noqa: E402
    PRIMARY_IDS, SUB_IDS, SutraExecutionPlan, VedicSutraEngine,
)
from primarysutra import SutraContext, SutraMode  # noqa: E402
from vedic.kernel.sutras_canonical import apply_sutra, compose  # noqa: E402
from vedic.kernel.tesseract import NUM_VERTICES  # noqa: E402


def _delta(psi_a, psi_b):
    return float(np.max(np.abs(np.asarray(psi_a) - np.asarray(psi_b))))


# ------------------------------------------------------- the arithmetic itself

def test_sutra_18_matches_the_formula_derived_by_hand():
    """S18 is MODULAR: Ψ'ᵢ = blend(Ψᵢ, mean(Ψ), α), α(18,100) = 18/435 = 6/145.

    For Ψ = (1, 0, ..., 0) the mean is 1/16, so

        Ψ'₀ = 1 + (1/16 - 1)·6/145 = 223/232
        Ψ'ᵢ = 0 + (1/16 - 0)·6/145 = 3/1160   for i > 0

    Worked from the published formulas, not read off the implementation.
    """
    psi = tuple([Fraction(1)] + [Fraction(0)] * (NUM_VERTICES - 1))
    got = apply_sutra(18, psi, Fraction(100))

    assert got[0] == Fraction(223, 232), f"Ψ'₀ = {got[0]}, expected 223/232"
    for i, v in enumerate(got[1:], start=1):
        assert v == Fraction(3, 1160), f"Ψ'_{i} = {v}, expected 3/1160"
    # Exact rationals, not floats that happen to compare equal.
    assert all(isinstance(v, Fraction) for v in got)


def test_strength_zero_is_exactly_the_identity():
    """α = 0 makes every operator the identity -- the kernel's §12Y guarantee.

    Exact equality, across every composition mode and the full 29. A tolerance
    here would pass for an operator that merely stays close to the identity.
    """
    psi = np.array([(v % 7) / 3.0 - 1.0 for v in range(NUM_VERTICES)])
    for mode in sorted(VedicSutraEngine.MODES):
        out = VedicSutraEngine.apply_all_29_sutras(psi, Fraction(0), mode)
        assert np.array_equal(out, psi), (
            f"strength=0 in {mode} moved Ψ by {_delta(out, psi)}; it must be "
            f"the identity exactly")


def test_a_nonzero_strength_actually_moves_the_state():
    """Guards the test above: an engine that ignored Ψ entirely would pass
    the identity check too."""
    psi = np.array([(v % 7) / 3.0 - 1.0 for v in range(NUM_VERTICES)])
    out = VedicSutraEngine.apply_all_29_sutras(psi, Fraction(100), "serial")
    assert _delta(out, psi) > 1e-9, "strength=100 left Ψ unchanged"


def test_the_bridge_does_not_corrupt_the_canonical_result():
    """The engine's only job is conversion; the arithmetic is the kernel's.

    Compared against `compose` called directly, so any clamp, reorder or
    rounding introduced by the wrapper shows up here.
    """
    psi = np.array([(v % 5) / 4.0 for v in range(NUM_VERTICES)])
    exact = tuple(Fraction(float(v)) for v in psi)
    for mode_name, canonical in VedicSutraEngine.MODES.items():
        got = VedicSutraEngine.apply_primary_sutras(psi, Fraction(100), mode_name)
        want = compose(canonical, exact, Fraction(100), PRIMARY_IDS)
        assert np.array_equal(got, np.array([float(v) for v in want])), (
            f"{mode_name}: the wrapper changed the kernel's answer")


def test_the_primary_and_sub_ids_partition_the_29():
    assert PRIMARY_IDS == tuple(range(1, 17))
    assert SUB_IDS == tuple(range(17, 30))
    assert len(PRIMARY_IDS) + len(SUB_IDS) == 29
    assert not set(PRIMARY_IDS) & set(SUB_IDS)


def test_float_to_exact_is_a_conversion_not_a_rounding():
    """Every binary64 is a rational, so the trip in and back is lossless.

    Includes subnormals and the extremes, where a naive rational conversion
    that went via a decimal string would lose bits.
    """
    vals = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-300, 5e-324, 1.7976931348623157e308,
            -2.2250738585072014e-308, 1/3, 7.83, 432.0, 1e5, -1e5, 2**53 + 0.0]
    psi = np.array((vals * 2)[:NUM_VERTICES])
    back = VedicSutraEngine.to_float(VedicSutraEngine.to_exact(psi))
    assert np.array_equal(back, psi), "round trip lost bits"


# ----------------------------------------------------------- it refuses loudly

def test_a_state_that_is_not_16_vertices_is_refused():
    """The canonical sutras are defined on the 4-cube's vertices. The old
    engine accepted any length, which is how a 5-element demo state ran."""
    for n in (0, 1, 5, 15, 17, 32):
        try:
            VedicSutraEngine.to_exact(np.zeros(n))
        except ValueError as exc:
            assert str(NUM_VERTICES) in str(exc), f"n={n}: {exc}"
        else:
            raise AssertionError(f"a {n}-vertex state was accepted")


def test_a_non_finite_amplitude_is_refused_and_named():
    """This is the nan_to_num replacement. A NaN must stop the run and say
    WHERE, not become 0.0; an infinity must not become 1e6."""
    for bad, label in ((np.nan, "NaN"), (np.inf, "+inf"), (-np.inf, "-inf")):
        psi = np.zeros(NUM_VERTICES)
        psi[11] = bad
        try:
            VedicSutraEngine.to_exact(psi)
        except ValueError as exc:
            assert "11" in str(exc), f"{label}: refusal does not name index 11: {exc}"
        else:
            raise AssertionError(f"{label} was accepted into Ψ")


def test_an_unknown_execution_mode_is_refused_rather_than_defaulted():
    """`compose` says "Unknown modes raise -- there is no default", and the
    wrapper must not reintroduce one."""
    psi = np.zeros(NUM_VERTICES)
    for mode in ("", "SERIES ", "quick", "smoke", "legacy", "auto"):
        try:
            VedicSutraEngine.apply_all_29_sutras(psi, Fraction(100), mode)
        except ValueError as exc:
            assert "execution mode" in str(exc)
        else:
            raise AssertionError(f"mode {mode!r} was silently accepted")


def test_a_large_state_never_yields_the_old_clip_sentinel():
    """The replaced engine returned exactly ±1e6 for any input that overflowed.

    At |Ψ| = 1e5 the old pipeline handed back 1000000.0 in place of 2.06e187.
    Whatever happens now, the sentinel must never appear: the result is either
    a real number or an explicit refusal.

    My first version of this asserted the output stays inside the input's
    range, "because the canonical operators are blends". That expectation was
    wrong and this test caught it: `_mult` is Ψᵢ·(1 + w·Ψ_{i⊕1}) and the module
    classes MULT and CONV as QUADRATIC_KINDS, so they amplify. Measured at
    Ψ = 1e12: the exact rational overflows binary64 on the way out, which is
    now a named OverflowError rather than numbers.py's "integer division
    result too large for a float".
    """
    for scale in (1e2, 1e5, 1e12, -1e5):
        psi = np.full(NUM_VERTICES, scale)
        psi[0] = -scale
        try:
            out = VedicSutraEngine.apply_all_29_sutras(psi, Fraction(100), "concurrent")
        except OverflowError as exc:
            assert "binary64" in str(exc), f"scale={scale}: opaque refusal: {exc}"
            continue
        assert np.all(np.isfinite(out)), (
            f"scale={scale}: returned a non-finite value instead of refusing")
        assert not np.any(np.abs(out) == 1e6), (
            f"scale={scale}: output contains the old clip sentinel exactly")


def test_the_quadratic_sutras_really_do_amplify():
    """Pins the fact the test above was rewritten around.

    S1 is MULT: Ψ'ᵢ = Ψᵢ·(1 + α·Ψ_{i⊕1}) with α(1,100) = 1/435. For a uniform
    Ψ = 10 that is 10·(1 + 10/435) = 10·445/435 = 890/87, strictly outside the
    input range -- so "blends cannot amplify" is false. Derived by hand.
    """
    psi = tuple(Fraction(10) for _ in range(NUM_VERTICES))
    got = apply_sutra(1, psi, Fraction(100))
    assert got[0] == Fraction(890, 87), f"S1 gave {got[0]}, expected 890/87"
    assert got[0] > Fraction(10), "MULT did not amplify a uniform state"


def test_a_missing_quantum_backend_raises_instead_of_downgrading():
    """`effective_context` used to return CLASSICAL under the QUANTUM label.

    The backend IS importable here, so the downgrade path is forced by taking
    the availability probe away -- the same condition the old code met when it
    silently substituted.
    """
    real = G._quantum_available
    G._quantum_available = lambda: False
    try:
        for mode in (SutraMode.QUANTUM, SutraMode.HYBRID):
            plan = SutraExecutionPlan(context=SutraContext(mode=mode))
            try:
                plan.effective_context()
            except RuntimeError as exc:
                assert mode.name in str(exc), f"refusal does not name {mode.name}"
            else:
                raise AssertionError(f"{mode.name} was downgraded, not refused")
        # CLASSICAL is unaffected: it needs no backend.
        plan = SutraExecutionPlan(context=SutraContext(mode=SutraMode.CLASSICAL))
        assert plan.effective_context().mode is SutraMode.CLASSICAL
    finally:
        G._quantum_available = real


def test_the_replaced_perturbation_functions_are_gone():
    """The invented sutras must not survive anywhere in the module, under any
    name -- a leftover would still be importable and still be wrong."""
    for dead in ("sutra01_ekadhikena", "sutra04_paravartya",
                 "subsutra05_stabilization", "apply_subsutras_serial",
                 "apply_subsutras_concurrent", "apply_subsutras_parallel"):
        assert not hasattr(G, dead), f"{dead} is still defined"
        assert not hasattr(VedicSutraEngine, dead), f"VedicSutraEngine.{dead} survives"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}\n         {exc}")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
