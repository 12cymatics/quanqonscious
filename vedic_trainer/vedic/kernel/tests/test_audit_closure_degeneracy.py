"""Audit closure is a function of the trace counter alone, for every Ψ in ℚ^16.

Why this exists
---------------
The README named an "audit-closure rate at inference" as falsification
criterion 2: ``full`` minus ``no_sutra`` < 10% absolute. A criterion that
two arms cannot possibly differ on is met by any two models whatsoever,
including two copies of the same one, so it discriminates nothing and must
not be reported as passed.

That is what these tests establish, and they establish it as a **proof over
all of ℚ^16**, not as a measurement on a corpus. The previous version of this
file ran two 480-string text corpora through ``encode_text_to_psi`` and
observed that they produced identical closure flags. That was evidence about
960 encoded vectors and silent about every other Ψ — and it could only be
run at all because a synthetic text encoder existed to produce the vectors.
The encoder is gone; the property never needed it.

The argument
------------
Write each residual as a map ℚ^16 → ℚ.

* ``R2`` and ``R3`` are **linear** in Ψ. R2 is a difference of two sums of
  components; R3 is ``mean(S29 Ψ) − mean(Ψ)`` and S29 is affine-linear with
  no constant term. A linear map that vanishes on ``{0} ∪ {eᵢ}`` is the zero
  map.
* ``R4 = ⟨S(Ψ), A(Ψ)⟩`` is **quadratic**: a product of two linear maps. A
  degree-≤2 map is determined by its values on
  ``SPANNING_SET = {0} ∪ {eᵢ} ∪ {eᵢ + eⱼ}`` — c = F(0),
  Q(eᵢ,eⱼ) = [F(eᵢ+eⱼ) − F(eᵢ) − F(eⱼ) + F(0)]/2, L(eᵢ) = F(eᵢ) − F(0) −
  Q(eᵢ,eᵢ) — so one that vanishes there has c = L = Q = 0 and is the zero
  map on all of ℚ^16.
* ``R1`` takes **no Ψ at all**: its signature is ``r1_trace_closure(trace_sum)``.

So closure — all four residuals zero — reduces to R1, which is a function of
the counter. ``TRIPLE_SET`` is checked too: a residual of degree three or
higher could vanish on the spanning set without being zero, and that is
exactly the premise the polarisation argument rests on. Vanishing on the
560 three-vertex sums as well is what rules it out.
"""
from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

from vedic.kernel.audit_filter import audit_closed, audit_psi
from vedic.kernel.conservation_exact import (
    T29,
    all_residuals,
    r1_trace_closure,
    r2_complement_pair_sum,
    r3_s29_mean_preservation,
    r4_beltrami_orthogonality,
)

from .psi_corpus import BASIS, DYADIC, PSI_CASES, SPANNING_SET, TRIPLE_SET, ZERO

#: Every Ψ this module reasons over: the labelled structural cases, the
#: dyadic cases, and the two enumerated sets the degree argument needs.
ALL_PSI: tuple[tuple, ...] = (
    tuple(psi for _, psi in PSI_CASES)
    + tuple(psi for _, psi in DYADIC)
    + SPANNING_SET
    + TRIPLE_SET
)

#: The three Ψ-dependent residuals, by name.
PSI_RESIDUALS = (
    ("R2", r2_complement_pair_sum),
    ("R3", r3_s29_mean_preservation),
    ("R4", r4_beltrami_orthogonality),
)


def test_the_corpus_is_large_enough_to_carry_the_argument() -> None:
    """Guards every assertion below: an empty set vanishes vacuously."""
    assert len(SPANNING_SET) == 1 + 16 + 120, (
        f"SPANNING_SET holds {len(SPANNING_SET)} vectors, not the 137 the "
        f"polarisation argument needs")
    assert len(TRIPLE_SET) == 560, (
        f"TRIPLE_SET holds {len(TRIPLE_SET)} vectors, not 560")
    assert len(ALL_PSI) > len(SPANNING_SET)


@pytest.mark.parametrize("name,fn", PSI_RESIDUALS)
def test_the_residual_vanishes_on_the_spanning_set(name, fn) -> None:
    """Determines the map: a degree-≤2 map zero here is zero everywhere."""
    nonzero = [i for i, psi in enumerate(SPANNING_SET) if fn(psi) != 0]
    assert not nonzero, (
        f"{name} is non-zero on {len(nonzero)} of {len(SPANNING_SET)} spanning "
        f"vectors (first at index {nonzero[0] if nonzero else '-'}). It is "
        f"documented as an algebraic identity; it is no longer one.")


@pytest.mark.parametrize("name,fn", PSI_RESIDUALS)
def test_the_residual_vanishes_on_the_three_vertex_sums(name, fn) -> None:
    """Establishes the degree-≤2 premise instead of assuming it.

    A cubic or higher map can vanish on ``SPANNING_SET`` and be non-zero off
    it. If that ever happened here, the test above would still pass and the
    conclusion drawn from it would be false.
    """
    nonzero = [i for i, psi in enumerate(TRIPLE_SET) if fn(psi) != 0]
    assert not nonzero, (
        f"{name} vanishes on the spanning set but not on the three-vertex "
        f"sums ({len(nonzero)} of {len(TRIPLE_SET)} non-zero), so it is not "
        f"of degree ≤ 2 and the polarisation argument does not apply to it.")


@pytest.mark.parametrize("name,fn", PSI_RESIDUALS)
def test_the_residual_vanishes_on_every_structural_case(name, fn) -> None:
    """The spanning argument already covers these; this reads the result off
    directly on the labelled cases, so a failure names a recognisable Ψ."""
    nonzero = [label for label, psi in PSI_CASES + DYADIC if fn(psi) != 0]
    assert not nonzero, f"{name} is non-zero on: {nonzero}"


def test_r1_takes_no_psi() -> None:
    """Structural, not measured: R1 cannot depend on Ψ because it is not given one."""
    params = list(inspect.signature(r1_trace_closure).parameters)
    assert params == ["trace_sum"], (
        f"r1_trace_closure now takes {params}. If it has gained a Ψ argument "
        f"the degeneracy argument in this module's docstring needs redoing.")


def test_r1_is_the_residual_that_moves() -> None:
    """Without this the conclusion would be 'nothing constrains anything'.

    Over the closed range 0..435 the only counters R1 vanishes on are the two
    multiples of T(29) it names; on the other 434 it is non-zero.
    """
    zeros = [t for t in range(T29 + 1) if r1_trace_closure(Fraction(t)) == 0]
    assert zeros == [0, T29], (
        f"R1 vanishes at {zeros}; expected only the multiples of {T29} in "
        f"0..{T29}")


@pytest.mark.parametrize("trace", [0, 1, 7, 434, 435, 436, 869, 870, 871])
def test_closure_is_the_same_verdict_for_every_psi(trace: int) -> None:
    """The degeneracy itself, at a fixed counter, over the whole corpus."""
    verdicts = {audit_psi(psi, Fraction(trace)).closed for psi in ALL_PSI}
    assert len(verdicts) == 1, (
        f"at trace {trace} the audit returns both verdicts across "
        f"{len(ALL_PSI)} inputs — closure has become Ψ-dependent, and the "
        f"README's falsification criterion 2 must be revisited")
    assert verdicts == {trace % T29 == 0}, (
        f"at trace {trace} closure is {verdicts.pop()}; R1 says "
        f"{trace % T29 == 0}")


#: Three Psi that are structurally as different as the corpus gets: the zero
#: field, a single-vertex spike, and a field with six-digit denominators. The
#: counter sweep below runs two full periods per vector, so it is linear in
#: corpus size at 871 audits each; sweeping all of ``ALL_PSI`` would be
#: 650,000 audits to re-establish, one Psi at a time, what the spanning-set
#: tests above already prove for every Psi at once.
PERIOD_SWEEP_PSI = (ZERO, BASIS[0], dict(PSI_CASES)["fine_denominators"])


@pytest.mark.parametrize("psi", PERIOD_SWEEP_PSI,
                         ids=["zero", "basis_0", "fine_denominators"])
def test_closure_holds_exactly_on_multiples_of_435(psi) -> None:
    """Over two full periods: the composition of the four residuals.

    The tests above prove R2, R3 and R4 are zero for every Psi; this checks
    that ``audit_closed`` therefore reduces to R1 and fires exactly on the
    multiples of T(29), rather than on some other schedule.
    """
    closed = [t for t in range(2 * T29 + 1) if audit_closed(psi, Fraction(t))]
    assert closed == [0, T29, 2 * T29], (
        f"closure fired at {closed}, not at the multiples of {T29}")


def test_two_models_cannot_differ_on_this_metric() -> None:
    """States the consequence as a test rather than leaving it to prose.

    Whatever two arms generate, the vectors they encode to are elements of
    ℚ^16 and the verdicts above depend on none of them. So the *rate* over
    any two equal-length runs at the same counters is identical, and a
    criterion phrased as a difference between arms is zero by construction.
    """
    counters = [Fraction(t) for t in range(900)]
    arm_a = [audit_closed(ZERO, t) for t in counters]
    arm_b = [audit_closed(psi, t)
             for psi, t in zip(BASIS * 60, counters)]
    assert arm_a == arm_b, (
        "two arms produced different closure sequences at identical "
        "counters, which this module's docstring says is impossible")
    assert sum(arm_a) == 3 and sum(arm_b) == 3


def test_audit_closed_has_no_default_trace() -> None:
    """A default made R1 vacuously closed — 0 is a multiple of 435."""
    params = inspect.signature(audit_closed).parameters
    assert params["trace_sum"].default is inspect.Parameter.empty, (
        "audit_closed's trace_sum has a default again; with one, a caller "
        "with no trace gets a quarter of the audit answered for them")


def test_a_non_integer_trace_is_rejected() -> None:
    """R1 is a modular reduction; it has no meaning off the integers."""
    with pytest.raises(ValueError):
        r1_trace_closure(Fraction(1, 2))
    with pytest.raises(ValueError):
        all_residuals(ZERO, Fraction(435, 2))
