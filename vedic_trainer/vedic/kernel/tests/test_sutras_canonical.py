"""The 29 canonical sutras — completeness, structure, and the §12Y guarantee.

Every assertion here traces to the Vedic protocol v4.0 or to
``vedic_v18.24_full_kernel.html`` (§12Z, STRICT_SUTRA_KERNEL, SUTRA_KIND,
ALPHA.computeQ, §12Y).

What this file replaced
-----------------------
The formula tests used to check **eleven of the twenty-nine** operators —
S1, S2, S3, S4, S5, S6, S7, S9, S18, S26, S29 — each on a single hardcoded
Ψ at a single strength. The remaining eighteen (S8, S10–S17, S19–S25, S27,
S28) had no check that they computed anything in particular: they were
covered only by "does not crash", "is not the identity at α≠0" and
"stays in ℚ", all of which a wrong formula satisfies comfortably.

Nothing about that was necessary. Each of the seven kinds has one formula,
and applying the kind's formula to every member of the kind reproduces all
twenty-nine operators exactly. The per-kind reference below is written once
and checked against every sutra, at several strengths, over a corpus that
includes the degenerate inputs a single positive monotone Ψ never exercises.

Two rules this file follows throughout:

* **No single-input tests.** A property asserted on one Ψ is a property of
  that Ψ. Every behavioural test runs the whole corpus and reports every
  failure, not the first.
* **Parametrise per operator.** A loop over 29 sutras inside one test stops
  at the first failure and hides the other 28; one failing operator should
  report as one failure with 28 still checked.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from vedic.kernel import sutras_canonical as K


# ────────────────────────────────────────────────────────────── the corpus
# Imported, not re-declared: test_sutra_operands.py runs against the same
# vectors, so neither file can drift onto a weaker set than the other.

from vedic.kernel.tests.psi_corpus import (        # noqa: E402
    BASIS, PSI_CASES, SPANNING_SET, STRENGTHS, TRIPLE_SET, MONOTONE_PSI, ZERO,
)

PSI_LABELS = [label for label, _ in PSI_CASES]

#: A single Ψ kept for the tests that are genuinely about one input.
PSI = MONOTONE_PSI
STRENGTH = Fraction(50)


def test_the_corpus_is_not_degenerate() -> None:
    """Guards every corpus-driven test below.

    An empty or all-identical corpus would make them pass without checking
    anything, and a corpus of one vector would silently reintroduce exactly
    the weakness this file was rewritten to remove.
    """
    assert len(PSI_CASES) == 29, f"corpus is {len(PSI_CASES)} vectors"
    assert len({psi for _, psi in PSI_CASES}) == len(PSI_CASES), \
        "corpus contains duplicate vectors"
    assert all(len(psi) == 16 for _, psi in PSI_CASES)
    assert all(isinstance(x, Fraction) for _, psi in PSI_CASES for x in psi)
    # The properties the structured cases exist to exercise.
    by = dict(PSI_CASES)
    assert K.mean(by["mean_zero"]) == 0
    assert len(set(by["constant"])) == 1
    assert any(x < 0 for x in by["negative"])
    assert len(STRENGTHS) >= 5 and Fraction(0) in STRENGTHS
    # The spanning geometry the completeness argument below rests on.
    assert len(SPANNING_SET) == 137, f"spanning set is {len(SPANNING_SET)}"
    assert len(TRIPLE_SET) == 560, f"triple set is {len(TRIPLE_SET)}"
    assert ZERO in SPANNING_SET and all(b in SPANNING_SET for b in BASIS)


# ───────────────────────────────────────────── the per-kind reference formula

def reference_output(sid: int, psi: tuple[Fraction, ...],
                     strength: Fraction) -> tuple[Fraction, ...]:
    """Independent implementation of the spec formula for sutra ``sid``.

    Written from §4.1/§12Z, one branch per kind, and deliberately *not* by
    calling anything in ``sutras_canonical`` beyond the shared primitives
    (``alpha``, ``blend``, ``mean``, ``edge_mean``, ``hw``). If this were
    written by calling ``apply_sutra`` it would agree with it by
    construction and check nothing.

    Every one of the 29 operators is produced by exactly one of these seven
    branches, so there is no operator this reference cannot express — which
    is why checking only eleven of them was a choice rather than a limit.
    """
    if not 1 <= sid <= 29:
        raise ValueError(f"sutra id out of range: {sid}")
    w = K.alpha(sid, strength)
    kind = K.SUTRA_KIND[sid]
    m = K.mean(psi)

    if kind == "MULT":
        # Ψ'ᵢ = Ψᵢ·(1 + α·Ψ_{i⊕1})
        return tuple(psi[i] * (1 + w * psi[i ^ 1]) for i in range(16))
    if kind == "REFL":
        if sid == 5:
            # S5 Śūnyam Sāmyasamuccaye: the zero-sum case, target = −Ψ_c.
            return tuple(K.blend(psi[i], -psi[i ^ 15], w) for i in range(16))
        return tuple(K.blend(psi[i], (psi[i] + psi[i ^ 15]) / 2, w)
                     for i in range(16))
    if kind == "CONV":
        conv = [sum((psi[j] * psi[i ^ j] for j in range(16)), Fraction(0)) / 16
                for i in range(16)]
        return tuple(K.blend(psi[i], conv[i], w) for i in range(16))
    if kind == "DIFF":
        return tuple(K.blend(psi[i], K.edge_mean(psi, i), w) for i in range(16))
    if kind == "PERM":
        step = 1 << ((sid + 1) & 3)
        return tuple(K.blend(psi[i], psi[i ^ step], w) for i in range(16))
    if kind == "DIV":
        return tuple(
            K.blend(psi[i], m + Fraction(K.hw(i), 4) * (K.edge_mean(psi, i) - m), w)
            for i in range(16))
    if kind == "MOD":
        return tuple(K.blend(psi[i], m, w) for i in range(16))
    raise AssertionError(f"sutra {sid} has unhandled kind {kind!r}")


def test_the_reference_covers_every_kind_in_the_table() -> None:
    """No sutra may fall through to the AssertionError branch."""
    kinds = {K.SUTRA_KIND[s.id] for s in K.SUTRAS}
    assert kinds == {"MULT", "REFL", "CONV", "DIV", "DIFF", "PERM", "MOD"}
    for s in K.SUTRAS:
        reference_output(s.id, PSI, STRENGTH)   # must not raise


@pytest.mark.parametrize("sid", K.ALL)
def test_every_sutra_matches_its_kind_formula(sid: int) -> None:
    """All 29 operators, the whole corpus, every strength.

    This is the check eighteen sutras never had. A mismatch names the
    operator, the input and the strength.
    """
    mismatches: list[str] = []
    for label, psi in PSI_CASES:
        for strength in STRENGTHS:
            got = K.apply_sutra(sid, psi, strength)
            want = reference_output(sid, psi, strength)
            if got != want:
                first = next(i for i in range(16) if got[i] != want[i])
                mismatches.append(
                    f"{label} @ strength {strength}: component {first} "
                    f"is {got[first]}, spec says {want[first]}")
    assert not mismatches, (
        f"S{sid} ({K.NAMES[sid]}, kind {K.SUTRA_KIND[sid]}) does not match "
        f"its kind formula in {len(mismatches)} case(s):\n  "
        + "\n  ".join(mismatches))


def _reconstruct_degree_two(fn, strength):
    """Recover (c, L, Q) of F(Ψ) = c + L(Ψ) + Q(Ψ,Ψ) from the spanning set.

    Only the 137 evaluations in ``SPANNING_SET`` are used, via the identities
    in ``psi_corpus``. If ``fn`` has degree at most two this determines it
    completely.
    """
    c = fn(ZERO)
    fi = [fn(b) for b in BASIS]
    q = {}
    for i in range(16):
        for j in range(i + 1, 16):
            fij = fn(tuple(a + b for a, b in zip(BASIS[i], BASIS[j])))
            q[(i, j)] = tuple((a - b - cc + d) / 2
                              for a, b, cc, d in zip(fij, fi[i], fi[j], c))
    lin = [tuple(a - b for a, b in zip(fi[i], c)) for i in range(16)]
    return c, lin, q


def _predict(c, lin, q, support):
    """Evaluate the reconstructed polynomial on a 0/1 vector's support."""
    out = list(c)
    for i in support:
        out = [o + v for o, v in zip(out, lin[i])]
    for a in range(len(support)):
        for b in range(a + 1, len(support)):
            out = [o + 2 * v for o, v in zip(out, q[(support[a], support[b])])]
    return tuple(out)


@pytest.mark.parametrize("sid", K.ALL)
def test_every_operator_has_degree_at_most_two_in_psi(sid: int) -> None:
    """Establishes the premise the completeness argument needs.

    The spanning set determines a map only if the map is degree ≤ 2. That is
    checked, not assumed: the polynomial reconstructed from the 137 spanning
    evaluations must reproduce the operator on all 560 three-vertex sums,
    which a map of degree three or higher does not.
    """
    strength = STRENGTH

    def fn(psi):
        return K.apply_sutra(sid, psi, strength)

    c, lin, q = _reconstruct_degree_two(fn, strength)
    failures = []
    for t in TRIPLE_SET:
        support = tuple(i for i, x in enumerate(t) if x != 0)
        if _predict(c, lin, q, support) != fn(t):
            failures.append(support)
    assert not failures, (
        f"S{sid} is not degree ≤ 2 in Ψ: its degree-2 reconstruction from the "
        f"spanning set disagrees on {len(failures)} of {len(TRIPLE_SET)} "
        f"three-vertex sums (first: {failures[0]}). The completeness argument "
        f"for the spanning set does not apply to it.")


@pytest.mark.parametrize("sid", K.ALL)
def test_the_kind_formula_agrees_on_all_of_q16(sid: int) -> None:
    """Agreement everywhere, proved rather than sampled.

    Both ``apply_sutra`` and the reference are degree ≤ 2 (established above),
    so agreeing on the 137-vector spanning set means they are the same map on
    all of ℚ^16. This replaces a sample of random vectors, which established
    the property on the vectors drawn and nothing about the rest.
    """
    for strength in STRENGTHS:
        mismatches = []
        for psi in SPANNING_SET:
            got = K.apply_sutra(sid, psi, strength)
            want = reference_output(sid, psi, strength)
            if got != want:
                support = tuple(i for i, x in enumerate(psi) if x != 0)
                mismatches.append(support)
        assert not mismatches, (
            f"S{sid} ({K.NAMES[sid]}, kind {K.SUTRA_KIND[sid]}) disagrees with "
            f"its kind formula at strength {strength} on {len(mismatches)} of "
            f"{len(SPANNING_SET)} spanning vectors (first support: "
            f"{mismatches[0]}) — so the two maps differ on ℚ^16")


@pytest.mark.parametrize("sid", K.ALL)
def test_the_reference_is_also_degree_at_most_two(sid: int) -> None:
    """The completeness argument needs *both* maps to be degree ≤ 2.

    Checking only ``apply_sutra`` would leave the possibility that the
    reference is higher-degree and merely coincides on the spanning set.
    """
    def fn(psi):
        return reference_output(sid, psi, STRENGTH)

    c, lin, q = _reconstruct_degree_two(fn, STRENGTH)
    failures = [t for t in TRIPLE_SET
                if _predict(c, lin, q,
                            tuple(i for i, x in enumerate(t) if x != 0)) != fn(t)]
    assert not failures, (
        f"the reference formula for S{sid} is not degree ≤ 2; agreement on "
        f"the spanning set would not imply agreement on ℚ^16")


@pytest.mark.parametrize("sid", K.ALL)
def test_alpha_zero_is_the_identity_for_every_input(sid: int) -> None:
    """§12Y, over the corpus rather than one vector.

    The guarantee the whole design rests on: α → 0 ⇒ every operator is the
    identity. Exact equality, not approximate, and on inputs including the
    zero vector, a constant field and a mean-zero field.
    """
    moved = [label for label, psi in PSI_CASES
             if K.apply_sutra(sid, psi, Fraction(0)) != psi]
    assert not moved, f"S{sid} moved at α=0 on: {moved}"


@pytest.mark.parametrize("sid", K.ALL)
def test_every_sutra_is_a_total_exact_endomorphism(sid: int) -> None:
    """ℚ^16 → ℚ^16 on every input at every strength, with no exceptions."""
    for label, psi in PSI_CASES:
        for strength in STRENGTHS:
            out = K.apply_sutra(sid, psi, strength)
            assert len(out) == 16, \
                f"S{sid} returned length {len(out)} on {label} @ {strength}"
            assert all(isinstance(x, Fraction) for x in out), \
                f"S{sid} left ℚ on {label} @ {strength}"


@pytest.mark.parametrize("sid", K.ALL)
def test_no_sutra_is_dead(sid: int) -> None:
    """Every operator must actually move some field at nonzero strength.

    Stated as "there exists an input it moves" rather than "it moves this
    input", because the zero vector and the constant field are fixed points
    of several operators by construction — a formulation that tested one
    input would have to either exclude those or assert something false.
    """
    movers = [label for label, psi in PSI_CASES
              if K.apply_sutra(sid, psi, STRENGTH) != psi]
    assert movers, (
        f"S{sid} is the identity on all {len(PSI_CASES)} corpus inputs at "
        f"strength {STRENGTH} — it cannot affect any computation")


def test_all_29_operators_are_pairwise_distinct() -> None:
    """29 operators, not 29 names for a smaller set.

    The claim is that no two ids denote the same map, so the test is: for
    every one of the 406 pairs there exists a corpus input on which they
    disagree. It is deliberately *not* "they disagree on every input" —
    distinct maps can and do coincide on particular fields, and asserting
    otherwise reports arithmetic as a defect.

    S7 and S28 are the worked example. α(n) is proportional to n and
    28 = 4·7, so α(28) = 4·α(7) exactly. On an alternating field the S7
    (PERM, step 1) target is −Ψᵢ and the S28 (DIFF) target is the edge mean,
    which there equals Ψᵢ/2; blending gives Ψᵢ(1 − 2α₇) and Ψᵢ(1 − α₂₈/2),
    equal precisely because α₂₈ = 4α₇. They are still different operators,
    and this test proves it by finding an input where they differ.

    The old form took one representative per *kind* — seven outputs on one
    Ψ — so it could not have caught two ids of the same kind denoting the
    same map, which is the collision actually worth catching.
    """
    identical: list[str] = []
    for a in K.ALL:
        for b in K.ALL:
            if b <= a:
                continue
            differs = any(
                K.apply_sutra(a, psi, STRENGTH) != K.apply_sutra(b, psi, STRENGTH)
                for _, psi in PSI_CASES)
            if not differs:
                identical.append(f"S{a} and S{b}")
    assert not identical, (
        f"{len(identical)} operator pair(s) agree on every corpus input, so "
        f"they are not distinct maps: {identical}")


def test_two_distinct_operators_may_still_coincide_on_a_special_field() -> None:
    """Pins the S7/S28 coincidence documented above.

    Without this, a future change to α or to the DIFF formula could remove
    the coincidence and nobody would know the reasoning in the test above
    had gone stale — or reintroduce it as a genuine duplication.
    """
    alternating = dict(PSI_CASES)["alternating"]
    assert K.alpha(28, STRENGTH) == 4 * K.alpha(7, STRENGTH), \
        "α is no longer proportional to id; the coincidence below is unexplained"
    assert K.apply_sutra(7, alternating, STRENGTH) == \
        K.apply_sutra(28, alternating, STRENGTH)
    # …and they are nonetheless different operators.
    assert any(K.apply_sutra(7, psi, STRENGTH) != K.apply_sutra(28, psi, STRENGTH)
               for _, psi in PSI_CASES)


# ──────────────────────────────────────────────── linearity, over the corpus

LINEAR_IDS = tuple(s.id for s in K.SUTRAS if K.is_linear(s.id))
QUADRATIC_IDS = tuple(s.id for s in K.SUTRAS if not K.is_linear(s.id))


def _corpus_pairs() -> tuple[tuple[str, tuple, tuple], ...]:
    """(label, f, g) pairs drawn from the corpus, plus random pairs."""
    out = []
    names = [n for n, _ in PSI_CASES]
    by = dict(PSI_CASES)
    # Every corpus vector paired with its successor, so each appears on both
    # sides. No random partners: a pair drawn from a PRNG tests the pair drawn.
    for i in range(len(names)):
        a, b = names[i], names[(i + 1) % len(names)]
        out.append((f"{a}|{b}", by[a], by[b]))
    # Every basis vector against every other: 120 pairs, exhaustive over the
    # two-vertex geometry that determines a degree-2 map.
    for i in range(16):
        for j in range(i + 1, 16):
            out.append((f"e{i}|e{j}", BASIS[i], BASIS[j]))
    return tuple(out)


PAIRS = _corpus_pairs()


def test_there_are_pairs_to_test_superposition_with() -> None:
    assert len(PAIRS) == len(PSI_CASES) + 120, f"{len(PAIRS)} pairs"
    assert all(f != g for _, f, g in PAIRS), \
        "a pair with f == g makes superposition trivially symmetric"


def test_linearity_status_is_declared_and_correct() -> None:
    """MULT multiplies Ψ by Ψ and CONV convolves Ψ with itself, so both are
    quadratic. The other five kinds are linear."""
    assert set(QUADRATIC_IDS) == {s.id for s in K.SUTRAS
                                  if s.kind in ("MULT", "CONV")}
    assert set(LINEAR_IDS) | set(QUADRATIC_IDS) == set(K.ALL)
    assert set(LINEAR_IDS) & set(QUADRATIC_IDS) == set()


@pytest.mark.parametrize("sid", LINEAR_IDS)
def test_linear_operators_really_are_linear(sid: int) -> None:
    """Superposition over the whole pair corpus at several strengths.

    The old version used one hand-built (f, g) pair at one strength. A
    linear-declared operator with a stray constant term agrees on plenty of
    individual pairs; it does not agree on twenty-six of them at six
    strengths.
    """
    failures: list[str] = []
    for label, f, g in PAIRS:
        fg = tuple(a + b for a, b in zip(f, g))
        for strength in STRENGTHS:
            lhs = K.apply_sutra(sid, fg, strength)
            rhs = tuple(a + b for a, b in zip(K.apply_sutra(sid, f, strength),
                                              K.apply_sutra(sid, g, strength)))
            if lhs != rhs:
                failures.append(f"{label} @ {strength}")
    assert not failures, (
        f"S{sid} is declared linear but fails superposition on "
        f"{len(failures)} case(s): {failures}")


def _quadratic_part(sid: int, strength: Fraction) -> dict:
    """Q(eᵢ,eⱼ) for every i<j, recovered exactly from the spanning set."""
    def fn(psi):
        return K.apply_sutra(sid, psi, strength)
    _, _, q = _reconstruct_degree_two(fn, strength)
    return q


@pytest.mark.parametrize("sid", QUADRATIC_IDS)
def test_quadratic_operators_have_a_nonzero_quadratic_part(sid: int) -> None:
    """The exact statement, replacing a count of failing pairs.

    An earlier version asserted that superposition failed on at least half of
    the sampled pairs. That threshold was arbitrary and wrong: for a MULT
    operator the quadratic term is Ψᵢ·Ψ_{i⊕1}, which is nonzero only when both
    i and i⊕1 lie in the support — 8 of the 120 basis pairs. "Linear on 127 of
    149 pairs" was correct arithmetic being reported as a defect.

    What actually distinguishes a quadratic operator from a linear one is
    whether its bilinear part Q is identically zero, and Q is recovered
    exactly from the spanning set. This says exactly that, over every
    strength, with no threshold.
    """
    for strength in STRENGTHS:
        q = _quadratic_part(sid, strength)
        nonzero = [(i, j) for (i, j), v in q.items() if any(x != 0 for x in v)]
        if strength == 0:
            # α = 0 is the identity by §12Y, so Q vanishes there for every
            # operator. Asserting otherwise would contradict that guarantee.
            assert not nonzero, f"S{sid} has a quadratic part at α = 0"
            continue
        assert nonzero, (
            f"S{sid} is declared quadratic but its bilinear part is "
            f"identically zero at strength {strength}: it is a linear map")


@pytest.mark.parametrize("sid", LINEAR_IDS)
def test_linear_operators_have_no_quadratic_part(sid: int) -> None:
    """The converse, and the stronger half of the linearity claim.

    Superposition on a set of pairs is evidence; Q ≡ 0 recovered from the
    spanning set is the property itself, and it holds for all of ℚ^16.
    """
    for strength in STRENGTHS:
        q = _quadratic_part(sid, strength)
        nonzero = [(i, j) for (i, j), v in q.items() if any(x != 0 for x in v)]
        assert not nonzero, (
            f"S{sid} is declared linear but has a nonzero bilinear part at "
            f"strength {strength} on {len(nonzero)} vertex pairs "
            f"(first {nonzero[0]})")


@pytest.mark.parametrize("sid", QUADRATIC_IDS)
def test_quadratic_operators_fail_superposition_somewhere(sid: int) -> None:
    """A concrete witness, in addition to Q ≠ 0 above.

    Kept because it exercises the operator through its public entry point
    rather than through a reconstruction, so a bug in the reconstruction
    cannot make both tests pass.
    """
    witnesses = [label for label, f, g in PAIRS
                 if K.apply_sutra(sid, tuple(a + b for a, b in zip(f, g)), STRENGTH)
                 != tuple(a + b for a, b in
                          zip(K.apply_sutra(sid, f, STRENGTH),
                              K.apply_sutra(sid, g, STRENGTH)))]
    assert witnesses, \
        f"S{sid} is declared quadratic but satisfies superposition on all " \
        f"{len(PAIRS)} pairs"


@pytest.mark.parametrize("sid", QUADRATIC_IDS)
def test_quadratic_operators_refuse_a_matrix_representation(sid: int) -> None:
    for strength in STRENGTHS:
        with pytest.raises(ValueError, match="quadratic"):
            K.operator_matrix(sid, strength)


@pytest.mark.parametrize("sid", LINEAR_IDS)
def test_linear_operator_matrices_reproduce_the_action(sid: int) -> None:
    """M·Ψ = operator(Ψ), for every Ψ in the corpus and every strength.

    One Ψ cannot distinguish a correct matrix from one that happens to agree
    on that vector — a 16×16 matrix has 256 entries and one equation per
    component gives 16 constraints.
    """
    failures: list[str] = []
    for strength in STRENGTHS:
        M = K.operator_matrix(sid, strength)
        assert len(M) == 16 and all(len(row) == 16 for row in M), \
            f"S{sid} matrix is not 16×16 at strength {strength}"
        for label, psi in PSI_CASES:
            got = tuple(sum((M[i][j] * psi[j] for j in range(16)), Fraction(0))
                        for i in range(16))
            if got != K.apply_sutra(sid, psi, strength):
                failures.append(f"{label} @ {strength}")
    assert not failures, \
        f"S{sid} matrix does not reproduce the action on: {failures}"


@pytest.mark.parametrize("sid", LINEAR_IDS)
def test_matrix_at_zero_strength_is_the_identity(sid: int) -> None:
    M = K.operator_matrix(sid, Fraction(0))
    for i in range(16):
        for j in range(16):
            assert M[i][j] == (1 if i == j else 0), \
                f"S{sid} matrix at α=0 has M[{i}][{j}] = {M[i][j]}"


@pytest.mark.parametrize("sid", LINEAR_IDS)
def test_linear_operators_are_reversible_at_every_strength(sid: int) -> None:
    """det ≠ 0. Checked across the strength range, not only at 50.

    Reversibility is a property of the matrix at a given α; an operator can
    be invertible at α(50) and singular at α(250).
    """
    singular = [str(s) for s in STRENGTHS if not K.is_reversible(sid, s)]
    assert not singular, f"S{sid} is singular at strength(s): {singular}"


# ──────────────────────────────────────────────────────── cascade behaviour

def test_full_cascade_at_zero_strength_is_the_identity() -> None:
    for label, psi in PSI_CASES:
        assert K.apply_all(psi, Fraction(0)) == psi, f"cascade moved {label} at α=0"


def test_full_series_cascade_does_not_annihilate() -> None:
    """The canonical operators do not collapse the field.

    An earlier non-canonical implementation had S20→S21→S22 acting as a
    rank-1 projection, an absolute value and a pair difference, which
    annihilated every input. The old test checked six random seeds and only
    that *some* component was nonzero; a cascade that mapped everything to
    (ε, 0, 0, …) would have passed. This runs the whole corpus and requires
    the output to differ from the zero vector on every non-degenerate input.
    """
    zero = tuple(Fraction(0) for _ in range(16))
    annihilated = [label for label, psi in PSI_CASES
                   if psi != zero and K.apply_all(psi, STRENGTH) == zero]
    assert not annihilated, f"cascade annihilated: {annihilated}"


def test_cascade_preserves_exactness_on_every_input() -> None:
    for label, psi in PSI_CASES:
        for strength in STRENGTHS:
            out = K.apply_all(psi, strength)
            assert len(out) == 16, f"{label} @ {strength}"
            assert all(isinstance(x, Fraction) for x in out), \
                f"cascade left ℚ on {label} @ {strength}"


def test_conservation_cores_drift_less_than_the_full_cascade() -> None:
    """§3.8: the wormhole and symmetry cores are conservation cores.

    Over the corpus. Inputs with zero norm are excluded because the relative
    drift is undefined there, and that exclusion is stated rather than
    achieved by picking one input where it does not arise.
    """
    checked = 0
    for label, psi in PSI_CASES:
        q0 = K.norm_sq(psi)
        if q0 == 0:
            continue
        def rel(order):
            return abs(K.norm_sq(K.apply_all(psi, STRENGTH, order)) - q0) / q0
        full = rel(K.ALL)
        assert rel(K.WORMHOLE_CORE) < full, f"wormhole core on {label}"
        assert rel(K.SYMMETRY_CORE) < full, f"symmetry core on {label}"
        checked += 1
    assert checked >= len(PSI_CASES) - 1, \
        f"only {checked} inputs had nonzero norm; the test is near-vacuous"


def test_conservation_cores_are_subsets_of_the_29() -> None:
    for core in (K.WORMHOLE_CORE, K.SYMMETRY_CORE):
        assert set(core) <= set(K.ALL)
        assert core, "an empty core would satisfy every drift comparison"


@pytest.mark.parametrize("sid", K.ALL)
def test_drift_is_zero_at_zero_strength(sid: int) -> None:
    for label, psi in PSI_CASES:
        assert K.drift(sid, psi, Fraction(0)) == 0, f"S{sid} on {label}"


def test_drift_ranker_covers_all_29_and_is_sorted() -> None:
    for label, psi in PSI_CASES:
        ranked = K.rank_by_drift(psi, STRENGTH)
        assert len(ranked) == 29, f"{label}: ranked {len(ranked)}"
        assert sorted(sid for sid, _ in ranked) == list(range(1, 30)), label
        assert [d for _, d in ranked] == sorted(d for _, d in ranked), label


# ────────────────────────────────────────────────────────────── completeness

def test_all_29_are_present() -> None:
    """All 29. Not 28, not 'the main 16' — the complete set."""
    assert K.N_SUTRAS == 29
    assert len(K.SUTRAS) == 29
    assert K.ALL == tuple(range(1, 30))
    assert [s.id for s in K.SUTRAS] == list(range(1, 30))


def test_every_sutra_has_name_sanskrit_kind_category_coefficient() -> None:
    for s in K.SUTRAS:
        assert s.name and s.sanskrit and s.kind and s.category, f"S{s.id}"
        assert isinstance(s.coefficient, Fraction), f"S{s.id}"


def test_category_census_matches_protocol_4_1() -> None:
    """§4.1 seven categories; the engine gives S29 its own CONSERVATION desc."""
    assert Counter(s.category for s in K.SUTRAS) == {
        "MULTIPLICATIVE": 4, "REFLECTIVE": 5, "CONVOLUTIVE": 3,
        "DIVISIVE": 5, "PERMUTATIVE": 3, "DIFFUSIVE": 4,
        "MODULAR": 4, "CONSERVATION": 1,
    }


def test_category_membership_matches_protocol_4_1() -> None:
    def ids(cat):
        return tuple(s.id for s in K.SUTRAS if s.category == cat)
    assert ids("MULTIPLICATIVE") == (1, 10, 14, 15)
    assert ids("REFLECTIVE") == (2, 5, 12, 22, 23)
    assert ids("CONVOLUTIVE") == (3, 11, 25)
    assert ids("DIVISIVE") == (4, 8, 13, 16, 19)
    assert ids("DIFFUSIVE") == (9, 17, 27, 28)
    assert ids("PERMUTATIVE") == (6, 7, 26)
    assert ids("MODULAR") + ids("CONSERVATION") == (18, 20, 21, 24, 29)


def test_sutra_kind_matches_engine_table() -> None:
    """SUTRA_KIND transcribed from vedic_v18.24_full_kernel.html:3558."""
    assert K.SUTRA_KIND[1:] == (
        "MULT", "REFL", "CONV", "DIV", "REFL", "PERM", "PERM", "DIV", "DIFF",
        "MULT", "CONV", "REFL", "DIV", "MULT", "MULT", "DIV", "DIFF", "MOD",
        "DIV", "MOD", "MOD", "REFL", "REFL", "MOD", "CONV", "PERM", "DIFF",
        "DIFF", "MOD",
    )


def test_category_and_kind_agree_for_every_sutra() -> None:
    """The two tables above are independent transcriptions of one fact.

    Checked against each other rather than only against their own literals,
    so a typo that agrees with itself in both places still has to survive a
    consistency constraint.
    """
    expected = {"MULTIPLICATIVE": "MULT", "REFLECTIVE": "REFL",
                "CONVOLUTIVE": "CONV", "DIVISIVE": "DIV",
                "PERMUTATIVE": "PERM", "DIFFUSIVE": "DIFF",
                "MODULAR": "MOD", "CONSERVATION": "MOD"}
    for s in K.SUTRAS:
        assert K.SUTRA_KIND[s.id] == expected[s.category], (
            f"S{s.id}: category {s.category} but kind {K.SUTRA_KIND[s.id]}")


def test_all_29_coefficients_are_exact_rationals() -> None:
    assert len(K.COEFFICIENT) == 29
    assert set(K.COEFFICIENT) == set(K.ALL)
    assert all(isinstance(v, Fraction) for v in K.COEFFICIENT.values())


def test_named_coefficients_match_protocol_4_3() -> None:
    assert K.COEFFICIENT[1] == Fraction(12586269025, 7778742049)   # φ
    assert K.COEFFICIENT[3] == Fraction(355, 113)                  # π Milü
    assert K.COEFFICIENT[4] == Fraction(577, 408)                  # √2 Pell
    assert K.COEFFICIENT[7] == Fraction(97, 56)                    # √3
    assert K.COEFFICIENT[21] == Fraction(6931472, 10000000)        # ln 2


def test_phi_coefficient_is_the_fibonacci_convergent() -> None:
    """S1's coefficient is F₅₀/F₄₉ — computed, not compared to a float.

    The old form asserted ``abs(float(coef) - 1.6180339887498949) == 0.0``,
    which checks that the coefficient rounds to a particular double. Many
    rationals do. It says nothing about the coefficient being a Fibonacci
    convergent, which is the actual claim.
    """
    a, b = 0, 1                      # F₀, F₁
    for _ in range(49):
        a, b = b, a + b              # after the loop: a = F₄₉, b = F₅₀
    assert (a, b) == (7778742049, 12586269025), \
        f"Fibonacci computation is wrong: F49={a}, F50={b}"
    assert K.COEFFICIENT[1] == Fraction(b, a)
    # And it really is a convergent of φ: F₅₀·F₄₈ − F₄₉² = ±1 (Cassini).
    x, y = 0, 1
    for _ in range(48):
        x, y = y, x + y              # after 48 steps: x = F₄₈, y = F₄₉
    assert (x, y) == (4807526976, 7778742049), f"F48={x}, F49={y}"
    assert b * x - a * a == -1, \
        "Cassini's identity fails; these are not consecutive Fibonacci numbers"


def test_other_named_coefficients_are_the_stated_approximants() -> None:
    """Each named constant is checked as the rational it claims to be.

    355/113 is π's Milü convergent, 577/408 the Pell approximant to √2,
    97/56 to √3. Verified by the defining property (cross-multiplied, exact)
    rather than by float comparison.
    """
    # 577/408 approximates √2: 577² − 2·408² = 1 (Pell).
    assert 577 ** 2 - 2 * 408 ** 2 == 1
    # 97/56 approximates √3: 97² − 3·56² = 1.
    assert 97 ** 2 - 3 * 56 ** 2 == 1
    # 355/113 is a convergent of π: |355/113 − π| < 1/113² by construction of
    # a continued-fraction convergent; checked against exact bounds on π.
    lo, hi = Fraction(31415926535, 10 ** 10), Fraction(31415926536, 10 ** 10)
    assert lo < Fraction(355, 113) < hi + Fraction(1, 10 ** 6)


# ────────────────────────────────────────────────────── triangular identity

def test_delta_sum_is_t29() -> None:
    """§8 invariants 1 and 2: Σδ(1..29) = T(29) = 435."""
    assert sum(s.delta for s in K.SUTRAS) == 435
    assert K.SUTRA_SUM == 435 == 29 * 30 // 2
    assert [s.delta for s in K.SUTRAS] == list(range(1, 30))


@pytest.mark.parametrize("sid", K.ALL)
def test_alpha_is_the_triangular_weight(sid: int) -> None:
    """α(n) = (n/435)·(strength/100) exactly, for every id and strength.

    The old test spot-checked three (id, strength) pairs. This is the closed
    form, checked everywhere it is used.
    """
    for strength in STRENGTHS:
        assert K.alpha(sid, strength) == \
            Fraction(sid, 435) * strength / 100, f"S{sid} @ {strength}"


def test_alpha_is_monotone_in_id() -> None:
    for strength in STRENGTHS:
        if strength == 0:
            continue
        a = [K.alpha(i, strength) for i in K.ALL]
        assert a == sorted(a) and a[0] < a[-1], f"strength {strength}"


def test_alpha_rejects_out_of_range_ids() -> None:
    for bad in (0, 30, -1, 1000):
        with pytest.raises(ValueError):
            K.alpha(bad, STRENGTH)


# ─────────────────────────────────────────────────────────────── substrate

def test_hamming_weight_and_complement() -> None:
    assert [K.hw(v) for v in range(16)] == [bin(v).count("1") for v in range(16)]
    assert all(K.comp(K.comp(v)) == v for v in range(16))
    assert all(K.hw(v) + K.hw(K.comp(v)) == 4 for v in range(16))


def test_every_vertex_has_four_neighbours_at_hamming_distance_one() -> None:
    for v in range(16):
        nb = K.neighbors(v)
        assert len(set(nb)) == 4, v
        assert all(bin(v ^ j).count("1") == 1 for j in nb), v


def test_edge_mean_is_the_mean_of_the_four_neighbours() -> None:
    """``edge_mean`` is used by the DIFF and DIV references above, so it is
    checked independently rather than inherited from them."""
    for label, psi in PSI_CASES:
        for v in range(16):
            want = sum((psi[j] for j in K.neighbors(v)), Fraction(0)) / 4
            assert K.edge_mean(psi, v) == want, f"{label} vertex {v}"


def test_mean_is_the_exact_arithmetic_mean() -> None:
    for label, psi in PSI_CASES:
        assert K.mean(psi) == sum(psi, Fraction(0)) / 16, label


def test_blend_is_exact_linear_interpolation() -> None:
    """blend(c, t, w) = c + (t − c)·w, at the endpoints and in between."""
    # An exhaustive grid rather than 200 draws: every combination of these
    # values is checked, so the result is a statement about the grid instead
    # of about whichever triples a PRNG produced.
    values = (Fraction(0), Fraction(1), Fraction(-1), Fraction(3, 5),
              Fraction(-7, 4), Fraction(9), Fraction(1, 100003))
    weights = values + (Fraction(1, 2), Fraction(5, 3), Fraction(-2))
    for c in values:
        for t in values:
            for w in weights:
                assert K.blend(c, t, w) == c + (t - c) * w, f"{c}, {t}, {w}"
    for c, t in ((Fraction(3, 5), Fraction(-2, 7)), (Fraction(0), Fraction(1))):
        assert K.blend(c, t, Fraction(0)) == c
        assert K.blend(c, t, Fraction(1)) == t


# ─────────────────────────────────────────────────────────────── contracts

@pytest.mark.parametrize("bad", [0, 30, -1, 1000])
def test_out_of_range_sutra_id_raises(bad: int) -> None:
    with pytest.raises(ValueError):
        K.apply_sutra(bad, PSI, STRENGTH)


@pytest.mark.parametrize("n", [0, 1, 8, 15, 17, 32])
def test_wrong_vertex_count_raises(n: int) -> None:
    """Every wrong length, not just 8."""
    bad = tuple(Fraction(0) for _ in range(n))
    for sid in K.ALL:
        with pytest.raises(ValueError):
            K.apply_sutra(sid, bad, STRENGTH)


# ═══════════════════════════════ Gate E — operator records (blueprint)

def test_every_sutra_has_a_complete_operator_record() -> None:
    recs = K.all_operator_records()
    assert len(recs) == 29
    for r in recs:
        assert r.domain == "ℚ^16 over V4 = Z₂⁴"
        assert r.codomain == "ℚ^16 over V4 = Z₂⁴"
        assert r.decomposition and r.intensional


def test_determinant_is_exact_and_matches_known_cases() -> None:
    ident = tuple(tuple(Fraction(1 if i == j else 0) for j in range(4))
                  for i in range(4))
    assert K.determinant(ident) == 1
    swap = (ident[1], ident[0], ident[2], ident[3])
    assert K.determinant(swap) == -1
    # A singular matrix and a scaled one, so the function is exercised beyond
    # permutations of the identity.
    singular = (ident[0], ident[0], ident[2], ident[3])
    assert K.determinant(singular) == 0
    scaled = tuple(tuple(x * 3 for x in row) for row in ident)
    assert K.determinant(scaled) == 3 ** 4


def test_intensional_evidence_is_labelled_uncertified() -> None:
    """The blueprint says these are generic constructions, not proven Vedic
    decompositions. That label must not be quietly upgraded."""
    for r in K.all_operator_records():
        assert "UNCERTIFIED" in r.intensional
