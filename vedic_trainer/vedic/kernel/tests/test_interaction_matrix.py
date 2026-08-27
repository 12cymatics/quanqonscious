"""All 30 algebraic identities in INTERACTIONS hold bit-exactly over ℚ.

What this file replaced
-----------------------
The whole file was nineteen lines and one loop::

    def test_all_identities_hold(q16_corpus_pairs):
        failures = []
        for i, (psi, phi) in enumerate(q16_corpus_pairs):
            for name, ok in verify_all(psi, phi):
                if not ok:
                    failures.append((name, i))
        assert not failures, f"identity failures: {failures[:10]}"

Three problems, in increasing order of seriousness:

* ``failures[:10]`` truncated the report, so a run where all thirty
  identities broke showed ten of them.
* One test for thirty identities: a failure reported as one red test, and
  the identity names were buried in a message rather than in test ids.
* **Nothing checked that ``verify_all`` returned anything.** If it had
  returned an empty tuple — a filtered list, an early return, a renamed
  attribute — the inner loop would not execute and the test would pass
  while checking nothing at all. The only other assertion in the file was
  ``len(INTERACTIONS) == 30``, which does not constrain what ``verify_all``
  does with them.

The inputs were also 50 random unit-scale pairs and nothing else. The
structured corpus is used here too, so the identities are exercised on the
zero field, a constant field, an alternating field and a mean-zero field.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from vedic.kernel.interaction_matrix import INTERACTIONS, verify_all
from vedic.kernel.q import Q16
from vedic.kernel.tests.psi_corpus import BASIS, PHI, PSI_CASES

IDENTITY_NAMES = tuple(i.name for i in INTERACTIONS)


def _pairs() -> tuple[tuple[str, Q16, Q16], ...]:
    """(label, Ψ, Φ) — structured and exhaustive, with nothing sampled.

    Three families, none of them drawn from a generator:

    * each corpus vector against its successor, so every one appears on both
      sides of a binary identity;
    * each corpus vector against the fixed second field ``PHI``, so no
      identity is only ever seen with two structured fields;
    * every unordered pair of basis vectors — all 120 — which is exhaustive
      over the two-vertex geometry the arity-2 identities range over.

    The previous version added thirty random pairs. Those established the
    identities on the thirty pairs drawn and said nothing about any other,
    and the count was a knob with no justification behind it.
    """
    # Deduplicated on the (Ψ, Φ) values, first label wins. The corpus
    # contains basis_0..basis_15, so its successor chain re-derives pairs the
    # exhaustive basis loop already yields; keeping both would inflate the
    # count without adding an input and would make the uniqueness guard below
    # something to relax rather than something to satisfy.
    seen: dict[tuple, tuple[str, Q16, Q16]] = {}

    def add(label: str, psi: Q16, phi: Q16) -> None:
        seen.setdefault((psi, phi), (label, psi, phi))

    labels = [n for n, _ in PSI_CASES]
    by = dict(PSI_CASES)
    for i, label in enumerate(labels):
        nxt = labels[(i + 1) % len(labels)]
        add(f"{label}|{nxt}", by[label], by[nxt])
        add(f"{label}|phi", by[label], PHI)
    for i in range(16):
        for j in range(i + 1, 16):
            add(f"e{i}|e{j}", BASIS[i], BASIS[j])
    return tuple(seen.values())


PAIRS = _pairs()

#: The full results matrix: every identity against every pair, computed once.
#: The two parametrised tests below are transposes of this same matrix, so
#: computing it per test would run every identity thirty times over. Coverage
#: is identical; only the redundant recomputation is gone.
RESULTS: dict[str, dict[str, bool]] = {
    label: dict(verify_all(psi, phi)) for label, psi, phi in PAIRS
}


# ─────────────────────────────────────────────────────── non-vacuity guards

def test_interactions_count() -> None:
    assert len(INTERACTIONS) == 30


def test_identity_names_are_unique() -> None:
    """Two identities sharing a name would make one of them unaddressable in
    the parametrised results below, and silently mask its failures."""
    assert len(set(IDENTITY_NAMES)) == 30, \
        f"duplicate identity names: {sorted(n for n in IDENTITY_NAMES if IDENTITY_NAMES.count(n) > 1)}"


def test_the_results_matrix_is_complete() -> None:
    """Every pair produced a verdict for every identity."""
    assert len(RESULTS) == len(PAIRS)
    for label, row in RESULTS.items():
        assert set(row) == set(IDENTITY_NAMES), \
            f"{label}: verdicts for {len(row)} identities, not 30"


def test_verify_all_reports_every_identity() -> None:
    """The guard the old file lacked entirely.

    ``verify_all`` returning an empty tuple, or a subset, would have made
    the previous single test pass without evaluating a single identity.
    """
    for label, psi, phi in PAIRS:
        results = verify_all(psi, phi)
        assert len(results) == 30, \
            f"{label}: verify_all returned {len(results)} results, not 30"
        assert tuple(n for n, _ in results) == IDENTITY_NAMES, \
            f"{label}: verify_all reported a different set of identities"


def test_there_are_pairs_to_check() -> None:
    # Every basis pair, plus the structured pairs that are not already one.
    assert len(PAIRS) == len({(p, q) for _, p, q in PAIRS}), "duplicate pairs"
    basis_pairs = {(BASIS[i], BASIS[j]) for i in range(16) for j in range(i + 1, 16)}
    present = {(p, q) for _, p, q in PAIRS}
    assert basis_pairs <= present, (
        f"{len(basis_pairs - present)} of the 120 basis pairs are missing; "
        f"the two-vertex geometry is no longer covered exhaustively")
    assert len(PAIRS) >= 120 + len(PSI_CASES)


def test_every_identity_declares_a_supported_arity() -> None:
    for ident in INTERACTIONS:
        assert ident.arity in (1, 2), \
            f"{ident.name} declares arity {ident.arity}, which verify_all " \
            f"cannot dispatch — it would silently take the arity-2 branch"


# ───────────────────────────────────────────────────────── the identities

@pytest.mark.parametrize("name", IDENTITY_NAMES)
def test_identity_holds_on_every_pair(name: str) -> None:
    """One identity, every pair, all failures reported.

    Parametrised by identity so a broken one reports under its own name and
    the other twenty-nine still run.
    """
    failures = [label for label in RESULTS if not RESULTS[label][name]]
    assert not failures, (
        f"identity {name!r} fails on {len(failures)} of {len(PAIRS)} pairs: "
        f"{failures}")


@pytest.mark.parametrize("label,psi,phi", PAIRS,
                         ids=[p[0] for p in PAIRS])
def test_all_thirty_identities_hold_on_this_pair(label, psi, phi) -> None:
    """The transpose of the test above: one pair, all thirty identities.

    Both directions are kept because they fail differently — a bad operator
    shows as one identity failing everywhere, a bad input class shows as one
    pair failing everything, and each view names the thing that changed.
    """
    row = RESULTS[label]
    assert len(row) == 30, f"{label}: only {len(row)} identities ran"
    failed = [n for n, ok in row.items() if not ok]
    assert not failed, f"{label}: {len(failed)} identities fail: {failed}"


CORRUPTIBLE_OPERATORS = (
    "s1_eka_adhikena", "s2_nikhilam", "s4_paravartya", "s5_shunyam_samya",
    "s10_yavadunam_tavadunikrtya", "s14_ekanyunena_purvena",
    "s15_gunitasamucchaya_product", "s16_gunaka_samucchaya",
    "s25_vestana_circular", "s29_mean_drive",
)


#: Several ways to be wrong. One perturbation is not enough: the S10
#: identity is elementwise non-negativity, which "+1" preserves, so a single
#: additive corruption would have reported S10 as unconstrained when it is
#: not. Each operator must be caught by at least one of these.
CORRUPTIONS = {
    "plus_one": lambda x: x + Fraction(1),
    "negate": lambda x: -x,
    "zero": lambda x: Fraction(0),
    "double": lambda x: x * 2,
    "minus_one": lambda x: x - Fraction(1),
}


@pytest.mark.parametrize("operator", CORRUPTIBLE_OPERATORS)
def test_the_identities_are_falsifiable(monkeypatch, operator: str) -> None:
    """Corrupting an operator must break at least one identity.

    These are identities: they hold for *every* Ψ by construction, so
    perturbing the input cannot break them — the sensitivity that matters is
    to the operators, not the field. Each named operator is replaced with a
    wrong version and at least one identity must go false.

    Without this, ``check`` returning True unconditionally — or an identity
    that quietly stopped calling the operator it names — would satisfy every
    assertion above, and the whole matrix would report thirty passes while
    constraining nothing.
    """
    import vedic.kernel.interaction_matrix as IM

    assert hasattr(IM, operator), (
        f"{operator} is no longer imported into interaction_matrix; the "
        f"identities that named it may have stopped exercising it")
    original = getattr(IM, operator)

    caught_by: list[str] = []
    for cname, corrupt in CORRUPTIONS.items():
        def wrong(psi, *args, _c=corrupt, **kwargs):
            out = original(psi, *args, **kwargs)
            if isinstance(out, Fraction):
                return _c(out)
            if isinstance(out, tuple) and out and isinstance(out[0], tuple):
                return tuple(tuple(_c(x) for x in part) for part in out)
            return tuple(_c(x) for x in out)

        monkeypatch.setattr(IM, operator, wrong)
        # Every pair, stopping at the first that catches the corruption.
        # A cap here would report an operator as unconstrained when the only
        # identity constraining it happened to need a later input.
        for _, psi, phi in PAIRS:
            if any(not ok for _, ok in verify_all(psi, phi)):
                caught_by.append(cname)
                break
        monkeypatch.setattr(IM, operator, original)

    assert caught_by, (
        f"none of the {len(CORRUPTIONS)} corruptions of {operator} broke any "
        f"identity on any of the {len(PAIRS)} pairs — no identity in the "
        f"matrix actually constrains it")
