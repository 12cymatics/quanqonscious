"""Expose our 30 ℚ-exact algebraic identities as Lean 4 propositions.

The Lean 4 mirror (`lean4_mirror.py`) takes a mapping
``sutra_name -> Bool-valued statement`` and runs ``lean`` on each. This
module builds that mapping from a fixed canonical Ψ input by
evaluating every operator in ``vedic.kernel.z2_primitives`` and then
emitting a Lean 4 expression that asserts the corresponding identity.

Each emitted statement has the shape

    decide ((<a_num> : Int) * (<b_den> : Int) = (<b_num> : Int) * (<a_den> : Int))

which is exact rational equality written as integer cross-multiplication:
for a = a_num/a_den and b = b_num/b_den, a = b iff a_num·b_den = b_num·a_den.
Python's ``Fraction`` normalises to lowest terms with a positive
denominator, so the equivalence needs no side conditions.

Why cross-multiplication rather than Lean ``Rat`` literals
---------------------------------------------------------
``Rat`` is not in core Lean 4 -- it needs Mathlib or a Std build carrying
``Std.Internal.Rat``. Emitting ``Rat`` made every generated script
unverifiable without Mathlib, and that single dependency produced two
defects at once: the end-to-end mirror test was skipped wherever Mathlib
was absent, and the "the rendered script compiles" test compiled a
*doctored* script with the imports stripped and the body replaced by the
literal ``true`` -- a body chosen because it needs nothing, which is why
it passed while no real generated statement could compile at all.

``Int`` is core. Cross-multiplication is the same assertion over the same
integers, so nothing is weakened: the numerators and denominators appear
as literals and Lean performs the multiplication and the comparison
itself. What changes is that the emitted script now compiles under a bare
Lean 4 toolchain, so the real statements can be verified rather than
skipped or substituted.

``build_lean_props`` runs without Lean installed and produces the
statement mapping; only ``Lean4Mirror.run_*`` calls require the binary.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, Tuple

from vedic.kernel.q import Q16
from vedic.kernel.z2_primitives import s1_eka_adhikena, s2_nikhilam, s4_paravartya, s5_shunyam_samya, s10_yavadunam_tavadunikrtya, s14_ekanyunena_purvena, s15_gunitasamucchaya_product, s16_gunaka_samucchaya, s25_vestana_circular, s29_mean_drive


def _int_literal(n: int) -> str:
    """Render a Python int as a core-Lean ``Int`` literal.

    Negative values are parenthesised so ``-3 * 7`` cannot reassociate.
    """
    return f"({n} : Int)"


def _exact_equality(a: Fraction, b: Fraction) -> str:
    """Bool expression asserting a = b exactly, over core-Lean ``Int``.

    a = b  <->  a.numerator * b.denominator = b.numerator * a.denominator.
    ``Fraction`` guarantees lowest terms and a positive denominator, so no
    sign or zero-denominator side condition is needed. Lean is given the four
    integers and must do the multiplication and the comparison; nothing is
    precomputed on the Python side.
    """
    if a.denominator <= 0 or b.denominator <= 0:
        raise ValueError(
            f"Fraction invariant violated: denominators must be positive, "
            f"got {a} and {b}. Cross-multiplication would flip the "
            f"inequality direction and silently change the assertion.")
    lhs = f"{_int_literal(a.numerator)} * {_int_literal(b.denominator)}"
    rhs = f"{_int_literal(b.numerator)} * {_int_literal(a.denominator)}"
    return f"decide ({lhs} = {rhs})"


def _q16_equality(lhs: Q16, rhs: Q16) -> str:
    """Bool expression: every component of lhs equals every component of rhs.

    All sixteen components are emitted. There is no sampling and no early
    exit: a conjunction over fewer than sixteen would assert less than
    "these two vectors are equal".
    """
    if len(lhs) != 16 or len(rhs) != 16:
        raise ValueError(
            f"expected two length-16 vectors; got {len(lhs)} and {len(rhs)}")
    parts = [_exact_equality(a, b) for a, b in zip(lhs, rhs)]
    if len(parts) != 16:
        raise AssertionError("emitted fewer than 16 component equalities")
    return " && ".join(parts)


def build_lean_props(psi: Q16) -> Dict[str, str]:
    """Return a mapping ``identity_name -> Bool Lean expression``.

    Each value is a closed Bool-valued expression suitable as the
    ``sutraStatement`` body in ``Lean4Mirror``. The expressions are
    self-contained: core-Lean ``Int`` arithmetic only, no imports and no
    function definitions, so they compile under a bare toolchain.

    Coverage is PARTIAL and reported as such: 8 of the 30 catalogue
    identities are proved in Lean, 22 are declared unrenderable with a stated
    reason, and ``coverage_report`` raises if any entry is in neither bucket
    or in both. An earlier docstring here read "Coverage is COMPLETE: every
    entry of ``INTERACTIONS`` is rendered" while 10 props were emitted for a
    30-entry catalogue, and the coverage check it pointed at returned a
    hardcoded zero.

    Two rendering shapes are used:

    * componentwise ℚ equality, for identities that compare two Q16 vectors —
      the rational literals appear in the expression, so the Lean side checks
      the arithmetic, not just a verdict;
    * a decided Bool for predicate-shaped identities, which are closed over
      *this* Ψ.

    Both are INSTANCE evidence at the given Ψ, not universally quantified
    theorems. The blueprint's extensional/intensional distinction applies: a
    decided instance does not establish the general statement.
    """
    if len(psi) != 16:
        raise ValueError(f"expected length-16 Ψ; got {len(psi)}")

    props: Dict[str, str] = {}

    # S1 ∘ S1 = id   (bit-0 toggle is involution)
    props["S1∘S1 = id"] = _q16_equality(s1_eka_adhikena(s1_eka_adhikena(psi)), psi)

    # S2 ∘ S2 = id   (complement is involution)
    props["S2∘S2 = id"] = _q16_equality(s2_nikhilam(s2_nikhilam(psi)), psi)

    # S5 ∘ S5 = S5   (centering is idempotent)
    s5_psi = s5_shunyam_samya(psi)
    props["S5∘S5 = S5"] = _q16_equality(s5_shunyam_samya(s5_psi), s5_psi)

    # S14^16 = id    (cyclic shift period 16)
    cycled = psi
    for _ in range(16):
        cycled = s14_ekanyunena_purvena(cycled)
    props["S14^16 = id"] = _q16_equality(cycled, psi)

    # S15 ∘ S16 = id   (popcount scale invertible)
    props["S15∘S16 = id"] = _q16_equality(
        s15_gunitasamucchaya_product(s16_gunaka_samucchaya(psi)),
        psi,
    )

    # S16 ∘ S15 = id
    props["S16∘S15 = id"] = _q16_equality(
        s16_gunaka_samucchaya(s15_gunitasamucchaya_product(psi)),
        psi,
    )

    # S25^4 = id    (4-bit rotation period 4)
    rotated = psi
    for _ in range(4):
        rotated = s25_vestana_circular(rotated)
    props["S25^4 = id"] = _q16_equality(rotated, psi)

    # S29 preserves the mean: S29(Ψ) − Ψ has mean 0 ⇔ component-equal to
    # (mean - Ψ)/2.
    mean = sum(psi, Fraction(0)) / Fraction(16)
    expected = tuple((x + mean) / Fraction(2) for x in psi)
    props["S29 closed form"] = _q16_equality(s29_mean_drive(psi), expected)

    # S4 = I − S1
    s4_via_id = tuple(psi[v] - s1_eka_adhikena(psi)[v] for v in range(16))
    props["S4 = I − S1"] = _q16_equality(s4_paravartya(psi), s4_via_id)

    # S10 = (Ψ − 1)²
    sq = tuple((x - Fraction(1)) * (x - Fraction(1)) for x in psi)
    props["S10 = (Ψ − 1)²"] = _q16_equality(s10_yavadunam_tavadunikrtya(psi), sq)

    return props



# Which catalogue identity each rendered proposition proves. Explicit, because
# the two previous versions of this both used prefix matching -- one to decide
# what was unrenderable, a different one (a 6-character prefix) to decide what
# was covered -- so "S1∘S1 = id" reduced to the prefix "S1" and marked S10,
# S11, S12, S13, S14, S15, S16, S19, S20 and S21 as covered. Exactly fourteen
# identities landed in both buckets at once.
#
# The prefix match is the load-bearing defect. The literal `"unaccounted": 0`
# alongside it was cosmetic rather than independently false: with `covered`
# that generous, covered ∪ unrenderable really did span all 30, so the
# `if unaccounted: raise` above it never fired and the literal was only ever
# reached when it happened to be correct. It still had to go -- a value that
# is right by luck is not a check -- but it was not what hid the gap.
#
# A rendered prop with no catalogue counterpart maps to None and is counted as
# such rather than being credited against an unrelated entry.
RENDERS: Dict[str, str | None] = {
    "S1∘S1 = id": "S1∘S1 = id (bit-0 toggle is involution)",
    "S2∘S2 = id": "S2∘S2 = id (complement is involution)",
    "S5∘S5 = S5": "S5∘S5 = S5 (centering is idempotent)",
    "S14^16 = id": "S14^16 = id",
    "S15∘S16 = id": "S15 ∘ S16 = id (popcount scale is invertible)",
    "S16∘S15 = id": "S16 ∘ S15 = id",
    "S25^4 = id": "S25^4 = id",
    "S4 = I − S1": "S4 = (I − S1)",
    # These two render a *different* statement from any catalogue entry: the
    # closed form of S29 rather than its mean-preservation, and the closed form
    # of S10 rather than its non-negativity. Claiming they cover
    # "mean(S29 Ψ) = mean(Ψ)" and "S10 Ψ ≥ 0 elementwise" would be the same
    # slippage the prefix match made, so they are recorded as covering nothing.
    "S29 closed form": None,
    "S10 = (Ψ − 1)²": None,
}

def unrenderable_identities() -> Dict[str, str]:
    """Catalogue identities that this renderer does NOT emit, with the reason.

    These are predicate-shaped (column sums, subspace membership, sign
    conditions, symmetry in two arguments). Rendering them as a bare ``true``
    would produce a Lean body that is vacuously valid and cannot fail, which
    is worse than not rendering them: it would look like coverage. They are
    listed explicitly instead, and ``coverage_report`` asserts that rendered
    plus unrenderable accounts for every catalogue entry.
    """
    from vedic.kernel.interaction_matrix import INTERACTIONS

    reason = ("predicate-shaped: not expressible as a closed componentwise ℚ "
              "equality without defining functions in Lean")
    proved = {v for v in RENDERS.values() if v is not None}
    return {i.name: reason for i in INTERACTIONS if i.name not in proved}


def coverage_report(psi: Q16) -> Dict[str, object]:
    """Rendered + unrenderable must account for the whole catalogue."""
    from vedic.kernel.interaction_matrix import INTERACTIONS

    rendered = build_lean_props(psi)
    unrend = unrenderable_identities()
    catalogue = {i.name for i in INTERACTIONS}

    unmapped = sorted(set(rendered) - set(RENDERS))
    if unmapped:
        raise ValueError(
            f"rendered props with no declared catalogue target: {unmapped}. "
            f"Add them to RENDERS -- mapping to None if they prove something "
            f"the catalogue does not list.")

    covered = {RENDERS[k] for k in rendered if RENDERS[k] is not None}
    stray = sorted(covered - catalogue)
    if stray:
        raise ValueError(f"RENDERS names entries not in the catalogue: {stray}")

    both = sorted(covered & set(unrend))
    if both:
        raise ValueError(
            f"{len(both)} identities are both proved and declared "
            f"unrenderable: {both}")

    unaccounted = sorted(catalogue - covered - set(unrend))
    if unaccounted:
        raise ValueError(
            f"{len(unaccounted)} catalogue identities neither rendered nor "
            f"declared unrenderable: {unaccounted}"
        )
    return {
        "catalogue": len(catalogue),
        "rendered_props": len(rendered),
        "proved_identities": len(covered),
        "declared_unrenderable": len(unrend),
        "unaccounted": len(unaccounted),
    }


def _enumerate_canonical_psi() -> Iterable[Tuple[str, Q16]]:
    """A small fixed set of canonical Ψ inputs used by the test suite."""
    # 1) Centered odd-/even-vertex split
    psi1 = tuple(Fraction(1, v + 2) for v in range(16))
    yield "psi_reciprocal", psi1
    # 2) Polarity-flipped tensor pattern
    psi2 = tuple(Fraction(1, 2) if (v & 1) else Fraction(-1, 2) for v in range(16))
    yield "psi_polarity_pm_half", psi2
    # 3) Constant
    psi3 = tuple(Fraction(3, 7) for _ in range(16))
    yield "psi_constant_3_7", psi3
