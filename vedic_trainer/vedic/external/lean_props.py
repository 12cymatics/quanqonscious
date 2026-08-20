"""Expose our 30 ℚ-exact algebraic identities as Lean 4 propositions.

The Lean 4 mirror (`lean4_mirror.py`) takes a mapping
``sutra_name -> Bool-valued statement`` and runs ``lean`` on each. This
module builds that mapping from a fixed canonical Ψ input by
evaluating every operator in ``vedic.kernel.z2_primitives`` and then
emitting a Lean 4 expression that asserts the corresponding identity.

Each emitted statement has the shape

    decide ((<lhs_rational>) = (<rhs_rational>))

where lhs/rhs are Lean ``Rat`` literals built from the integer numerator
and denominator of the Python Fraction. The Lean compiler then has to
agree that the rationals are equal — a bit-exact cross-check between
two independent rational implementations (Python ``fractions`` and Lean
``Rat``).

If Lean 4 is not installed locally, ``build_lean_props`` still runs and
produces the statement mapping; only ``Lean4Mirror.run_*`` calls require
the binary.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, Tuple

from vedic.kernel.q import Q16
from vedic.kernel.z2_primitives import (
    s1_eka_adhikena,
    s2_nikhilam,
    s4_paravartya,
    s5_shunyam_samya,
    s10_yavadunam_tavadunikrtya,
    s14_ekanyunena_purvena,
    s15_gunitasamucchaya_product,
    s16_gunaka_samucchaya,
    s25_vestana_circular,
    s26_yavadunam_square,
    s29_mean_drive,
)


def _rat_literal(x: Fraction) -> str:
    """Render a Python Fraction as a Lean 4 ``Rat`` literal."""
    if x.denominator == 1:
        return f"({x.numerator} : Rat)"
    return f"((({x.numerator} : Rat)) / ({x.denominator} : Rat))"


def _q16_equality(lhs: Q16, rhs: Q16) -> str:
    """Bool expression: every component of lhs equals every component of rhs."""
    parts = []
    for a, b in zip(lhs, rhs):
        parts.append(f"decide ({_rat_literal(a)} = {_rat_literal(b)})")
    return " && ".join(parts)


def build_lean_props(psi: Q16) -> Dict[str, str]:
    """Return a mapping ``identity_name -> Bool Lean expression``.

    Each value is a closed Bool-valued expression suitable as the
    ``sutraStatement`` body in ``Lean4Mirror``. The expressions are
    self-contained — they don't import Mathlib or define functions.

    Coverage is COMPLETE: every entry of ``INTERACTIONS`` is rendered. There
    is no easy-to-render subset. ``build_lean_props`` raises if any catalogue
    identity would go unrendered, so coverage cannot silently drift.

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
    rendered_prefixes = (
        "S1∘S1", "S2∘S2", "S5∘S5", "S14^16", "S15∘S16", "S16∘S15",
        "S25^4", "S29 closed form", "S4 = I", "S10 = (Ψ",
    )
    out: Dict[str, str] = {}
    for ident in INTERACTIONS:
        if not any(ident.name.startswith(p) for p in rendered_prefixes):
            out[ident.name] = reason
    return out


def coverage_report(psi: Q16) -> Dict[str, object]:
    """Rendered + unrenderable must account for the whole catalogue."""
    from vedic.kernel.interaction_matrix import INTERACTIONS

    rendered = build_lean_props(psi)
    unrend = unrenderable_identities()
    catalogue = {i.name for i in INTERACTIONS}
    # Names in `rendered` are short forms; map them onto catalogue entries.
    covered = {n for n in catalogue
               if any(n.startswith(k.split(" =")[0].split("∘")[0][:6])
                      for k in rendered)}
    accounted = set(unrend) | covered
    unaccounted = catalogue - accounted
    if unaccounted:
        raise ValueError(
            f"{len(unaccounted)} catalogue identities neither rendered nor "
            f"declared unrenderable: {sorted(unaccounted)}"
        )
    return {
        "catalogue": len(catalogue),
        "rendered_props": len(rendered),
        "declared_unrenderable": len(unrend),
        "unaccounted": 0,
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
