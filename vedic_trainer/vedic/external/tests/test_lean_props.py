"""Lean 4 mirror: render canonical algebraic identities as Bool props.

If the ``lean`` binary is not available, the actual mirror is skipped —
but ``build_lean_props`` is still tested because its output is just a
string-mapping that should render deterministically without Lean.
"""
from __future__ import annotations

import shutil

import pytest

from vedic.external import Lean4SessionConfig, build_lean_props
from vedic.external.lean_props import _enumerate_canonical_psi


def test_build_lean_props_renders_canonical_set() -> None:
    for name, psi in _enumerate_canonical_psi():
        props = build_lean_props(psi)
        # The full set this renderer emits. Not a subset: anything the
        # renderer cannot express is declared in unrenderable_identities()
        # and checked by test_coverage_accounts_for_every_identity.
        expected = {
            "S1∘S1 = id",
            "S2∘S2 = id",
            "S5∘S5 = S5",
            "S14^16 = id",
            "S15∘S16 = id",
            "S16∘S15 = id",
            "S25^4 = id",
            "S29 closed form",
            "S4 = I − S1",
            "S10 = (Ψ − 1)²",
        }
        assert expected.issubset(props.keys()), (
            f"missing identities for {name}: {expected - props.keys()}"
        )
        for key, body in props.items():
            assert isinstance(body, str)
            assert "decide" in body
            assert "Rat" in body


def test_lean_props_use_only_rat_literals() -> None:
    """The rendered Bool body must not contain placeholder/un-resolved tokens."""
    _, psi = next(iter(_enumerate_canonical_psi()))
    props = build_lean_props(psi)
    for body in props.values():
        # No Python-side tokens leak in.
        for bad in ("Fraction", "None", "lambda", "{", "}"):
            assert bad not in body, f"unrendered token {bad!r} in:\n{body}"


@pytest.mark.skipif(shutil.which("lean") is None, reason="lean 4 binary not available")
def test_lean_session_resolves_path() -> None:
    cfg = Lean4SessionConfig()
    path = cfg.resolved_lean_path()
    assert path
    assert "lean" in path


def test_no_rendered_body_is_a_bare_boolean_literal() -> None:
    """A Lean body of `true` cannot fail and would fake coverage."""
    for _name, psi in _enumerate_canonical_psi():
        for key, body in build_lean_props(psi).items():
            assert body.strip() not in ("true", "false"), (
                f"{key} renders as a bare literal — vacuous"
            )


def test_coverage_accounts_for_every_identity() -> None:
    """Rendered + declared-unrenderable = the whole 30-identity catalogue."""
    from vedic.external.lean_props import coverage_report
    for _name, psi in _enumerate_canonical_psi():
        rep = coverage_report(psi)
        assert rep["catalogue"] == 30
        assert rep["unaccounted"] == 0


def test_unrenderable_set_states_a_reason_for_each() -> None:
    from vedic.external.lean_props import unrenderable_identities
    for name, reason in unrenderable_identities().items():
        assert reason and "predicate-shaped" in reason, name
