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
        # Expected identity keys (a stable subset of the 30-identity catalogue
        # that is renderable as Bool over Rat literals).
        expected = {
            "S1∘S1 = id",
            "S2∘S2 = id",
            "S5∘S5 = S5",
            "S14^16 = id",
            "S15∘S16 = id",
            "S16∘S15 = id",
            "S25^4 = id",
            "S26 = S26",
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
