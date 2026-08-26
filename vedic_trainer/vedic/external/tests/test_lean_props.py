"""Lean 4 mirror: render canonical algebraic identities as Bool props.

Nothing here is skipped and nothing is substituted. Every statement the
renderer emits is compiled by the real ``lean`` binary, unmodified, through
the same ``Lean4Mirror`` path production uses.

What that replaced
------------------
Two tests used to stand in for this one, and neither could have failed:

* the compile test **stripped** ``import Mathlib`` and
  ``open scoped BigOperators`` from the rendered script and compiled a body
  of the literal ``true`` -- a body chosen because it needs nothing. No real
  generated statement was ever compiled by it; every one of them would have
  failed, because ``Rat`` is not in core Lean.
* the end-to-end test was **skipped** wherever Mathlib was absent, which is
  everywhere this package runs.

Underneath both sat a defect neither could reach: ``Lean4Mirror`` writes its
scripts to a temp directory, and ``elan`` resolves a toolchain by walking up
from the invocation directory, so every ``lean`` call returned "no default
toolchain configured". The mirror could not verify anything at all, and the
skip meant nobody found out.

The renderer now emits core-Lean ``Int`` cross-multiplication -- the same
exact rational equality, no library needed -- and the mirror writes the
committed toolchain pin beside its scripts. So the real path runs, and these
tests exercise it.
"""
from __future__ import annotations


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
        assert props.keys() == expected, (
            f"renderer output for {name} is not the declared set: "
            f"missing {expected - props.keys()}, unexpected {props.keys() - expected}")
        for key, body in props.items():
            assert isinstance(body, str)
            assert "decide" in body, f"{key} renders no decidable proposition"
            assert "Int" in body, f"{key} renders no Int arithmetic"
            assert "Rat" not in body, (
                f"{key} renders a Rat literal. Rat is not in core Lean, so "
                f"the emitted script needs Mathlib and cannot be verified by "
                f"a bare toolchain -- which is what made the compile test "
                f"strip its imports and the end-to-end test skip.")


def test_every_rendered_body_asserts_all_sixteen_components() -> None:
    """A conjunction over fewer than 16 asserts less than vector equality.

    Both rendering shapes compare two Q16 vectors, so each body must carry
    exactly sixteen component comparisons. A renderer that emitted, say,
    the first four would still read as a passing Lean proof.
    """
    for name, psi in _enumerate_canonical_psi():
        for key, body in build_lean_props(psi).items():
            assert body.count("decide") == 16, (
                f"{name}/{key} compares {body.count('decide')} components, "
                f"not 16")


def test_rendered_bodies_carry_no_python_side_tokens() -> None:
    """Every Ψ, not one: a leak could be input-dependent."""
    for name, psi in _enumerate_canonical_psi():
        for key, body in build_lean_props(psi).items():
            for bad in ("Fraction", "None", "lambda", "{", "}", "Rat"):
                assert bad not in body, \
                    f"unrendered token {bad!r} in {name}/{key}:\n{body}"


def test_cross_multiplication_is_exact_rational_equality() -> None:
    """The rewrite's correctness, checked rather than asserted.

    a/b = c/d iff a·d = c·b for positive b, d. This exercises the renderer's
    own comparison on pairs that are equal, unequal, and equal-only-after-
    reduction, so a renderer that emitted a always-true or always-false
    comparison fails here.
    """
    from fractions import Fraction

    from vedic.external.lean_props import _exact_equality

    import re

    def evaluates_true(expr: str) -> bool:
        nums = [int(n) for n in re.findall(r"\(\s*(-?\d+)\s*: Int\)", expr)]
        assert len(nums) == 4, f"expected four Int literals, got {nums}"
        return nums[0] * nums[1] == nums[2] * nums[3]

    assert evaluates_true(_exact_equality(Fraction(3, 6), Fraction(1, 2)))
    assert evaluates_true(_exact_equality(Fraction(-2, 4), Fraction(-1, 2)))
    assert not evaluates_true(_exact_equality(Fraction(1, 2), Fraction(1, 3)))
    assert not evaluates_true(_exact_equality(Fraction(1, 2), Fraction(-1, 2)))



def test_no_rendered_body_is_a_bare_boolean_literal() -> None:
    """A Lean body of `true` cannot fail and would fake coverage."""
    for _name, psi in _enumerate_canonical_psi():
        for key, body in build_lean_props(psi).items():
            assert body.strip() not in ("true", "false"), (
                f"{key} renders as a bare literal — vacuous"
            )


def test_coverage_accounts_for_every_identity() -> None:
    """Proved + declared-unrenderable = the whole catalogue, with no overlap.

    `unaccounted` used to be the literal `0` in the returned dict, so
    asserting it was 0 tested nothing. It is now computed from the sets, and
    `coverage_report` raises before returning if the partition is broken --
    so reaching this assertion is itself part of the check.
    """
    from vedic.kernel.interaction_matrix import INTERACTIONS
    from vedic.external.lean_props import coverage_report

    for _name, psi in _enumerate_canonical_psi():
        rep = coverage_report(psi)
        assert rep["catalogue"] == len(INTERACTIONS)
        assert rep["proved_identities"] + rep["declared_unrenderable"] == \
            rep["catalogue"], "the two buckets do not partition the catalogue"
        assert rep["unaccounted"] == 0


def test_coverage_is_partial_and_says_so() -> None:
    """Guards against the claim drifting back to 'complete'.

    8 of 30 identities are proved. If that ratio changes, the docstring and
    any prose quoting it have to change with it — which is the point.
    """
    from vedic.external.lean_props import coverage_report
    _name, psi = next(iter(_enumerate_canonical_psi()))
    rep = coverage_report(psi)
    assert rep["proved_identities"] == 8
    assert rep["declared_unrenderable"] == 22
    assert rep["proved_identities"] < rep["catalogue"], \
        "coverage is claimed complete; verify that before saying so"


def test_a_rendered_prop_cannot_be_credited_to_an_unrelated_identity() -> None:
    """The mapping is exact: no prefix matching, no silent credit."""
    from vedic.external.lean_props import RENDERS, build_lean_props
    from vedic.kernel.interaction_matrix import INTERACTIONS

    catalogue = {i.name for i in INTERACTIONS}
    _name, psi = next(iter(_enumerate_canonical_psi()))
    for key in build_lean_props(psi):
        assert key in RENDERS, f"{key!r} claims no catalogue target"
        target = RENDERS[key]
        assert target is None or target in catalogue, \
            f"{key!r} maps to {target!r}, which is not a catalogue identity"


def test_unrenderable_set_states_a_reason_for_each() -> None:
    from vedic.external.lean_props import unrenderable_identities
    for name, reason in unrenderable_identities().items():
        assert reason and "predicate-shaped" in reason, name



def test_the_error_branch_keeps_its_string_literal():
    """The regression guard for the actual bug — runs everywhere, no Lean needed.

    `_render_script` wrote its error branch as two adjacent Python string
    literals, so the inner quotes were consumed by implicit concatenation and
    every generated script was a Lean syntax error. `_interpret_result` then
    set success=False for every sutra, and nothing noticed, because no test
    had ever compiled one.

    Rendering is pure string work, so this assertion must not be gated behind
    having a compiler. It was, and CI (which has no Lean) failed on the
    construction rather than skipping — which is how `Lean4Mirror.__init__`
    resolving the binary eagerly turned out to be a defect of its own.
    """
    from vedic.external.lean4_mirror import Lean4Mirror

    body = Lean4Mirror()._render_script("S1test", "true")
    assert 'IO.userError "mirror validation failed"' in body, \
        "the error branch lost its string literal again"


def test_rendering_needs_no_lean_toolchain():
    """Constructing a mirror to render must not require a compiler."""
    import shutil as _shutil

    from vedic.external.lean4_mirror import Lean4Mirror

    real_which = _shutil.which
    _shutil.which = lambda name, *a, **k: (None if name == "lean"
                                           else real_which(name, *a, **k))
    try:
        body = Lean4Mirror()._render_script("S1test", "true")
        assert "def sutraStatement : Bool :=" in body
    finally:
        _shutil.which = real_which


def test_running_without_lean_still_raises():
    """Laziness must not turn a missing compiler into a silent no-op."""
    import shutil as _shutil

    from vedic.external.lean4_mirror import Lean4Mirror

    real_which = _shutil.which
    _shutil.which = lambda name, *a, **k: (None if name == "lean"
                                           else real_which(name, *a, **k))
    try:
        with pytest.raises(FileNotFoundError):
            Lean4Mirror().run_serial({"S1test": "true"})
    finally:
        _shutil.which = real_which


def test_lean_session_resolves_path() -> None:
    """No skip guard. Lean is a declared requirement of this package.

    The pin lives at the package root in ``lean-toolchain``; if the toolchain
    is absent this must fail rather than report a skip, because a skipped
    Lean check reads as "not applicable here" when it means "the one
    independent cross-check of the kernel did not run".
    """
    cfg = Lean4SessionConfig()
    path = cfg.resolved_lean_path()
    assert path
    assert "lean" in path


def test_the_package_pins_a_lean_toolchain() -> None:
    """The pin is what makes the generated scripts compile from any cwd."""
    from vedic.external.lean4_mirror import Lean4Mirror

    pin = Lean4Mirror.TOOLCHAIN_PIN
    assert pin.is_file(), f"{pin} is missing"
    text = pin.read_text(encoding="utf-8").strip()
    assert text.startswith("leanprover/lean4:"), \
        f"toolchain pin does not name a Lean 4 toolchain: {text!r}"


def test_the_artifact_directory_carries_the_pin() -> None:
    """elan resolves upward from the invocation directory.

    Without this file beside the scripts, every ``lean`` call in a temp
    directory fails with "no default toolchain configured" -- which is
    exactly what the mirror did, undetected, for as long as the only test
    that ran it was skipped.
    """
    from vedic.external.lean4_mirror import Lean4Mirror

    m = Lean4Mirror()
    written = m._artifact_root / "lean-toolchain"
    assert written.is_file(), "artifact root has no toolchain pin"
    assert written.read_text(encoding="utf-8").strip() == \
        Lean4Mirror.TOOLCHAIN_PIN.read_text(encoding="utf-8").strip()


def test_the_default_config_requests_no_imports() -> None:
    """A Mathlib default makes the emitted script unverifiable everywhere
    this package actually runs, and nothing in the emitted body used it."""
    assert tuple(Lean4SessionConfig().imports) == ()
    assert "BigOperators" not in Lean4SessionConfig().prelude


def test_the_rendered_script_compiles_unmodified() -> None:
    """The real rendered script, byte for byte, through the real compiler.

    Nothing is stripped and no body is substituted. The previous version of
    this test removed the two Mathlib lines and rendered the statement
    ``"true"``; it passed for years while every genuine statement failed to
    compile.
    """
    import subprocess

    from vedic.external.lean4_mirror import Lean4Mirror

    mirror = Lean4Mirror()
    statement = "decide ((1 : Int) * (7 : Int) = (1 : Int) * (7 : Int))"
    path = mirror._write_script("S1test", statement)
    assert path.read_text(encoding="utf-8") == \
        mirror._render_script("S1test", statement), \
        "the compiled file is not what the renderer produced"
    r = subprocess.run(["lean", path.name], cwd=str(path.parent),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, \
        f"generated Lean does not compile:\n{r.stderr or r.stdout}"
    assert r.stdout.strip() == "true"


def _all_generated_statements():
    """(psi_label, identity_name, body) for every canonical Ψ. No sampling."""
    out = []
    for label, psi in _enumerate_canonical_psi():
        for name, body in build_lean_props(psi).items():
            out.append((label, name, body))
    return out


GENERATED = _all_generated_statements()


def test_there_are_generated_statements_to_compile() -> None:
    """Guards the parametrised test below against an empty renderer."""
    assert len(GENERATED) == 30, \
        f"expected 3 canonical Ψ × 10 identities = 30, got {len(GENERATED)}"


@pytest.mark.parametrize("label,name,body", GENERATED,
                         ids=[f"{a}:{b}" for a, b, _ in GENERATED])
def test_every_generated_statement_is_verified_by_lean(label, name, body):
    """Each statement compiled and evaluated separately.

    Parametrised rather than looped so one failing identity reports as one
    failure and the other twenty-nine still run -- a loop stops at the first
    and hides the rest.
    """
    from vedic.external.lean4_mirror import Lean4Mirror

    results = Lean4Mirror().run_serial({name: body})
    assert len(results) == 1
    r = results[0]
    assert r.success, (
        f"{label}/{name} was not verified by Lean:\n"
        f"stdout: {r.stdout}\nstderr: {r.stderr}")


def test_mirror_run_serial_actually_verifies() -> None:
    """End-to-end, unskipped: render, compile, read the verdict.

    Includes a false statement, because a mirror that reports success for
    everything would pass every test above.
    """
    from vedic.external.lean4_mirror import Lean4Mirror

    results = Lean4Mirror().run_serial({
        "S1true": "decide ((1 : Int) * (7 : Int) = (1 : Int) * (7 : Int))",
        "S1false": "decide ((1 : Int) * (7 : Int) = (2 : Int) * (7 : Int))",
    })
    by_name = {r.sutra: r for r in results}
    assert by_name["S1true"].success, by_name["S1true"]
    assert not by_name["S1false"].success, (
        "a statement that is false must not be reported as verified")


def _corrupt_one_component(body: str) -> str:
    """Return `body` with exactly one component comparison made false.

    Rewrites the right-hand side of the first ``decide`` so the comparison is
    arithmetically false, and verifies that in Python before returning, so a
    corruption that happened to stay true can never be handed to Lean as if
    it were a real falsification.
    """
    import re

    m = re.search(r"decide \(\((-?\d+) : Int\) \* \((-?\d+) : Int\) = "
                  r"\((-?\d+) : Int\) \* \((-?\d+) : Int\)\)", body)
    if m is None:
        raise AssertionError(f"no component comparison found in:\n{body}")
    a, b, c, d = (int(g) for g in m.groups())
    assert a * b == c * d, "the identity was already false before corruption"
    c_bad = c + 1
    assert a * b != c_bad * d, "corruption did not change the truth value"
    corrupted = (f"decide (({a} : Int) * ({b} : Int) = "
                 f"({c_bad} : Int) * ({d} : Int))")
    out = body[:m.start()] + corrupted + body[m.end():]
    assert out != body
    return out


@pytest.mark.parametrize("label,name,body", GENERATED,
                         ids=[f"{a}:{b}" for a, b, _ in GENERATED])
def test_lean_rejects_a_corrupted_form_of_every_identity(label, name, body):
    """Falsification on real content, for all thirty statements.

    Every statement above is one Lean accepts. Without this, a mirror that
    reported success unconditionally -- or a renderer emitting comparisons
    that are true by construction regardless of the kernel -- would pass the
    whole file. One component of each identity is made arithmetically false
    and Lean must refuse it.
    """
    from vedic.external.lean4_mirror import Lean4Mirror

    r = Lean4Mirror().run_serial({name: _corrupt_one_component(body)})[0]
    assert not r.success, (
        f"Lean accepted a corrupted form of {label}/{name} — the mirror "
        f"cannot distinguish a true identity from a false one")
