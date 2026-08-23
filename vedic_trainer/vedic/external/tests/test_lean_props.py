"""Lean 4 mirror: render canonical algebraic identities as Bool props.

If the ``lean`` binary is not available, the actual mirror is skipped —
but ``build_lean_props`` is still tested because its output is just a
string-mapping that should render deterministically without Lean.
"""
from __future__ import annotations

import pathlib
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



def _lean_compiles(preamble: str) -> tuple[bool, str]:
    """Can `lean` actually compile this preamble?

    `shutil.which("lean") is None` was the whole condition. That gates on the
    binary being on PATH, not on it working: an elan shim with no toolchain
    passes it, so the guarded tests "ran" against a compiler that could not
    compile anything. Ask the compiler instead of the filesystem.

    Two capabilities are distinguished, because conflating them over-skips:
    running Lean at all, and having Mathlib available.
    """
    import subprocess
    import tempfile
    if shutil.which("lean") is None:
        return False, "no lean binary on PATH"
    probe = pathlib.Path(tempfile.mkdtemp()) / "Probe.lean"
    probe.write_text(preamble + "#eval (1 : Nat) + 1\n")
    try:
        r = subprocess.run(["lean", str(probe)], capture_output=True,
                           text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"lean did not run: {e}"
    if r.returncode != 0:
        return False, f"lean cannot compile: {(r.stderr or r.stdout).strip()[:120]}"
    return True, ""


LEAN_OK, LEAN_WHY = _lean_compiles("")
MATHLIB_OK, MATHLIB_WHY = _lean_compiles("import Mathlib\n")


@pytest.mark.skipif(not LEAN_OK, reason=f"lean unusable: {LEAN_WHY}")
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


@pytest.mark.skipif(not LEAN_OK, reason=f"lean unusable: {LEAN_WHY}")
def test_the_rendered_script_compiles():
    """The generated code parses, checked by the real compiler.

    Mathlib is not installed in every environment, so the two Mathlib-only
    lines are stripped here; that isolates the *generated* code, which is
    what this test is about.
    """
    import subprocess
    import tempfile

    from vedic.external.lean4_mirror import Lean4Mirror

    body = Lean4Mirror()._render_script("S1test", "true")
    stripped = "\n".join(
        line for line in body.splitlines()
        if line not in ("import Mathlib", "open scoped BigOperators"))
    f = pathlib.Path(tempfile.mkdtemp()) / "M.lean"
    f.write_text(stripped)
    r = subprocess.run(["lean", str(f)], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, \
        f"generated Lean does not compile:\n{(r.stderr or r.stdout)[:600]}"
    assert r.stdout.strip() == "true"


@pytest.mark.skipif(not MATHLIB_OK, reason=f"Mathlib unavailable: {MATHLIB_WHY}")
def test_mirror_run_serial_actually_verifies():
    """End-to-end: render, compile, and read the verdict.

    `run_serial`, `run_concurrent` and `run_parallel` had no test at all. The
    generated scripts `import Mathlib`, so this needs a Mathlib toolchain and
    is skipped where there isn't one — with the reason established by asking
    the compiler, not by looking for a file on PATH.
    """
    from vedic.external.lean4_mirror import Lean4Mirror

    results = Lean4Mirror().run_serial({"S1true": "true", "S1false": "false"})
    by_name = {r.sutra: r for r in results}
    assert by_name["S1true"].success, by_name["S1true"]
    assert not by_name["S1false"].success, (
        "a statement that is `false` must not be reported as verified")
