"""No test in this repository may skip itself.

Why this exists
---------------
A skip reports as "not run" and reads as covered. Both of the real instances
this repository has had were found by accident, long after the fact, and both
had been hiding work that was broken rather than work that was optional:

* the Lean mirror's tests carried `skipif` on the compiler being present. It
  was absent everywhere, so the one independent cross-check of the exact-ℚ
  kernel reported green in CI while being unable to resolve a toolchain at
  all. Nobody reads "3 skipped".
* `pcfe-v3/tests/test_integration.py` opened with three `pytest.importorskip`
  calls, one of them for `cudaq`. That is not installable without an NVIDIA
  GPU stack, so on every machine this repository has ever run on — CI
  included — all five of its checks collapsed into a single "1 skipped".

Neither was fixed by making the skip conditional or better-worded. The Lean
one was fixed by installing the compiler in CI; the cudaq one by admitting the
file was an environment probe rather than a test and making it a script that
exits non-zero. This gate exists so the third instance is caught when it is
written instead of years later.

There is deliberately no allowlist. An exemption here would be the mechanism
being reintroduced under a different name — and an environment a test needs is
either installed (Lean) or the thing is not a test (cudaq). Those are the two
answers; "skip" is not a third.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Every way pytest or unittest can be told not to run something, as dotted
#: names. Matched against the **parsed syntax tree**, never against the text:
#: this file has to name the mechanisms in order to forbid them, and a
#: text search flags its own prose. An AST walk sees code and not strings, so
#: there is no self-exclusion to maintain and no comment-stripping heuristic
#: to get wrong.
FORBIDDEN = frozenset({
    "pytest.importorskip",
    "pytest.skip",
    "pytest.mark.skip",
    "pytest.mark.skipif",
    "unittest.skip",
    "unittest.skipIf",
    "unittest.skipUnless",
    "unittest.SkipTest",
})


def _dotted(node: ast.AST) -> str | None:
    """`pytest.mark.skipif` from the Attribute chain, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def tracked_python() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.py"],
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.split() if p]


def skips_in(source: str) -> list[str]:
    """Forbidden dotted names actually *used* in this source. Pure."""
    found = []
    for node in ast.walk(ast.parse(source)):
        name = _dotted(node)
        if name and name in FORBIDDEN:
            found.append(f"line {getattr(node, 'lineno', '?')}: {name}")
    return found


def offenders() -> list[str]:
    out = []
    for rel in tracked_python():
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        try:
            hits = skips_in(text)
        except SyntaxError:
            continue          # test_scripts_are_valid.py owns "does it parse"
        out.extend(f"{rel}:{h}" for h in hits)
    return out


def test_there_are_python_files_to_check():
    """Guards the rest: an empty file list would pass vacuously."""
    files = tracked_python()
    assert len(files) > 50, f"only {len(files)} tracked .py files — the glob is wrong"


def test_the_detector_matches_the_real_mechanisms():
    """Without this the gate could pass by matching nothing at all."""
    for probe in ('np = pytest.importorskip("numpy")',
                  "pytest.skip('no gpu')",
                  "@pytest.mark.skipif(not HAVE_LEAN, reason='x')\ndef test_x(): pass",
                  "@pytest.mark.skip\ndef test_y(): pass",
                  "pytestmark = pytest.mark.skipif(True, reason='x')"):
        assert skips_in(probe), f"detector missed: {probe}"


def test_the_detector_ignores_prose_that_names_the_mechanisms():
    """The reason this walks the AST. A text search flags this very file:
    it has to write `pytest.importorskip` down in order to forbid it."""
    for prose in ('"""This file used to call pytest.importorskip here."""',
                  "# pytest.mark.skipif was removed from this module",
                  "MESSAGE = 'do not use pytest.skip in this repository'"):
        assert not skips_in(prose), f"detector flagged prose: {prose}"


def test_no_tracked_test_skips_itself():
    found = offenders()
    assert not found, (
        "a test can skip itself:\n  " + "\n  ".join(found)
        + "\n\nA skip reports as not-run and reads as covered. Install what "
          "the test needs, or — if it is probing the environment rather than "
          "testing this code — make it a script that exits non-zero, as "
          "pcfe-v3/tests/check_cudaq_environment.py does.")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  ✓ {name}")
            except AssertionError as exc:
                fails += 1; print(f"  ✗ {name}\n      {exc}")
    print(f"\n{'FAILED' if fails else 'OK'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)
