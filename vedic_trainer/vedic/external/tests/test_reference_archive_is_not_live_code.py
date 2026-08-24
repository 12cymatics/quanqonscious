"""`docs/external/reference/` is an archive, and archives do not get imported.

The one file there, `extended_subsutras_palindrome.py`, was carried over
from another branch. It computes `x * 2 ** steps` under the comment
"dominant eigenvalue approximation" and dispatches through three identical
branches — an approximation and dead branching, both of which this package
forbids in live code.

It is kept byte-identical anyway. It records what that branch contains, and
rewriting a record to meet a standard it never met is falsifying it, not
fixing it — the same reasoning that keeps the withdrawn subset evaluations
in `runs/*.json` while forbidding anyone to quote them.

So the rule is not "clean it up"; the rule is that nothing may depend on it.
That is what is checked here. Without this, the archive is one `import`
away from being load-bearing, and the standard would have been dropped
without anyone deciding to drop it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = REPO / "docs" / "external" / "reference"

# Module names the archive would be imported under, however it is spelled.
ARCHIVE_MODULES = {p.stem for p in ARCHIVE_DIR.glob("*.py")}


def _package_sources() -> list[Path]:
    """Every source file that ships as part of the package."""
    return sorted(
        p for root in ("vedic", "scripts")
        for p in (REPO / root).rglob("*.py")
        if "__pycache__" not in p.parts
    )


SOURCES = _package_sources()


def test_the_archive_exists_and_this_test_knows_its_contents():
    """Guards the rest: with no archive modules, every check below is vacuous."""
    assert ARCHIVE_DIR.is_dir(), f"{ARCHIVE_DIR} is gone; update or delete this test"
    assert ARCHIVE_MODULES, (
        f"no modules found under {ARCHIVE_DIR} — this test would pass "
        f"without checking anything")


def test_there_are_package_sources_to_scan():
    assert SOURCES, "found no package sources; the import scan would be vacuous"


@pytest.mark.parametrize(
    "source", SOURCES, ids=[str(p.relative_to(REPO)) for p in SOURCES])
def test_no_package_source_imports_the_archive(source: Path):
    """Parsed, not grepped: a string mentioning the name is not an import."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            head = name.split(".")[-1]
            if head in ARCHIVE_MODULES or name.startswith("docs."):
                offenders.append(f"line {node.lineno}: {name}")
    assert not offenders, (
        f"{source.relative_to(REPO)} imports the reference archive:\n  "
        + "\n  ".join(offenders)
        + "\n\ndocs/external/reference/ holds code that does not meet this "
          "package's standards. Port what you need into vedic/ and hold it "
          "to the same bar; do not import the archive.")


def test_the_archive_is_outside_the_collected_test_paths():
    """pytest must not collect it either, or its content becomes a gate."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'testpaths = ["vedic"]' in pyproject, (
        "testpaths changed; confirm docs/external/reference is still "
        "outside what pytest collects")
