"""Every script in scripts/ must compile and declare a main entry point.

Why this exists
---------------
`scripts/run_ablation_eval.py` was shipped with an IndentationError. Nothing
caught it because no test imported or compiled anything under scripts/ --
the whole directory was outside the suite. A syntax error in a driver is
invisible until someone runs it, which is exactly when it costs the most.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[3] / "scripts").glob("*.py"))


def test_scripts_directory_is_not_empty():
    assert SCRIPTS, "no scripts found — the glob is wrong, not the directory"


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_compiles(path: Path):
    """Parse every script. A driver that does not compile is broken."""
    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src, filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path.name} does not parse: line {e.lineno}: {e.msg}")


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_script_has_an_entry_point(path: Path):
    """Each script is runnable: it defines main() and guards __main__."""
    if path.name == "__init__.py":
        pytest.skip("package marker, not a driver")
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    has_main = any(isinstance(n, ast.FunctionDef) and n.name == "main"
                   for n in tree.body)
    assert has_main, f"{path.name} defines no main()"
    assert '__main__' in src, f"{path.name} has no __main__ guard"
