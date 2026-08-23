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

# `__init__.py` is a package marker, not a driver, so the entry-point
# obligation does not apply to it. That exclusion lives here, in the
# parameter list, rather than as a runtime pytest.skip: a skip reports as
# "not run" and looks the same whether the reason is sound or the test
# quietly stopped working. The marker is checked on its own terms in
# test_package_marker_is_only_a_marker.
DRIVERS = [p for p in SCRIPTS if p.name != "__init__.py"]


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


def test_there_are_drivers_to_check():
    assert DRIVERS, "scripts/ contains no drivers — the exclusion is too broad"


def test_package_marker_is_only_a_marker():
    """`__init__.py` is exempt from the entry-point rule only while it is
    genuinely empty. If code appears in it, the exemption stops applying."""
    init = Path(__file__).resolve().parents[3] / "scripts" / "__init__.py"
    if not init.exists():
        return
    body = ast.parse(init.read_text(encoding="utf-8")).body
    code = [n for n in body if not isinstance(n, ast.Expr)
            or not isinstance(n.value, ast.Constant)]
    assert not code, (
        "scripts/__init__.py now contains code, so it is a module and must "
        "meet the same obligations as every other script")


@pytest.mark.parametrize("path", DRIVERS, ids=[p.name for p in DRIVERS])
def test_script_has_an_entry_point(path: Path):
    """Each script is runnable: it defines main() and guards __main__."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    has_main = any(isinstance(n, ast.FunctionDef) and n.name == "main"
                   for n in tree.body)
    assert has_main, f"{path.name} defines no main()"
    assert '__main__' in src, f"{path.name} has no __main__ guard"
