"""Every script in scripts/ must compile and declare a main entry point.

Why this exists
---------------
`scripts/run_ablation_eval.py` was shipped with an IndentationError. Nothing
caught it because no test imported or compiled anything under scripts/ --
the whole directory was outside the suite. A syntax error in a driver is
invisible until someone runs it, which is exactly when it costs the most.

The same glob then only matched `*.py`, so the shell driver
`reproduce_ablation.sh` went on invoking `run_ablation_eval.py` for the
entire life of that file and for some time after it was deleted. The scripts
a driver references are therefore checked too: a driver that calls something
which no longer exists is broken whether or not it parses.

There are no `.sh` drivers left. `reproduce_ablation.sh` was the only one,
and it went with the synthetic corpus its first step generated — steps 1 and
4 read files nothing in this repository can now produce. The two tests that
globbed `*.sh` are gone rather than left asserting over an empty set: a
`test_there_are_shell_drivers_to_check` that must fail, or be weakened to
pass on nothing, is worse than no test. `test_referenced_scripts_exist`
stays and covers every Python driver, which is what caught the dangling
reference in the first place.
"""
from __future__ import annotations

import ast
import re
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
    assert init.exists(), (
        "scripts/__init__.py is gone. It was exempted from the entry-point "
        "rule as a package marker; with the file absent the exemption has "
        "nothing to cover and should be removed from this test.")
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


# ------------------------------------------------------- cross-references

_REF = re.compile(r"scripts/([A-Za-z0-9_-]+\.(?:py|sh))")


def dead_references(source: str, scripts_dir: Path) -> list[str]:
    """scripts/<name> referenced by this source that do not exist. Pure."""
    return sorted({r for r in _REF.findall(source)
                   if not (scripts_dir / r).exists()})


def syntax_error_in(source: str, name: str = "<probe>") -> str | None:
    """The SyntaxError message for this source, or None if it parses. Pure."""
    try:
        ast.parse(source, filename=name)
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"
    return None


@pytest.mark.parametrize("path", DRIVERS, ids=[p.name for p in DRIVERS])
def test_referenced_scripts_exist(path: Path):
    """Every scripts/<name> a driver names must be there to be called."""
    root = path.resolve().parents[1]
    missing = dead_references(path.read_text(encoding="utf-8"), root / "scripts")
    assert not missing, (
        f"{path.name} invokes scripts that do not exist: {missing}. Either "
        f"they were deleted without updating this driver, or the name is a "
        f"typo — both make the driver dead on arrival.")


# ------------------------------------------------- runnable without PYTHONPATH

_INSERT = "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))"


def _imports_vedic_at(tree: ast.Module) -> int | None:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "vedic":
            return node.lineno
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "vedic" for a in node.names):
            return node.lineno
    return None


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_a_script_that_imports_vedic_puts_the_repository_on_its_path(path):
    """Run as ``python scripts/<name>.py``, Python puts scripts/ on sys.path,
    not the repository root, so ``import vedic`` fails without PYTHONPATH.
    Four scripts lacked the insert -- among them verify_bit_exact.py, which
    train_lora.py runs as a mandatory gate, so training could not start
    outside an environment that happened to set PYTHONPATH."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    first = _imports_vedic_at(tree)
    if first is None:
        return
    root = "Path(__file__).resolve().parents[1]"
    # Names bound to the repository root, so `sys.path.insert(0, str(REPO))`
    # counts when REPO is that path.
    roots = {t.id for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.value) == root
             for t in n.targets if isinstance(t, ast.Name)}
    accepted = {_INSERT} | {f"sys.path.insert(0, str({name}))" for name in roots}
    inserts = [n.lineno for n in tree.body
               if isinstance(n, ast.Expr) and ast.unparse(n) in accepted]
    assert inserts and min(inserts) < first, (
        f"{path.name} imports vedic at line {first} without first running {_INSERT}")


def test_the_gate_train_lora_runs_works_without_pythonpath(tmp_path):
    """The real invocation: train_lora.py's subprocess call, from another
    directory, with PYTHONPATH removed from the environment."""
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[3]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(repo / "scripts" / "verify_bit_exact.py")],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "bit-exact" in r.stdout
