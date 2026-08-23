"""Every ablation number in ABLATION_RESULTS.md is verified against runs/.

The document and the pull request that quotes it are prose; ``runs/*.json``
are the measurements. Prose drifts. This test makes the drift a test failure
rather than something a reader has to notice.

It calls ``scripts/verify_ablation.py`` in-process (never as a pytest
subprocess -- that would recurse through the suite that contains this file).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "_verify_ablation", REPO / "scripts" / "verify_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules, so the module has
    # to be registered before it is executed.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


VA = _load_gate()
SETS = {rs.key: m for rs in VA.RUN_SETS if (m := VA.load(rs)) is not None}


def test_the_evidence_travels_with_the_repository():
    """runs/*.json must be tracked: a number nobody can re-read is a claim."""
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files", "runs/"], cwd=REPO,
        capture_output=True, text=True).stdout.split()
    assert tracked, "runs/ is not tracked; ABLATION_RESULTS.md would be unbacked"
    on_disk = {p.name for p in (REPO / "runs").glob("*.json")}
    assert on_disk <= {Path(t).name for t in tracked}, (
        "some run JSONs exist locally but are not committed: "
        f"{sorted(on_disk - {Path(t).name for t in tracked})}")


def test_at_least_one_run_set_is_present():
    assert SETS, "no ablation runs found under runs/ — nothing to verify against"


def test_document_matches_the_measurements():
    problems = VA.check((REPO / "ABLATION_RESULTS.md").read_text("utf-8"), SETS)
    assert not problems, "ABLATION_RESULTS.md disagrees with runs/:\n" + \
        "\n".join(f"  - {p}" for p in problems)


@pytest.mark.parametrize("key", sorted(SETS))
def test_every_run_json_is_a_complete_record(key):
    """A truncated or partially-written run must not be quotable."""
    rs = next(r for r in VA.RUN_SETS if r.key == key)
    for seed in VA.SEEDS:
        for arm in ("base", "full"):
            path = VA._path(rs, arm, seed)
            rec = json.loads(path.read_text())
            assert "heldout" in rec, f"{path.name} has no heldout block"
            h = rec["heldout"]
            for field in ("ce_loss", "ppl", "n_tokens"):
                assert field in h, f"{path.name} is missing {field}"
            assert h["n_tokens"] > 0, f"{path.name} evaluated zero tokens"
            assert h["ce_loss"] > 0, f"{path.name} has a non-positive CE"


@pytest.mark.parametrize("key", sorted(SETS))
def test_both_arms_were_evaluated_on_the_same_token_count(key):
    """Comparing CE across arms is only meaningful on identical held-out data."""
    counts = set()
    rs = next(r for r in VA.RUN_SETS if r.key == key)
    for seed in VA.SEEDS:
        for arm in ("base", "full"):
            counts.add(json.loads(VA._path(rs, arm, seed).read_text())
                       ["heldout"]["n_tokens"])
    assert len(counts) == 1, f"{key}: arms saw different token counts {counts}"


def test_the_gate_rejects_a_wrong_number():
    """Regeneration check: the gate must fail on a perturbed document.

    Without this, a gate that silently passes everything would look identical
    to a gate that works.
    """
    text = (REPO / "ABLATION_RESULTS.md").read_text("utf-8")
    assert not VA.check(text, SETS), "precondition: the real document passes"
    key = sorted(SETS)[0]
    m = SETS[key]
    real = f"{m.seeds[42][0]:.4f}"
    perturbed = text.replace(f"| 42 | {real} |", f"| 42 | {real[:-1]}9 |", 1)
    assert perturbed != text, f"expected to find a seed-42 cell quoting {real}"
    assert VA.check(perturbed, SETS), "the gate passed a document it should reject"


RUN_FILES = sorted((REPO / "runs").glob("*.json"))


def test_there_are_run_files():
    assert RUN_FILES, "runs/ holds no JSON — the evidence directory is empty"


@pytest.mark.parametrize("path", RUN_FILES, ids=[p.name for p in RUN_FILES])
def test_run_file_is_actually_json(path: Path):
    """A .json that is captured stdout is not evidence, it is a transcript.

    Two probe outputs were committed here as `.json` while actually holding
    piped terminal output -- one with a trailing human-readable line, one
    that was entirely a torch warning. Both are unreadable by anything that
    trusts the extension.
    """
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AssertionError(
            f"{path.name} is named .json but does not parse: {e}. If it is a "
            f"terminal capture, it belongs in a .log (which .gitignore "
            f"excludes) and the producing script should write real JSON.")
