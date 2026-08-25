"""Every gate is proven to REJECT, on every run — not once, by hand.

Why this exists
---------------
The pattern this package is built on has one rule: a check you have never
seen fail is not evidence. Each gate here was regeneration-tested when it was
written — the defect it exists to catch was reintroduced, the gate was
required to go red, and only then was its green believed.

But that loop lived in a terminal, not in the suite. Exactly one gate
(`test_reported_ablation.test_the_gate_rejects_a_wrong_number`) encoded it
permanently; the rest rested on the author having done it once. That is the
same "trust the author's carefulness" assumption the whole pattern exists to
remove, and it is how a gate silently rots into a decoration: someone
refactors the comparison, every existing test still passes, and nothing ever
asks whether it can still fail.

So each gate's *judgment* is separated from its *measurement*, and the
judgment is exercised here against a deliberately broken input. Nothing in
this file writes to the repository — a regeneration test that edits real
files leaves the tree dirty when it fails midway, which is a worse defect
than the one it was checking for.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(rel: str):
    """Import a script by path, registered in sys.modules for dataclasses."""
    spec = importlib.util.spec_from_file_location(
        "_gate_" + Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════ gate 1 — verify_counts

VC = _load("scripts/verify_counts.py")


def test_counts_gate_accepts_a_consistent_table():
    """Precondition: without this, the rejections below prove nothing."""
    assert VC.reconcile({"A": 3, "B": 2}, 5, {"A": 3, "B": 2}) == []


def test_counts_gate_rejects_a_wrong_number():
    problems = VC.reconcile({"A": 3, "B": 2}, 5, {"A": 3, "B": 9})
    assert problems and "measured 2" in problems[0]


def test_counts_gate_rejects_a_layer_the_readme_omits():
    problems = VC.reconcile({"A": 3, "B": 2}, 5, {"A": 3})
    assert any("no row for layer" in p for p in problems)


def test_counts_gate_rejects_a_readme_row_with_no_layer():
    problems = VC.reconcile({"A": 3}, 3, {"A": 3, "Ghost": 1})
    assert any("maps to no layer" in p for p in problems)


def test_counts_gate_rejects_tests_belonging_to_no_layer():
    """The check that keeps a new test file from being invisible."""
    problems = VC.reconcile({"A": 3, "B": 2}, 7, {"A": 3, "B": 2})
    assert any("belong to no layer" in p for p in problems)


def test_counts_gate_rejects_a_red_suite_behind_a_correct_count():
    """A count of collected tests must not launder a failing suite."""
    problems = VC.reconcile({"A": 3}, 3, {"A": 3}, 2, "2 failed, 1 passed")
    assert any("do not pass" in p for p in problems)


# ═══════════════════════════════════════ gate 2 — verify_ablation

VA = _load("scripts/verify_ablation.py")
SETS = {rs.key: m for rs in VA.RUN_SETS if (m := VA.load(rs)) is not None}
DOC = (REPO / "ABLATION_RESULTS.md").read_text(encoding="utf-8")


def test_ablation_gate_accepts_the_real_document():
    assert SETS, "no run sets loaded; the rejections below would be vacuous"
    assert VA.check(DOC, SETS) == []


def _first_seed_cell() -> tuple[str, str]:
    key = sorted(SETS)[0]
    return key, f"{SETS[key].seeds[42][0]:.4f}"


def test_ablation_gate_rejects_a_digit_changed_in_its_last_place():
    _key, real = _first_seed_cell()
    broken = DOC.replace(f"| 42 | {real} |", f"| 42 | {real[:-1]}9 |", 1)
    assert broken != DOC, f"expected a seed-42 cell quoting {real}"
    assert VA.check(broken, SETS), "a wrong digit passed the gate"


def test_ablation_gate_rejects_prose_in_a_numeric_column():
    _key, real = _first_seed_cell()
    broken = DOC.replace(f"| 42 | {real} |", "| 42 | ~1.65 |", 1)
    assert broken != DOC
    problems = VA.check(broken, SETS)
    assert any("declared numeric but reads" in p for p in problems)


def test_ablation_gate_rejects_an_undeclared_numeric_column():
    """The rule that caught an unchecked sd-of-delta column on its first run.

    A cell beyond the declared columns is not skipped as unrecognised -- it is
    reported, because a number nobody verifies is exactly how the figures
    drifted apart in the first place.
    """
    broken = DOC.replace("| sd | 0.0064 | 0.0786 | | | |",
                         "| sd | 0.0064 | 0.0786 | | | | 0.5 |", 1)
    assert broken != DOC
    problems = VA.check(broken, SETS)
    assert any("no column is declared" in p for p in problems), problems


def test_ablation_gate_rejects_a_wrong_value_in_a_declared_spread_column():
    """Every reported spread must be derivable, including sd of the deltas."""
    broken = DOC.replace("| sd | 0.0064 | 0.0786 | | | |",
                         "| sd | 0.0064 | 0.0786 | 0.9999 | | |", 1)
    assert broken != DOC
    problems = VA.check(broken, SETS)
    assert any("delta says 0.9999" in p for p in problems), problems


def test_ablation_gate_rejects_a_missing_sd_row():
    broken = DOC.replace("| sd | 0.0064 | 0.0786 | | | |", "", 1)
    assert broken != DOC
    assert any("no sd row" in p for p in VA.check(broken, SETS))


def test_ablation_gate_rejects_a_run_file_in_the_wrong_arm():
    """The arm comes from the adapter each file records, not its name."""
    swapped = json.loads((REPO / "runs" / "fixed_seed42_full.json").read_text())
    tmp = Path(tempfile.mkdtemp()) / "fixed_seed42_base.json"
    tmp.write_text(json.dumps(swapped))
    with pytest.raises(ValueError, match="the file and its slot disagree"):
        VA.ce(tmp, r"no_sutra$")


def test_ablation_gate_rejects_a_run_file_with_no_adapter():
    tmp = Path(tempfile.mkdtemp()) / "anon.json"
    tmp.write_text(json.dumps({"heldout": {"ce_loss": 1.0}}))
    with pytest.raises(ValueError, match="records no adapter"):
        VA.ce(tmp, r"no_sutra$")


# ════════════════════════════════════ gate 3 — verify_bit_exact

VB = _load("scripts/verify_bit_exact.py")


def test_bit_exact_gate_refuses_to_run_without_its_reference(monkeypatch):
    """It must never rebuild the reference from the code under test."""
    monkeypatch.setattr(VB, "FIXTURE_DIR", Path(tempfile.mkdtemp()))
    with pytest.raises(SystemExit) as e:
        VB._require_fixtures()
    assert "cannot falsify anything" in str(e.value)


def test_bit_exact_gate_proceeds_when_the_reference_is_present():
    VB._require_fixtures()          # the real fixtures/, committed


# ═══════════════════════════════ documented paths / script drivers

DP = _load("vedic/kernel/tests/test_documented_paths.py")
SV = _load("vedic/kernel/tests/test_scripts_are_valid.py")


def test_path_gate_accepts_paths_that_exist():
    assert DP.missing({"README.md"}, DP.TRACKED, set()) == []


def test_path_gate_rejects_a_renamed_module_still_cited():
    assert DP.missing({"vedic/kernel/sutras_exact.py"}, DP.TRACKED, set()) == \
        ["vedic/kernel/sutras_exact.py"]


def test_path_gate_honours_only_declared_externals():
    assert DP.missing({"someones_kernel.html"}, DP.TRACKED,
                      {"someones_kernel.html"}) == []
    assert DP.missing({"someones_kernel.html"}, DP.TRACKED, set()) != []


def test_path_gate_rejects_a_generated_artifact_that_is_present_locally():
    """The defect that reached CI: `data/*` is gitignored, so a document
    naming a generated file passed on a machine that had run the generator
    and failed in a fresh clone. Resolving against the tracked set makes the
    verdict identical in both places, which is the whole point."""
    generated = "data/synthetic_eval.jsonl"
    assert generated not in DP.TRACKED, \
        f"{generated} is tracked now — pick another gitignored path here"
    assert DP.missing({generated}, DP.TRACKED, set()) == [generated], \
        "the path gate accepted a file git does not track"


def test_path_gate_reads_a_real_tracked_set():
    """Without this, an empty TRACKED would make the two rejection tests
    above pass for the wrong reason."""
    assert "README.md" in DP.TRACKED and len(DP.TRACKED) > 50


def test_script_gate_accepts_a_driver_whose_references_exist():
    assert SV.dead_references("python scripts/train_lora.py",
                              REPO / "scripts") == []


def test_script_gate_rejects_a_driver_calling_a_deleted_script():
    assert SV.dead_references("python scripts/run_ablation_eval.py --x",
                              REPO / "scripts") == ["run_ablation_eval.py"]


def test_script_gate_accepts_source_that_parses():
    assert SV.syntax_error_in("def main():\n    return 0\n") is None


def test_script_gate_rejects_the_indentation_error_that_shipped():
    """The exact defect class that merged because nothing compiled scripts/."""
    bad = "def main():\n    t0 = 1\n" + "  x = 2\n"
    assert SV.syntax_error_in(bad) is not None


# ═════════════════════════════════════════ exact-arithmetic boundary

BG = _load("vedic/kernel/tests/test_blueprint_gates.py")


def test_float_gate_accepts_a_module_whose_only_float_is_the_boundary():
    src = ("def to_float(self):\n    return float(self.a) * _SQRT2_F\n"
           "def __add__(self, o):\n    return Q2(self.a + o.a)\n")
    assert BG.float_offenders(src) == []


def test_float_gate_rejects_a_float_literal_outside_the_boundary():
    src = ("def to_float(self):\n    return float(self.a)\n"
           "def __add__(self, o):\n    _ = 1.5\n    return o\n")
    assert BG.float_offenders(src) == ["__add__: 1.5"]


def test_float_gate_rejects_reading_the_irrational_constant_elsewhere():
    src = ("def to_float(self):\n    return _SQRT2_F\n"
           "def __mul__(self, o):\n    return _SQRT5_F * o\n")
    assert BG.float_offenders(src) == ["__mul__ reads _SQRT5_F"]


# ═══════════════════════════════════════════ live-objective detector

PROBE = _load("scripts/probe_aux_gradients.py")


def test_dead_loss_detector_reports_all_four_live_on_the_real_losses():
    weights = PROBE.weights_from(REPO / "configs" / "ablations" / "cpu_full.yaml")
    assert PROBE.probe(weights)["dead"] == []


def test_dead_loss_detector_names_a_loss_detached_from_psi(monkeypatch):
    """The check that should have stopped the first ablation and did not."""
    import torch
    monkeypatch.setattr(PROBE, "L_cons", lambda psi: torch.tensor(3.0))
    weights = PROBE.weights_from(REPO / "configs" / "ablations" / "cpu_full.yaml")
    report = PROBE.probe(weights)
    assert report["dead"] == ["L_cons"]
    assert report["losses"]["L_cons"]["grad_l1"] == 0.0


# ══════════════════════════════════════════ structural verifier

SHOW = _load("scripts/show_sutras.py")


def test_structural_verifier_passes_on_the_real_operators(capsys):
    sys.argv = ["show_sutras", "--verify"]
    assert SHOW.main() == 0
    assert "FAILED" not in capsys.readouterr().out


def test_structural_verifier_exits_nonzero_when_operators_go_inert(
        monkeypatch, capsys):
    """It printed True/False and returned 0 either way."""
    monkeypatch.setattr(SHOW.K, "apply_sutra", lambda sid, psi, st: psi)
    sys.argv = ["show_sutras", "--verify"]
    assert SHOW.main() == 1
    assert "every sutra moves the field" in capsys.readouterr().out
