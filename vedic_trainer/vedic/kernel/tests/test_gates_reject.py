"""Every gate is proven to REJECT, on every run — not once, by hand.

Why this exists
---------------
The pattern this package is built on has one rule: a check you have never
seen fail is not evidence. Each gate here was regeneration-tested when it was
written — the defect it exists to catch was reintroduced, the gate was
required to go red, and only then was its green believed.

But that loop lived in a terminal, not in the suite. Exactly one gate
encoded it permanently; the rest rested on the author having done it once. That is the
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


#: A consistent set of arguments: two layers summing to the suite total, a
#: README table agreeing with both, nothing failing, and prose quoting the
#: same total. Each rejection below perturbs exactly one part of it.
CONSISTENT = dict(measured={"A": 3, "B": 2}, total=5, claimed={"A": 3, "B": 2},
                  n_failed=0, summary="5 passed", prose=(5, 5))


def _counts(**overrides):
    return VC.reconcile(**{**CONSISTENT, **overrides})


def test_counts_gate_accepts_a_consistent_table():
    """Precondition: without this, the rejections below prove nothing."""
    assert _counts() == []


def test_counts_gate_rejects_a_wrong_number():
    problems = _counts(claimed={"A": 3, "B": 9})
    assert problems and "measured 2" in problems[0]


def test_counts_gate_rejects_a_layer_the_readme_omits():
    assert any("no row for layer" in p for p in _counts(claimed={"A": 3}))


def test_counts_gate_rejects_a_readme_row_with_no_layer():
    problems = _counts(measured={"A": 3}, total=3,
                       claimed={"A": 3, "Ghost": 1}, prose=(3, 3))
    assert any("maps to no layer" in p for p in problems)


def test_counts_gate_rejects_tests_belonging_to_no_layer():
    """The check that keeps a new test file from being invisible."""
    problems = _counts(total=7, prose=(7, 7))
    assert any("belong to no layer" in p for p in problems)


def test_counts_gate_rejects_a_red_suite_behind_a_correct_count():
    """A count of collected tests must not launder a failing suite."""
    problems = _counts(n_failed=2, summary="2 failed, 3 passed", prose=(5, 3))
    assert any("do not pass" in p for p in problems)


def test_counts_gate_rejects_a_stale_prose_total():
    """The defect that got past the table check: the README's headline
    sentence sat two behind the suite while every row in the table matched."""
    problems = _counts(prose=(7, 7))
    assert any("prose says 7 tests are collected" in p for p in problems)


def test_counts_gate_rejects_a_prose_pass_count_that_ignores_failures():
    """'N collected and N pass' must not survive a red suite."""
    problems = _counts(n_failed=1, summary="1 failed, 4 passed")
    assert any("prose says 5 pass, measured 4" in p for p in problems)


def test_counts_gate_rejects_a_readme_with_no_prose_total_at_all():
    """Deleting the sentence must not be the way to satisfy the check."""
    assert any("states no prose total" in p for p in _counts(prose=None))


def test_counts_gate_runs_its_children_on_this_machine():
    """The nested pytest must see the same PATH the outer suite does.

    `verify_counts.py` shells out to pytest twice. It used to hand the child
    a hand-built environment -- PYTHONPATH plus a PATH hardcoded to three
    directories, and no HOME -- so a toolchain installed anywhere else was
    invisible to the child and visible to the parent. In CI that is Lean:
    elan installs it under `$HOME/.elan/bin`, the Lean mirror's tests no
    longer skip when the compiler is missing, and every one of them would
    fail in the child while passing in the parent. A gate whose verdict
    depends on which of two environments ran it is the defect this whole
    file is about.
    """
    import os
    env = VC.child_env()
    assert env.get("PATH") == os.environ.get("PATH"), \
        "the nested pytest gets a different PATH from this process"
    assert env.get("PYTHONPATH") == ".", \
        "the nested pytest cannot import the package under test"
    for key in ("HOME", "PATH"):
        assert key in env, f"the child environment drops {key}"


def test_counts_gate_reads_the_real_prose_total():
    """Without this the prose checks could pass on a regex matching nothing."""
    assert VC.readme_prose_total() is not None, \
        "the README's 'N tests are collected and N pass' sentence is gone or " \
        "reworded, so the prose check silently stopped applying"


# ═════════════════════════════ gate 2 — the withdrawn-figure detector

WD = _load("vedic/kernel/tests/test_no_withdrawn_number_is_quoted.py")
DOC = (REPO / "ABLATION_RESULTS.md").read_text(encoding="utf-8")


def test_withdrawal_gate_has_figures_to_look_for():
    """Precondition: with an empty set every rejection below is vacuous."""
    assert WD.CE, "no held-out CE values were read out of runs/"
    assert WD.PAIRS, "no arm pairs were reconstructed"
    assert WD.PERCENTAGES, "no percentage deltas were reconstructed"


def test_withdrawal_gate_accepts_the_withdrawal_document():
    assert WD.quoted_withdrawn(DOC) == []


def test_withdrawal_gate_rejects_a_reinstated_cross_entropy():
    """The figures come back the way they left: pasted into a table."""
    value = sorted(WD.CE.values())[0]
    broken = DOC + f"\n\n| 42 | {value:.4f} | 1.0000 |\n"
    assert WD.quoted_withdrawn(broken), "a reinstated held-out CE passed the gate"


def test_withdrawal_gate_rejects_a_reinstated_arm_mean():
    """The means appear in no run file, so a string search would miss them."""
    treatment = max(WD.ARM_MEANS.values())
    assert treatment not in set(WD.CE.values()), \
        "this mean coincides with a raw run; pick another for the test"
    assert WD.quoted_withdrawn(f"mean held-out CE {treatment:.4f}")


def test_withdrawal_gate_rejects_a_reinstated_percentage():
    """The number a reader remembers, reconstructed from the pairs."""
    treatment, control = WD.PAIRS[0]
    pct = 100.0 * (treatment - control) / control
    assert WD.quoted_withdrawn(f"worse by {pct:+.2f}%")


def test_withdrawal_gate_accepts_the_same_digits_written_as_a_frequency():
    """`7.83` is both the withdrawn percentage and the Schumann resonance.

    The `%` is what separates them, so this pins the distinction rather than
    leaving the gate to flag a frequency as a result.
    """
    treatment, control = WD.PAIRS[0]
    pct = 100.0 * (treatment - control) / control
    assert WD.quoted_withdrawn(f"{pct:.2f}%")
    assert WD.quoted_withdrawn(f"a resonance at {pct:.2f} Hz") == []


def test_withdrawal_gate_rejects_the_corpus_pipeline_coming_back(monkeypatch):
    """The withdrawal rests on the data being unbuildable; if a generator
    reappears the notice must be revisited, not left standing beside it."""
    monkeypatch.setattr(WD, "_tracked",
                        lambda pattern: [pattern] if "generate_synthetic" in pattern
                        else [])
    with pytest.raises(AssertionError, match="pipeline is back"):
        WD.test_the_corpus_those_runs_used_cannot_be_rebuilt()


def test_withdrawal_gate_rejects_a_document_that_drops_the_notice(monkeypatch):
    """Deleting the tables without saying why reads as 'never measured'."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "ABLATION_RESULTS.md").write_text("# Ablation\n\nNothing here.\n")
    monkeypatch.setattr(WD, "REPO", tmp)
    with pytest.raises(AssertionError, match="does not say the figures are withdrawn"):
        WD.test_the_withdrawal_is_stated_where_the_results_were()


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


def test_path_gate_accepts_a_path_that_exists():
    assert DP.is_live("README.md", "README.md")


def test_path_gate_rejects_a_renamed_module_still_cited():
    """The defect the whole file exists for: a pointer left behind by a rename."""
    assert not DP.resolves("vedic/kernel/sutras_never_existed.py", "README.md")


def test_path_gate_resolves_a_bare_filename_in_prose():
    """Documents write `losses.py`, not the full path, and that is a live
    pointer exactly when one tracked file carries the name."""
    assert DP.resolves("losses.py", "docs/ARCHITECTURE.md")
    assert not DP.resolves("definitely_not_a_file_here.py", "docs/ARCHITECTURE.md")


def test_path_gate_resolves_above_the_package():
    """CI_AUTOMATION.md cites workflow files that live above this package.
    Resolving against the package alone reported every one of them dead."""
    assert DP.resolves(".github/workflows/submit-pypi.yml", "docs/CI_AUTOMATION.md")


def test_path_gate_resolves_relative_to_the_citing_document():
    """`docs/external/README.md` writes `reference/foo.py`, meaning beside it."""
    assert DP.resolves("reference/extended_subsutras_palindrome.py",
                       "docs/external/README.md")


def test_path_gate_refuses_an_ambiguous_bare_filename(monkeypatch):
    """The basename rule fires only when exactly one tracked file carries the
    name; two or more is not a pointer a reader can follow.

    Exercised on a synthetic name rather than a real one. `__init__.py` has
    fourteen matches here and still resolves — because a file of that name
    also sits at the work-tree root, so the *work-tree-relative* rule catches
    it first and the answer is right for a different reason. Asserting
    against it would have tested that coincidence instead of this rule.
    """
    monkeypatch.setitem(DP.BY_NAME, "twin.py", ["a/twin.py", "b/twin.py"])
    monkeypatch.setitem(DP.BY_NAME, "lone.py", ["a/lone.py"])
    assert not DP.resolves("twin.py", "README.md")
    assert DP.resolves("lone.py", "README.md")


def test_path_gate_honours_only_declared_externals():
    assert DP.is_live(sorted(DP.EXTERNAL)[0], "README.md")
    assert not DP.is_live("someones_kernel.html", "README.md")


def test_path_gate_honours_only_declared_removals():
    """A path named because it is gone passes; an undeclared dead one does not."""
    assert DP.is_live("vedic/kernel/sutras_exact.py", "docs/SUTRA_CATALOGUE.md")
    assert not DP.is_live("vedic/kernel/sutras_imaginary.py", "docs/SUTRA_CATALOGUE.md")


def test_path_gate_rejects_a_generated_artifact_that_is_present_locally():
    """The defect that reached CI: `data/*` is gitignored, so a document
    naming a generated file passed on a machine that had run the generator
    and failed in a fresh clone. Resolving against the tracked set makes the
    verdict identical in both places, which is the whole point."""
    generated = "data/train.jsonl"
    assert generated not in DP.TRACKED, \
        f"{generated} is tracked now — pick another gitignored path here"
    assert not DP.is_live(generated, "README.md"), \
        "the path gate accepted a file git does not track"


def test_path_gate_reads_a_real_tracked_set():
    """Without this, an empty TRACKED would make the rejections above pass
    for the wrong reason."""
    assert "README.md" in DP.TRACKED and len(DP.TRACKED) > 50
    assert len(DP.ROOT_TRACKED) > len(DP.TRACKED)


def test_path_gate_covers_every_tracked_document():
    """DOCS was a hand-written two-element list, so everything under docs/
    sat outside the gate for the life of the project. It is discovered now,
    and this fails if anyone narrows it back."""
    import subprocess
    tracked_md = set(subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.md"],
        capture_output=True, text=True, check=True).stdout.split())
    assert set(DP.DOCS) == tracked_md, (
        f"DOCS misses {sorted(tracked_md - set(DP.DOCS))} — a document nothing "
        f"checks is a document that drifts")
    assert any(d.startswith("docs/") for d in DP.DOCS)


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
