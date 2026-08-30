"""The withdrawn ablation figures must not be quoted anywhere.

Why this exists
---------------
`ABLATION_RESULTS.md` reported held-out cross-entropy for two arms across
three seeds and four weightings. Those runs were executed on a *synthetic*
corpus — 5,120 records expanded by template from a seed file, with Ψ produced
by a hand-written text encoder — and that whole pipeline has been removed.
This repository's own `.gitignore` had already written down the standard:
the seed corpus was kept precisely because *without it nothing here can
regenerate or falsify the numbers in ABLATION_RESULTS.md*. That is now the
case, so the numbers are withdrawn.

The `runs/*.json` files stay. They are an honest record of what was executed,
and deleting evidence is not the same as withdrawing a claim. What is
forbidden is quoting them — and quoting them is easy to do by accident,
because the figures a reader remembers are means and percentage deltas that
appear in no run file verbatim.

So this gate does not string-match. It reconstructs the *derived* quantities
too — per-arm means, per-seed and mean-to-mean absolute deltas, the relative
deltas as percentages, and the seed spreads — then reads every decimal number
out of every tracked document and asks whether it is any of them, rounded to
its own printed precision. `+7.83%` and `1.7240` are caught even though
neither appears in any JSON file.

This is modelled on `vedic/eval/tests/test_no_subset_is_quoted_as_a_benchmark.py`,
which does the same job for the withdrawn SCAN/COGS subset scores.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev, stdev

import pytest

REPO = Path(__file__).resolve().parents[3]

#: A control-arm run: the sutra weights were all zero. Named by the adapter
#: or the file, both of which the run files carry.
_CONTROL = ("no_sutra", "_base")


def _tracked(pattern: str) -> list[str]:
    """Files git actually tracks. `Path.exists()` would also see untracked
    scratch files, which are not what a reader of this repository gets."""
    out = subprocess.run(["git", "ls-files", pattern], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def _run_files() -> list[Path]:
    return [REPO / name for name in _tracked("runs/*.json")]


def _heldout() -> dict[str, float]:
    """`name -> held-out ce_loss` for every run file that carries one."""
    out: dict[str, float] = {}
    for path in _run_files():
        # No try/except. A run file that will not parse is a broken record,
        # and skipping it would quietly shrink the set this gate covers.
        data = json.loads(path.read_text(encoding="utf-8"))
        block = data.get("heldout")
        if isinstance(block, dict) and "ce_loss" in block:
            out[path.name] = float(block["ce_loss"])
    return out


def _perplexities() -> list[float]:
    out = []
    for path in _run_files():
        block = json.loads(path.read_text(encoding="utf-8")).get("heldout")
        if isinstance(block, dict) and "ppl" in block:
            out.append(float(block["ppl"]))
    return out


CE = _heldout()
PPL = _perplexities()


def _arm(name: str) -> str:
    """Control arms have all four sutra weights at zero; ``eval_base`` is the
    untuned model, which the document also compared treatment runs against."""
    return "control" if any(m in name for m in _CONTROL) else "treatment"


def _family(name: str) -> str:
    """The run family: the prefix each weighting's runs were written under
    (``disjoint_``, ``fixed_``, ``rerun_``, ``scaled_``, ``heldout_``,
    ``eval_``), which is how the four weightings were kept apart."""
    return name.split("_")[0]


def _groups() -> dict[tuple[str, str], list[float]]:
    groups: dict[tuple[str, str], list[float]] = {}
    for name, value in CE.items():
        groups.setdefault((_family(name), _arm(name)), []).append(value)
    return groups


GROUPS = _groups()


def _seed(name: str) -> str:
    """The seed a run was made at, or "" for the unseeded first family."""
    m = re.search(r"seed(\d+)", name)
    return m.group(1) if m else ""


def _arm_means() -> dict[tuple[str, str], float]:
    return {key: mean(members) for key, members in GROUPS.items()}


ARM_MEANS = _arm_means()

#: The untuned base model. The document compared tuned arms against it
#: ("fine-tuning cuts held-out CE by ..."), so those comparisons are
#: reconstructed too.
BASE_CE = CE.get("eval_base.json")


def comparable_pairs() -> list[tuple[float, float]]:
    """(treatment, control) pairs a document could plausibly have quoted.

    Deliberately not every pair. An all-pairs set over 28 runs and their
    means is dense enough that ordinary numbers fall inside it — `1.35%` of
    trainable parameters lands in it, and a gate that flags that is not a
    gate. The pairs here are the ones the tables were actually built from:

    * within a run family, each seed's treatment against its own control;
    * the arm means, across families as well as within one, because the
      `scaled` arm was compared against the `fixed` controls it reused;
    * the untuned base against everything, for the reduction figures.
    """
    pairs: list[tuple[float, float]] = []

    by_slot = {}
    for name, value in CE.items():
        by_slot[(_family(name), _seed(name), _arm(name))] = value
    for (family, seed, arm), value in by_slot.items():
        if arm != "treatment":
            continue
        control = by_slot.get((family, seed, "control"))
        if control is not None:
            pairs.append((value, control))

    treatments = [v for (_, a), v in ARM_MEANS.items() if a == "treatment"]
    controls = [v for (_, a), v in ARM_MEANS.items() if a == "control"]
    pairs += [(t, c) for t in treatments for c in controls]

    if BASE_CE is not None:
        others = list(CE.values()) + list(ARM_MEANS.values())
        pairs += [(v, BASE_CE) for v in others if v != BASE_CE]

    return pairs


PAIRS = comparable_pairs()


def withdrawn_magnitudes() -> set[float]:
    """Figures quoted as a cross-entropy, a perplexity, a spread or a
    difference of two of them — the document wrote these to four decimals."""
    values: set[float] = set(CE.values()) | set(PPL) | set(ARM_MEANS.values())
    for members in GROUPS.values():
        if len(members) > 1:
            values.add(pstdev(members))
            values.add(stdev(members))
    for treatment, control in PAIRS:
        values.add(treatment - control)
        values.add(control - treatment)
    return values


def withdrawn_percentages() -> set[float]:
    """Figures quoted as a percentage change between an arm and its control.

    These are the numbers a reader remembers — `+7.83%`, `+25.63%`, the
    `73.8%` reduction against the untuned base — and none of them appears in
    any run file. They are reconstructed from the pairs that produced them.
    """
    values: set[float] = set()
    for treatment, control in PAIRS:
        if control:
            values.add(100.0 * (treatment - control) / control)
            values.add(100.0 * (control - treatment) / control)
        if treatment:
            values.add(100.0 * (control - treatment) / treatment)
    return values


MAGNITUDES = withdrawn_magnitudes()
PERCENTAGES = withdrawn_percentages()

#: Decimal numbers, with their printed precision. Thousands separators are
#: stripped first so `1,843,200` does not read as three numbers.
_NUMBER = re.compile(r"(?<![\w.])(\d+)\.(\d+)(?![\w.])")

#: A magnitude has to be printed to at least three places to be a quotation
#: rather than a coincidence: the withdrawn tables used four, and two-decimal
#: numbers are far too common in ordinary prose to carry a signal.
MIN_MAGNITUDE_DECIMALS = 3

#: A percentage is matched at two places, because that is how the withdrawn
#: percentages were written — but only when the number is actually written as
#: a percentage, which is what keeps `1.35%` of trainable parameters and a
#: `+7.83%` regression apart when neither the set nor the precision can.
MIN_PERCENTAGE_DECIMALS = 2


def _rounds_to(value: float, printed: float, decimals: int) -> bool:
    """True when `value`, printed to `decimals` places, reads as `printed`."""
    return abs(value - printed) < 0.5 * 10.0 ** -decimals


def quoted_withdrawn(text: str) -> list[str]:
    """Numbers in `text` that are a withdrawn figure at their own precision.

    Pure, and exercised on both sides of the distinction below so the gate
    cannot pass by matching nothing at all.
    """
    flat = text.replace(",", "")
    hits: list[str] = []
    for m in _NUMBER.finditer(flat):
        decimals = len(m.group(2))
        printed = float(m.group(0))
        is_percentage = flat[m.end():m.end() + 1] == "%"
        if is_percentage and decimals >= MIN_PERCENTAGE_DECIMALS:
            if any(_rounds_to(v, printed, decimals) for v in PERCENTAGES):
                hits.append(m.group(0) + "%")
                continue
        if decimals >= MIN_MAGNITUDE_DECIMALS:
            if any(_rounds_to(v, printed, decimals) for v in MAGNITUDES):
                hits.append(m.group(0))
    return hits


def test_there_are_run_records_to_check() -> None:
    """Guards everything below: with no records the gate is vacuous."""
    assert CE, "no runs/*.json carries a heldout.ce_loss — either the evidence " \
               "was deleted (it should not be) or this reader is broken"
    assert PAIRS, "no comparable arm pairs were reconstructed"
    assert len(MAGNITUDES) > len(CE), "no derived magnitudes were reconstructed"
    assert PERCENTAGES, "no percentage deltas were reconstructed"


def test_the_evidence_travels_with_the_repository() -> None:
    """A measurement nobody can re-read is a claim, not a record."""
    tracked = set(_tracked("runs/*.json"))
    assert tracked, "runs/*.json is untracked; the withdrawal has no evidence behind it"
    on_disk = {f"runs/{p.name}" for p in (REPO / "runs").glob("*.json")}
    assert on_disk <= tracked, f"untracked run files: {sorted(on_disk - tracked)}"


#: `git ls-files "*.md"` matches at any depth, so this is every tracked
#: Markdown file in the repository, deduplicated and ordered.
DOCS = sorted(set(_tracked("*.md")))


@pytest.mark.parametrize("doc", DOCS)
def test_no_document_quotes_a_withdrawn_figure(doc: str) -> None:
    """The withdrawal itself, over every tracked Markdown file."""
    text = (REPO / doc).read_text(encoding="utf-8")
    hits = quoted_withdrawn(text)
    assert not hits, (
        f"{doc} quotes withdrawn ablation figures: {sorted(set(hits))}\n\n"
        f"These were measured on a synthetic corpus this repository can no "
        f"longer regenerate, under two loss definitions it no longer "
        f"implements. Report a figure measured on data that is here, or "
        f"claim nothing.")


@pytest.mark.parametrize("quotation", [
    "held-out CE 1.7240 on the eval split",   # reconstructed arm mean
    "the treatment arm reached 1.8591",       # reconstructed arm mean
    "seed 42 scored 1.7315",                  # a single run's CE
    "and 1.8616 with the losses on",          # the matching treatment run
    "the untuned base sits at 6.2825",        # base CE
    "perplexity 535.110 against it",          # base perplexity
    "worse by +7.83%",                        # mean-to-mean relative delta
    "a penalty of 25.63% at the original weights",
])
def test_the_detector_catches_the_figures_that_were_withdrawn(quotation: str) -> None:
    """Without this the gate above could pass by detecting nothing.

    Every figure here appeared in the withdrawn tables. The arm means and
    every percentage appear in no run file: they are reconstructed.
    """
    assert quoted_withdrawn(quotation), \
        f"the detector missed a withdrawn figure in: {quotation!r}"


@pytest.mark.parametrize("sentence", [
    "pi is 3.14159 and always was",           # unrelated constant
    "beta_cons: 0.05 in every arm config",    # a loss weight
    "1,843,200 trainable params (1.35%)",     # adapter size
    "leanprover/lean4:v4.10.0",               # a toolchain pin
    "8 inputs x 29 sutras x 3 strengths",     # a fixture count
    "the Schumann resonance at 7.83 Hz",      # the same digits, not a percentage
])
def test_the_detector_leaves_unrelated_numbers_alone(sentence: str) -> None:
    """The other side of the distinction: a gate that flags everything is no
    more useful than one that flags nothing.

    The last case is the one that decides the design. `7.83` is exactly the
    percentage that was withdrawn, and it is also a frequency this project
    uses. Requiring the `%` is what separates them — the withdrawn figure was
    a percentage, and a number that is not written as one is not it.
    """
    assert not quoted_withdrawn(sentence), \
        f"the detector flagged an unrelated number in: {sentence!r}"


def test_the_corpus_those_runs_used_cannot_be_rebuilt() -> None:
    """Pins the fact the withdrawal rests on.

    If a corpus generator reappears, this fails — and the withdrawal notice
    in ABLATION_RESULTS.md should then be revisited rather than left standing
    beside a pipeline that could falsify it.
    """
    gone = ("data/seed_corpus.txt", "scripts/generate_synthetic.py",
            "scripts/split_corpus.py", "vedic/data/tesseract_encode.py",
            "vedic/data/synthetic_contradiction.py",
            "vedic/data/synthetic_paraphrase.py")
    present = [name for name in gone if _tracked(name)]
    assert not present, (
        f"the synthetic corpus pipeline is back: {present}. The ablation "
        f"figures were withdrawn because nothing here could regenerate the "
        f"data they were measured on; that reason no longer holds, so "
        f"ABLATION_RESULTS.md needs revisiting rather than this test "
        f"weakening.")


def test_the_withdrawal_is_stated_where_the_results_were() -> None:
    """Silence reads as 'no result', not as 'result withdrawn'."""
    text = (REPO / "ABLATION_RESULTS.md").read_text(encoding="utf-8").lower()
    assert "withdrawn" in text, "ABLATION_RESULTS.md does not say the figures are withdrawn"
    assert "unmeasured" in text, "ABLATION_RESULTS.md does not record the gap the withdrawal leaves"
