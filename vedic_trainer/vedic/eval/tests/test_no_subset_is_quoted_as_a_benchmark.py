"""A subset run must never be quoted as a benchmark result.

Why this exists
---------------
`runs/eval_*.json` carry SCAN and COGS blocks recording `n_total` of 30 and
20, against real splits of 3,920–21,000. Those came from
`--scan-subset 30 --cogs-subset 20` on a script that no longer exists, and
they were quoted in ABLATION_RESULTS.md and the README under a heading
asserting what the benchmark can and cannot do — 0.1–0.7% of the data,
presented as the data.

`scripts/eval_benchmarks.py` replaced that script and opens by saying "There
is no --subset and no --skip: a truncated benchmark is not the benchmark,
and a flag that shortens it is how a partial result gets reported as a
complete one." The subset figures survived that change anyway, because
removing the *mechanism* does nothing about *numbers already published*.

The run files are kept: they are an honest record of what was executed, and
deleting evidence is not the same as withdrawing a claim. What is forbidden
is quoting them. This test enforces that, and would equally catch a future
subset run being written up as a benchmark.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = ("ABLATION_RESULTS.md", "README.md")

# Real split sizes for the benchmarks this package names, from the canonical
# releases. A run whose n_total is below its split size is a subset run.
FULL_SPLIT_SIZE = {
    ("scan", "simple"): 4182,
    ("scan", "length"): 3920,
    ("scan", "addprim_jump"): 3920,
    ("cogs", "test"): 3000,
    ("cogs", "gen"): 21000,
}


def _benchmark_blocks() -> list[tuple[str, str, str, dict]]:
    """(file, benchmark, split, record) for every benchmark block in runs/."""
    out = []
    for path in sorted((REPO / "runs").glob("*.json")):
        # No try/except: a run file that will not parse is a broken record,
        # and skipping it here would quietly shrink the set this gate checks.
        # Deferring to another test that "covers it" is how a file ends up
        # checked by nothing.
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"{path.name} is not a JSON object")
        for bench in ("scan", "cogs"):
            for split, rec in (data.get(bench) or {}).items():
                if isinstance(rec, dict) and "n_total" in rec:
                    out.append((path.name, bench, split, rec))
    return out


BLOCKS = _benchmark_blocks()


def test_there_are_benchmark_records_to_check():
    """Guards the rest: with none, every assertion below passes vacuously."""
    assert BLOCKS, (
        "no SCAN/COGS blocks found under runs/ — either the evidence was "
        "deleted (it should not be) or this test's reader is broken")


@pytest.mark.parametrize(
    "name,bench,split,rec", BLOCKS,
    ids=[f"{n}:{b}.{s}" for n, b, s, _ in BLOCKS])
def test_every_benchmark_record_declares_its_size(name, bench, split, rec):
    """A record with no n_total cannot be judged subset or complete."""
    assert isinstance(rec["n_total"], int) and rec["n_total"] >= 0, \
        f"{name} {bench}.{split} has a non-integer n_total"


def test_the_committed_records_are_known_subsets():
    """Pins the fact the withdrawal rests on.

    If a full run ever lands, this fails and the withdrawal notice in
    ABLATION_RESULTS.md should be revisited — which is the point.
    """
    subsets = [(n, b, s, r["n_total"], FULL_SPLIT_SIZE[(b, s)])
               for n, b, s, r in BLOCKS
               if (b, s) in FULL_SPLIT_SIZE
               and r["n_total"] < FULL_SPLIT_SIZE[(b, s)]]
    assert subsets, (
        "no subset records remain — if the benchmarks were run in full, "
        "remove the withdrawal notice and report the real numbers")


def test_no_document_quotes_a_subset_accuracy():
    """The withdrawal itself: no `k/n` where n is a subset size.

    Matches the shape those figures were written in (`0/30`, `0/20`). Prose
    that *describes* the subset — "a 30/20-example subset" — is not a score
    and does not match.
    """
    sizes = {r["n_total"] for _, b, s, r in BLOCKS
             if (b, s) in FULL_SPLIT_SIZE
             and r["n_total"] < FULL_SPLIT_SIZE[(b, s)]}
    assert sizes, "no subset sizes to search for; the guard would be vacuous"

    pattern = re.compile(r"\b(\d+)\s*/\s*(" + "|".join(str(n) for n in sorted(sizes)) + r")\b")
    offenders: list[str] = []
    for doc in DOCS:
        for i, line in enumerate((REPO / doc).read_text(encoding="utf-8").splitlines(), 1):
            for m in pattern.finditer(line):
                correct, total = int(m.group(1)), int(m.group(2))
                # An exact-match score cannot exceed its denominator. "30/20"
                # describing the subset sizes is prose, not a result. This is
                # a property of what a score IS, not an exception carved out
                # to let particular wording through.
                if correct > total:
                    continue
                offenders.append(f"{doc}:{i}: {m.group(0)}  in  {line.strip()[:70]}")
    assert not offenders, (
        "a subset accuracy is quoted as a result:\n" + "\n".join(offenders)
        + "\n\nThese figures were withdrawn. Report the full-split numbers "
          "from scripts/eval_benchmarks.py, or claim nothing.")


# A sentence boundary: end punctuation, a blank line, or the start of a
# heading, list item, or table row. Markdown prose does not run on across
# these, so neither does a claim.
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n\s*\n|\n(?=\s*(?:#{1,6}\s|\||[-*]\s))")

_NAMES_A_BENCHMARK = re.compile(r"\b(SCAN|COGS)\b")

# A word that introduces a result: "scored", "achieved", "accuracy of".
_SCORING_WORD = re.compile(
    r"\b(scor(?:e|es|ed)|achiev(?:e|es|ed)|reach(?:es|ed)|got"
    r"|exact[-\s]match(?:es)?\s+of|accuracy\s+of)\b", re.IGNORECASE)

# A quantity that could be a result value.
_QUANTITY = re.compile(
    r"\bzero\b|\bnil\b|\d+\s*/\s*\d+|\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\b",
    re.IGNORECASE)

# A preposition between the scoring word and the quantity re-points the
# number at the *scope* of the measurement rather than its value: "scored
# over 30 examples" counts examples, "scored 0" states a result. This is a
# property of what the sentence says, not a phrasing allowed through — a
# sentence that names a real score has nothing between verb and number.
_SCOPE_PREPOSITION = re.compile(
    r"\b(over|across|on|from|per|among|within|of|for|against|in)\b",
    re.IGNORECASE)


def _score_claim(sentence: str) -> str | None:
    """The span asserting a score, or None. See _SCOPE_PREPOSITION."""
    for word in _SCORING_WORD.finditer(sentence):
        tail = sentence[word.end():word.end() + 40]
        quantity = _QUANTITY.search(tail)
        if quantity is None:
            continue
        between = tail[:quantity.start()]
        if _SCOPE_PREPOSITION.search(between):
            continue
        return (word.group(0) + tail[:quantity.end()]).strip()
    return None


def test_no_document_states_a_benchmark_score_in_prose():
    """The `k/n` guard above only catches the shape the table used.

    The same withdrawn result restates perfectly well in words — "scored 0
    for every arm" carries exactly the claim the table did, and slips a
    numeric-cell check entirely. SCAN and COGS are unmeasured for this
    package; there is no score to state in any form, so a sentence that
    both names one of them and asserts a score is a defect regardless of
    what number it lands on.

    There is deliberately no keyword that exempts a sentence. "Withdrawn"
    appearing nearby would make the guard bypassable by adding a word.
    """
    offenders: list[str] = []
    for doc in DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        for sentence in _SENTENCE.split(text):
            if not _NAMES_A_BENCHMARK.search(sentence):
                continue
            claim = _score_claim(sentence)
            if claim:
                flat = " ".join(sentence.split())
                offenders.append(f"{doc}: ...{flat[:120]}...  [{claim}]")
    assert not offenders, (
        "a SCAN/COGS score is stated in prose:\n" + "\n".join(offenders)
        + "\n\nThese benchmarks are unmeasured for this package. Withdrawing "
          "the table does not license restating its numbers in words.")


@pytest.mark.parametrize("sentence,is_claim", [
    ("SCAN/COGS scored 0 for every arm.", True),
    ("COGS scored zero on the gen split.", True),
    ("The SCAN run achieved 12 exact matches.", True),
    ("SCAN exact-match of 0.4% across every arm.", True),
    ("SCAN accuracy of 0 for the base model.", True),
    ("They were exact-match scores over 30 SCAN and 20 COGS examples.", False),
    ("SCAN and COGS are unmeasured for this package.", False),
    ("Running SCAN means roughly 36,000 greedy decodes.", False),
    ("The SCAN length split runs to 3,920 records.", False),
])
def test_the_prose_detector_separates_a_result_from_a_sample_size(
        sentence, is_claim):
    """The detector itself, exercised on both sides of the distinction.

    Without this the gate could pass by matching nothing at all.
    """
    assert (_score_claim(sentence) is not None) is is_claim, \
        f"detector got {sentence!r} wrong"


def test_the_documents_say_the_benchmark_is_unmeasured():
    """Withdrawing is not enough; the gap has to be stated."""
    for doc in DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        assert "unmeasured" in text.lower() or "withdrawn" in text.lower(), (
            f"{doc} neither withdraws the subset figures nor records "
            f"SCAN/COGS as unmeasured — silence reads as 'not applicable'")


def test_the_replacement_evaluator_offers_no_subset_flag():
    """The mechanism must stay gone, or the numbers come back."""
    src = (REPO / "scripts" / "eval_benchmarks.py").read_text(encoding="utf-8")
    for flag in ("--subset", "--scan-subset", "--cogs-subset", "--limit",
                 "--max-examples", "--skip"):
        assert f'"{flag}"' not in src and f"'{flag}'" not in src, \
            f"eval_benchmarks.py accepts {flag}, which is how the subset got reported"
