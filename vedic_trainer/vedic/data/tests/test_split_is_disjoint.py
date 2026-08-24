"""The train/eval split must not share source sentences.

Why this exists
---------------
The generator emits ten records per seed sentence: one contradiction pair
plus four axis-paraphrase pairs. The committed split had been made by
shuffling RECORDS, so nine of a sentence's ten records landed in train and
the tenth in eval. Measured on that split: **all 332 eval source sentences
also appeared in train** -- a 100% source-level leak.

Held-out cross-entropy is the measure the whole ablation turns on. Under
that split it was scoring the model on near-duplicate paraphrases of text it
had memorised, which is not what "held-out" means and not what the
conclusions said it meant.

Nothing caught it because no split code existed in the package at all --
`reproduce_ablation.sh` carried a `# 90/10 split` comment above a command
that writes one file. `scripts/split_corpus.py` is that missing step; this
checks the FILES the training configs actually read, so it fails whether the
split was made by that script, by hand, or by something uncommitted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "data" / "synthetic_train.jsonl"
EVAL = REPO / "data" / "synthetic_eval.jsonl"

pytestmark = pytest.mark.skipif(
    not (TRAIN.exists() and EVAL.exists()),
    reason="corpus not generated; run scripts/generate_synthetic.py then "
           "scripts/split_corpus.py")


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


TRAIN_R = _load(TRAIN) if TRAIN.exists() else []
EVAL_R = _load(EVAL) if EVAL.exists() else []


def test_both_splits_are_non_empty():
    """Guards every assertion below: empty sets are trivially disjoint."""
    assert TRAIN_R, "train split is empty"
    assert EVAL_R, "eval split is empty"


def test_every_record_declares_its_source():
    """Without a source field, leakage cannot even be measured."""
    for name, rows in (("train", TRAIN_R), ("eval", EVAL_R)):
        missing = [r.get("idx") for r in rows if "source" not in r]
        assert not missing, f"{name}: {len(missing)} records carry no source"


def test_no_source_sentence_appears_in_both_splits():
    """The check that the old split failed 332 times out of 332."""
    shared = {r["source"] for r in TRAIN_R} & {r["source"] for r in EVAL_R}
    assert not shared, (
        f"{len(shared)} source sentences appear in BOTH splits, so held-out "
        f"loss is scoring paraphrases of memorised text. Re-split with "
        f"scripts/split_corpus.py, which partitions sources rather than "
        f"records. Examples: {sorted(shared)[:3]}")


def test_no_generated_text_appears_in_both_splits():
    """Weaker than the source check, and implied by it — but a split built
    some other way could pass one and fail the other."""
    shared = {r["text"] for r in TRAIN_R} & {r["text"] for r in EVAL_R}
    assert not shared, f"{len(shared)} generated texts appear in both splits"


def test_the_splits_partition_the_corpus_without_loss():
    """Records must be split, not sampled: nothing duplicated, nothing dropped."""
    idx_tr = [r["idx"] for r in TRAIN_R]
    idx_ev = [r["idx"] for r in EVAL_R]
    assert len(set(idx_tr)) == len(idx_tr), "duplicate idx within train"
    assert len(set(idx_ev)) == len(idx_ev), "duplicate idx within eval"
    assert not set(idx_tr) & set(idx_ev), "an idx appears in both splits"


def test_every_source_keeps_all_of_its_records_together():
    """A source split across sides is the leak in its partial form."""
    from collections import defaultdict
    side = defaultdict(set)
    for r in TRAIN_R:
        side[r["source"]].add("train")
    for r in EVAL_R:
        side[r["source"]].add("eval")
    straddling = [s for s, sides in side.items() if len(sides) > 1]
    assert not straddling, f"{len(straddling)} sources straddle the split"


def test_the_eval_split_is_large_enough_to_measure_anything():
    """A handful of sources cannot separate two arms."""
    n_src = len({r["source"] for r in EVAL_R})
    assert n_src >= 20, (
        f"eval holds only {n_src} source sentences; a difference measured on "
        f"that few is not a difference")
