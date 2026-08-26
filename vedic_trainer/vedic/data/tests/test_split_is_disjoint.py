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
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TRAIN = REPO / "data" / "synthetic_train.jsonl"
EVAL = REPO / "data" / "synthetic_eval.jsonl"

SEED_CORPUS = REPO / "data" / "seed_corpus.txt"
ALL_JSONL = REPO / "data" / "synthetic_all.jsonl"


def _build_corpus_if_absent() -> None:
    """Ensure the split exists, by running the committed pipeline.

    This file used to carry ``pytestmark = pytest.mark.skipif(not
    (TRAIN.exists() and EVAL.exists()), ...)``. ``data/*`` is gitignored, so
    in CI neither file exists and every test here skipped -- meaning the
    train/eval split, the thing a leaking split had already invalidated once,
    was checked on developer machines only and never on a clean checkout.
    A skip is not a pass, but it is not a check either, and nobody reads
    "7 skipped".

    The inputs are committed (``data/seed_corpus.txt``) and both generators
    are deterministic, so the split can simply be built. If a generator
    fails, that failure surfaces here rather than being converted into a
    skip.
    """
    if TRAIN.exists() and EVAL.exists():
        return
    if not SEED_CORPUS.is_file():
        raise FileNotFoundError(
            f"{SEED_CORPUS} is missing. It is the committed input the "
            f"synthetic corpus is generated from; without it the split "
            f"cannot be built or checked.")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for cmd in (
        [sys.executable, str(REPO / "scripts" / "generate_synthetic.py"),
         "--input", str(SEED_CORPUS), "--output", str(ALL_JSONL)],
        [sys.executable, str(REPO / "scripts" / "split_corpus.py"),
         "--input", str(ALL_JSONL), "--train", str(TRAIN), "--eval", str(EVAL)],
    ):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO),
                           env=env)
        if r.returncode != 0:
            raise RuntimeError(
                f"corpus build failed: {' '.join(cmd)}\n"
                f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    if not (TRAIN.exists() and EVAL.exists()):
        raise RuntimeError(
            "the corpus pipeline reported success but did not write "
            f"{TRAIN} and {EVAL}")


_build_corpus_if_absent()


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


TRAIN_R = _load(TRAIN)
EVAL_R = _load(EVAL)


def test_both_splits_are_non_empty():
    """Guards every assertion below: empty sets are trivially disjoint."""
    assert TRAIN_R, "train split is empty"
    assert EVAL_R, "eval split is empty"


def test_rebuilding_the_corpus_reproduces_this_exact_split() -> None:
    """The build path above must not be able to substitute different data.

    ``_build_corpus_if_absent`` runs on any clean checkout, which is the
    whole point — but it would be worthless, and worse than the skip it
    replaced, if the corpus it produced differed from the one the results in
    ``runs/`` were measured on. Both stages are rebuilt into a temp
    directory and compared against the files on disk:

    * ``generate_synthetic.py`` is compared record-by-record keyed on ``idx``,
      not by file bytes, because the committed corpus is stored shuffled;
    * ``split_corpus.py`` sorts sources before seeding its shuffle, so its
      output is independent of input order — this asserts that rather than
      assuming it.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    gen = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_synthetic.py"),
         "--input", str(SEED_CORPUS), "--output", str(tmp / "all.jsonl")],
        capture_output=True, text=True, cwd=str(REPO), env=env)
    assert gen.returncode == 0, gen.stderr
    spl = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "split_corpus.py"),
         "--input", str(tmp / "all.jsonl"),
         "--train", str(tmp / "train.jsonl"), "--eval", str(tmp / "eval.jsonl")],
        capture_output=True, text=True, cwd=str(REPO), env=env)
    assert spl.returncode == 0, spl.stderr

    rebuilt_all = {r["idx"]: json.dumps(r, sort_keys=True)
                   for r in _load(tmp / "all.jsonl")}
    on_disk_all = {r["idx"]: json.dumps(r, sort_keys=True)
                   for r in _load(ALL_JSONL)}
    assert set(rebuilt_all) == set(on_disk_all), (
        "rebuilding produced a different set of record indices than "
        f"{ALL_JSONL.name}: the generator is not deterministic")
    differing = [i for i in rebuilt_all if rebuilt_all[i] != on_disk_all[i]]
    assert not differing, (
        f"{len(differing)} records differ between the rebuilt corpus and "
        f"{ALL_JSONL.name} (first: idx {differing[0]})")

    def sources(path: Path) -> set[str]:
        return {r["source"] for r in _load(path)}

    assert sources(tmp / "eval.jsonl") == sources(EVAL), \
        "rebuilding produced a different eval split"
    assert sources(tmp / "train.jsonl") == sources(TRAIN), \
        "rebuilding produced a different train split"


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
        f"records. All shared sources: {sorted(shared)}")


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
