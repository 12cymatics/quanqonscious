"""Split the synthetic corpus into train/eval with NO source-sentence overlap.

Why this exists
---------------
`reproduce_ablation.sh` said "# 1. Seed corpus -> synthetic corpus -> 90/10
split" and then ran a command that writes one file. No split code existed
anywhere in the package. The committed `synthetic_train.jsonl` /
`synthetic_eval.jsonl` were therefore produced by something uncommitted --
and it split at the RECORD level.

That matters because the generator emits ten records per seed sentence (one
contradiction pair plus four axis-paraphrase pairs). A record-level shuffle
puts nine of a sentence's ten records in train and the tenth in eval, so the
model is scored on a near-duplicate paraphrase of text it memorised.
Measured on the old split: **all 332 eval source sentences also appeared in
train**. "Held-out" cross-entropy was measuring paraphrase memorisation.

The split is therefore taken over SOURCES, never records. Every record
derived from a source travels with it. Sources are shuffled with an explicit
seed so the split is reproducible, and disjointness is asserted before
anything is written -- see `vedic/data/tests/test_split_is_disjoint.py`,
which checks the committed files rather than trusting this script to have
been run.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def group_by_source(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        try:
            groups[r["source"]].append(r)
        except KeyError:
            raise SystemExit(
                "a record carries no 'source' field, so it cannot be assigned "
                "to a side of the split without risking leakage")
    return dict(groups)


def split_sources(sources: list[str], eval_frac: float, seed: int
                  ) -> tuple[list[str], list[str]]:
    """Partition source sentences. Deterministic given `seed`."""
    if not 0.0 < eval_frac < 1.0:
        raise SystemExit(f"eval_frac must be in (0, 1); got {eval_frac}")
    ordered = sorted(sources)                 # sort first: dict order is not a seed
    random.Random(seed).shuffle(ordered)
    n_eval = round(len(ordered) * eval_frac)
    if n_eval == 0 or n_eval == len(ordered):
        raise SystemExit(
            f"eval_frac {eval_frac} over {len(ordered)} sources gives an empty "
            f"side; nothing can be measured on an empty split")
    return ordered[n_eval:], ordered[:n_eval]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="jsonl of all generated records")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = [json.loads(line) for line in
               args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"{args.input} holds no records")

    groups = group_by_source(records)
    train_src, eval_src = split_sources(list(groups), args.eval_frac, args.seed)

    overlap = set(train_src) & set(eval_src)
    if overlap:
        raise SystemExit(f"BUG: {len(overlap)} sources on both sides: "
                         f"{sorted(overlap)[:5]}")

    for path, srcs in ((args.train, train_src), (args.eval, eval_src)):
        rows = [r for s in srcs for r in groups[s]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                        encoding="utf-8")
        print(f"wrote {len(rows):>5} records from {len(srcs):>4} sources "
              f"-> {path}")

    print(f"\nsources: {len(train_src)} train / {len(eval_src)} eval, "
          f"0 shared (seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
