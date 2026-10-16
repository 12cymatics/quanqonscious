"""Write one SCAN split as the JSONL corpora the trainer and evaluators read.

    python scripts/build_scan_corpus.py --split length --output-dir data/scan_length

writes, from the copy vendored in ``vedic/eval/data/scan`` (SHA-256-checked):

- ``train.jsonl``   every training example of the split, in file order
- ``test.jsonl``    every test example, for ``scripts/eval_heldout.py``
- ``monitor.jsonl`` the first ``MONITOR_SIZE`` training examples, for the
  Trainer's periodic in-training evaluation. That number is a progress
  monitor on data the model is trained on. It is not a held-out measurement
  and no result is read from it; it exists because the Trainer requires an
  eval set, and evaluating 3,920 test examples every ``eval_steps`` would
  cost hours per run for a number nothing uses.

Each line is ``{"text": scan_data.record_text(command, actions)}``: the
evaluator's prompt, the gold actions, and a ``"\\n"`` terminator. Nothing in
the repository did this conversion before, so the format the scorer depends
on was stated nowhere. Now both sides import it from ``scan_data``.

The output is deterministic, byte for byte. ``train_lora.py`` records the
SHA-256 of what it trained on, so a run can be traced to its exact corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic.eval.scan_data import load_scan, record_text  # noqa: E402
from vedic.eval.scan_splits import SCAN_SPLITS  # noqa: E402

MONITOR_SIZE = 256


def jsonl(pairs: list[tuple[str, str]]) -> str:
    return "".join(json.dumps({"text": record_text(c, a)}, ensure_ascii=True) + "\n"
                   for c, a in pairs)


def build(split: str, output_dir: Path) -> dict[str, Path]:
    train = load_scan(split, "train")
    test = load_scan(split, "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {"train": output_dir / "train.jsonl",
           "test": output_dir / "test.jsonl",
           "monitor": output_dir / "monitor.jsonl"}
    out["train"].write_text(jsonl(train), encoding="ascii")
    out["test"].write_text(jsonl(test), encoding="ascii")
    out["monitor"].write_text(jsonl(train[:MONITOR_SIZE]), encoding="ascii")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=SCAN_SPLITS)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    for name, path in build(a.split, a.output_dir).items():
        n = sum(1 for _ in path.open(encoding="ascii"))
        print(f"{name}: {n} records -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
