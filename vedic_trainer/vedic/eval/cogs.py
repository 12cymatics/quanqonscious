"""COGS benchmark runner — exact-match accuracy on the in-distribution and gen splits.

COGS is not on the HF Hub. ``COGS_DIR`` must name the ``data/`` directory of
the Kim & Linzen release (najoungkim/COGS), which holds ``test.tsv`` and
``gen.tsv``. This loader used to open ``cogs_test.tsv`` and ``cogs_gen.tsv``,
names that release does not use, so pointing it at the release raised
FileNotFoundError.

How an answer is scored, and why the token budget is computed from the data,
is in ``vedic/eval/first_line.py``. The budget here used to be a fixed 256.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from vedic.eval.first_line import Example, score_first_lines
from vedic.eval.scan_splits import COGS_SPLITS

#: Rows per split in the release; a file with a different count is refused.
COGS_SPLIT_SIZE = {"test": 3000, "gen": 21000}


@dataclass
class CogsResult:
    split: str
    n_total: int
    n_correct: int
    max_new_tokens: int
    n_budget_exhausted: int

    @property
    def accuracy(self) -> float:
        """Exact-match accuracy. Undefined on an empty split, and says so.

        This returned 0.0 when n_total was 0, which is indistinguishable from
        a split where the model got everything wrong -- the difference between
        "the model failed" and "the data never loaded"."""
        if self.n_total == 0:
            raise ZeroDivisionError(
                f"{self.split!r} evaluated 0 examples, so it has no accuracy. "
                f"The split did not load; fix that rather than reading a rate "
                f"off an empty denominator.")
        return self.n_correct / self.n_total


def cogs_dir() -> Path:
    # Defaulting to "." silently pointed the loader at whatever directory the
    # process happened to start in, which either fails obscurely or picks up
    # an unrelated file.
    try:
        return Path(os.environ["COGS_DIR"])
    except KeyError:
        raise RuntimeError(
            "COGS_DIR is not set. It must name the data/ directory of the Kim "
            "& Linzen release (najoungkim/COGS), which holds test.tsv and "
            "gen.tsv; `cogs` is not on the HF Hub.") from None


def cogs_examples(split: str) -> list[Example]:
    """Every row of one COGS split: (sentence, logical form, generalization type)."""
    if split not in COGS_SPLITS:
        raise ValueError(f"unknown COGS split: {split!r}")
    path = cogs_dir() / f"{split}.tsv"
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    examples = []
    for i, line in enumerate(lines, 1):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{path}:{i}: {len(fields)} tab-separated fields, expected 3")
        sentence, logical_form, _split_type = fields
        examples.append(Example(prompt=f"Sentence: {sentence}\nLogical form:",
                                target=logical_form.strip()))
    if len(examples) != COGS_SPLIT_SIZE[split]:
        raise ValueError(f"{path}: {len(examples)} rows, the release has "
                         f"{COGS_SPLIT_SIZE[split]}")
    return examples


def evaluate_cogs(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    splits: Iterable[str] = COGS_SPLITS,
    device: str | torch.device = "cpu",
) -> Mapping[str, CogsResult]:
    """Greedy-decode every row of each requested COGS split, in full."""
    results: dict[str, CogsResult] = {}
    for split in splits:
        s = score_first_lines(model, tokenizer, cogs_examples(split), device)
        results[split] = CogsResult(split=split, n_total=s.n_total,
                                    n_correct=s.n_correct,
                                    max_new_tokens=s.max_new_tokens,
                                    n_budget_exhausted=s.n_budget_exhausted)
    return results
