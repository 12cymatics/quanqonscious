"""SCAN benchmark runner — exact-match accuracy on the simple/length/addprim_jump splits.

The test sets are read from the copy vendored in this package
(``vedic/eval/data/scan``, checked against pinned SHA-256s by
``scan_data.load_scan``), not from the HF Hub: ``scan`` there is a loading
script, and ``datasets`` >= 4 cannot run loading scripts, so the old
``load_dataset("scan", ..., trust_remote_code=True)`` could not load SCAN at
all.

How an answer is scored, and why the token budget is computed from the data
rather than passed in, is in ``vedic/eval/first_line.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from vedic.eval.first_line import Example, score_first_lines
from vedic.eval.scan_data import load_scan, prompt
from vedic.eval.scan_splits import SCAN_SPLITS


@dataclass
class ScanResult:
    split: str
    n_total: int
    n_correct: int
    #: The budget used: the longest gold continuation of the split, in tokens.
    max_new_tokens: int
    #: Examples still a correct prefix, unterminated, when the budget ran out.
    #: The only examples whose verdict the budget could have affected.
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


def scan_examples(split: str) -> list[Example]:
    """Every test example of one SCAN split, as prompts and targets."""
    if split not in SCAN_SPLITS:
        raise ValueError(f"unknown SCAN split: {split!r}")
    return [Example(prompt=prompt(c), target=a) for c, a in load_scan(split, "test")]


def evaluate_scan(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    splits: Iterable[str] = SCAN_SPLITS,
    device: str | torch.device = "cpu",
) -> Mapping[str, ScanResult]:
    """Greedy-decode every test example of each requested SCAN split.

    Every split is evaluated in full: there is no subset or smoke mode,
    because a truncated benchmark is not the benchmark.
    """
    results: dict[str, ScanResult] = {}
    for split in splits:
        s = score_first_lines(model, tokenizer, scan_examples(split), device)
        results[split] = ScanResult(split=split, n_total=s.n_total,
                                    n_correct=s.n_correct,
                                    max_new_tokens=s.max_new_tokens,
                                    n_budget_exhausted=s.n_budget_exhausted)
    return results
