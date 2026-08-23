"""SCAN benchmark runner — exact-match accuracy on simple/length/jump splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from datasets import load_dataset
from torch import nn
from transformers import PreTrainedTokenizerBase

from vedic.eval.scan_splits import SCAN_SPLITS  # noqa: F401


@dataclass
class ScanResult:
    split: str
    n_total: int
    n_correct: int

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


def _decode(tokenizer: PreTrainedTokenizerBase, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def evaluate_scan(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    splits: Iterable[str] = SCAN_SPLITS,
    max_new_tokens: int = 96,
    device: str | torch.device = "cpu",
) -> Mapping[str, ScanResult]:
    """Run greedy decoding on each requested SCAN split.

    The dataset id ``scan/{split}`` follows the canonical Lake & Baroni
    distribution. Every split is evaluated in full: there is no subset or
    smoke mode, because a truncated benchmark is not the benchmark.
    """
    model.eval()
    results: dict[str, ScanResult] = {}
    for split in splits:
        if split not in SCAN_SPLITS:
            raise ValueError(f"unknown SCAN split: {split!r}")
        ds = load_dataset("scan", split, split="test", trust_remote_code=True)
        n_total = 0
        n_correct = 0
        for example in ds:
            command = example["commands"]
            target = example["actions"].strip()
            prompt = f"Command: {command}\nActions:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].size(1):]
            pred = _decode(tokenizer, new_tokens).split("\n", 1)[0].strip()
            n_total += 1
            if pred == target:
                n_correct += 1
        results[split] = ScanResult(split=split, n_total=n_total, n_correct=n_correct)
    return results
