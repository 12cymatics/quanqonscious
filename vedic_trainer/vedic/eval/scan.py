"""SCAN benchmark runner — exact-match accuracy on simple/length/jump splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from datasets import load_dataset
from torch import nn
from transformers import PreTrainedTokenizerBase

SCAN_SPLITS: tuple[str, ...] = ("simple", "length", "addprim_jump")


@dataclass
class ScanResult:
    split: str
    n_total: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0


def _decode(tokenizer: PreTrainedTokenizerBase, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def evaluate_scan(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    splits: Iterable[str] = SCAN_SPLITS,
    max_new_tokens: int = 96,
    device: str | torch.device = "cpu",
    eval_subset: int | None = None,
) -> Mapping[str, ScanResult]:
    """Run greedy decoding on each requested SCAN split.

    The dataset id ``scan/{split}`` follows the canonical Lake & Baroni
    distribution. ``eval_subset`` truncates each split for quick
    smoke-runs but is None by default (full evaluation).
    """
    model.eval()
    results: dict[str, ScanResult] = {}
    for split in splits:
        if split not in SCAN_SPLITS:
            raise ValueError(f"unknown SCAN split: {split!r}")
        ds = load_dataset("scan", split, split="test", trust_remote_code=True)
        if eval_subset is not None:
            ds = ds.select(range(min(eval_subset, len(ds))))
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
