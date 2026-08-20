"""COGS benchmark runner — exact-match accuracy on the in-distribution and gen splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from datasets import load_dataset
from torch import nn
from transformers import PreTrainedTokenizerBase

COGS_SPLITS: tuple[str, ...] = ("test", "gen")


@dataclass
class CogsResult:
    split: str
    n_total: int
    n_correct: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0


def evaluate_cogs(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    splits: Iterable[str] = COGS_SPLITS,
    max_new_tokens: int = 256,
    device: str | torch.device = "cpu",
) -> Mapping[str, CogsResult]:
    model.eval()
    results: dict[str, CogsResult] = {}
    for split in splits:
        if split not in COGS_SPLITS:
            raise ValueError(f"unknown COGS split: {split!r}")
        # `cogs` does not exist on the HF Hub. Load the canonical Kim & Linzen
        # release (najoungkim/COGS) from local TSVs instead.
        import os
        _dir = os.environ.get("COGS_DIR", ".")
        ds = load_dataset(
            "csv",
            data_files={split: os.path.join(_dir, f"cogs_{split}.tsv")},
            split=split,
            delimiter="\t",
            column_names=["sentence", "logical_form", "split_type"],
            quoting=3,
        )
        n_total = 0
        n_correct = 0
        for example in ds:
            sentence = example["sentence"]
            target = example["logical_form"].strip()
            prompt = f"Sentence: {sentence}\nLogical form:"
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
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True).split("\n", 1)[0].strip()
            n_total += 1
            if pred == target:
                n_correct += 1
        results[split] = CogsResult(split=split, n_total=n_total, n_correct=n_correct)
    return results
