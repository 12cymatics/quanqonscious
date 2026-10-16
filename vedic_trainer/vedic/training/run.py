"""Everything a training run is, built in one place from its config.

``scripts/train_lora.py`` trains what this builds, and
``scripts/calibrate_aux_weights.py`` measures what this builds at
initialisation. They share the function so that "the run's initial state"
means one thing. The LoRA initialisation and the Ψ projection are drawn from
the global RNG in a fixed order (``torch.manual_seed(cfg.seed)``, model
load, LoRA, then the Trainer's ``set_seed(cfg.seed)`` before the projection
is drawn), and a calibration that rebuilt that order by hand could
silently measure a different projection than the one trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import TrainingConfig
from .hf_args import build_training_arguments
from .lora_config import build_lora_config


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


@dataclass
class Run:
    cfg: TrainingConfig
    tokenizer: Any
    model: Any
    trainer: Any
    train_ds: Any
    eval_ds: Any


def load_corpus(path: Path, tokenizer: Any, max_seq_length: int) -> Any:
    """A ``{"text": ...}`` JSONL file, tokenised whole.

    No truncation and no max_length. Training on a silently-cut prefix of an
    example teaches the model a different distribution from the one the
    corpus describes, and the held-out evaluation this feeds would then be
    comparing against a target nobody stated.

    ``max_seq_length`` is read as an assertion about the corpus rather than a
    cutter applied to it: an example longer than the declared budget stops the
    run and names itself, instead of being quietly shortened.
    """
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(path), split="train")

    def _tokenize(batch):
        out = tokenizer(batch["text"], truncation=False, padding=False)
        over = [(i, len(ids)) for i, ids in enumerate(out["input_ids"])
                if len(ids) > max_seq_length]
        if over:
            worst = max(n for _, n in over)
            raise ValueError(
                f"{len(over)} example(s) exceed max_seq_length={max_seq_length}; "
                f"longest is {worst} tokens. Raise max_seq_length in the "
                f"config to cover the corpus, or shorten the corpus. This "
                f"used to truncate them, which trains on a prefix and "
                f"reports the result as if the whole example had been seen.")
        # Labels come from DataCollatorForLanguageModeling(mlm=False), which
        # sets labels = input_ids and then -100 wherever the token id equals
        # pad_token_id. That is every *pad-id token*, not only padding: for
        # SmolLM2 pad == eos, so a real EOS inside a record is masked too and
        # never trained. Records therefore end in "\n", which is trained
        # (vedic/eval/scan_data.py). Pre-setting labels here would make
        # tokenizer.pad() fail on the ragged list.
        return out

    return ds.map(_tokenize, batched=True, remove_columns=ds.column_names)


def build_run(cfg: TrainingConfig) -> Run:
    """Tokenizer, LoRA model, datasets and VedicTrainer, in the run's RNG order."""
    from peft import get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling)

    from .trainer import VedicTrainer

    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, revision=cfg.base_model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        revision=cfg.base_model_revision,
        dtype=_dtype_from_str(cfg.dtype),
        device_map={"": cfg.device},
    )
    model = get_peft_model(model, build_lora_config(cfg.lora))

    train_ds = load_corpus(cfg.data.train_path, tokenizer, cfg.data.max_seq_length)
    eval_ds = load_corpus(cfg.data.eval_path, tokenizer, cfg.data.max_seq_length)

    trainer = VedicTrainer(
        model=model,
        args=build_training_arguments(cfg),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        vedic_config=cfg,
        d_model=int(model.config.hidden_size),
    )
    return Run(cfg=cfg, tokenizer=tokenizer, model=model, trainer=trainer,
               train_ds=train_ds, eval_ds=eval_ds)
