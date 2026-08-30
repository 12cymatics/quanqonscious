"""Fine-tune an open-weights LLM with LoRA + the four sutra losses.

Pipeline (run from the repo root):

    python scripts/verify_bit_exact.py     # bit-exact gate (mandatory)
    python scripts/train_lora.py --config configs/ablations/full.yaml

The trainer subclass injects L_χ + L_cons + L_curv + L_dual into the
total loss. LoRA is applied to the q/k/v/o projection matrices of the
base model. Hidden states are pulled from the last layer and projected
through ``TesseractWM`` to a 16-vertex Ψ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from vedic.training import (
    TrainingConfig,
    build_lora_config,
    load_yaml,
)
from vedic.training.trainer import VedicTrainer


REPO = Path(__file__).resolve().parents[1]


def _run_bit_exact_gate() -> None:
    cmd = [sys.executable, str(REPO / "scripts" / "verify_bit_exact.py")]
    subprocess.check_call(cmd, cwd=REPO)


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vedic LoRA fine-tune.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    # The bit-exact gate is mandatory. There is no override: a kernel that does
    # not reproduce the committed fixtures must not train.
    _run_bit_exact_gate()

    cfg: TrainingConfig = load_yaml(args.config)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=_dtype_from_str(cfg.dtype),
        device_map={"": cfg.device},
    )
    lora_cfg = build_lora_config(cfg.lora)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def _load_jsonl(path: Path):
        return load_dataset("json", data_files=str(path), split="train")

    train_ds = _load_jsonl(cfg.data.train_path)
    eval_ds = _load_jsonl(cfg.data.eval_path)

    def _tokenize(batch):
        # No truncation and no max_length. Training on a silently-cut prefix
        # of an example teaches the model a different distribution from the
        # one the corpus describes, and the held-out evaluation this feeds
        # would then be comparing against a target nobody stated.
        #
        # cfg.data.max_seq_length is still read, but as an assertion about
        # the corpus rather than a cutter applied to it: an example longer
        # than the declared budget stops the run and names itself, instead of
        # being quietly shortened.
        out = tokenizer(
            batch["text"],
            truncation=False,
            padding=False,
        )
        budget = cfg.data.max_seq_length
        over = [(i, len(ids)) for i, ids in enumerate(out["input_ids"])
                if len(ids) > budget]
        if over:
            worst = max(n for _, n in over)
            raise ValueError(
                f"{len(over)} example(s) exceed max_seq_length={budget}; "
                f"longest is {worst} tokens. Raise max_seq_length in the "
                f"config to cover the corpus, or shorten the corpus. This "
                f"used to truncate them, which trains on a prefix and "
                f"reports the result as if the whole example had been seen.")
        # NOTE: labels are produced by DataCollatorForLanguageModeling(mlm=False),
        # which sets labels = input_ids with pad positions masked to -100.
        # Pre-setting them here makes tokenizer.pad() fail on the ragged list.
        return out

    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(_tokenize, batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.data.train_batch_size,
        per_device_eval_batch_size=cfg.data.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        warmup_ratio=cfg.optim.warmup_ratio,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        bf16=(cfg.dtype == "bf16"),
        fp16=(cfg.dtype == "fp16"),
        report_to=[],
        remove_unused_columns=False,
    )

    d_model = int(model.config.hidden_size)

    trainer = VedicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        vedic_config=cfg,
        d_model=d_model,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    with (cfg.output_dir / "vedic_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2, default=str)


if __name__ == "__main__":
    main()
