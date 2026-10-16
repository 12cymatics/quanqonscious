"""The ``TrainingArguments`` a ``TrainingConfig`` means, built in one place.

``scripts/train_lora.py`` used to construct these inline, which left them
untested: CI did not install transformers, and nothing else built them. When
transformers 5 removed the ``warmup_ratio`` keyword, every run of every arm
died with ``TypeError: ... unexpected keyword argument 'warmup_ratio'`` after
the bit-exact gate, the model load and the dataset map, before a single
optimizer step -- and the README still said the pipeline ran end to end.

Moving construction here lets ``vedic/training/tests/test_hf_args.py`` build
exactly what the script builds and pin what each field resolves to.

Warmup
------
transformers 5.19 has only ``warmup_steps``. ``TrainingArguments.
get_warmup_steps(n)`` returns ``int(warmup_steps)`` when it is >= 1 and
``ceil(n * warmup_steps)`` when it is below 1. A ratio is therefore passed
as ``warmup_steps`` directly, and ``OptimHyperparams`` bounds it strictly
below 1, because 1.0 would be read as one step.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import TrainingConfig

if TYPE_CHECKING:  # pragma: no cover
    from transformers import TrainingArguments


def build_training_arguments(cfg: TrainingConfig) -> "TrainingArguments":
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(cfg.output_dir),
        seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.data.train_batch_size,
        per_device_eval_batch_size=cfg.data.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.max_grad_norm,
        warmup_steps=cfg.optim.warmup_ratio,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        bf16=(cfg.dtype == "bf16"),
        fp16=(cfg.dtype == "fp16"),
        use_cpu=(cfg.device == "cpu"),
        report_to=[],
        remove_unused_columns=False,
    )
