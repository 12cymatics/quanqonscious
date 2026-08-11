"""peft.LoraConfig builder. Single-line wrapper kept separate so the trainer
file can stay focused on training logic and so the test suite can construct
LoRA configs without instantiating peft modules."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import LoRAHyperparams

if TYPE_CHECKING:  # pragma: no cover
    from peft import LoraConfig


def build_lora_config(cfg: LoRAHyperparams) -> "LoraConfig":
    from peft import LoraConfig, TaskType  # local import to avoid mandatory dep at import

    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
