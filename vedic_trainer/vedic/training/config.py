"""Pydantic v2 configuration models for vedic_trainer.

Every training run loads a single YAML and instantiates ``TrainingConfig``.
No defaults are silently substituted: a missing field raises a validation
error so the run cannot start with a partial config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class LoRAHyperparams(BaseModel):
    r: int = Field(..., ge=1)
    lora_alpha: int = Field(..., ge=1)
    target_modules: list[str] = Field(..., min_length=1)
    lora_dropout: float = Field(..., ge=0.0, le=1.0)


class LossWeights(BaseModel):
    """Coefficients for the four sutra-derived auxiliary losses.

    Setting any weight to 0.0 effectively disables that loss; the trainer
    still computes it for logging.
    """
    alpha_chi: float = Field(..., ge=0.0)
    beta_cons: float = Field(..., ge=0.0)
    gamma_curv: float = Field(..., ge=0.0)
    delta_dual: float = Field(..., ge=0.0)


class OptimHyperparams(BaseModel):
    learning_rate: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)
    # A fraction of the optimizer steps, strictly below 1. transformers 5
    # removed `warmup_ratio` and reads a float `warmup_steps` < 1 as this same
    # ratio -- but reads 1.0 as *one step*, so the old `le=1.0` bound admitted
    # a value that would have meant something else entirely. See
    # vedic/training/hf_args.py.
    warmup_ratio: float = Field(..., ge=0.0, lt=1.0)
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant"
    ] = "cosine"
    # Gradient clipping, over every trainable parameter including the Ψ
    # projection. It was transformers' default 1.0, unstated, and it binds at
    # every step of every shipped config, so it shapes the update as much as
    # the learning rate does.
    max_grad_norm: float = Field(..., gt=0.0)


class DataConfig(BaseModel):
    train_path: Path
    eval_path: Path
    max_seq_length: int = Field(..., ge=8)
    train_batch_size: int = Field(..., ge=1)
    eval_batch_size: int = Field(..., ge=1)


class TrainingConfig(BaseModel):
    """Top-level config consumed by ``scripts/train_lora.py``."""

    seed: int
    output_dir: Path
    base_model: str
    # The Hub commit of base_model, in full. A model id alone names whatever
    # the repository holds on the day the run starts, so two runs of one
    # config could train different weights and nothing would record it.
    base_model_revision: str = Field(..., pattern=r"^[0-9a-f]{40}$")
    device: Literal["cuda", "mps", "cpu"]
    dtype: Literal["bf16", "fp16", "fp32"]
    # torch's intra-op thread count. CPU float reductions are summed in an
    # order that depends on it: the same seed and config at 4 threads and at 1
    # gave different weights in all 240 adapter tensors (max |diff| 9.5e-7),
    # while two runs at 4 threads were bit-identical. So it is part of what a
    # run is, and is set and recorded rather than left to the machine.
    num_threads: int = Field(..., ge=1)
    num_train_epochs: float = Field(..., gt=0.0)
    gradient_accumulation_steps: int = Field(..., ge=1)
    logging_steps: int = Field(..., ge=1)
    eval_steps: int = Field(..., ge=1)
    save_steps: int = Field(..., ge=1)
    lora: LoRAHyperparams
    loss_weights: LossWeights
    optim: OptimHyperparams
    data: DataConfig

    @model_validator(mode="after")
    def _check_save_eval_align(self) -> "TrainingConfig":
        if self.save_steps % self.eval_steps != 0:
            raise ValueError(
                f"save_steps ({self.save_steps}) must be a multiple of "
                f"eval_steps ({self.eval_steps})"
            )
        return self


def load_yaml(path: Path) -> TrainingConfig:
    """Load a YAML file into a TrainingConfig (raises on schema violation)."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw)}")
    return TrainingConfig.model_validate(raw)
