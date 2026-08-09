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
    warmup_ratio: float = Field(..., ge=0.0, le=1.0)
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant"
    ] = "cosine"


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
    device: Literal["cuda", "mps", "cpu"]
    dtype: Literal["bf16", "fp16", "fp32"]
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
