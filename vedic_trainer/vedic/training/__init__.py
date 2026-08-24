"""Training layer: Pydantic config, LoRA builder, sutra losses, HF Trainer subclass."""
from __future__ import annotations

from .config import (
    DataConfig,
    LoRAHyperparams,
    LossWeights,
    OptimHyperparams,
    TrainingConfig,
    load_yaml,
)
from .lora_config import build_lora_config
from .losses import (
    L_chi,
    L_cons,
    L_curv,
    L_dual,
    total_loss,
)

__all__ = [
    "DataConfig",
    "LoRAHyperparams",
    "LossWeights",
    "OptimHyperparams",
    "TrainingConfig",
    "load_yaml",
    "build_lora_config",
    "L_chi",
    "L_cons",
    "L_curv",
    "L_dual",
    "total_loss",
]
