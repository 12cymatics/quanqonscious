"""The TrainingArguments every shipped config resolves to.

Why this exists
---------------
``train_lora.py`` passed ``warmup_ratio=`` to ``TrainingArguments``.
transformers 5 removed that keyword, so every run of every arm died with a
TypeError after the bit-exact gate and the model load, before one optimizer
step. Nothing built those arguments in CI, so nothing noticed.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from vedic.training.config import TrainingConfig, load_yaml
from vedic.training.hf_args import build_training_arguments

REPO = Path(__file__).resolve().parents[3]
CONFIGS = sorted((REPO / "configs").rglob("*.yaml"))
# TrainingArguments checks the dtype against the hardware it is built on: a
# bf16 config raises "Your setup doesn't support bf16/gpu" on a machine with
# no GPU. So the arguments are built here for the fp32 CPU configs -- the ones
# this experiment runs -- and every config, whatever its device, is held to
# the schema. The MPS/CUDA configs' arguments are checked by nothing in CI,
# which has neither.
CPU_CONFIGS = [p for p in CONFIGS if load_yaml(p).device == "cpu"]


def test_there_are_configs_to_check():
    assert len(CONFIGS) >= 17
    assert len(CPU_CONFIGS) >= 9


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_config_satisfies_the_schema(path):
    load_yaml(path)


@pytest.mark.parametrize("path", CPU_CONFIGS, ids=[p.name for p in CPU_CONFIGS])
def test_every_cpu_config_builds_its_training_arguments(path):
    cfg = load_yaml(path)
    args = build_training_arguments(cfg)
    assert args.gradient_accumulation_steps == cfg.gradient_accumulation_steps
    assert args.learning_rate == cfg.optim.learning_rate
    assert args.max_grad_norm == cfg.optim.max_grad_norm
    # The warmup the config states, as transformers resolves it: ceil of the
    # ratio times the step count, for a step count a SCAN length epoch gives.
    assert args.get_warmup_steps(1062) == math.ceil(1062 * cfg.optim.warmup_ratio)


def test_a_three_percent_warmup_of_one_scan_epoch_is_32_steps():
    """16,990 records / (4 per batch x 4 accumulated) = 1,062 steps; 3% of
    that is 31.86, which transformers rounds up."""
    cfg = load_yaml(REPO / "configs" / "scan_length" / "no_sutra_seed0.yaml")
    assert cfg.optim.warmup_ratio == 0.03
    assert build_training_arguments(cfg).get_warmup_steps(1062) == 32


def _raw(**optim) -> dict:
    import yaml
    raw = yaml.safe_load((REPO / "configs" / "scan_length" / "no_sutra_seed0.yaml").read_text())
    raw["optim"].update(optim)
    return raw


def test_every_config_field_reaches_the_trainer_arguments():
    """Each field set to a value unlike transformers' default, so a field the
    builder drops cannot pass by coinciding with that default.

    The per-config test above cannot see a dropped max_grad_norm: every
    shipped config sets 1.0, which is also the default (found by regeneration
    testing). The same held for weight decay, scheduler, seed, epochs, batch
    sizes and step intervals, which nothing asserted at all.
    """
    from transformers import TrainingArguments
    raw = _raw(learning_rate=3e-4, weight_decay=0.07, warmup_ratio=0.25,
               lr_scheduler_type="polynomial", max_grad_norm=0.37)
    raw.update(seed=11, num_train_epochs=2.0, gradient_accumulation_steps=3,
               logging_steps=7, eval_steps=13, save_steps=26)
    raw["data"].update(train_batch_size=5, eval_batch_size=6)
    cfg = TrainingConfig.model_validate(raw)
    got = build_training_arguments(cfg)
    want = {"seed": 11, "num_train_epochs": 2.0,
            "per_device_train_batch_size": 5, "per_device_eval_batch_size": 6,
            "gradient_accumulation_steps": 3, "learning_rate": 3e-4,
            "weight_decay": 0.07, "max_grad_norm": 0.37, "warmup_steps": 0.25,
            "logging_steps": 7, "eval_steps": 13, "save_steps": 26,
            "use_cpu": True, "remove_unused_columns": False}
    default = TrainingArguments(output_dir=str(cfg.output_dir))
    for field, value in want.items():
        assert getattr(default, field) != value, f"{field}: test value is the default"
        assert getattr(got, field) == value, f"{field} did not reach TrainingArguments"
    assert str(got.lr_scheduler_type.value) == "polynomial"
    assert str(default.lr_scheduler_type.value) != "polynomial"
    assert str(got.eval_strategy.value) == "steps"
    assert str(got.save_strategy.value) == "steps"
    assert got.report_to == []


def test_a_full_warmup_is_refused():
    """transformers reads warmup_steps=1.0 as one step, not the whole run."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(_raw(warmup_ratio=1.0))


def test_the_base_model_revision_is_required_and_a_full_commit():
    raw = _raw()
    del raw["base_model_revision"]
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(raw)
    raw["base_model_revision"] = "main"
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(raw)


def test_the_thread_count_and_clip_norm_are_required():
    for section, key in ((None, "num_threads"), ("optim", "max_grad_norm")):
        raw = _raw()
        del (raw[section] if section else raw)[key]
        with pytest.raises(ValidationError):
            TrainingConfig.model_validate(raw)
