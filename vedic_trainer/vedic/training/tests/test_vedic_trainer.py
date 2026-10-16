"""VedicTrainer's per-step mechanics, through the real transformers Trainer.

Why this exists
---------------
Nothing imported ``VedicTrainer`` in CI, because CI did not install
transformers. Four defects sat on the path every training run takes:

1. The Ψ projection's gradient was never zeroed. It lived on the Trainer, not
   in the model, and transformers 5.19 zeroes and clips ``model`` only, so
   AdamW was fed the running sum of every gradient since step 1, unclipped.
2. The step loss and gradient were ``gradient_accumulation_steps`` times too
   large: ``loss_is_scaled_for_ga`` was left at ``None``, transformers inferred
   "already scaled" from the model accepting loss kwargs, and skipped the
   division.
3. Components were logged once per micro-batch, and once per *eval* batch
   under the training keys and step.
4. ``eval_loss`` was CE plus the weighted auxiliary terms in one arm and CE in
   the other, so it could not be compared across arms.

What the model is
-----------------
A one-layer Llama with a 32-token vocabulary and hidden size 16, built from a
config with no download. These mechanics are the Trainer's bookkeeping and do
not depend on model size, and every assertion is an exact value. This is not
a training run, and no result is read from it.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

from vedic.training import TrainingConfig, build_lora_config
from vedic.training.aux_state import load_aux_state
from vedic.training.hf_args import build_training_arguments
from vedic.training.trainer import AUX_SUBMODULE, VedicTrainer

D_MODEL = 16
PAD = 0


def _config(tmp_path: Path, weights=(0.5, 0.05, 0.05, 0.5)) -> TrainingConfig:
    return TrainingConfig.model_validate({
        "seed": 0,
        "output_dir": str(tmp_path / "out"),
        "base_model": "synthetic/one-layer-llama",
        "base_model_revision": "0" * 40,
        "device": "cpu",
        "dtype": "fp32",
        "num_threads": 4,
        "num_train_epochs": 1.0,
        "gradient_accumulation_steps": 2,
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "lora": {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj", "v_proj"],
                 "lora_dropout": 0.0},
        "loss_weights": dict(zip(("alpha_chi", "beta_cons", "gamma_curv",
                                  "delta_dual"), weights)),
        "optim": {"learning_rate": 1e-2, "weight_decay": 0.01,
                  "warmup_ratio": 0.0, "lr_scheduler_type": "constant",
                  "max_grad_norm": 1.0},
        "data": {"train_path": "unused", "eval_path": "unused",
                 "max_seq_length": 64, "train_batch_size": 2,
                 "eval_batch_size": 2},
    })


def _dataset(n: int, offset: int) -> Dataset:
    rows = []
    for i in range(n):
        length = 5 + (i + offset) % 6
        ids = [3 + (7 * i + 3 * j + offset) % 29 for j in range(length)]
        rows.append({"input_ids": ids, "attention_mask": [1] * length})
    return Dataset.from_list(rows)


def _collate(batch):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), PAD)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        mask[i, :len(b["input_ids"])] = 1
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class _Recording(VedicTrainer):
    """Keeps the loss compute_loss returns on each training micro-batch."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.micro_totals: list[torch.Tensor] = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if model.training:
            loss = out[0] if return_outputs else out
            self.micro_totals.append(loss.detach().clone())
        return out


class _Probe(TrainerCallback):
    def __init__(self, w: torch.nn.Parameter) -> None:
        self.w = w
        self.grad_at_step_begin: dict[int, torch.Tensor | None] = {}
        self.w_grad_norm_before_step: dict[int, float] = {}

    def on_step_begin(self, args, state, control, **kw):
        g = self.w.grad
        self.grad_at_step_begin[state.global_step] = None if g is None else g.clone()

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self.w_grad_norm_before_step[state.global_step] = float(self.w.grad.norm())


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trainer")
    cfg = _config(tmp)
    torch.manual_seed(0)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=D_MODEL, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, pad_token_id=PAD))
    model = get_peft_model(base, build_lora_config(cfg.lora))
    trainer = _Recording(model=model, args=build_training_arguments(cfg),
                         train_dataset=_dataset(8, 0), eval_dataset=_dataset(2, 5),
                         data_collator=_collate, vedic_config=cfg, d_model=D_MODEL)
    w = trainer.tesseract_wm.proj.weight
    w0 = w.detach().clone()
    probe = _Probe(w)
    trainer.add_callback(probe)
    trainer.train()
    return {"trainer": trainer, "probe": probe, "w0": w0, "cfg": cfg}


def test_two_optimizer_steps_of_two_micro_batches(run):
    t = run["trainer"]
    assert t.state.global_step == 2
    assert len(t.micro_totals) == 4


def test_the_projection_is_a_submodule_and_in_the_optimizer(run):
    t = run["trainer"]
    assert getattr(t.model, AUX_SUBMODULE) is t.register_buffer_module
    held = {id(p) for g in t.optimizer.param_groups for p in g["params"]}
    assert id(t.tesseract_wm.proj.weight) in held


def test_the_projection_gradient_is_zeroed_between_steps(run):
    """Defect 1. Before the fix this held the whole of step 1's gradient."""
    probe = run["probe"]
    assert 1 in probe.grad_at_step_begin
    g = probe.grad_at_step_begin[1]
    assert g is None or not g.any(), "step 2 began with step 1's gradient still in place"
    assert probe.w_grad_norm_before_step[0] > 0, "vacuous: the projection got no gradient"


def test_the_projection_gradient_is_clipped_with_the_rest(run):
    """Defect 1, its other half: clipping covers the projection."""
    probe, t = run["probe"], run["trainer"]
    max_norm = t.args.max_grad_norm
    for step, norm in probe.w_grad_norm_before_step.items():
        assert norm <= max_norm, f"step {step}: |grad W| = {norm} exceeds max_grad_norm"
    grad_norms = [e["grad_norm"] for e in t.state.log_history if "grad_norm" in e]
    assert grad_norms and max(grad_norms) > max_norm, \
        "vacuous: clipping never engaged, so this shows nothing about it"


def test_the_projection_trained(run):
    assert not torch.equal(run["w0"], run["trainer"].tesseract_wm.proj.weight.detach())


def test_the_step_loss_is_the_mean_over_micro_batches(run):
    """Defect 2. The Trainer accumulates loss/GA per micro-batch in a float32
    tensor; replay that arithmetic on the recorded micro-batch losses. Before
    the fix the logged loss was the sum, twice this value at GA = 2."""
    t = run["trainer"]
    logged = [e for e in t.state.log_history if "loss" in e]
    assert [e["step"] for e in logged] == [1, 2]
    for entry, pair in zip(logged, (t.micro_totals[0:2], t.micro_totals[2:4])):
        acc = torch.tensor(0.0)
        for loss in pair:
            acc += loss / 2
        assert entry["loss"] == acc.item()


def test_components_are_logged_once_per_step_as_step_means(run):
    """Defect 3. Each training line carries the mean over the micro-batches
    its loss covers; nothing else carries component keys."""
    t = run["trainer"]
    for e in t.state.log_history:
        has_parts = any(k.startswith("L_") or k == "psi_rms" for k in e)
        assert has_parts == ("loss" in e), f"component keys on a non-training line: {e}"
        if "loss" in e:
            assert e["psi_rms"] > 0
    logged = [e for e in t.state.log_history if "loss" in e]
    for entry, pair in zip(logged, (t.micro_totals[0:2], t.micro_totals[2:4])):
        assert entry["L_total"] == (0.5 * float(pair[0]) + 0.5 * float(pair[1])) / 1.0


def test_eval_loss_is_cross_entropy_alone(run):
    """Defect 4. The eval set is one batch of two, so eval_loss is that batch's
    CE exactly, and eval_L_total exceeds it by the weighted auxiliary terms."""
    t = run["trainer"]
    evals = [e for e in t.state.log_history if "eval_loss" in e]
    assert len(evals) == 2
    for e in evals:
        assert e["eval_loss"] == e["eval_L_CE"]
        assert e["eval_L_total"] > e["eval_L_CE"]


def test_create_optimizer_takes_the_model_argument():
    """transformers 5.19 calls create_optimizer(model) when optimizer
    creation is delayed; a zero-argument override raised TypeError there."""
    assert "model" in inspect.signature(VedicTrainer.create_optimizer).parameters


def test_checkpoints_carry_the_trained_projection(run):
    t, cfg = run["trainer"], run["cfg"]
    restored = load_aux_state(Path(cfg.output_dir) / "checkpoint-2", d_model=D_MODEL)
    assert torch.equal(restored["tesseract_wm"].proj.weight,
                       t.tesseract_wm.proj.weight.detach())
