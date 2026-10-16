"""HuggingFace Trainer subclass: routes hidden states through TesseractWM and adds the four sutra losses."""
from __future__ import annotations

from typing import Any, Mapping

from torch import Tensor, nn
from transformers import Trainer

from vedic.kernel.wht import wht_axis_torch

from .aux_state import AUX_STATE_FILE, build_aux_modules, save_aux_state
from .config import TrainingConfig
from .losses import total_loss

#: Name of the auxiliary ModuleDict inside the model being trained.
AUX_SUBMODULE = "vedic_aux"


class VedicTrainer(Trainer):
    """LoRA fine-tuning trainer that adds the four sutra-derived auxiliary losses.

    The base HuggingFace ``Trainer`` is reused for the optimizer, scheduler,
    gradient accumulation, clipping and checkpointing. Overridden:

    - ``compute_loss`` injects the auxiliary losses during training, and
      returns cross-entropy alone during evaluation;
    - ``log`` attaches the per-component means, aggregated per optimizer step;
    - ``create_optimizer`` refuses to proceed unless every auxiliary
      parameter is in the optimizer;
    - ``_save`` writes the auxiliary parameters next to the adapter.

    The auxiliary modules are a submodule of the model
    ------------------------------------------------------
    ``TesseractWM`` holds the learned d_model -> 16 projection. It used to live
    on the Trainer only, beside ``self.model``. The Trainer's per-step
    machinery operates on ``model``: transformers 5.19 calls
    ``model.zero_grad()`` after each optimizer step and clips
    ``model.parameters()``. So the projection's gradient was never zeroed and
    never clipped. At step k AdamW was fed the *sum* of every gradient since
    step 1 (measured: 5,475 → 10,791 → 15,346 → … → 32,880 over ten steps),
    while the LoRA gradients were clipped to norm 1. The "full" arm was not
    trained on the objective it declared.

    Registering the modules as ``model.vedic_aux`` puts them under every one of
    those mechanisms at once: zeroing, clipping, device placement and the
    optimizer's own parameter collection. Clipping is therefore joint, over the
    LoRA weights and the projection together, as it is for any other trainable
    parameter. PEFT's adapter save is unaffected: it writes ``lora_`` tensors
    only, which is why ``_save`` writes the auxiliary state separately.

    Gradient accumulation
    ---------------------
    ``loss_is_scaled_for_ga = False``. ``compute_loss`` returns a per-micro-
    batch mean and does not pass ``num_items_in_batch`` to the model, so the
    Trainer must divide by the accumulation count. Left at ``None``,
    transformers infers "already scaled" from the model accepting loss kwargs
    (LlamaForCausalLM does), skips the division, and the step loss and its
    gradient came out ``gradient_accumulation_steps`` times too large: 4x in
    every shipped config, which also made ``max_grad_norm`` 4x tighter than
    configured.
    """

    #: Re-exported from .aux_state so callers can find it on the trainer.
    AUX_STATE_FILE = AUX_STATE_FILE

    loss_is_scaled_for_ga = False

    def __init__(
        self,
        *args: Any,
        vedic_config: TrainingConfig,
        d_model: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vedic_config = vedic_config
        self.d_model = d_model

        aux = build_aux_modules(d_model)
        if hasattr(self.model, AUX_SUBMODULE):
            raise ValueError(f"the model already has an attribute {AUX_SUBMODULE!r}")
        self.model.add_module(AUX_SUBMODULE, aux)
        aux.to(self.args.device)
        self.register_buffer_module = aux
        self.tesseract_wm = aux["tesseract_wm"]
        self.hessian = aux["hessian"]
        self.s5 = aux["s5"]
        self.s7 = aux["s7"]
        self.s11 = aux["s11"]
        self.wht_axis = wht_axis_torch(device=self.args.device)

        # Per-component sums awaiting the next log line. Training parts are
        # weighted like the Trainer's own `loss` (1/accumulation per micro-
        # batch, so their weights sum to optimizer steps); eval parts by
        # batch size, as `eval_loss` is.
        self._train_parts: dict[str, float] = {}
        self._train_weight = 0.0
        self._eval_parts: dict[str, float] = {}
        self._eval_weight = 0.0

    def _save(self, output_dir: str | None = None,
              state_dict: Any = None) -> None:
        """Write the adapter, then the auxiliary modules beside it."""
        super()._save(output_dir, state_dict)
        save_aux_state(self.register_buffer_module,
                       output_dir if output_dir is not None
                       else self.args.output_dir)

    def create_optimizer(self, model: nn.Module | None = None) -> Any:
        """Build the optimizer, and refuse if any auxiliary parameter is missing.

        The projection is collected by the base class because it is a
        submodule of the model. This used to add it by hand as a separate
        parameter group -- a fix for its having been silently frozen at its
        random initialisation, since the base class builds the optimizer from
        the model's parameters and the projection was not among them. The
        check below keeps that defect from returning quietly.

        ``model`` is accepted and passed through: transformers 5.19 calls
        ``create_optimizer(model)`` on its delayed-creation path (FSDP,
        SageMaker MP), which the previous zero-argument override turned into
        a TypeError.
        """
        optimizer = super().create_optimizer(model)
        held = {id(p) for group in optimizer.param_groups for p in group["params"]}
        missing = [n for n, p in self.register_buffer_module.named_parameters()
                   if p.requires_grad and id(p) not in held]
        if missing:
            raise RuntimeError(
                f"auxiliary parameters not in the optimizer: {missing}. They "
                f"would stay at their initialisation for the whole run.")
        return optimizer

    @property
    def loss_weights(self) -> tuple[float, float, float, float]:
        w = self.vedic_config.loss_weights
        return (w.alpha_chi, w.beta_cons, w.gamma_curv, w.delta_dual)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> Tensor | tuple[Tensor, Any]:
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("VedicTrainer requires `labels` in the batch.")
        forward_inputs = dict(inputs)
        # Hidden states are needed for Ψ regardless of the model's config.
        outputs = model(**forward_inputs, output_hidden_states=True, use_cache=False)
        ce_loss = outputs.loss

        last_hidden = outputs.hidden_states[-1]            # (B, T, d_model)
        try:
            attn_mask: Tensor = forward_inputs["attention_mask"]
        except KeyError:
            raise ValueError(
                "the batch carries no attention_mask; TesseractWM cannot pool "
                "without one and the auxiliary losses would be computed over "
                "padding. Check the collator."
            ) from None
        psi = self.tesseract_wm(last_hidden, attn_mask)    # (B, 16)

        total, components = total_loss(
            ce_loss=ce_loss,
            psi=psi,
            weights=self.loss_weights,
            s5=self.s5,
            s7=self.s7,
            s11=self.s11,
            hessian=self.hessian,
            wht_axis=self.wht_axis,
        )

        values = {k: float(v.detach().item()) for k, v in components.items()}
        # Every auxiliary term is homogeneous in Ψ (degree 2, or 4 for
        # L_cons), and the projection producing Ψ is read by nothing else, so
        # the terms can be paid off by shrinking Ψ's response on the data at
        # no cost to cross-entropy. Logging Ψ's scale keeps that visible: a
        # falling L_* with a falling psi_rms is that, not structure.
        values["psi_rms"] = float(psi.detach().pow(2).mean().sqrt().item())
        if model.training:
            w = 1.0 / self.current_gradient_accumulation_steps
            for k, v in values.items():
                self._train_parts[k] = self._train_parts.get(k, 0.0) + w * v
            self._train_weight += w
            loss = total
        else:
            # Evaluation reports cross-entropy, the one quantity comparable
            # across arms. It used to report `total`, so the "full" arm's
            # eval_loss carried its weighted auxiliary terms (about +30 at
            # initialisation) and could not be set beside the other arm's.
            b = float(labels.size(0))
            for k, v in values.items():
                self._eval_parts[k] = self._eval_parts.get(k, 0.0) + b * v
            self._eval_weight += b
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Attach component means to the Trainer's own train and eval lines.

        This used to call ``self.log`` from inside ``compute_loss`` on every
        micro-batch whose ``global_step`` was a multiple of ``logging_steps``:
        four single-micro-batch entries per logged step, plus one unprefixed
        entry per evaluation batch filed under the same step as training. The
        curves in ``trainer_state.json`` were then dominated by eval batches.
        Now each training line carries the mean over exactly the micro-batches
        its ``loss`` covers, and each eval line carries ``eval_``-prefixed
        means over the eval set.
        """
        if "loss" in logs and self._train_weight > 0:
            for k, v in self._train_parts.items():
                logs[k] = v / self._train_weight
            self._train_parts = {}
            self._train_weight = 0.0
        if "eval_loss" in logs and self._eval_weight > 0:
            for k, v in self._eval_parts.items():
                logs[f"eval_{k}"] = v / self._eval_weight
            self._eval_parts = {}
            self._eval_weight = 0.0
        super().log(logs, start_time)
