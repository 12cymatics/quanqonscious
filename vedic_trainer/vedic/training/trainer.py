"""HuggingFace Trainer subclass: routes hidden states through TesseractWM and adds the four sutra losses."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from transformers import Trainer

from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.kernel.wht import wht_axis_torch
from vedic.memory import TesseractWM

from .config import TrainingConfig
from .aux_state import AUX_STATE_FILE, save_aux_state  # noqa: F401
from .losses import CONS_TRACE_KEY, total_loss


class VedicTrainer(Trainer):
    """LoRA fine-tuning trainer that adds the four sutra-derived auxiliary losses.

    The base HuggingFace ``Trainer`` is reused unchanged for optimizer,
    scheduler, gradient accumulation, mixed precision, checkpointing, etc.
    Three methods are overridden:

    - ``compute_loss`` to inject the auxiliary losses
    - ``create_optimizer`` to put the auxiliary parameters in a param group,
      without which TesseractWM's projection never trains
    - ``_save`` to write those auxiliary parameters next to the adapter

    The third used to read: "``_save`` is unchanged; the TesseractWM and
    HessianModule are registered as ``self.aux_modules`` so they are saved
    with the checkpoint via the standard PyTorch state-dict path." Every
    clause of that was false. There is no ``aux_modules`` attribute anywhere
    in the package, ``_save`` was not overridden, and PEFT's ``_save`` writes
    adapter tensors only -- verified: the committed checkpoints hold 240
    tensors, none of them TesseractWM or Hessian.

    The consequence was that the 9,216-parameter Ψ projection was trained and
    then discarded. Reloading a checkpoint silently produced a fresh random
    orthogonal projection, so Ψ -- and every auxiliary loss computed from it
    -- could not be reproduced from a saved run.
    """

    #: Re-exported from .aux_state so callers can find it on the trainer.
    AUX_STATE_FILE = AUX_STATE_FILE

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

        # Auxiliary modules — registered so they live on the same device as the model.
        self.tesseract_wm = TesseractWM(d_model=d_model)
        self.hessian = HessianModule()
        self.s5 = S5()
        self.s7 = S7()
        self.s11 = S11()
        self.register_buffer_module = nn.ModuleDict(
            {
                "tesseract_wm": self.tesseract_wm,
                "hessian": self.hessian,
                "s5": self.s5,
                "s7": self.s7,
                "s11": self.s11,
            }
        )
        self.register_buffer_module.to(self.args.device)
        self.wht_axis = wht_axis_torch(device=self.args.device)

        # Running integer counter for R1 (CONS_TRACE_KEY). Diagnostic only —
        # it is reported in the log, never added to the loss (see L_cons).
        self._trace_sum = 0

    def _save(self, output_dir: str | None = None,
              state_dict: Any = None) -> None:
        """Write the adapter, then the auxiliary modules beside it."""
        super()._save(output_dir, state_dict)
        save_aux_state(self.register_buffer_module,
                       output_dir if output_dir is not None
                       else self.args.output_dir)

    def create_optimizer(self):
        """Include the auxiliary modules' parameters in the optimizer.

        ``TesseractWM`` holds the learned d_model -> 16 projection that maps
        hidden states onto the tesseract, and it lives on the Trainer rather
        than inside ``self.model``. HuggingFace builds the optimizer from
        ``self.model.parameters()``, so those 9,216 weights were silently
        frozen at their random initialisation for the whole run: gradients
        flowed through the projection to the LoRA weights, but the projection
        itself never moved. Adding the parameter group here trains it.
        """
        optimizer = super().create_optimizer()
        aux = [p for p in self.register_buffer_module.parameters() if p.requires_grad]
        if aux:
            seen = {id(p) for group in optimizer.param_groups for p in group["params"]}
            fresh = [p for p in aux if id(p) not in seen]
            if fresh:
                optimizer.add_param_group({
                    "params": fresh,
                    "weight_decay": self.args.weight_decay,
                })
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
        # Force hidden-state output regardless of model config.
        labels = inputs.get("labels")
        forward_inputs = dict(inputs)
        outputs = model(**forward_inputs, output_hidden_states=True, use_cache=False)

        if labels is None:
            raise ValueError("VedicTrainer requires `labels` in the batch.")
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

        # Maintain the integer trace counter (R1).
        batch_size = psi.size(0)
        self._trace_sum = (self._trace_sum + batch_size) % (29 * 30 // 2 * 1000)
        trace_sum = torch.tensor(
            [self._trace_sum] * batch_size,
            dtype=torch.long,
            device=psi.device,
        )

        total, components = total_loss(
            ce_loss=ce_loss,
            psi=psi,
            trace_sum=trace_sum,
            weights=self.loss_weights,
            s5=self.s5,
            s7=self.s7,
            s11=self.s11,
            hessian=self.hessian,
            wht_axis=self.wht_axis,
        )

        if self.state.global_step % self.args.logging_steps == 0:
            log_payload = {f"{k}": float(v.item()) for k, v in components.items()}
            log_payload[CONS_TRACE_KEY] = self._trace_sum
            self.log(log_payload)

        return (total, outputs) if return_outputs else total
