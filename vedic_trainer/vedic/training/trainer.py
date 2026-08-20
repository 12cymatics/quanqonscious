"""HuggingFace Trainer subclass: routes hidden states through TesseractWM and adds the four sutra losses."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
from transformers import Trainer

from vedic.kernel.hessian import HessianModule
from vedic.kernel.sutras_torch import S5, S7, S11
from vedic.kernel.wht import wht_axis_torch
from vedic.memory import TesseractWM

from .config import TrainingConfig
from .losses import CONS_TRACE_KEY, total_loss


class VedicTrainer(Trainer):
    """LoRA fine-tuning trainer that adds the four sutra-derived auxiliary losses.

    The base HuggingFace ``Trainer`` is reused unchanged for optimizer,
    scheduler, gradient accumulation, mixed precision, checkpointing, etc.
    We override exactly two things:

    - ``compute_loss`` to inject the auxiliary losses
    - ``_save`` is unchanged; the TesseractWM and HessianModule are
      registered as ``self.aux_modules`` so they are saved with the
      checkpoint via the standard PyTorch state-dict path.
    """

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
        attn_mask: Tensor | None = forward_inputs.get("attention_mask")
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
