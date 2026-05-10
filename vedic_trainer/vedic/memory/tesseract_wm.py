"""TesseractWM: project hidden states ``(B, T, d_model) → (B, 16)``.

The projection has two stages:

    1.  Mask-aware mean-pool over the sequence axis. The HuggingFace
        attention mask is required (every example has at least one
        unmasked token; we assert this rather than clamping).

    2.  An ``nn.Linear(d_model, 16, bias=False)`` whose weight is
        initialised with ``nn.init.orthogonal_`` so each of the 16 output
        slots starts as a random orthogonal direction in d_model space.
        The weight is trained jointly with the LoRA adapters.

No fail-safes: if the attention mask is all zeros for any example, the
forward asserts and the trainer is expected to produce a clean stack
trace rather than hide the bug behind a clamp.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from vedic.kernel.tesseract import NUM_VERTICES


class TesseractWM(nn.Module):
    """Projection layer feeding the 16-vertex working memory."""

    def __init__(self, d_model: int, init_orthogonal: bool = True) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive; got {d_model}")
        self.d_model = d_model
        self.proj = nn.Linear(d_model, NUM_VERTICES, bias=False)
        if init_orthogonal:
            nn.init.orthogonal_(self.proj.weight)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be (B, T, d_model); got {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.d_model:
            raise ValueError(
                f"hidden_states last dim {hidden_states.size(-1)} != d_model {self.d_model}"
            )

        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                    f"hidden_states[:, :, :] shape {tuple(hidden_states.shape[:2])}"
                )
            mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1)
            if (denom == 0).any():
                raise ValueError(
                    "attention_mask has at least one example with all-zero rows"
                )
            pooled = (hidden_states * mask).sum(dim=1) / denom
        return self.proj(pooled)
