# Architecture

## Layer map

```
                        ┌──────────────────┐
                        │  user prompts    │
                        │  + base LLM      │
                        └────────┬─────────┘
                                 │ logits, hidden states
                  ┌──────────────┴──────────────┐
                  │                             │
            ┌─────▼──────┐               ┌──────▼─────┐
            │  L_CE      │               │ TesseractWM│
            │  (HF Trnr) │               │ d_model→16 │
            └─────┬──────┘               └──────┬─────┘
                  │                             │ Ψ ∈ ℝ^16
                  │              ┌──────────────┼──────────────┐
                  │              ▼              ▼              ▼
                  │       ┌──────────┐   ┌────────────┐  ┌─────────┐
                  │       │ S5 / S7  │   │ HessianMod │  │ S11/S5  │
                  │       │ S11 etc. │   │ g_ab       │  │ + WHT   │
                  │       └────┬─────┘   └─────┬──────┘  └────┬────┘
                  │            │               │              │
                  │            ▼               ▼              ▼
                  │        L_χ, L_cons     L_curv          L_dual
                  │            │               │              │
                  │            └───────────────┴──────────────┘
                  │                            │
                  │                            ▼
                  └─────────────────►  total = L_CE + α·χ + β·cons + γ·curv + δ·dual
```

## Component boundaries

- **kernel/** is pure mathematics. No torch in the ℚ path; the torch port
  is a separate file using the same integer indices and rational
  constants. No fail-safes, no epsilons, no clamps. Tests are 100 %
  Fraction-vs-Fraction comparisons.
- **memory/** is the only PyTorch trainable layer outside LoRA: a single
  `nn.Linear(d_model, 16)` with orthogonal init.
- **training/** wires HuggingFace Trainer + peft LoRA + the four sutra
  losses. Loss aggregation is the only place where small numerical
  stabilisers appear, and each is documented in `losses.py`.
- **data/** is deterministic: same input → same Fraction output. No LLM
  is invoked at generation time; the encoder is closed-form linguistic
  feature extraction over a Cartesian product of four binary axes.
- **eval/** consumes a HuggingFace model + tokenizer and produces
  per-split exact-match accuracy plus an audit-closure rate.

## Data flow during a training step

1. Batch arrives with `input_ids`, `attention_mask`, `labels`.
2. The HuggingFace model runs forward with `output_hidden_states=True`.
3. The last hidden state goes through `TesseractWM` → `Ψ ∈ ℝ^{B×16}`.
4. The Trainer's running integer counter `_trace_sum` is incremented by
   `B` and passed as `trace_sum` (a length-B long tensor).
5. The four auxiliary losses are computed; CE + α·χ + β·cons + γ·curv +
   δ·dual is the optimiser objective.
6. Each loss is logged separately; the running trace counter is logged
   under `vedic_trace_sum`.

## Dependency hierarchy

```
kernel.q   →  kernel.tesseract  →  kernel.{wht, hessian, sutras_*, conservation_*}
                                                 ↓
                                       kernel.interaction_matrix
                                                 ↓
                                              memory
                                                 ↓
                                              training
data depends on kernel only.
eval depends on data + training-loaded model.
scripts depend on everything.
```
