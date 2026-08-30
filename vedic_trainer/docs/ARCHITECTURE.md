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
                  │       │ S7 / S29 │   │ HessianMod │  │ S11/S5  │
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
  constants. No fail-safes, no epsilons, no clamps.

  Its tests are *not* all Fraction-vs-Fraction, and this line used to say
  they were. `test_conservation_torch.py` and `test_torch_buffers.py` live
  under `vedic/kernel/tests/` and compare torch float buffers against the
  exact-ℚ reference — that comparison is the point of the port. What holds
  is the stronger and checkable claim: **no comparison anywhere in the suite
  uses a tolerance.** The float side is fed dyadic inputs (every component
  k/2^m), which float64 represents exactly, so the residuals are exactly
  0.0 and the assertions are equalities.
- **memory/** is the only PyTorch trainable layer outside LoRA: a single
  `nn.Linear(d_model, 16)` with orthogonal init.
- **training/** wires HuggingFace Trainer + peft LoRA + the four sutra
  losses. It contains **no numerical stabilisers at all** — no epsilon, no
  clamp, and no division by any quantity measured from the data. This line
  used to say loss aggregation was "the only place where small numerical
  stabilisers appear"; `losses.py` opens by saying the opposite, and
  `losses.py` is right. The two denominators that once existed (the `L_chi`
  energy ratio, the `L_curv` Rayleigh quotient and its batch-mean baseline)
  were removed, which is why the file has nothing left to stabilise.
- **eval/** consumes a HuggingFace model + tokenizer and produces per-split
  exact-match accuracy. Nothing else: it used to also produce an
  audit-closure rate, a metric that could not distinguish two models and has
  been removed rather than reported.

There is no **data/** layer. One existed and generated a synthetic corpus;
it was a stand-in for real training data and is gone, along with the figures
measured on it. See `ABLATION_RESULTS.md`.

## Data flow during a training step

1. Batch arrives with `input_ids`, `attention_mask`, `labels`.
2. The HuggingFace model runs forward with `output_hidden_states=True`.
3. The last hidden state goes through `TesseractWM` → `Ψ ∈ ℝ^{B×16}`.
4. The four auxiliary losses are computed from Ψ alone; CE + α·χ + β·cons
   + γ·curv + δ·dual is the optimiser objective.
5. Each loss is logged separately.

There is no trace counter on this path. Steps 4 and 6 here used to describe
a `_trace_sum` incremented per batch and logged as `vedic_trace_sum`, passed
into `L_cons` so it could sum R1..R4. R1 is a step counter with no Ψ in it
and R2..R4 are algebraic identities, so that sum had identically zero
gradient while growing quadratically in the counter. `L_cons` no longer
takes it and the trainer no longer keeps it; `grep vedic_trace_sum` matches
nothing outside the checkpoints of the withdrawn runs.

## Dependency hierarchy

```
kernel.q   →  kernel.tesseract  →  kernel.{wht, hessian, sutras_*, conservation_*}
                                                 ↓
                              kernel.{interaction_matrix, audit_filter}
                                                 ↓
                                              memory
                                                 ↓
                                              training
eval depends on a training-loaded model only.
scripts depend on everything.
```
