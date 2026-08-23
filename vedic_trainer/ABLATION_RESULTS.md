# Sutra auxiliary-loss ablation — results

First end-to-end execution of the `vedic_trainer` LoRA pipeline. The
question: do the four sutra-derived auxiliary losses improve
compositional generalization?

**Answer: no. Two of the four cannot affect training at all, and the two
that can make held-out language modelling consistently worse.**

Reproduce with `scripts/reproduce_ablation.sh`.

## Setup

| | |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (`meta-llama/Llama-3.2-1B-Instruct` is gated) |
| Adapter | LoRA r=16, α=32, dropout 0.05, on q/k/v/o — 1,843,200 trainable params (1.35%) |
| Data | 5,120 synthetic records from 512 seed sentences; 4,608 train / 512 eval |
| Schedule | 1 epoch, 288 optimizer steps, lr 2e-4 cosine, batch 4 × grad-accum 4 |
| Hardware | 4 CPU cores, fp32 |
| Arms | `full` (α=0.10, β=0.05, γ=0.02, δ=0.05) vs `no_sutra` (all four = 0) |

The two arm configs differ **only** in the four loss weights and
`output_dir`. Seeds 42, 1, 2.

## Are the four losses differentiable w.r.t. Ψ?

`scripts/probe_aux_gradients.py`:

| loss | value | weighted | ‖∇Ψ‖₁ | reaches Ψ |
|---|---|---|---|---|
| `L_chi` | 5.09e-01 | 5.09e-02 | 7.23e-01 | yes |
| `L_cons` | 1.75e+01 | 8.75e-01 | **0.00** | **no** |
| `L_curv` | 1.34e-05 | 2.67e-07 | **no grad_fn** | **no** |
| `L_dual` | 1.22e+01 | 6.12e-01 | 2.10e+01 | yes |

- **`L_curv` is identically zero.** It power-iterates from
  `torch.randn_like(psi)` — a random vector, not Ψ — against `g_ab`.
  `vedic/kernel/hessian.py`'s own docstring states *"because every contributing
  operator is linear, g_ab is independent of Ψ"*, confirmed here:
  `g_ab` is bit-identical for different Ψ. Every batch row therefore
  shares one matrix, so `relu(kappa - kappa.detach().mean())` is 0.
- **`L_cons` reduces to `trace_sum²`**, a function of the step counter
  alone. From the training log: 16, 64, 144, 256 at `trace_sum` =
  4, 8, 12, 16. It contributes zero gradient but grows quadratically,
  so it corrupts every reported loss (below).

The advertised four losses are effectively two; the ablation tests
`L_CE + 0.10·L_chi + 0.05·L_dual`.

## Reported loss is not the loss being optimised

`L_cons` inflates the logged numbers by ~2000×:

| arm | reported `train_loss` | HF `eval_loss` | true CE |
|---|---|---|---|
| `no_sutra` | 9.85 | 1.69 | 1.69 |
| `full` | **12,293.7** | **3,269.8** | **1.71** |

At `trace_sum = 336`, `L_cons` = 112,896 against a true CE of 4.9 — over
99% of the logged loss is a zero-gradient constant. Loss curves and any
loss-based checkpoint selection or early stopping are unusable on the
`full` arm. This is why the tables below use a separately computed
pure-CE held-out loss.

## Held-out CE (the discriminating measure)

| seed | `no_sutra` | `full` | Δ | rel |
|---|---|---|---|---|
| 42 | 1.6477 | 1.7109 | +0.0632 | +3.84% |
| 1 | 1.6556 | 1.7084 | +0.0528 | +3.19% |
| 2 | 1.6631 | 1.7182 | +0.0550 | +3.31% |
| **mean** | **1.6555** | **1.7125** | **+0.0570** | **+3.44%** |
| sd | 0.0077 | 0.0051 | 0.0055 | |

- The effect is **7.4× the baseline seed spread** (sd 0.0077).
- The two arms' ranges are **disjoint**: worst `no_sutra` (1.6631) is
  still better than best `full` (1.7084).
- Direction is unanimous across all three seeds.

The adapters are genuinely different — all 240 tensors differ, 19.29%
relative L1 — so `L_chi` and `L_dual` do move the model. They move it
the wrong way.

## The pipeline itself works

| arm | held-out CE | PPL |
|---|---|---|
| base (untuned) | 6.2825 | 535.11 |
| `no_sutra` | 1.6477 | 5.195 |
| `full` | 1.7109 | 5.534 |

Fine-tuning cuts held-out CE by **73.8%** (perplexity 535 → 5.2). The
harness is sound; that is what makes the negative result above
trustworthy rather than an artifact of a broken setup.

## SCAN / COGS cannot test this hypothesis

Exact-match, greedy decoding, 30 SCAN and 20 COGS examples per split:

| arm | SCAN simple | SCAN length | SCAN addprim_jump | COGS test | COGS gen |
|---|---|---|---|---|---|
| base (untuned) | 0/30 | 0/30 | 0/30 | 0/20 | 0/20 |
| `no_sutra` | 0/30 | 0/30 | 0/30 | 0/20 | 0/20 |
| `full` | 0/30 | 0/30 | 0/30 | 0/20 | 0/20 |

Zero everywhere, including the untuned base. The training corpus is
English declaratives with polarity flips and axis paraphrases; SCAN is
`command → action-sequence` and COGS is `sentence → logical form`. The
distributions are disjoint, so no amount of training on this corpus
moves either benchmark, and neither can detect an effect in either
direction. **The benchmark named in the README as the validation target
cannot validate the claim.**

## Notes

- `scripts/verify_bit_exact.py` passes (32 inputs, 32 sutra records, 96
  conservation records). The exact-ℚ kernel and its torch port do agree;
  that is a separate, genuine property and is not in question here.
- 12 of 5,120 generated records come back `audit_closed=True`, worth
  reconciling with the README's "audit-closed by construction".
- `TesseractWM` is instantiated outside `self.model`, so its parameters
  never enter the optimizer — the Ψ projection stays frozen at init.
  Gradients still flow through it to the LoRA weights.
- Single base model and one dataset scale. The negative result is solid
  for this setup; it does not rule out an effect at larger scale or on
  an in-distribution compositional task.

---

## Re-run on the reworked kernel (operands + composition algebra)

After parameterising all 29 sutras (every operand explicit) and adding the
SERIES / PARALLEL / CONCURRENT composition algebra, the whole ablation was
re-run from scratch: fresh adapters, all three seeds, both arms.

| seed | `no_sutra` | `full` | Δ | rel | reproduces prior run |
|---|---|---|---|---|---|
| 42 | 1.647661 | 1.710895 | +0.0632 | +3.84% | yes |
| 1 | 1.655605 | 1.708396 | +0.0528 | +3.19% | yes |
| 2 | 1.663117 | 1.718151 | +0.0550 | +3.31% | yes |
| **mean** | **1.655461** | **1.712481** | **+0.0570** | **+3.44%** | |
| sd | 0.007729 | 0.005067 | | | |

All six runs reproduce the pre-rework held-out CE **bit-for-bit**. The
operand refactor is behaviour-neutral end to end — not merely at the fixture
gate but through a full training run — and the conclusion is unchanged:

- effect is **7.4×** the baseline seed sd
- ranges **disjoint** (worst `no_sutra` 1.6631 < best `full` 1.7084)
- direction unanimous across seeds

`L_cons` and `L_curv` remain dead after the rework, as expected: their defect
is structural, not a matter of which operands the sutras take. `L_cons`
depends only on the step counter; `L_curv` power-iterates a Ψ-independent
matrix from a random vector.

### Composition modes on the full 29-sutra queue

`python scripts/run_composition.py`

| mode | result | max denominator digits |
|---|---|---|
| SERIES | **zero map** | 1 |
| PARALLEL | 16/16 nonzero | 7 |
| CONCURRENT | 16/16 nonzero | 220 |
| CANONICAL | 16/16 nonzero | 7 |
| COMPOSITE | raises (S17 precondition) | — |

SERIES over the whole queue annihilates every input: S20 projects onto one
Walsh row (image `c·h₀`), S21 takes absolute values (constant vector `|c|`),
S22 takes differences over (v, v̄) pairs (exactly 0 on a constant). Any queue
containing that ordered run is the zero map. CONCURRENT's 220-digit
denominators show the real cost of exact-ℚ composition.

---

## After fixing the structural defects

The runs above were not a fair test of the hypothesis: two of the four
auxiliary losses had zero gradient and the Tesseract projection was frozen at
random initialisation. With all four defects fixed (`44ce458`), the ablation
was re-run from scratch.

| seed | `no_sutra` | `full` | Δ | rel | prior rel (2 dead) |
|---|---|---|---|---|---|
| 42 | 1.6501 | 2.1220 | +0.4719 | **+28.60%** | +3.84% |
| 1 | 1.6566 | 2.1310 | +0.4744 | **+28.64%** | +3.19% |
| 2 | 1.6630 | 1.9906 | +0.3276 | **+19.70%** | +3.31% |
| **mean** | **1.6566** | **2.0812** | **+0.4246** | **+25.63%** | +3.44% |
| sd | 0.0064 | 0.0786 | | | |

- effect is **66×** the baseline seed sd
- ranges **disjoint** (worst `no_sutra` 1.6630 < best `full` 1.9906)
- direction unanimous
- the penalty is **7.4× larger** than when half the objective was inert

Making the losses work made the result worse, not better. The earlier +3.44%
understated the damage precisely because `L_cons` and `L_curv` contributed
nothing.

### Caveats

- **Baseline shifted** 1.6477 → 1.6501. Expected: `TesseractWM`'s parameters
  now enter the optimizer, so AdamW weight-decays them even in the `no_sutra`
  arm where no auxiliary gradient flows.
- **`full` variance grew** (sd 0.0051 → 0.0786). Seed 2 lands at +19.7% while
  42 and 1 sit at +28.6%, so the live objective is less stable across seeds
  than the inert one was.
- **Loss scale not retuned.** `L_cons` now contributes ≈1.99 weighted against
  a CE of ≈1.7 at the configured β = 0.05, i.e. comparable to the main
  objective. The weights were chosen for the old (inert) scales and were run
  unchanged rather than retuned to flatter the result. A fair follow-up would
  rescale the auxiliary weights so each sits an order of magnitude below CE,
  which is the usual convention. **This run does not settle whether a
  well-scaled version of these losses would still hurt.**

