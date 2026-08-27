# vedic_trainer

A 29-sutra LoRA fine-tuning kernel for small open-weights LLMs. The
package exposes:

- An exact-ℚ reference implementation of the 29 Vedic sutras (the
  "structuring algebra") and four conservation residuals over **Z₂⁴**.
  Every sutra takes its own operands explicitly (mask, base, reference
  index, modulus, rotation, axis set, blend weight) with named canonical
  defaults. The 29 α-weighted sutra operators live in
  `vedic/kernel/sutras_canonical.py`, which is the single authority for
  their definitions; `vedic/kernel/z2_primitives.py` holds the unweighted
  Z₂⁴ primitives the residuals and generators are built from, and is *not*
  the 29. `vedic/kernel/sutras_canonical.py` is a port of the STRICT_SUTRA_KERNEL,
  SUTRA_KIND table and §12Z coefficients from the user's
  `vedic_v18.24_full_kernel.html`; that file is the upstream definition
  and lives on the user's machine, not in this repository.
- A composition algebra (`vedic/kernel/composition.py`) that runs any sutra
  queue in **SERIES**, **PARALLEL**, **CONCURRENT** (BSP wavefront,
  W = ⌈√N⌉), **CANONICAL** or **COMPOSITE**, all in exact ℚ with a
  deterministic, queue-seeded scheduler.
- A bit-exact torch port (autograd-enabled) used by the training path.
- A `TesseractWM` working-memory projection from hidden states to a
  16-vertex Boolean cube.
- Four sutra-derived auxiliary losses applied during LoRA fine-tuning:
  - `L_χ`    — contradiction (S7 antisymmetric energy, ‖A(Ψ)‖²)
  - `L_cons` — conservation (drift of mass, ‖S‖² and ‖A‖² under S29)
  - `L_curv` — curvature (the quadratic form ⟨Ψ, g_ab Ψ⟩ at Ψ)
  - `L_dual` — dual-basis coherence ((S5∘S11) Ψ vs WHT axes)

  None of the four divides by a quantity measured from the data. `L_χ` was
  the antisymmetric *share* of Ψ's energy and `L_curv` a Rayleigh quotient
  hinged against the batch mean; both normalisations are gone, which makes
  both terms scale-dependent — see the module docstring in
  `vedic/training/losses.py`, which states that trade-off rather than
  burying it. The two divisions by 16 left in `L_dual` are
  orthogonal-projection coefficients onto basis vectors of norm squared 16,
  not normalisations.
- SCAN / COGS evaluators.

**No training data ships with this package, and none is generated.** A
synthetic corpus used to: 5,120 records expanded by template from 512 seed
sentences, with Ψ produced by a hand-written text encoder. It was a stand-in
for real data and it has been removed, along with the figures measured on it
— see `ABLATION_RESULTS.md`. Point `data.train_path` and `data.eval_path` at
your own JSONL, one object per line with a `text` field:

```json
{"text": "the sentence to train on"}
```

`scripts/train_lora.py` reads only `text`. Nothing truncates: an example
longer than the config's `max_seq_length` stops the run and names itself
rather than being cut to fit. Partition train and eval so that no source
document contributes to both — the original split did not, and every
held-out number measured under it was scoring paraphrases of memorised
text.

## Status

| Layer            | Implemented | Tested locally         |
| ---------------- | ----------- | ---------------------- |
| Kernel (ℚ)       | yes         | 1284 tests             |
| Operands         | yes         | 169 tests              |
| Composition      | yes         | 37 tests               |
| Canonical 29     | yes         | 435 tests              |
| Blueprint gates  | yes         | 35 tests               |
| Kernel (torch)   | yes         | 22 buffer tests        |
| External sidecar | yes         | 180 tests              |
| Script validity  | yes         | 40 tests               |
| Withdrawn numbers | yes        | 26 tests               |
| Documented paths | yes         | 39 tests               |
| Conservation (torch) | yes         | 42 tests               |
| Audit closure    | yes         | 27 tests               |
| Benchmark honesty | yes        | 30 tests               |
| Gates reject     | yes         | 37 tests               |
| Aux checkpoint   | yes         | 7 tests                |
| Auxiliary losses | yes         | 15 tests               |
| Memory           | yes         | covered by trainer     |
| Training         | yes         | requires HF + LoRA     |
| Eval             | yes         | requires SCAN/COGS     |
| Fixtures         | committed   | bit-exact gate, all 29 |

Counts above are not hand-maintained. `scripts/verify_counts.py --check`
measures the suite and exits 1 if this table disagrees, because these numbers
were previously wrong: they had been read off wrapped `pytest -q` dots, and
`-q` prints no summary line, so the real figure was never on screen.

2425 tests are collected and 2425 pass. **Nothing is skipped**, here or in
CI. The counts above are *collected* rather than *passed* so that the same
README is correct on every machine — a passed count moves with the
environment, and a README that is right on one box and wrong on another is
not a claim about the suite. `verify_counts.py --check` measures collection
and separately fails if any test does not pass, so neither question can hide
behind the other.

There are no skips left to explain. The Lean mirror's tests used to carry
`skipif` guards on a toolchain being present, which meant the one independent
cross-check of the exact-ℚ kernel reported green in CI while never compiling
anything; the guards are gone and CI installs the compiler, pinned by
`vedic_trainer/lean-toolchain`. The training pipeline has also been run end
to end on 4 CPU cores in fp32, so CPU is a supported path for the full
experiment and not only for the gates — though the figures that run produced
are withdrawn, for reasons `ABLATION_RESULTS.md` sets out.

## Quick reference

```bash
# Bit-exact gate (Fraction kernel ↔ committed fixtures)
python scripts/verify_bit_exact.py

# All local tests
python -m pytest vedic -q

# Are the four auxiliary losses differentiable w.r.t. Psi?
python scripts/probe_aux_gradients.py

# Run a sutra queue through every composition mode
python scripts/run_composition.py
python scripts/run_composition.py --mode CONCURRENT --show-waves

# LoRA fine-tune (Mac Pro / MPS); bring your own corpus, see above
python scripts/train_lora.py --config configs/ablations/cpu_full.yaml

# Evaluate
# Held-out cross-entropy (fast; this is the discriminating measure)
python scripts/eval_heldout.py \
    --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --adapter checkpoints/cpu_full \
    --device cpu \
    --heldout data/eval.jsonl \
    --output runs/full_eval.json

# SCAN / COGS exact-match (slow: greedy decoding)
python scripts/eval_benchmarks.py \
    --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --adapter checkpoints/cpu_full \
    --device cpu \
    --output runs/full_bench.json

# Do the documents still match the suite?
python scripts/verify_counts.py --check
```

## Hardware target

- Mac Pro 2019 16" with 32 GB unified memory (Apple Silicon MPS).
- Llama-3.2-1B-Instruct or Qwen2.5-1.5B-Instruct as the intended base
  model. The runs that were executed used
  `HuggingFaceTB/SmolLM2-135M-Instruct`, because Llama-3.2 is gated; their
  figures have been withdrawn (`ABLATION_RESULTS.md`) and nothing here has
  been reproduced on Llama.
- LoRA rank-16 on q/k/v/o projection matrices.

CPU runs the whole pipeline, slowly: roughly five minutes per LoRA arm and
ten seconds per held-out evaluation at this model size. Full SCAN/COGS
generation is the one part that is impractical on CPU (~36k greedy decodes).

## Falsification criteria — and the verdict

These were declared before the experiment ran. The experiment has now run,
so each one carries a result rather than an expectation.

**1. `configs/ablations/full.yaml` does not beat `configs/ablations/no_sutra.yaml`
by ≥ 2% absolute on SCAN length-split exact-match.**

*Unmeasured.* This was previously recorded as "met, but vacuously" on the
strength of 30 SCAN and 20 COGS examples per split — 0.1–0.7% of splits that
run to 3,920–21,000. Those figures came from `--scan-subset`/`--cogs-subset`
flags on a script that no longer exists, and have been **withdrawn**; see
`ABLATION_RESULTS.md`. A subset that small cannot establish the criterion in
either direction, so nothing is claimed for it.

Measuring it means the full splits through `scripts/eval_benchmarks.py`,
which has no flag to shorten the work: roughly 36,000 greedy decodes.

Note also that this criterion names `configs/ablations/full.yaml`, a config targeting a gated
base model that **was never run**. Every executed result uses the `cpu_*` and
`scaled*` configs on `HuggingFaceTB/SmolLM2-135M-Instruct`.

The measure that does discriminate is held-out cross-entropy, and that too
is now **unmeasured**. It was measured, across four weightings and three
seeds — but on a synthetic corpus this repository no longer contains and
cannot regenerate, and under two loss definitions it no longer implements.
Those figures are **withdrawn**; `ABLATION_RESULTS.md` is the withdrawal and
says what it would take to ask the question again.
`vedic/kernel/tests/test_no_withdrawn_number_is_quoted.py` fails if any
document here quotes one of them.

**2. Audit-closure rate at inference for `full` minus `no_sutra` < 10% absolute.**

*Unmeasurable — the criterion cannot discriminate, and this is proved, not
assumed.* R2, R3 and R4 are algebraic identities: exactly zero for **every**
Ψ in ℚ¹⁶. Closure therefore reduces to R1, which takes no Ψ at all and
vanishes exactly when the trace counter is a multiple of T(29) = 435.
**Audit closure is a function of the counter alone**, so two arms are
guaranteed the same number and a `full` − `no_sutra` delta below 10% is
satisfied by any two models whatsoever, including two copies of one.

`vedic/kernel/tests/test_audit_closure_degeneracy.py` establishes this over
all of ℚ¹⁶ rather than on a sample. R2 and R3 are linear and R4 is quadratic
in Ψ, so vanishing on `{0} ∪ {eᵢ} ∪ {eᵢ+eⱼ}` — 137 vectors — determines each
of them as the zero map; the 560 three-vertex sums are checked as well,
because a cubic map could vanish on the spanning set without being zero and
that is the premise the argument rests on.

An earlier version of this section measured the degeneracy instead: two
480-string corpora, one English and one not, producing identical closure
flags. That was evidence about 960 encoded vectors and silent about the rest
of ℚ¹⁶, and it needed a synthetic text encoder to produce them at all. The
metric itself has been removed rather than reported.

**3. Any bit-exactness mismatch between the ℚ kernel and the committed
fixtures.**

*Not triggered.* `scripts/verify_bit_exact.py` checks all 30 fixture keys —
32 inputs, 32 sutra records, 96 conservation records — and an unchecked key
is itself a failure. It passes.

Its scope was narrower than this criterion implied. Those fixtures cover
`vedic/kernel/z2_primitives.py`, which states in its own first paragraph that
it is **not** an implementation of the 29 sutras and uses a conflicting
numbering. `vedic/kernel/sutras_canonical.py` — named above as the single authority — had
no fixture-backed gate at all. It now has one: 8 inputs × 29 sutras × 3
strengths = 696 records, including α → 0 (the §12Y identity guarantee) and
the α value itself.

**What that gate can and cannot show.** The fixtures are written by the same
kernel they are compared against, so they detect **drift, not error** — they
are a regression reference. Correctness against the upstream definition is a
separate question, answered by exporting from the user's
`vedic_v18.24_full_kernel.html`, which is external to this repository. The
distinction matters: an earlier version of this gate rebuilt its own missing
fixtures, which made it unfalsifiable rather than merely narrow.

The strict ℚ reference layer is what makes criterion 3 honest: float
tolerance does not enter the verification path, and every comparison in
`vedic/kernel/tests/` is exact rational equality. It is worth being clear
that this buys correctness of the *kernel*, not correctness of the
*hypothesis* — criterion 1 shows those are separate questions, and the
kernel passing tells you nothing about whether the sutras help.

## Bit-exact protocol with v18.16

The Fraction kernel is the authoritative simulator. The committed
fixtures (`fixtures/*.json`) are produced by
`scripts/build_fixtures.py`. The user's `vedic_v18.16_strict_kernel.html` (an external file, not in
this repository) implements the same operators in JavaScript over BigInt rationals; when
the user runs `scripts/export_simulator_fixtures.py` against it, the
exported JSON must equal the committed fixtures byte-for-byte. Any
mismatch is a kernel bug.

See `docs/BIT_EXACT_PROTOCOL.md` for the JavaScript hooks the simulator
must expose (`window.exportFixtures(seed, n)`).

## Repository layout

```
vedic_trainer/
├── vedic/
│   ├── kernel/                 # ℚ reference + torch port + tests
│   ├── memory/                 # TesseractWM, slot map
│   ├── training/               # config, lora, losses, trainer
│   └── eval/                   # SCAN, COGS
├── scripts/
├── configs/                    # 2 base models + 15 ablation arms
├── fixtures/                   # ℚ JSON (bit-exact reference)
└── docs/                       # ARCHITECTURE, SUTRA_CATALOGUE, BIT_EXACT_PROTOCOL
```

## License

Apache-2.0 (see top-level LICENSE).
