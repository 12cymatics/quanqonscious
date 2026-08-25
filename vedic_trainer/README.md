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
  - `L_χ`   — contradiction (S7 antisymmetric energy)
  - `L_cons` — conservation (R1..R4)
  - `L_curv` — curvature (top eigenvalue of `g_ab`)
  - `L_dual` — dual-basis coherence ((S5∘S11) Ψ vs WHT axes)
- Synthetic data generators (contradiction pairs, axis-emphasis
  paraphrases) that are deterministic. They are **not** audit-closed by
  construction: 12 of the 5,120 generated records (0.23%) satisfy the
  audit predicate. An earlier revision of this file claimed otherwise.
- SCAN / COGS evaluators + at-inference audit-closure rate.

## Status

| Layer            | Implemented | Tested locally         |
| ---------------- | ----------- | ---------------------- |
| Kernel (ℚ)       | yes         | 20 tests               |
| Operands         | yes         | 44 tests               |
| Composition      | yes         | 37 tests               |
| Canonical 29     | yes         | 45 tests               |
| Blueprint gates  | yes         | 35 tests               |
| Kernel (torch)   | yes         | 22 buffer tests        |
| Data             | yes         | 5 tests                |
| Split integrity  | yes         | 7 tests                |
| External sidecar | yes         | 116 tests              |
| Script validity  | yes         | 52 tests               |
| Reported numbers | yes         | 44 tests               |
| Documented paths | yes         | 27 tests               |
| Conservation (torch) | yes     | 111 tests              |
| Audit closure    | yes         | 6 tests                |
| Benchmark honesty | yes        | 30 tests               |
| Gates reject     | yes         | 32 tests               |
| Aux checkpoint   | yes         | 7 tests                |
| Memory           | yes         | covered by trainer     |
| Training         | yes         | requires HF + LoRA     |
| Eval             | yes         | requires SCAN/COGS     |
| Fixtures         | committed   | bit-exact gate, all 29 |

Counts above are not hand-maintained. `scripts/verify_counts.py --check`
measures the suite and exits 1 if this table disagrees, because these numbers
were previously wrong: they had been read off wrapped `pytest -q` dots, and
`-q` prints no summary line, so the real figure was never on screen.

507 tests are collected. The counts above are **collected**, not passed:
three tests need a Lean toolchain, so a "passed" count would be 506 here (1
skipped) and 504 in CI (3 skipped) — the same README correct on one machine and
wrong on the other. Collection is 507 in both.
`verify_counts.py --check` measures collection and separately fails if any
test does not pass, so neither question can hide behind the other.

Every skip left is an environment capability, and each reason comes from
asking the compiler rather than looking for a file on `PATH`: an elan shim
with no toolchain satisfies `shutil.which("lean")`. On this machine Lean 4.10
is present but Mathlib is not, so one test skips; in CI all three do. The
assertion that guards the string-literal defect in the generated Lean runs
everywhere — rendering is pure string work and must not be gated on having a
compiler. The training
pipeline has also been run end to end on 4 CPU cores in fp32 — see
`ABLATION_RESULTS.md` — so CPU is a supported path for the full
experiment, not only for the gates.

## Quick reference

```bash
# Bit-exact gate (Fraction kernel ↔ committed fixtures)
python scripts/verify_bit_exact.py

# All local tests (kernel + data; ℚ-only, no floats)
python -m pytest vedic/kernel/tests vedic/data/tests -q

# Are the four auxiliary losses differentiable w.r.t. Psi?
python scripts/probe_aux_gradients.py

# Run a sutra queue through every composition mode
python scripts/run_composition.py
python scripts/run_composition.py --mode CONCURRENT --show-waves

# Generate the synthetic LoRA corpus
python scripts/generate_synthetic.py \
    --input data/seed_corpus.txt \
    --output data/synthetic_train.jsonl

# LoRA fine-tune (Mac Pro / MPS)
python scripts/train_lora.py --config configs/ablations/cpu_full.yaml

# Evaluate
# Held-out cross-entropy (fast; this is the discriminating measure)
python scripts/eval_heldout.py \
    --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --adapter checkpoints/cpu_full \
    --device cpu \
    --heldout data/synthetic_eval.jsonl \
    --output runs/full_eval.json

# SCAN / COGS exact-match (slow: greedy decoding)
python scripts/eval_benchmarks.py \
    --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
    --adapter checkpoints/cpu_full \
    --device cpu \
    --output runs/full_bench.json

# Do the documents still match the measurements?
python scripts/verify_ablation.py --check
python scripts/verify_counts.py --check
```

## Hardware target

- Mac Pro 2019 16" with 32 GB unified memory (Apple Silicon MPS).
- Llama-3.2-1B-Instruct or Qwen2.5-1.5B-Instruct as the intended base
  model. The runs actually executed use `HuggingFaceTB/SmolLM2-135M-Instruct`,
  because Llama-3.2 is gated; every number in `ABLATION_RESULTS.md` is on
  SmolLM2, and no result here has been reproduced on Llama.
- LoRA rank-16 on q/k/v/o projection matrices.
- Synthetic corpus of ~10k pairs for one epoch (~30 min on MPS).

CPU runs the whole pipeline, slowly: ~5 min per LoRA arm and ~11 s per
held-out evaluation at this model size. Full SCAN/COGS generation is the
one part that is impractical on CPU (~36k greedy decodes).

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

The measure that does discriminate is held-out cross-entropy on a
source-disjoint split, and there the auxiliary losses make the model **worse
by +7.83%** (three seeds, disjoint ranges, 17.4× the baseline seed spread).
See `ABLATION_RESULTS.md`; every figure there is checked against
`runs/*.json` by `scripts/verify_ablation.py --check`.

**2. Audit-closure rate at inference for `full` minus `no_sutra` < 10% absolute.**

*Unmeasurable — the criterion cannot discriminate, and this was checked
rather than assumed.* R2, R3 and R4 are algebraic identities on
tensor-product-encoded Ψ: exactly zero for every input. Closure therefore
reduces to R1, which closes when the trace counter is a multiple of
T(29) = 435 — and the counter is the position in the list. **Audit closure
is a function of the loop index alone.**

Measured: 480 English sentences and 480 strings of random consonants produce
*identical* closure flags and an identical rate of 0.0042 (2 of 480). Across
960 distinct texts at a fixed trace index the verdict takes exactly one
value. Two arms are guaranteed the same number, so a `full` − `no_sutra`
delta below 10% is satisfied by any two models at all, including two copies
of the same one.

`vedic/eval/tests/test_audit_closure_degeneracy.py` pins this down. If the
residuals ever become text-dependent those tests fail, and this criterion
becomes worth measuring.

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
│   ├── data/                   # encoder, contradiction/paraphrase, audit
│   ├── training/               # config, lora, losses, trainer
│   └── eval/                   # SCAN, COGS, audit-closure rate
├── scripts/
├── configs/                    # 2 base models + 5 ablations
├── fixtures/                   # ℚ JSON (bit-exact reference)
└── docs/                       # ARCHITECTURE, SUTRA_CATALOGUE, BIT_EXACT_PROTOCOL
```

## License

Apache-2.0 (see top-level LICENSE).
