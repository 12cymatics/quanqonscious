# vedic_trainer

A 29-sutra LoRA fine-tuning kernel for small open-weights LLMs. The
package exposes:

- An exact-ℚ reference implementation of the 29 Vedic sutras (the
  "structuring algebra") and four conservation residuals over **Z₂⁴**.
  Every sutra takes its own operands explicitly (mask, base, reference
  index, modulus, rotation, axis set, blend weight) with named canonical
  defaults — see `vedic/kernel/sutras_exact.py`.
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
  paraphrases) that are deterministic and audit-closed by construction.
- SCAN / COGS evaluators + at-inference audit-closure rate.

## Status

| Layer            | Implemented | Tested locally       |
| ---------------- | ----------- | -------------------- |
| Kernel (ℚ)       | yes         | 13 tests (pytest)    |
| Operands         | yes         | 44 tests             |
| Composition      | yes         | 35 tests             |
| Kernel (torch)   | yes         | 22 buffer tests      |
| Conservation     | yes         | 8 tests              |
| Interaction lat. | 30 ids      | 2 tests, 50 pairs    |
| Memory           | yes         | covered by trainer   |
| Training         | yes         | requires HF + LoRA   |
| Data             | yes         | 5 tests              |
| Eval             | yes         | requires SCAN/COGS   |
| Fixtures         | committed   | bit-exact gate ✓     |
| External sidecar | yes         | 15 tests (1 lean-skip) |

All ℚ-only tests pass on this machine (CPU). The training pipeline runs
on the user's Mac Pro 2019 (32 GB unified memory, MPS).

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
python scripts/train_lora.py --config configs/ablations/full.yaml

# Evaluate
python scripts/run_eval.py \
    --base-model meta-llama/Llama-3.2-1B-Instruct \
    --adapter checkpoints/ablation_full \
    --device mps \
    --output runs/full_eval.json
```

## Hardware target

- Mac Pro 2019 16" with 32 GB unified memory (Apple Silicon MPS).
- Llama-3.2-1B-Instruct or Qwen2.5-1.5B-Instruct as the base model.
- LoRA rank-16 on q/k/v/o projection matrices.
- Synthetic corpus of ~10k pairs for one epoch (~30 min on MPS).

CPU is supported only for the bit-exact gate and the kernel/data tests.

## Falsification criteria

These are the conditions under which we declare the experiment a null:

1. **`full.yaml` does not beat `no_sutra.yaml`** by ≥ 2 % absolute on
   SCAN length-split exact-match. The auxiliary losses do not deliver.
2. **Audit-closure rate at inference for `full` minus `no_sutra` < 10 %
   absolute**. The conservation laws are not structuring the
   distribution.
3. **Any bit-exactness mismatch** between the ℚ kernel and the
   committed fixtures (or between ℚ and the v18.16 simulator export).
   The kernel is wrong; stop, fix, restart.

The strict ℚ reference layer is what makes those criteria honest. Float
tolerance does not enter the verification path: every comparison in
`vedic/kernel/tests/` is exact rational equality.

## Bit-exact protocol with v18.16

The Fraction kernel is the authoritative simulator. The committed
fixtures (`fixtures/*.json`) are produced by
`scripts/build_fixtures.py`. The user's `vedic_v18.16_strict_kernel.html`
implements the same operators in JavaScript over BigInt rationals; when
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
