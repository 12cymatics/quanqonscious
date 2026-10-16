# Pre-registration: the four sutra auxiliary losses on SCAN (length split)

This file is committed **before any training step of this experiment**. The
registration is two consecutive commits, both made before training starts.
The first adds this file, the code and the control configs. The second adds
the calibration measurements and the treatment configs. Those are produced
from the first commit's clean tree, so their provenance names a committed
state. Training runs at the second. Anything decided after a metric has been
seen goes into the results document as a deviation, never back into this
file.

Under rule C6 of `CODEX_OPENAI_MATH.md` (repository root), everything this
experiment produces is an **empirical result, not a formalized one**. The
premises proved by Comparator are listed under *Premises* below; the outcome
is measured, not proved.

## Question

Does adding the four sutra-derived auxiliary losses (L_χ, L_cons, L_curv,
L_dual), as `vedic/training/losses.py` defines them, change held-out
performance of a LoRA fine-tune on real data, relative to an otherwise
identical run with all four weights at zero?

## Data

- SCAN (Lake & Baroni 2018), brendenlake/SCAN @
  `c4b756cbc010d75c912f16c42c8f15dc6b7e6c8f`, vendored gzip-compressed with its
  BSD licence under `vedic/eval/data/scan/`. `vedic/eval/scan_data.py` pins the
  SHA-256 and the example count of each decompressed file and refuses any file
  that does not match. Each vendored file was compared byte for byte with the
  upstream file at that commit.
- Length split. Train: 16,990 examples. Test: 3,920.
- Corpus: `scripts/build_scan_corpus.py --split length` writes train, test and
  monitor files under data/scan_length/. The output is byte-deterministic, and
  `vedic/eval/tests/test_scan_data.py` pins the SHA-256 of all three files.
- Record format: `{"text": "Command: <cmd>\nActions: <actions>\n"}`, built by
  `scan_data.record_text`. That is the evaluator's prompt, then the gold
  actions, then a newline. The terminator is a newline, not EOS: for this
  tokenizer pad == eos, and the collator masks every pad-id token out of the
  labels, so an EOS terminator would never be trained
  (`vedic/eval/tests/test_first_line.py` pins both facts).
- Trainer's periodic evaluation set ("monitor"): the first 256 training
  records. It is a progress monitor on training data. No result is read from
  it and no decision depends on it.

## Model and training

- `HuggingFaceTB/SmolLM2-135M-Instruct` @
  `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
- LoRA r=16, α=32, dropout 0.05 on q/k/v/o. The Ψ projection (TesseractWM,
  576 → 16) is trained jointly, as a submodule of the model. It is clipped and
  zeroed with the LoRA weights.
- One epoch, which is 1,062 optimizer steps. Batch 4 × gradient accumulation 4;
  lr 2e-4, cosine schedule, 3% warmup (32 steps); weight decay 0.01;
  max_grad_norm 1.0; fp32 on CPU; 2 intra-op threads per run.
- Library versions are pinned exactly in `requirements.txt`: torch 2.14.1,
  transformers 5.19.0, peft 0.21.2, datasets 5.1.0, accelerate 1.15.0. Each
  run writes its library versions, thread count, commit and data hashes to
  run_provenance.json.
- Seeds 0, 1 and 2. All three are run and all three are reported.

## Arms

- **control**: `configs/scan_length/no_sutra_seed0.yaml`,
  `configs/scan_length/no_sutra_seed1.yaml`,
  `configs/scan_length/no_sutra_seed2.yaml`. All four weights are 0.
- **treatment**: full_seed0.yaml, full_seed1.yaml and full_seed2.yaml in the
  same directory, added by the second registration commit. Each is the control
  config of its seed with the weights from the calibration rule below. Nothing
  else differs, and test_calibration_configs.py (added with them, under
  vedic/training/tests/) checks that.

## Calibration rule (fixed here; computed before any training step)

    α_k = 0.1 · ‖∇_LoRA L_CE‖ / ‖∇_LoRA L_k‖

Every quantity is measured at the run's exact initial state. That state is
built by `vedic/training/run.py`, the same function `train_lora.py` trains
with. The model is in eval mode (no dropout). Each gradient is of the mean
over the first 64 training records in file order, taken in micro-batches of 8.
Each weighted term then starts with a LoRA gradient one tenth of
cross-entropy's, so at most 0.4 of it in total.
`scripts/calibrate_aux_weights.py` implements the rule and writes each seed's
measurement (calibration_seed0.json and so on) beside the configs.

The rule calibrates **per seed**. The Ψ projection is a random orthogonal
draw and Ψ lies almost entirely along one direction, so at fixed weights the
auxiliary dose at initialisation varied 2.5x across seeds through that draw
alone (pre-flight audit measurement). A seed-0 trial of this rule, run before
this registration on uncommitted code, gave CE₀ = 3.9718 for every seed, as it
must, since LoRA's B starts at zero and the initial model does not depend on
the seed. It gave psi_rms at initialisation of 2.39, 1.53 and 1.44 for seeds
0, 1 and 2, and a second measurement of seed 0 reproduced the first bit for
bit. That trial was discarded with the container it ran in; the committed
measurements are the ones that count.

The shipped configs' weights are not used. At initialisation they made the
auxiliary terms 94–99% of the loss and 40–163x cross-entropy's LoRA-gradient
norm, depending on the seed's projection draw and the batch (two independent
measurements in the pre-flight audit).
With clipping active at every step, a difference between the arms would then
have measured cross-entropy under-training.

## Metrics

1. **Primary: held-out cross-entropy.** The token-mean over all 3,920 test
   records, from `scripts/eval_heldout.py --batch-size 8 --num-threads 2`. The
   mean is over scored tokens exactly: the first token of each record is
   predicted by nothing and is not counted
   (`vedic/eval/tests/test_heldout_ce.py`).
2. **Criterion 1: exact match.** All 3,920 length test examples, greedy, from
   `scripts/eval_benchmarks.py --benchmark scan --split length --num-threads 2`.
   The answer is the generated text before the first newline (or EOS),
   stripped. The token budget is computed from the data: 265, the longest gold
   continuation plus its terminator. A model emitting the canonical
   tokenization of every correct answer scores 100%. Decoding stops once the
   verdict cannot change. This is exact: it never changes a verdict
   (`vedic/eval/tests/test_first_line.py`). The count of examples still open at
   the budget is reported with every score.
3. **Reference**: the untuned base model, on both metrics.
4. **Logged, every 10 steps, both arms**: L_CE, the four auxiliary terms
   (unweighted), L_total and psi_rms.

## Execution

`scripts/run_scan_length_experiment.py` runs every step above with the flags
named here. It trains each seed's control and treatment side by side (two
runs at 2 threads each on a 4-core machine), then evaluates the base model and
the six adapters. Records go to results/scan_length/: for each model a
held-out record and an exact-match record, and for each run its provenance
and its Trainer log history. Each evaluation record carries the adapter's
SHA-256.

## Decision rules

- **Primary.** The effect for seed s is CE(full_s) − CE(no_sutra_s). The claim
  "the losses reduce held-out CE" is made only if the effect is negative for
  all three seeds **and** |mean effect| exceeds the larger of the two arms'
  across-seed ranges (max − min of the three CEs). "Increase" is decided the
  same way with the sign reversed. Otherwise the conclusion is "no detectable
  effect at this scale".
- **Criterion 1** (README, falsification criterion 1, instantiated at this
  scale). Let d = the mean over seeds of EM(full_s) − EM(no_sutra_s), in
  percentage points. The criterion is **triggered** (the hypothesis is
  falsified on this experiment) iff d < 2.0. If both arms' mean EM is below
  1.0%, the outcome is reported as floor-limited. The criterion then still
  falls as computed, but the comparison is uninformative, and the report says
  so.
- No re-runs, no seed substitution, and no reweighting after any metric is
  seen. A run that dies (for example, the container is reclaimed) is restarted
  from scratch with its unchanged config and the restart is recorded. At a
  fixed thread count the code is bit-deterministic, so a restart cannot
  select a different outcome. No run is resumed from a checkpoint and none
  is excluded.

## Known limitations, stated in advance

These come from the pre-flight audit. They are reasons the experiment may show
nothing even if sutra structure could help, and they are recorded so a null
result is not over-read.

1. **The auxiliary terms have a trivial minimiser.** Each is homogeneous in Ψ
   (degree 2; degree 4 for L_cons). The projection producing Ψ is read by
   nothing but these terms, so all four can be lowered by turning the
   projection away from the data at no cost to cross-entropy. For L_curv this
   is guaranteed: g_ab is positive definite (smallest eigenvalue 73/16,
   `vedic/kernel/tests/test_hessian_definite.py`), so its only minimiser is
   Ψ = 0. The pre-flight audit measured this directly (not reproduced
   here): with the LM frozen and only the projection trained, the weighted
   auxiliary total fell 98% within 100 steps. The treatment arm may therefore
   test a transient. psi_rms is logged so that, if this happens, it shows.
2. **Ψ is mostly one constant vector.** Mean-pooled final hidden states point
   almost the same way for every SCAN example (the pre-flight audit measured a
   median cosine to the corpus mean of 0.994). So most of each auxiliary value
   scores the projection of one corpus-wide direction, not the example.
3. **L_cons overlaps L_χ.** L_cons is exactly (9/16)(‖S − mean·𝟙‖⁴ + ‖A‖⁴), a
   quartic penalty on Ψ's non-constant part
   (`vedic/training/tests/test_losses.py`). Its antisymmetric half is the
   square of the per-example L_χ.
4. **Scale.** One 135M model, one split, one epoch, CPU fp32. Nothing here
   transfers to Llama-3.2-1B, to other splits, or to bitwise reproduction at
   another thread count or on another machine.

## Premises proved by Comparator

Statement and proof are checked by Comparator under the permitted axioms
(`formal/ComparatorChallenges/VedicResiduals.json`; scope in
`formal/docs/VedicResiduals.md`):

- `auditClosed_indep`: falsification criterion 2 cannot discriminate between
  any two models;
- `lDual_eq_fourier_weight`: L_dual is 16 times the Walsh–Fourier weight above
  degree one of S11(S5 Ψ);
- `dMass_eq_zero`: the mass drift that L_cons used to include is identically
  zero, so its removal changed no value.
