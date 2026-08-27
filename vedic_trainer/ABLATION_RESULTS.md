# Sutra auxiliary-loss ablation — withdrawn

**Every held-out cross-entropy figure this document reported has been
withdrawn.** Nothing is claimed in their place: the effect of the four
sutra-derived auxiliary losses on held-out language modelling is
**unmeasured** for this package.

The measurements happened. `runs/*.json` still hold them, tracked, and are an
honest record of what was executed. What has gone is the licence to quote
them, and this file is the withdrawal rather than the results.

## What was measured

A LoRA fine-tune of `HuggingFaceTB/SmolLM2-135M-Instruct` (r=16, α=32, on
q/k/v/o), one epoch, three seeds, two arms — `full` with the four auxiliary
losses weighted and `no_sutra` with all four weights zero — scored by a
separately computed pure-CE held-out loss. Four weightings were run in
sequence: with two of the losses inert, with all four live at the original
weights, with all four live at conventional weights, and finally on a
source-disjoint train/eval split after the original split was found to leak
every one of its eval source sentences into train.

The reported direction was consistent across all four: adding the losses
cost held-out cross-entropy. That was the finding, and it is the finding
being withdrawn.

## Why the figures are withdrawn

**1. The data they were measured on is no longer in this repository, and
nothing here can regenerate it.**

Every arm trained and evaluated on a *synthetic* corpus: 5,120 records
expanded by template from 512 seed sentences, ten records per sentence, with
Ψ vectors produced by a hand-written text encoder. That pipeline —
`data/seed_corpus.txt`, `scripts/generate_synthetic.py`,
`scripts/split_corpus.py`, `vedic/data/tesseract_encode.py`,
`vedic/data/synthetic_contradiction.py` and
`vedic/data/synthetic_paraphrase.py` — has been removed. It was a stand-in for real training
data, and it carried the defects a stand-in carries: three of the four
encoder axes asserted a positive feature when no marker was found in the
text at all, the polarity axis computed a normalised marker tally and then
discarded it unread, and the contradiction generator fell back to prefixing
"It is not the case that" whenever it could not find an auxiliary verb.

This repository's own `.gitignore` stated the standard being applied here.
The generated corpus was ignored and the seed corpus was not, with the reason
written beside it: *without it, nothing in the repository can regenerate or
falsify the numbers in ABLATION_RESULTS.md*. That is now the case. A figure
that cannot be reproduced or falsified from the repository that publishes it
is a claim, not a result.

**2. The objective is no longer the one that was run.**

`L_chi` and `L_curv` have both changed definition since those runs.
`L_chi` was the antisymmetric *share* of Ψ's energy, ‖A‖²/‖Ψ‖²; it is now
‖A‖². `L_curv` was a Rayleigh quotient hinged against `kappa.detach().mean()`
— a per-example loss shifted by a statistic of whichever examples shared its
batch; it is now ⟨Ψ, g_ab Ψ⟩. Both denominators were normalisations and both
are gone, along with the batch-relative baseline. So even with the corpus
restored, the recorded numbers would describe an objective this package no
longer implements. See `vedic/training/losses.py`.

Re-running with the corpus rebuilt and the old losses restored would not
rescue the numbers either. It would restore the placeholder data and the
normalisations in order to reproduce a figure about them, which is the
opposite of the reason both were removed.

## What survives, and where it is established

Three of the things this document reported were never measurements. They are
properties of the code, they were established by tests rather than by runs,
and they are unaffected by the withdrawal:

- **`L_cons` summing R1..R4 was a constant.** R2, R3 and R4 are algebraic
  identities that vanish for every Ψ in ℚ¹⁶ and R1 takes no Ψ at all, so the
  sum had identically zero gradient while growing quadratically in the step
  counter. Proved over all of ℚ¹⁶ in
  `vedic/kernel/tests/test_conservation_laws.py` and, for the
  composition, in
  `vedic/kernel/tests/test_audit_closure_degeneracy.py`. `L_cons` no longer
  sums the residuals; its docstring derives what it does instead.
- **The power-iteration `L_curv` had no gradient path to Ψ at all.** `g_ab`
  is independent of Ψ — every contributing operator is linear — so iterating
  toward its top eigenvector from a random vector produced a quantity
  constant across the batch. `vedic/kernel/hessian.py` states it and
  `vedic/kernel/tests/test_conservation_laws.py` verifies it. `scripts/probe_aux_gradients.py`
  is the detector, and it exits non-zero when a loss does not reach Ψ.
- **Audit closure cannot distinguish two models.** The README named an
  audit-closure rate at inference as falsification criterion 2. Closure is a
  function of the trace counter alone, so any two arms — including two copies
  of one model — are guaranteed the same number and the criterion is met by
  anything. Proved over all of ℚ¹⁶ in
  `vedic/kernel/tests/test_audit_closure_degeneracy.py`. The metric has been
  removed rather than reported.

## The raw records

`runs/*.json` are kept. Deleting evidence is not the same as withdrawing a
claim, and a reader checking this withdrawal needs to see what was actually
run. Each file records its model, its adapter, and the held-out `ce_loss`,
`ppl`, `n_tokens` and wall time of one evaluation.

`vedic/kernel/tests/test_no_withdrawn_number_is_quoted.py` fails if any
tracked document quotes a figure sourced from them, and fails if the corpus
pipeline reappears without this withdrawal being revisited.

## SCAN / COGS — withdrawn

**The SCAN and COGS figures that were here have also been withdrawn**, and
for a different reason. They were exact-match scores over 30 SCAN and 20 COGS
examples per split, against real splits of 3,920–21,000 — that is 0.1–0.7% of
the benchmark, presented as the benchmark.

They were produced by `--scan-subset 30 --cogs-subset 20` flags on a script
that no longer exists. `scripts/eval_benchmarks.py`, which replaced it,
states in its first paragraph: *"There is no `--subset` and no `--skip`: a
truncated benchmark is not the benchmark, and a flag that shortens it is how
a partial result gets reported as a complete one."*

**SCAN and COGS are unmeasured for this package.** Running them means the
full splits — roughly 36,000 greedy decodes — via
`scripts/eval_benchmarks.py`, which has no flag to shorten it. The raw
records remain in `runs/eval_*.json` and are honest about what was executed:
each carries `"n_total": 30` or `"n_total": 20`. They are a record of a
subset run, not of a benchmark, and
`vedic/eval/tests/test_no_subset_is_quoted_as_a_benchmark.py` fails if any
document quotes an accuracy sourced from them.

## What it would take to measure this again

The question the ablation asked is a good one and remains open. Answering it
needs, in order:

1. **A real corpus.** Not a template expansion of a seed file — text that
   exists independently of this package, with the record schema
   `scripts/train_lora.py` reads (see the README), and a train/eval split
   partitioned over *sources* rather than records, so that held-out means
   held out.
2. **Both arms retrained from scratch** on it, at three or more seeds, with
   the auxiliary weights scaled so no term dominates cross-entropy — and with
   that scaling chosen before the results are seen, not after.
3. **Held-out cross-entropy computed separately from the training loss.**
   The logged `train_loss` was unusable on the `full` arm for the whole first
   run, because the then-current `L_cons` added a step-counter square that
   grew to over 99% of it.
4. **The full SCAN and COGS splits**, if a compositional claim is to be made
   at all, through `scripts/eval_benchmarks.py`.

Until that exists, this package makes no claim about what the four auxiliary
losses do to held-out performance.
