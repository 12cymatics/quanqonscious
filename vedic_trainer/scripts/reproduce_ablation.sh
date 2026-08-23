#!/usr/bin/env bash
# Reproduce the full vs no_sutra ablation on CPU with an open base model.
#
# Toolchain pin matters: the package targets the transformers 4.x API.
#   pip install "transformers==4.46.3" "peft==0.13.2" "datasets==3.1.0" "accelerate==1.1.1"
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Runtime on 4 CPU cores: ~14 min per training arm, ~45 s per held-out
# evaluation. Steps 1-4 are the experiment. Step 5 (SCAN/COGS) runs the FULL
# splits -- roughly 36k greedy decodes, i.e. days on CPU -- and is opt-in via
# RUN_BENCHMARKS=1. It is not part of the result: the benchmarks scored 0
# everywhere including the untuned base, so they cannot discriminate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
BASE=HuggingFaceTB/SmolLM2-135M-Instruct   # Llama-3.2-1B-Instruct is gated

# 1. Seed corpus -> synthetic corpus -> 90/10 split
python scripts/generate_synthetic.py --input data/seed_corpus.txt \
                                     --output data/synthetic_all.jsonl

# 2. Are the four auxiliary losses actually differentiable w.r.t. Psi?
#    This exits non-zero if any of them is dead, which stops the run here.
python scripts/probe_aux_gradients.py \
    --config configs/ablations/cpu_full.yaml \
    --output runs/aux_gradient_probe.json

# 3. Train both arms (identical but for the four loss weights, seed 42)
for arm in no_sutra full; do
  python scripts/train_lora.py --config "configs/ablations/cpu_${arm}.yaml"
done

# 4. Held-out cross-entropy -- the measure that discriminates
for arm in no_sutra full; do
  python scripts/eval_heldout.py \
      --base-model "$BASE" --adapter "checkpoints/cpu_${arm}" --device cpu \
      --heldout data/synthetic_eval.jsonl \
      --output "runs/eval_${arm}.json"
done
python scripts/eval_heldout.py --base-model "$BASE" --device cpu \
    --heldout data/synthetic_eval.jsonl --output runs/eval_base.json

# 5. Do the documents still agree with what was just measured?
python scripts/verify_ablation.py --check

# 6. Optional: full SCAN/COGS exact-match. Needs COGS_DIR and a lot of time.
if [ "${RUN_BENCHMARKS:-0}" = "1" ]; then
  : "${COGS_DIR:?set COGS_DIR to a dir holding cogs_test.tsv / cogs_gen.tsv}"
  for arm in no_sutra full; do
    python scripts/eval_benchmarks.py \
        --base-model "$BASE" --adapter "checkpoints/cpu_${arm}" --device cpu \
        --output "runs/bench_${arm}.json"
  done
  python scripts/eval_benchmarks.py --base-model "$BASE" --device cpu \
      --output runs/bench_base.json
fi
