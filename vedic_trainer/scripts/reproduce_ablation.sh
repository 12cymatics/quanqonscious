#!/usr/bin/env bash
# Reproduce the full vs no_sutra ablation on CPU with an open base model.
#
# Toolchain pin matters: the package targets the transformers 4.x API.
#   pip install "transformers==4.46.3" "peft==0.13.2" "datasets==3.1.0" "accelerate==1.1.1"
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
BASE=HuggingFaceTB/SmolLM2-135M-Instruct   # Llama-3.2-1B-Instruct is gated
: "${COGS_DIR:?set COGS_DIR to a dir holding cogs_test.tsv / cogs_gen.tsv}"

# 1. Seed corpus -> synthetic corpus -> 90/10 split
python scripts/generate_synthetic.py --input data/seed_corpus.txt \
                                     --output data/synthetic_all.jsonl

# 2. Are the four auxiliary losses actually differentiable w.r.t. Psi?
python scripts/probe_aux_gradients.py | tee runs/aux_gradient_probe.json

# 3. Train both arms (identical but for the four loss weights, seed 42)
for arm in no_sutra full; do
  python scripts/train_lora.py --config "configs/ablations/cpu_${arm}.yaml"
done

# 4. Evaluate both arms + the untuned base model
for arm in no_sutra full; do
  python scripts/run_ablation_eval.py \
      --base-model "$BASE" --adapter "checkpoints/cpu_${arm}" --device cpu \
      --scan-subset 30 --cogs-subset 20 \
      --heldout data/synthetic_eval.jsonl \
      --output "runs/eval_${arm}.json"
done
python scripts/run_ablation_eval.py --base-model "$BASE" --device cpu \
    --scan-subset 30 --cogs-subset 20 --heldout data/synthetic_eval.jsonl \
    --output runs/eval_base.json
