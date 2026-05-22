# Month Work Bundle - Sutra / LLM Reasoning Experiment

Created: 2026-05-22, Australia/Brisbane.

This bundle collects the local work artifacts from the Vedic/sutra simulator review and the LLM sutra-trace fine-tuning experiment.

## Contents

- `llm_sutra_reasoning_experiment/`
  - Dataset/oracle source code.
  - Training and inference scripts.
  - Local smoke tests.
  - Full 39k generated reasoning dataset.
  - SFT formatted train/val/test/OOD files for trace, answer-only, and shuffled-trace variants.
  - Reports and status pages.
  - Valid uploaded 3B trace-adapter checkpoints:
    - `remote_checkpoints/incoming/runs__gpu_trace_lora__checkpoint-5.tar.gz`
    - `remote_checkpoints/incoming/runs__gpu_trace_lora__checkpoint-10.tar.gz`
  - Local macOS dependency file:
    - `requirements-local-mac.txt`
  - GPU/Colab dependency file:
    - `requirements-train.txt`

- `source_artifacts/original_markdown/`
  - Original markdown artifact used to structure the LLM training method.

- `source_artifacts/vedic_html_and_recordings/`
  - Vedic simulator HTML artifacts and `vedic_v18_rec_*.json` recording fixtures from the last month.

## Important Status

The local tests and oracle checks pass. The full model-improvement claim is not proven yet.

What is complete:

- Full curvature-inclusive dataset: 39,000 rows.
- Trace, answer-only, and shuffled-trace SFT datasets: 30,000 training rows each.
- Oracle held-out test evaluation: 5,000 rows at 100% answer accuracy, trace validity, check match, and contradiction localization F1.
- 3B QLoRA trace-adapter training started on Colab T4 and produced valid checkpoints through step 10 of 938.

What remains:

- Complete the trace-adapter fine-tune.
- Complete answer-only and shuffled-trace adapters.
- Run base, answer-only, sutra-trace, and shuffled-trace held-out/OOD inference.
- Compare answer accuracy, contradiction F1, invariant check validity, JSON parse rate, trace validity, and OOD depth generalization.

## Recreate Local Mac Environment

From `llm_sutra_reasoning_experiment/`:

```bash
PYENV_VERSION=3.12.7 python -m venv .venv312
.venv312/bin/python -m pip install --upgrade pip setuptools wheel
.venv312/bin/python -m pip install -r requirements-local-mac.txt
PYTHONPATH=src .venv312/bin/python -m pytest -q
```

## Resume GPU Training

Use a Linux/CUDA GPU worker for QLoRA:

```bash
python3 -m pip install -r requirements-train.txt
bash scripts/run_gpu_qlora.sh
```

To resume from the included checkpoint, restore:

```bash
tar -xzf remote_checkpoints/incoming/runs__gpu_trace_lora__checkpoint-10.tar.gz -C .
```

Then run the GPU trainer with `--resume-if-possible` via `scripts/run_gpu_qlora.sh`.
