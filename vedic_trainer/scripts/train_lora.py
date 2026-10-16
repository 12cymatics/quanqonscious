"""Fine-tune an open-weights LLM with LoRA + the four sutra losses.

Pipeline (run from the repo root):

    python scripts/verify_bit_exact.py     # bit-exact gate (mandatory)
    python scripts/train_lora.py --config configs/ablations/full.yaml

The trainer subclass injects L_χ + L_cons + L_curv + L_dual into the
total loss. LoRA is applied to the q/k/v/o projection matrices of the
base model. Hidden states are pulled from the last layer and projected
through ``TesseractWM`` to a 16-vertex Ψ.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vedic.provenance import environment, sha256_file  # noqa: E402
from vedic.training import TrainingConfig, load_yaml  # noqa: E402
from vedic.training.run import build_run  # noqa: E402


def _run_bit_exact_gate() -> None:
    cmd = [sys.executable, str(REPO / "scripts" / "verify_bit_exact.py")]
    subprocess.check_call(cmd, cwd=REPO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vedic LoRA fine-tune.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    # The bit-exact gate is mandatory. There is no override: a kernel that does
    # not reproduce the committed fixtures must not train.
    _run_bit_exact_gate()

    cfg: TrainingConfig = load_yaml(args.config)
    run = build_run(cfg)
    run.model.print_trainable_parameters()

    # Written before training starts, so a run that dies still says what it
    # was. The data hashes are what tie a result to the exact corpus.
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": cfg.model_dump(mode="json"),
        "config_file": str(args.config),
        "config_sha256": sha256_file(args.config),
        "data_sha256": {"train": sha256_file(cfg.data.train_path),
                        "eval": sha256_file(cfg.data.eval_path)},
        "n_train_examples": len(run.train_ds),
        "n_eval_examples": len(run.eval_ds),
        "environment": environment(),
    }
    (cfg.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    run.trainer.train()
    # The full log history -- every step's components and psi_rms -- goes to
    # output_dir/trainer_state.json. The Trainer writes that file only inside
    # checkpoint-N/ folders, and the last of those precedes the final step.
    run.trainer.save_state()
    run.trainer.save_model(str(cfg.output_dir))
    run.tokenizer.save_pretrained(str(cfg.output_dir))
    with (cfg.output_dir / "vedic_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2, default=str)


if __name__ == "__main__":
    main()
