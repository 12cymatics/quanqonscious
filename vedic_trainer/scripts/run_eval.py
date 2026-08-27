"""Evaluate a trained checkpoint on SCAN + COGS.

There is no audit-closure option. ``--audit-corpus`` used to take a file of
generated text and write an ``audit_closure_rate`` into the results JSON.
That metric could not measure a model: R2, R3 and R4 are algebraic identities
that vanish for every Ψ in ℚ^16 and R1 takes no Ψ at all, so the verdict was
a function of the loop index — two arms, or two copies of one model, were
guaranteed the same number. It is proved over all of ℚ^16 in
``vedic/kernel/tests/test_audit_closure_degeneracy.py``.

The rate also needed a text→Ψ encoder to exist, and the synthetic encoder
that supplied one has been removed along with the rest of the generated
corpus. Both halves are gone rather than one being left as a flag nobody
can pass an input to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from vedic.eval import evaluate_cogs, evaluate_scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Vedic LoRA evaluation.")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="LoRA adapter directory (a TrainingConfig output_dir).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model = model.to(args.device)

    scan_results = evaluate_scan(model, tokenizer, device=args.device)
    cogs_results = evaluate_cogs(model, tokenizer, device=args.device)

    payload = {
        "scan": {k: {"n_total": v.n_total, "n_correct": v.n_correct, "accuracy": v.accuracy}
                 for k, v in scan_results.items()},
        "cogs": {k: {"n_total": v.n_total, "n_correct": v.n_correct, "accuracy": v.accuracy}
                 for k, v in cogs_results.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
