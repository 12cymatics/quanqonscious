"""Evaluate a trained checkpoint on SCAN + COGS + audit-closure rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from vedic.eval import (
    audit_closure_rate,
    evaluate_cogs,
    evaluate_scan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vedic LoRA evaluation.")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="LoRA adapter directory (a TrainingConfig output_dir).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--scan-subset", type=int, default=None)
    parser.add_argument("--cogs-subset", type=int, default=None)
    parser.add_argument("--audit-corpus", type=Path, default=None,
                        help="Optional file with one generated text per line for audit-closure rate.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model = model.to(args.device)

    scan_results = evaluate_scan(
        model, tokenizer, device=args.device, eval_subset=args.scan_subset
    )
    cogs_results = evaluate_cogs(
        model, tokenizer, device=args.device, eval_subset=args.cogs_subset
    )

    audit_rate = None
    if args.audit_corpus is not None:
        with args.audit_corpus.open("r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        audit_rate = audit_closure_rate(texts)

    payload = {
        "scan": {k: {"n_total": v.n_total, "n_correct": v.n_correct, "accuracy": v.accuracy}
                 for k, v in scan_results.items()},
        "cogs": {k: {"n_total": v.n_total, "n_correct": v.n_correct, "accuracy": v.accuracy}
                 for k, v in cogs_results.items()},
        "audit_closure_rate": audit_rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
