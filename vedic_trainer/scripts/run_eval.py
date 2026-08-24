"""Evaluate a trained checkpoint on SCAN + COGS, and optionally audit closure.

``audit_closure_rate`` is written to the results JSON only when
``--audit-corpus`` supplied a corpus to measure it on. It used to be emitted
as ``null`` on every run that omitted the flag, which records a measurement
that was never taken as though it were a value: a reader aggregating result
files sees a numeric field that is sometimes null and reasonably reads it as
0.0, or as a measurement that ran and failed.

Absence is the only faithful encoding of "not measured". A consumer that
needs the rate now gets a ``KeyError`` from ``payload["audit_closure_rate"]``
instead of a ``None`` that arithmetic will happily turn into a number.

The flag stays optional rather than becoming required, because the rate needs
an input the caller may legitimately not have -- a corpus of generated text,
which is a separate generation step from the SCAN/COGS benchmarks. Requiring
it would couple those benchmarks to that step, and the likely response would
be a placeholder corpus passed only to satisfy argparse, which yields a
real-looking rate computed from nothing. That is worse than an absent key.
"""
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

    scan_results = evaluate_scan(model, tokenizer, device=args.device)
    cogs_results = evaluate_cogs(model, tokenizer, device=args.device)

    payload = {
        "scan": {k: {"n_total": v.n_total, "n_correct": v.n_correct, "accuracy": v.accuracy}
                 for k, v in scan_results.items()},
        "cogs": {k: {"n_total": v.n_total, "n_correct": v.n_correct, "accuracy": v.accuracy}
                 for k, v in cogs_results.items()},
    }

    # The key exists only when it was measured. See the module docstring.
    if args.audit_corpus is not None:
        with args.audit_corpus.open("r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        payload["audit_closure_rate"] = audit_closure_rate(texts)
    else:
        print("audit closure: not measured (--audit-corpus not given); "
              "'audit_closure_rate' is omitted from the results file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
