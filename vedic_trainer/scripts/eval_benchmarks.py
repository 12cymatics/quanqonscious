"""Full SCAN + COGS exact-match evaluation. No subsetting.

Every split is evaluated in full: SCAN simple/length/addprim_jump and COGS
test/gen. There is no --subset and no --skip: a truncated benchmark is not
the benchmark, and a flag that shortens it is how a partial result gets
reported as a complete one.

COST. This is greedy autoregressive decoding over roughly 36k examples
(SCAN 4182 + 3920 + 3920, COGS 3000 + 21000). On CPU that is days; it needs
a GPU. That is a hardware requirement, not a reason to add a subset flag.

COGS is not on the HF Hub. Set COGS_DIR to a directory holding
cogs_test.tsv and cogs_gen.tsv from najoungkim/COGS.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic.eval import evaluate_cogs, evaluate_scan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="omit to evaluate the untuned base model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.float32)
    tag = "base"
    if a.adapter is not None:
        model = PeftModel.from_pretrained(model, str(a.adapter))
        tag = str(a.adapter)
    model = model.to(a.device)

    payload = {"model": a.base_model, "adapter": tag}

    t0 = time.time()
    s = evaluate_scan(model, tok, device=a.device)
    payload["scan"] = {k: {"n_total": v.n_total, "n_correct": v.n_correct,
                           "accuracy": v.accuracy} for k, v in s.items()}
    payload["scan_secs"] = round(time.time() - t0, 1)
    print("scan:", json.dumps(payload["scan"]))

    t0 = time.time()
    c = evaluate_cogs(model, tok, device=a.device)
    payload["cogs"] = {k: {"n_total": v.n_total, "n_correct": v.n_correct,
                           "accuracy": v.accuracy} for k, v in c.items()}
    payload["cogs_secs"] = round(time.time() - t0, 1)
    print("cogs:", json.dumps(payload["cogs"]))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
