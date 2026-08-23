"""Held-out cross-entropy on the synthetic eval split. Runs in full.

This is a *different evaluation* from scripts/eval_benchmarks.py, not a
faster mode of it. Neither script has a flag that skips work: a run either
completes or fails.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def heldout_loss(model, tok, path: Path, device: str, max_len: int = 512, bs: int = 8):
    ds = load_dataset("json", data_files=str(path), split="train")
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len,
                              padding=False),
                batched=True, remove_columns=ds.column_names)
    dl = DataLoader(ds, batch_size=bs,
                    collate_fn=DataCollatorForLanguageModeling(tok, mlm=False))
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n = int((batch["labels"] != -100).sum())
            tot += float(out.loss) * n
            ntok += n
    if ntok == 0:
        raise ValueError(f"{path} produced no scored tokens")
    return tot / ntok, ntok


def perplexity(ce: float) -> float:
    """exp(CE), with no clamp.

    This read ``math.exp(min(loss, 40))``, so any cross-entropy above 40 was
    reported as e^40 -- a plausible-looking number standing in for a
    divergent run. A CE that high means the evaluation is broken, and that
    should surface here rather than be rounded into the record.
    """
    if ce > 709:                      # math.exp overflows just above this
        raise ValueError(
            f"held-out cross-entropy is {ce:.1f}; perplexity is not "
            f"representable and the run is not usable. Investigate the "
            f"evaluation rather than recording a capped value.")
    return math.exp(ce)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="omit to evaluate the untuned base model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heldout", type=Path, required=True)
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

    t0 = time.time()
    loss, ntok = heldout_loss(model, tok, a.heldout, a.device)
    payload = {"model": a.base_model, "adapter": tag,
               "heldout": {"ce_loss": loss, "ppl": perplexity(loss),
                           "n_tokens": ntok, "secs": round(time.time() - t0, 1)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2))
    print("heldout:", json.dumps(payload["heldout"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
