"""Ablation eval: SCAN + COGS exact-match (via vedic.eval) + held-out synthetic LM loss.

The LM-loss arm is added because SCAN/COGS are out-of-distribution for this
training corpus; held-out CE on the synthetic eval split is the in-distribution
measure that can actually discriminate the two arms.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)
from torch.utils.data import DataLoader

from vedic.eval import evaluate_cogs, evaluate_scan


def heldout_loss(model, tok, path, device, max_len=512, bs=8):
    ds = load_dataset("json", data_files=str(path), split="train")
    def tk(b):
        return tok(b["text"], truncation=True, max_length=max_len, padding=False)
    ds = ds.map(tk, batched=True, remove_columns=ds.column_names)
    dl = DataLoader(ds, batch_size=bs,
                    collate_fn=DataCollatorForLanguageModeling(tok, mlm=False))
    model.eval(); tot, ntok = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            n = int((batch["labels"] != -100).sum())
            tot += float(out.loss) * n; ntok += n
    return tot / ntok, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--heldout", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.float32)
    tag = "base"
    if a.adapter is not None:
        model = PeftModel.from_pretrained(model, str(a.adapter)); tag = str(a.adapter)
    model = model.to(a.device)

    payload = {"model": a.base_model, "adapter": tag}

    if a.heldout is not None:
        t0 = time.time()
        loss, ntok = heldout_loss(model, tok, a.heldout, a.device)
        payload["heldout"] = {"ce_loss": loss, "ppl": math.exp(min(loss, 40)),
                              "n_tokens": ntok, "secs": round(time.time() - t0, 1)}
        print("heldout:", json.dumps(payload["heldout"]))

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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
