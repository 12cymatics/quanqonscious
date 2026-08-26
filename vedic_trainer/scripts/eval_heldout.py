"""Held-out cross-entropy on the synthetic eval split. Runs in full.

This is a *different evaluation* from scripts/eval_benchmarks.py, not a
faster mode of it. Neither script has a flag that skips work: a run either
completes or fails.

Truncation
----------
Examples longer than ``--max-len`` tokens are truncated to that length, and
the count of truncated examples is written into the results JSON as
``heldout.n_truncated`` alongside ``max_len`` and ``max_tokens_seen``. It is
not a silent operation: a run that truncated anything says so on stdout and
carries the count in its record.

Truncating is reported rather than raised on, because a held-out CE over a
truncated prefix is still a well-defined number and both arms of the
comparison this script exists for (base vs. adapter) truncate identically, so
the comparison stays sound. What was not sound was leaving it unrecorded --
``truncation=True`` under a docstring promising no skipped work meant a
corpus of long documents could have most of its tokens dropped and the
resulting CE would look like a full-corpus measurement. Raising instead would
make a legitimate long-document corpus simply unevaluable; recording keeps
the measurement available and makes its scope impossible to miss.
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
    """Return (mean CE per token, n scored tokens, n truncated, max tokens seen).

    Tokenises without truncation so the true length of every example is known,
    then cuts to ``max_len`` and counts the cuts. The previous form passed
    ``truncation=True`` straight to the tokenizer, which discards exactly the
    information needed to report how much was dropped.
    """
    ds = load_dataset("json", data_files=str(path), split="train")

    n_truncated = 0
    max_tokens_seen = 0

    def _encode(batch):
        nonlocal n_truncated, max_tokens_seen
        full = tok(batch["text"], truncation=False, padding=False)
        kept = []
        for seq in full["input_ids"]:
            if len(seq) > max_len:
                n_truncated += 1
            if len(seq) > max_tokens_seen:
                max_tokens_seen = len(seq)
            kept.append(seq[:max_len])
        return {"input_ids": kept,
                "attention_mask": [[1] * len(s) for s in kept]}

    # load_from_cache_file=False is load-bearing: a cached map would skip
    # _encode entirely and leave the counters at zero, reporting "nothing
    # truncated" for a run that truncated plenty.
    ds = ds.map(_encode, batched=True, remove_columns=ds.column_names,
                load_from_cache_file=False)
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
    return tot / ntok, ntok, n_truncated, max_tokens_seen


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
    ap.add_argument("--max-len", type=int, default=512,
                    help="token cap per example; truncated examples are "
                         "counted into heldout.n_truncated (default 512)")
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
    loss, ntok, n_trunc, max_seen = heldout_loss(
        model, tok, a.heldout, a.device, max_len=a.max_len)
    payload = {"model": a.base_model, "adapter": tag,
               "heldout": {"ce_loss": loss, "ppl": perplexity(loss),
                           "n_tokens": ntok, "max_len": a.max_len,
                           "n_truncated": n_trunc,
                           "max_tokens_seen": max_seen,
                           "secs": round(time.time() - t0, 1)}}
    if n_trunc:
        print(f"NOTE: {n_trunc} example(s) exceeded --max-len {a.max_len} and "
              f"were truncated; longest was {max_seen} tokens. The CE below "
              f"is over the truncated prefixes, and n_truncated records this "
              f"in the results file.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2))
    print("heldout:", json.dumps(payload["heldout"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
