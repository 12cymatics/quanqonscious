"""Held-out cross-entropy on the synthetic eval split. Runs in full.

This is a *different evaluation* from scripts/eval_benchmarks.py, not a
faster mode of it. Neither script has a flag that skips work: a run either
completes or fails.

No truncation
-------------
Every token of every example is scored. There is no ``--max-len``, no cap,
and no token is dropped, so the reported cross-entropy is over the whole
corpus by construction rather than by a flag someone remembered to set.

The previous version capped examples at 512 tokens and *recorded* how many
it cut. Recording is better than hiding, but it still left a knob whose
default silently decided what the headline number covered, and it made every
result carry a scope caveat a reader had to check. The cap never bound on
this corpus -- the longest example in ``data/synthetic_train.jsonl``,
``synthetic_eval.jsonl`` and ``synthetic_all.jsonl`` is 14 tokens against
that 512 -- so removing it changes no recorded number and removes the
mechanism by which a future corpus could be cut.

Memory is bounded by batching, not by truncating: ``--batch-size`` controls
the footprint, and every example in every batch is scored whole. Batches are
formed in corpus order and padded to the longest member, so a corpus with
very uneven lengths pays in padding — which costs time, not correctness, and
is the right trade against dropping tokens from the measurement.
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


def heldout_loss(model, tok, path: Path, device: str, bs: int = 8):
    """Return (mean CE per scored token, n scored tokens, n examples, max tokens).

    Every example is tokenised in full and every token is scored. The token
    count is returned so the caller can record the exact size of what was
    measured; it is a description of the corpus, not a limit applied to it.
    """
    ds = load_dataset("json", data_files=str(path), split="train")

    max_tokens_seen = 0

    def _encode(batch):
        nonlocal max_tokens_seen
        # truncation=False, and no max_length: the tokenizer is given no
        # opportunity to drop anything.
        full = tok(batch["text"], truncation=False, padding=False)
        for seq in full["input_ids"]:
            if len(seq) > max_tokens_seen:
                max_tokens_seen = len(seq)
        return {"input_ids": full["input_ids"],
                "attention_mask": [[1] * len(s) for s in full["input_ids"]]}

    # load_from_cache_file=False is load-bearing: a cached map would skip
    # _encode entirely and leave max_tokens_seen at zero, so the record would
    # describe a corpus this run never actually read.
    ds = ds.map(_encode, batched=True, remove_columns=ds.column_names,
                load_from_cache_file=False)
    n_examples = len(ds)
    dl = DataLoader(ds, batch_size=bs,
                    collate_fn=DataCollatorForLanguageModeling(tok, mlm=False))
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            n = int((b["labels"] != -100).sum())
            tot += float(out.loss) * n
            ntok += n
    if ntok == 0:
        raise ValueError(f"{path} produced no scored tokens")
    if n_examples == 0:
        raise ValueError(f"{path} contains no examples")
    return tot / ntok, ntok, n_examples, max_tokens_seen


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
    ap.add_argument("--batch-size", type=int, default=8,
                    help="examples per forward pass; controls memory "
                         "footprint without discarding any data (default 8)")
    # There is deliberately no --max-len. A token cap decides what the
    # headline number covers, and a default cap decides it silently.
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
    loss, ntok, n_examples, max_seen = heldout_loss(
        model, tok, a.heldout, a.device, bs=a.batch_size)
    # n_examples and max_tokens_seen describe the corpus that was scored in
    # full; they are not a cap and nothing was dropped to satisfy them.
    payload = {"model": a.base_model, "adapter": tag,
               "heldout": {"ce_loss": loss, "ppl": perplexity(loss),
                           "n_tokens": ntok,
                           "n_examples": n_examples,
                           "max_tokens_seen": max_seen,
                           "truncated": False,
                           "secs": round(time.time() - t0, 1)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2))
    print("heldout:", json.dumps(payload["heldout"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
