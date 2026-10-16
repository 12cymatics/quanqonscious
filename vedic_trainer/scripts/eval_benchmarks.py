"""Full SCAN or COGS exact-match evaluation. No subsetting.

Every named split is evaluated in full: SCAN simple/length/addprim_jump, COGS
test/gen. There is no --subset and no --skip: a truncated benchmark is not the
benchmark, and a flag that shortens it is how a partial result gets reported
as a complete one. ``--split`` chooses *which* splits run, never how much of
one; each record carries ``n_total``, which is checked against the published
size by ``vedic/eval/tests/test_no_subset_is_quoted_as_a_benchmark.py``.

One benchmark per invocation. This used to run SCAN and then COGS and write
``--output`` only after both finished, so a COGS failure -- ``COGS_DIR``
unset, say -- discarded a SCAN pass that had taken hours. The record is now
rewritten after every split, so what finished is kept.

COST. Greedy decoding, one example at a time. SCAN's test splits are 4,182 +
3,920 + 7,706 examples and COGS's 3,000 + 21,000 (39,808 in all). Decoding
stops once each verdict is decided (``vedic/eval/first_line.py``), so a wrong
answer costs only the tokens up to where it goes wrong.

SCAN is read from the copy vendored in ``vedic/eval/data/scan``. COGS is not
on the HF Hub: set ``COGS_DIR`` to the ``data/`` directory of najoungkim/COGS,
which holds ``test.tsv`` and ``gen.tsv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic.eval import COGS_SPLITS, SCAN_SPLITS, evaluate_cogs, evaluate_scan  # noqa: E402
from vedic.provenance import environment, sha256_file  # noqa: E402

BENCHMARKS = {"scan": (SCAN_SPLITS, evaluate_scan),
              "cogs": (COGS_SPLITS, evaluate_cogs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--split", action="append", required=True, dest="splits",
                    help="a whole split to evaluate; repeat for several")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--revision", required=True,
                    help="full Hub commit of --base-model")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="omit to evaluate the untuned base model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-threads", type=int, required=True,
                    help="torch intra-op threads; float reductions depend on it, "
                         "so it is set and recorded rather than inherited")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.num_threads)

    known, evaluate = BENCHMARKS[a.benchmark]
    for split in a.splits:
        if split not in known:
            ap.error(f"{a.benchmark} has no split {split!r}; known: {known}")
    if len(set(a.splits)) != len(a.splits):
        ap.error("a split is named twice")

    tok = AutoTokenizer.from_pretrained(a.base_model, revision=a.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base_model, revision=a.revision,
                                                 dtype=torch.float32)
    tag = "base"
    adapter_sha256 = None
    if a.adapter is not None:
        model = PeftModel.from_pretrained(model, str(a.adapter))
        tag = str(a.adapter)
        adapter_sha256 = {p.name: sha256_file(p)
                          for p in sorted(a.adapter.iterdir()) if p.is_file()}
    model = model.to(a.device)

    payload = {"model": a.base_model, "revision": a.revision, "adapter": tag,
               "adapter_sha256": adapter_sha256, "environment": environment(),
               a.benchmark: {}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        t0 = time.time()
        r = evaluate(model, tok, splits=[split], device=a.device)[split]
        payload[a.benchmark][split] = {
            "n_total": r.n_total, "n_correct": r.n_correct,
            "accuracy": r.accuracy, "max_new_tokens": r.max_new_tokens,
            "n_budget_exhausted": r.n_budget_exhausted,
            "secs": round(time.time() - t0, 1)}
        a.output.write_text(json.dumps(payload, indent=2))
        print(f"{a.benchmark}.{split}:", json.dumps(payload[a.benchmark][split]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
