"""Held-out cross-entropy is the mean over exactly the scored tokens.

Why this exists
---------------
``eval_heldout.py`` weighted each batch's mean loss by
``(labels != -100).sum()``. A causal LM never predicts the first token of an
example, so that count included one unscored token per example. The result
was not the token-mean CE, it moved with ``--batch-size``, and the reported
``n_tokens`` overstated the scored count by exactly the number of examples.

The reference here is computed independently, one example at a time with no
padding, from the logits. The model is a random one-layer Llama over the real
SmolLM2 vocabulary: what is checked is the bookkeeping, which does not depend
on the weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from vedic.eval.scan_data import load_scan, record_text

REPO = Path(__file__).resolve().parents[3]


def _heldout_loss():
    sys.path.insert(0, str(REPO))
    from scripts.eval_heldout import heldout_loss
    return heldout_loss


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Seven records of very different lengths, so padding is uneven."""
    pairs = load_scan("length", "test")
    pick = [pairs[i] for i in (0, 1, 7, 100, 1000, 2000, 3919)]
    path = tmp_path_factory.mktemp("heldout") / "heldout.jsonl"
    path.write_text("".join(json.dumps({"text": record_text(c, a)}) + "\n"
                            for c, a in pick), encoding="ascii")
    return path, [record_text(c, a) for c, a in pick]


@pytest.fixture(scope="module")
def model(smollm2_tokenizer):
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=len(smollm2_tokenizer), hidden_size=8, intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
        max_position_embeddings=512)).eval()


def _reference(model, tok, texts):
    tot, n = 0.0, 0
    with torch.no_grad():
        for text in texts:
            ids = torch.tensor([tok(text)["input_ids"]])
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            nll = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
            tot += float(nll.double().sum())
            n += ids.size(1) - 1
    return tot / n, n


@pytest.mark.parametrize("bs", [1, 3, 7])
def test_the_scored_token_count_excludes_each_first_token(
        model, smollm2_tokenizer, corpus, bs):
    path, texts = corpus
    _, ntok, n_examples, max_seen = _heldout_loss()(model, smollm2_tokenizer, path, "cpu", bs=bs)
    lengths = [len(smollm2_tokenizer(t)["input_ids"]) for t in texts]
    assert n_examples == 7
    assert max_seen == max(lengths)
    assert ntok == sum(n - 1 for n in lengths)


def test_unpadded_ce_equals_the_per_example_reference(model, smollm2_tokenizer, corpus):
    """At batch size 1 there is no padding, so the same operations in the same
    order must give the same float, bit for bit."""
    path, texts = corpus
    ce, ntok, _, _ = _heldout_loss()(model, smollm2_tokenizer, path, "cpu", bs=1)
    ref_ce, ref_n = _reference(model, smollm2_tokenizer, texts)
    assert ntok == ref_n
    assert ce == ref_ce
