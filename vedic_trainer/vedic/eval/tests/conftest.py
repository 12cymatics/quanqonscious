"""The real tokenizer the experiment uses, at the revision its config pins.

The format and budget checks run against the SmolLM2 tokenizer itself, not a
stand-in: whether the prompt's tokens are a prefix of the record's, and how
many tokens the longest answer takes, are properties of that tokenizer. CI
downloads it at the pinned revision; a run without network access fails here
rather than passing on a substitute.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vedic.training.config import load_yaml

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG = REPO / "configs" / "scan_length" / "no_sutra_seed0.yaml"


@pytest.fixture(scope="session")
def smollm2_tokenizer():
    from transformers import AutoTokenizer

    cfg = load_yaml(EXPERIMENT_CONFIG)
    tok = AutoTokenizer.from_pretrained(cfg.base_model,
                                        revision=cfg.base_model_revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
