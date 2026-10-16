"""The vendored SCAN files, their checks, and the one text format.

Why this exists
---------------
The evaluator loaded SCAN with ``load_dataset("scan", ...,
trust_remote_code=True)``. ``scan`` on the Hub is a loading script, and
``datasets`` >= 4 cannot run loading scripts, so under the installed 5.1 no
SCAN number could be produced at all. And no code converted SCAN into training
records, so the prompt, answer and terminator the scorer depends on were
stated nowhere.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from vedic.eval import scan_data
from vedic.eval.scan_data import (
    SCAN_FILES,
    TERMINATOR,
    gold_continuation,
    load_scan,
    prompt,
    record_text,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("split,part", sorted(SCAN_FILES))
def test_every_vendored_file_is_the_published_one(split, part):
    """Hash and count, recomputed here from the archive bytes."""
    path = scan_data.vendored_path(split, part)
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == SCAN_FILES[(split, part)].sha256
    pairs = load_scan(split, part)
    assert len(pairs) == SCAN_FILES[(split, part)].n_examples
    assert raw.count(b"\n") == len(pairs)


def test_the_published_split_sizes():
    """From the SCAN release, not from this code: the test split of
    addprim_jump is 7,706, which this repository used to misstate as 3,920."""
    sizes = {k: f.n_examples for k, f in SCAN_FILES.items()}
    assert sizes == {
        ("simple", "train"): 16728, ("simple", "test"): 4182,
        ("length", "train"): 16990, ("length", "test"): 3920,
        ("addprim_jump", "train"): 14670, ("addprim_jump", "test"): 7706,
    }


def test_the_licence_travels_with_the_data():
    text = (scan_data.DATA_DIR / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("BSD License")
    assert "Redistributions in binary form must reproduce" in text


def test_a_modified_file_is_refused(tmp_path, monkeypatch):
    """One changed byte in one example, and the loader must refuse it."""
    for f in scan_data.DATA_DIR.iterdir():
        shutil.copy(f, tmp_path / f.name)
    target = tmp_path / "tasks_test_length.txt.gz"
    raw = gzip.decompress(target.read_bytes())
    tampered = raw.replace(b"I_JUMP", b"I_LOOK", 1)
    assert tampered != raw
    target.write_bytes(gzip.compress(tampered))
    monkeypatch.setattr(scan_data, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="not the published"):
        load_scan("length", "test")
    load_scan("length", "train")                # an untouched file still loads


def test_the_first_test_examples_of_the_length_split():
    """Pinned from the upstream file, read by eye: the parse is not shifted."""
    pairs = load_scan("length", "test")
    assert pairs[0] == (
        "run around left twice and run around right",
        " ".join(["I_TURN_LEFT", "I_RUN"] * 8 + ["I_TURN_RIGHT", "I_RUN"] * 4))
    assert pairs[1][0] == "look around left twice and jump around right"


def test_the_record_format_is_prompt_then_gold_then_newline():
    assert prompt("jump twice") == "Command: jump twice\nActions:"
    assert gold_continuation("I_JUMP I_JUMP") == " I_JUMP I_JUMP\n"
    assert record_text("jump twice", "I_JUMP I_JUMP") == \
        "Command: jump twice\nActions: I_JUMP I_JUMP\n"
    assert TERMINATOR == "\n"


def _builder():
    sys.path.insert(0, str(REPO))
    from scripts import build_scan_corpus
    return build_scan_corpus


def test_the_corpus_builder_is_byte_for_byte_reproducible(tmp_path):
    """The length-split corpus the experiment trains and evaluates on, pinned
    by hash. A change to the format, the data or the builder moves these."""
    out = _builder().build("length", tmp_path)
    digests = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in out.items()}
    assert digests == {
        "train": "0c6f7628058cc2207881373d7d20494a5005111cb255ab4f6678bcf82308a8cb",
        "test": "c61ff25a2f308d72becceb54303bfeef307920686ab9a33f566e58963ccaf874",
        "monitor": "522d01cd03a330d7b4b70fc49c04bcbd86854dc6011140341075a6c847aa3d8d",
    }


def test_every_corpus_line_is_a_record_of_its_example(tmp_path):
    out = _builder().build("length", tmp_path)
    for part, path in (("train", out["train"]), ("test", out["test"])):
        lines = path.read_text(encoding="ascii").splitlines()
        pairs = load_scan("length", part)
        assert len(lines) == len(pairs)
        for line, (c, a) in zip(lines, pairs):
            assert json.loads(line) == {"text": record_text(c, a)}
    monitor = out["monitor"].read_text(encoding="ascii").splitlines()
    assert monitor == out["train"].read_text(encoding="ascii").splitlines()[:256]
