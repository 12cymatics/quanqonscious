"""Every file path named in the documentation must exist.

Why this exists
---------------
The README pointed at `vedic/kernel/sutras_exact.py` for months after that
module was renamed to `z2_primitives.py`. Nothing failed, because nothing
reads the README. A reader following the pointer finds nothing and has no
way to tell whether the file moved or the feature was never there.

Renames are the common case and they are silent by nature: the code keeps
working, so only the prose breaks. This test makes a stale pointer a test
failure at the moment of the rename.

External files -- things that live on the user's machine rather than in this
repository -- are declared below. A path is exempt only by appearing in that
list, never by failing to be found.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = ("README.md", "ABLATION_RESULTS.md")

# Paths deliberately named but not present: they belong to the user, not here.
EXTERNAL = {
    "vedic_v18.16_strict_kernel.html",   # the user's JS reference simulator
    "vedic_v18.24_full_kernel.html",     # ditto, the later revision
}

_PATH = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|yaml|yml|"
                   r"jsonl|sh|html|txt|md|hpp|cpp))`")


def _cited(doc: str) -> set[str]:
    return set(_PATH.findall((REPO / doc).read_text(encoding="utf-8")))


ALL = sorted({(d, p) for d in DOCS for p in _cited(d)})


def test_the_documents_cite_some_paths():
    """Guards the regex: a pattern that matches nothing passes vacuously."""
    assert len(ALL) > 10, f"only {len(ALL)} paths found — the regex is wrong"


@pytest.mark.parametrize("doc,path", ALL, ids=[f"{d}:{p}" for d, p in ALL])
def test_documented_path_exists(doc: str, path: str):
    if path in EXTERNAL:
        return
    assert (REPO / path).exists(), (
        f"{doc} points at {path}, which does not exist. Either the file was "
        f"renamed and the document was not updated, or the path belongs in "
        f"EXTERNAL in this test with a note saying whose machine it is on.")


def test_external_list_has_no_dead_entries():
    """An exemption for a path nobody cites any more is a lie waiting to be
    reused. If the citation is gone, so is the exemption."""
    cited = {p for _, p in ALL}
    stale = EXTERNAL - cited
    assert not stale, f"EXTERNAL exempts paths no document mentions: {sorted(stale)}"


def test_no_external_entry_actually_exists_in_the_repo():
    """If an 'external' file turns up in the repo, the exemption is wrong."""
    present = {p for p in EXTERNAL if (REPO / p).exists()}
    assert not present, (
        f"{sorted(present)} are marked external but exist here — remove the "
        f"exemption so they are checked like everything else")
