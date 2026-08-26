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

Tracked, not present
--------------------
Existence is decided by ``git ls-files``, not by ``Path.exists()``. The two
differ exactly where it matters: ``data/*`` is gitignored, so a document
naming ``data/synthetic_eval.jsonl`` passed on any machine that had run the
generator and failed in CI, which checks out only tracked files. That is the
same environment-dependence ``verify_counts.py`` was fixed for -- a gate
whose verdict depends on the machine is not a gate.

A reader clones this repository. A path that reaches them only if they first
run something is not a pointer they can follow, so naming the generator is
the correct fix, not exempting the artifact.
"""
from __future__ import annotations

import re
import subprocess
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


def _tracked(repo: Path) -> frozenset[str]:
    """Every path git tracks, relative to `repo`.

    Directories are included as well as files, so a document may cite a
    directory that contains tracked files.
    """
    out = subprocess.run(["git", "-C", str(repo), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    paths: set[str] = set()
    for line in out.splitlines():
        paths.add(line)
        parent = Path(line).parent
        while str(parent) != ".":
            paths.add(str(parent))
            parent = parent.parent
    return frozenset(paths)


TRACKED = _tracked(REPO)


def missing(cited: set[str], tracked: frozenset[str],
            external: set[str]) -> list[str]:
    """Cited paths that git does not track and that are not declared external.

    Pure, and takes the tracked set rather than reading the filesystem, so
    it returns the same answer on a developer machine carrying generated
    artifacts as in a fresh clone. `test_gates_reject.py` uses it to prove
    this rejects a stale path without editing the real README -- a
    regeneration test that mutates the repo can leave it dirty if it fails
    midway.
    """
    return sorted(p for p in cited if p not in external and p not in tracked)


ALL = sorted({(d, p) for d in DOCS for p in _cited(d)})


def test_the_documents_cite_some_paths():
    """Guards the regex: a pattern that matches nothing passes vacuously."""
    assert len(ALL) > 10, f"only {len(ALL)} paths found — the regex is wrong"


@pytest.mark.parametrize("doc,path", ALL, ids=[f"{d}:{p}" for d, p in ALL])
def test_documented_path_is_tracked(doc: str, path: str):
    if path in EXTERNAL:
        return
    assert path in TRACKED, (
        f"{doc} points at {path}, which git does not track"
        + (" -- though it is present here, so this passes on your machine "
           "and fails in a fresh clone. It is a generated artifact: name the "
           "script that writes it instead."
           if (REPO / path).exists() else
           ". Either the file was renamed and the document was not updated, "
           "or the path belongs in EXTERNAL in this test with a note saying "
           "whose machine it is on."))


def test_the_tracked_set_was_read():
    """Guards the rest: an empty tracked set would fail everything, and a
    subprocess that quietly returned nothing would be indistinguishable from
    a repository with no files."""
    assert len(TRACKED) > 50, (
        f"git ls-files returned {len(TRACKED)} paths — the tracked set was "
        f"not read, so every path check above is meaningless")


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
