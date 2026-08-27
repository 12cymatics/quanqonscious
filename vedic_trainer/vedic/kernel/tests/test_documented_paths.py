"""Every file path named in the documentation must exist.

Why this exists
---------------
The README pointed at `vedic/kernel/sutras_exact.py` for months after that
module was renamed to `z2_primitives.py`. Nothing failed, because nothing
reads the README. A reader following the pointer finds nothing and has no
way to tell whether the file moved or the feature was never there.

**And this test then reproduced the same defect at one level up.** ``DOCS``
was a hand-written two-element tuple, ``("README.md", "ABLATION_RESULTS.md")``.
Everything under ``docs/`` was outside it, so those files drifted for the whole
life of the project while the two gated ones were corrected repeatedly. When
the list was finally widened, three documents were still pointing at
``vedic/kernel/sutras_exact.py`` -- the very path in the paragraph above, in
the very file written to stop it. A gate that covers a hand-listed subset
reports on the subset and reads as covering the whole.

``DOCS`` is therefore **discovered, not enumerated**: every Markdown file git
tracks. A new document is covered by existing, not by remembering.

Renames are the common case and they are silent by nature: the code keeps
working, so only the prose breaks. This test makes a stale pointer a test
failure at the moment of the rename.

Two kinds of path are exempt, and only by being declared below -- never by
failing to be found. ``EXTERNAL`` is for files that live on the user's
machine rather than in this repository. ``REMOVED`` is for files a document
names *because they are gone*: a withdrawal notice that cannot say what it
removed is not a withdrawal. Both lists are checked in both directions --
an entry nobody cites is deleted, an "external" file that turns up here loses
its exemption, and a "removed" file that comes back loses its exemption too,
so neither list can quietly cover a real rename.

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


def _git(*args: str, cwd: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(cwd), *args],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


#: The git work-tree root. Documents here cite files that live above this
#: package -- the CI workflows under `.github/`, the sibling modules the
#: archived reference material came from -- and a reader follows those
#: pointers in the repository, not in this subdirectory. Resolving against
#: the package alone would report them all dead.
GIT_ROOT = Path(_git("rev-parse", "--show-toplevel", cwd=REPO)[0])

#: Every Markdown file git tracks in this package. Discovered, never listed:
#: see the module docstring for what a hand-listed DOCS cost.
DOCS: tuple[str, ...] = tuple(sorted(_git("ls-files", "*.md", cwd=REPO)))

# Paths deliberately named but not present: they belong to the user, not here.
EXTERNAL = {
    "vedic_v18.16_strict_kernel.html",   # the user's JS reference simulator
    "vedic_v18.24_full_kernel.html",     # ditto, the later revision
}

# Paths named *because they no longer exist*. Two kinds of document need to
# do this. ABLATION_RESULTS.md withdraws the figures measured on the synthetic
# corpus and has to name the pipeline it withdrew with them; a reader cannot
# check a withdrawal against a description that will not say what was removed.
# And a correction has to name what it corrected -- "this heading used to read
# X" is the sentence that lets a reader tell a fix from a rewrite.
#
# The gate cannot tell either of those from a live pointer, and should not
# try: backticks look the same both ways. Declaring them here is the
# mechanism, and it can only be abused in the direction of honesty, because
# every entry is asserted below to be both cited and absent. A rename cannot
# hide in this list.
REMOVED = {
    "data/seed_corpus.txt",                    # the 512 seed sentences
    "scripts/generate_synthetic.py",           # expanded them into 5,120 records
    "scripts/split_corpus.py",                 # partitioned them by source
    "vedic/data/tesseract_encode.py",          # the text -> Psi encoder
    "vedic/data/synthetic_contradiction.py",   # (P, not-P) pair generator
    "vedic/data/synthetic_paraphrase.py",      # axis-emphasis pair generator
    # Renamed to z2_primitives.py. Named by docs/SUTRA_CATALOGUE.md and
    # docs/BIT_EXACT_PROTOCOL.md in the sentences recording that they used
    # to point at it — which is how a reader tells a moved pointer from a
    # vanished feature. It is the defect this whole file was written for,
    # and it survived in three documents because DOCS listed two.
    "vedic/kernel/sutras_exact.py",
    "sutras_exact.py",
}

_PATH = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|yaml|yml|"
                   r"jsonl|sh|html|txt|md|hpp|cpp))`")


def _cited(doc: str) -> set[str]:
    return set(_PATH.findall((REPO / doc).read_text(encoding="utf-8")))


def _tracked(repo: Path) -> frozenset[str]:
    """Every path git tracks under `repo`, relative to it.

    Directories are included as well as files, so a document may cite a
    directory that contains tracked files.
    """
    paths: set[str] = set()
    for line in _git("ls-files", cwd=repo):
        paths.add(line)
        parent = Path(line).parent
        while str(parent) != ".":
            paths.add(str(parent))
            parent = parent.parent
    return frozenset(paths)


TRACKED = _tracked(REPO)
ROOT_TRACKED = _tracked(GIT_ROOT)

#: basename -> the tracked paths carrying it, across the whole work tree.
BY_NAME: dict[str, list[str]] = {}
for _p in ROOT_TRACKED:
    BY_NAME.setdefault(Path(_p).name, []).append(_p)


def resolves(citation: str, doc: str) -> bool:
    """Would a reader following this pointer land on a tracked file?

    Documents cite paths the way people write them, and all four of these
    forms appear in this repository:

    1. package-relative -- ``vedic/kernel/q.py``;
    2. work-tree-relative -- ``.github/workflows/submit-pypi.yml``, which is
       above this package and invisible to a package-only listing;
    3. relative to the citing document -- ``reference/foo.py`` inside
       ``docs/external/README.md``;
    4. a bare filename in prose -- ``losses.py`` -- which is a live pointer
       exactly when one tracked file carries that name and ambiguous
       otherwise.

    Only a citation that satisfies none of them is dead. Checking form 1
    alone would have reported forty live pointers as broken and buried the
    two real ones.

    Pure: takes no filesystem reading of its own, so
    ``test_gates_reject.py`` can exercise it on synthetic inputs.
    """
    if citation in TRACKED:
        return True
    if citation in ROOT_TRACKED:
        return True
    if f"vedic_trainer/{citation}" in ROOT_TRACKED:
        return True
    beside = (Path(doc).parent / citation).as_posix()
    if beside in TRACKED:
        return True
    return len(BY_NAME.get(Path(citation).name, ())) == 1


def is_live(citation: str, doc: str) -> bool:
    """The gate's whole decision, in one pure function.

    True when a reader following this citation lands somewhere real: it
    resolves to a tracked file, or it is declared in EXTERNAL (someone
    else's machine) or REMOVED (named because it is gone).

    This replaced a helper named ``missing()`` that took a pre-computed
    tracked set. When the resolution ladder was added, the gate stopped
    calling ``missing()`` and started calling ``resolves()`` -- but five
    regeneration tests in ``test_gates_reject.py`` went on exercising
    ``missing()``, which by then judged nothing. A tested function no
    caller uses is a decoration, and it reads in a coverage report exactly
    like a gate. One function makes the decision now, and it is the one the
    regeneration tests exercise.
    """
    return citation in EXTERNAL or citation in REMOVED or resolves(citation, doc)


ALL = sorted({(d, p) for d in DOCS for p in _cited(d)})


def test_the_documents_cite_some_paths():
    """Guards the regex: a pattern that matches nothing passes vacuously."""
    assert len(ALL) > 10, f"only {len(ALL)} paths found — the regex is wrong"


@pytest.mark.parametrize("doc,path", ALL, ids=[f"{d}:{p}" for d, p in ALL])
def test_documented_path_is_tracked(doc: str, path: str):
    assert is_live(path, doc), (
        f"{doc} points at {path}, which resolves to nothing git tracks"
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


def test_removed_list_has_no_dead_entries():
    """Same rule as EXTERNAL: an exemption nobody uses is one waiting to be
    reused for something it was not written for."""
    cited = {p for _, p in ALL}
    stale = REMOVED - cited
    assert not stale, f"REMOVED exempts paths no document mentions: {sorted(stale)}"


def test_no_removed_entry_is_back_in_the_repository():
    """The exemption says these are gone. If one returns, the document that
    calls it removed is now wrong, and this list would otherwise hide that —
    which is exactly the stale-pointer defect the whole file exists for."""
    back = sorted(p for p in REMOVED if p in TRACKED or (REPO / p).exists())
    assert not back, (
        f"{back} are listed as removed but are present again. Update the "
        f"document that describes them as gone, then drop them from REMOVED.")


def test_the_two_exemption_lists_are_disjoint():
    """A path cannot be both someone else's file and a file deleted here."""
    both = EXTERNAL & REMOVED
    assert not both, f"declared both external and removed: {sorted(both)}"
