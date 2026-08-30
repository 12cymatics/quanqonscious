"""Every file path named in this repository's documentation must resolve.

Why this exists
---------------
`vedic_trainer/` has had a gate like this for a while, and it caught real
defects the moment it was widened past its original two-file list. Everything
*outside* that package — 29 tracked Markdown files, roughly 11,900 lines,
including the two instruction files a future agent reads first — had no gate
at all.

When one was finally run over them it found eight dead pointers, and one of
them mattered: `agent.md` documented a "standardised" repository layout that
does not exist, and closed with *"Directories listed above must exist; the
agent auto-creates missing paths."* That directive had already run. It left
eight directories whose entire tracked contents are a `.gitkeep` — including
`src/julia/` and `src/verilog/`, into which no Julia or Verilog has ever been
written. A wish in the voice of fact, plus an instruction to make the
filesystem agree, produces scaffolding that makes the wish look satisfied.

A dead pointer is cheap to write and expensive to follow. This makes writing
one a test failure.

Running it
----------
    python -m pytest tests/test_documented_paths.py
    python tests/test_documented_paths.py        # same checks, no pytest

The second form matches `tests/test_invariants.py`, which is how this
repository has always been run.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), *args],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


#: Every tracked Markdown file outside `vedic_trainer/`, which has its own
#: gate (`vedic_trainer/vedic/kernel/tests/test_documented_paths.py`) and
#: whose docs would otherwise be checked twice under different rules.
#: Discovered, never enumerated: a hand-written list is how the trainer's
#: version came to cover two files out of eight.
DOCS: tuple[str, ...] = tuple(
    d for d in sorted(_git("ls-files", "*.md")) if not d.startswith("vedic_trainer/"))

#: Paths named *because they do not exist*. Each is asserted below to be both
#: cited and genuinely absent, so this list cannot quietly absorb a rename.
#: A document that names a thing it does not have is doing the right thing —
#: what it must not do is let the reader mistake that for a pointer.
ASPIRATIONAL = {
    # CLAUDE.md prints these as the shape a PCFE config takes. No `config/`
    # directory exists at the repository root; the real ones are under
    # `pcfe-v3/config/` and `vedic_trainer/configs/`.
    "config/minimal.yaml",
    "config/production.yaml",
    # README_BUNDLE.md describes a separate local working bundle. None of it
    # is in this repository, which that document now says in its second
    # paragraph.
    "requirements-local-mac.txt",
    "requirements-train.txt",
    "scripts/run_gpu_qlora.sh",
    # agent.md §7 describes a CI workflow that was never built. `ci/` holds
    # only `.gitkeep` files.
    "ci/linux_gpu.yml",
    # agent.md §5.2 sets a Julia convention against a test file in a language
    # the repository does not contain. Same aspiration as `src/julia/`, which
    # holds one `.gitkeep`.
    "test/runtests.jl",
}

_PATH = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|json|yaml|yml|jsonl|"
                   r"sh|html|txt|md|hpp|cpp|ipynb|toml|cfg|jl|adoc))`")

TRACKED = frozenset(_git("ls-files"))
DIRS = set()
for _t in TRACKED:
    _q = Path(_t).parent
    while str(_q) != ".":
        DIRS.add(str(_q))
        _q = _q.parent
KNOWN = TRACKED | DIRS

BY_NAME: dict[str, list[str]] = {}
for _t in TRACKED:
    BY_NAME.setdefault(Path(_t).name, []).append(_t)


def cited(doc: str) -> set[str]:
    return set(_PATH.findall((REPO / doc).read_text(encoding="utf-8", errors="replace")))


def resolves(citation: str, doc: str) -> bool:
    """Would a reader following this pointer land on a tracked file?

    Three forms, all of which appear in these documents: repository-relative,
    relative to the citing document, and a bare filename in prose — which is
    a live pointer exactly when one tracked file carries that name, and
    ambiguous otherwise. Checking only the first would report most of the
    live pointers here as broken and bury the real ones.
    """
    if citation in KNOWN:
        return True
    if (Path(doc).parent / citation).as_posix() in KNOWN:
        return True
    return len(BY_NAME.get(Path(citation).name, ())) == 1


def is_live(citation: str, doc: str) -> bool:
    """The whole decision, in one pure function — see the trainer's gate for
    what happens when the tested helper and the called helper drift apart."""
    return citation in ASPIRATIONAL or resolves(citation, doc)


def dead_pointers() -> list[str]:
    """Every citation in every document that resolves to nothing."""
    return sorted(f"{d}: {c}" for d in DOCS for c in cited(d) if not is_live(c, d))


# --------------------------------------------------------------- the checks

def test_there_are_documents_and_citations_to_check():
    """Guards the rest: a regex matching nothing passes vacuously."""
    assert len(DOCS) >= 20, f"only {len(DOCS)} documents found — the glob is wrong"
    total = sum(len(cited(d)) for d in DOCS)
    assert total >= 50, f"only {total} citations found — the pattern is wrong"


def test_no_document_points_at_something_that_is_not_there():
    dead = dead_pointers()
    assert not dead, (
        "documented paths that resolve to nothing:\n  " + "\n  ".join(dead)
        + "\n\nEither the file was renamed and the document was not updated, "
          "or the path is named because it does not exist — in which case add "
          "it to ASPIRATIONAL here, with the reason.")


def test_every_aspirational_entry_is_still_cited():
    """An exemption nobody uses is one waiting to be reused for something it
    was not written for."""
    everything = {c for d in DOCS for c in cited(d)}
    stale = ASPIRATIONAL - everything
    assert not stale, f"ASPIRATIONAL exempts paths no document mentions: {sorted(stale)}"


def test_no_aspirational_entry_actually_exists():
    """The exemption says these are absent. If one appears, the document
    calling it absent is now wrong — and this list would otherwise hide that,
    which is exactly the defect the file exists to catch."""
    present = sorted(p for p in ASPIRATIONAL if p in KNOWN or (REPO / p).exists())
    assert not present, (
        f"{present} are listed as not existing but are present. Update the "
        f"document that describes them as absent, then drop them from here.")


# ---------------------------------------------------------------------------
# Line-range citations
# ---------------------------------------------------------------------------
#
# A citation of the form `path.py:START-END` makes a claim the path check
# above cannot see: that the named construct is at those lines. Nothing
# checked it, and every such citation in the repository had drifted --
# ALL_29_VEDIC_SUTRAS.md's 29 were exact at the commit that added them and
# then moved +3 to +52 lines as the file changed beneath them, and
# COMPLETE_SUTRA_DEFINITIONS_SUPERIOR.md's five MSTVQ ranges matched no
# revision in that file's history at all, tiling it into contiguous blocks
# that ended exactly on the claimed file length.
#
# Checking that START merely lands on some `class`/`def` is not enough: one
# of those five cited line 412 for MSTVQCouplingOperator, and 412 is a class
# line -- it is `MSTVQCompositeOperator`. So where the document names a
# construct in the heading above the citation, the definition at START must
# be that one.

CITATION = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:py|hpp|cpp|lean|js)):(\d+)-(\d+)`"
)
DEFINITION = re.compile(r"^(?:class|def)\s+(\w+)")


def _cited_ranges(doc: str):
    """Yield (path, start, end, heading) for each line-range citation."""
    text = (REPO / doc).read_text(errors="replace")
    heading = ""
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            heading = line
        for path, start, end in CITATION.findall(line):
            yield path, int(start), int(end), heading


def test_line_range_citations_point_at_what_they_name():
    """`path.py:START-END` must start on a definition, and on the right one."""
    problems = []
    checked = 0

    for doc in DOCS:
        for path, start, end, heading in _cited_ranges(doc):
            target = REPO / path
            if not target.exists():
                continue          # the path check above owns this case
            lines = target.read_text(errors="replace").splitlines()

            if not 1 <= start <= end <= len(lines):
                problems.append(
                    f"{doc}: {path}:{start}-{end} is outside the file "
                    f"({len(lines)} lines)")
                continue

            checked += 1
            match = DEFINITION.match(lines[start - 1])
            if match is None:
                problems.append(
                    f"{doc}: {path}:{start} is not a definition, it is "
                    f"{lines[start - 1].strip()[:60]!r}")
                continue

            # If the heading names something defined in that file, the
            # definition at START has to be it.
            named = [w for w in re.findall(r"\w+", heading)
                     if re.search(rf"^(?:class|def)\s+{re.escape(w)}\b",
                                  "\n".join(lines), re.M)]
            if named and match.group(1) not in named:
                problems.append(
                    f"{doc}: {path}:{start} defines {match.group(1)}, but the "
                    f"heading names {' / '.join(named)}")

    assert checked, "no line-range citations were checked; the regex is wrong"
    assert not problems, (
        f"{len(problems)} line-range citation(s) do not point at what they name:\n  "
        + "\n  ".join(problems)
        + "\n\nRecompute the range from the file rather than adjusting it by eye."
    )

def test_the_tracked_set_was_read():
    """An empty TRACKED would fail everything above for the wrong reason."""
    assert len(TRACKED) > 100, f"git ls-files returned {len(TRACKED)} paths"
    assert "CLAUDE.md" in TRACKED


def test_the_gate_covers_the_instruction_files():
    """These are what an agent reads before touching anything, so a dead
    pointer in one propagates into work rather than just misleading a reader."""
    for name in ("CLAUDE.md", "AGENTS.md", "agent.md", "README.md"):
        assert name in DOCS, f"{name} is outside this gate"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ✓ {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  ✗ {name}\n      {exc}")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
