"""The exact-ℚ kernel, checked against the upstream JavaScript that defines it.

Why this exists
---------------
Falsification criterion 3 asked whether this package's 29 operators agree with
the upstream definition. Every document answered that the question could not be
reached, because `vedic_v18.24_full_kernel.html` "is external to this
repository" and "lives on the user's machine".

**It is tracked at the work-tree root, and has been all along.** The claim was
never true, and it did more than mislead: the path gate carried the file in its
`EXTERNAL` exemption list on the strength of it, and that entry short-circuits
resolution — so the one check that would have noticed never looked. An
exemption granted on a false premise stopped the premise being tested.

So the comparison is possible. This makes it, and it is the only check here
that can establish *correctness* rather than drift: `verify_bit_exact.py` says
in its own docstring that its canonical fixtures are written by the kernel they
are compared against, and therefore detect drift only.

How it works
------------
`scripts/extract_upstream_kernel.js` lifts `STRICT_SUTRA_KERNEL` and its
dependencies (`Q`, `VTX`, `ALPHA`, `SUTRA_KIND`) out of the HTML **by source
slicing** and evaluates them under node. Nothing is reimplemented: a
reimplementation would compare this package against itself, which is precisely
the weakness the fixture gate documents about its own reference.

Which upstream path is the definition
-------------------------------------
The HTML carries two. `SUTRAS[].evolve()` is the display path: it round-trips Ψ
through IEEE-754 via `Q.fl`, uses `Math.log10`/`floor`/`pow`, carries an
epsilon (`+1 // prevent division by zero`) and re-quantises to 1e-4 through
`Bi(x*10000)/10000n`. `STRICT_SUTRA_KERNEL` (line 6527) is the definition, is
float-free, and is what `sutras_canonical.py` says it ports. Comparing against
`evolve()` would be comparing against the wrong half of the file.

No skips
--------
node is required, exactly as `lean` is required for the Lean mirror. A missing
interpreter fails here rather than turning into a green run — the Lean mirror
sat broken for months behind a `skipif` and this file will not repeat it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from vedic.kernel.sutras_canonical import apply_sutra
from vedic.kernel.tests.psi_corpus import BY_LABEL, DYADIC, PSI_CASES

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "scripts" / "extract_upstream_kernel.js"
UPSTREAM_HTML = REPO.parent / "vedic_v18.24_full_kernel.html"

#: Upstream reduces strength with ``BigInt(Math.round(strength))`` before
#: forming α, so it can only be compared on integers. The non-integer case is
#: a real difference and is pinned separately below rather than avoided.
INTEGER_STRENGTHS = (0, 1, 50, 100, 250)

CASES = tuple(PSI_CASES) + tuple(DYADIC)


def test_the_upstream_definition_is_in_this_repository() -> None:
    """The premise every other test here rests on, and the one the docs got wrong."""
    assert UPSTREAM_HTML.is_file(), (
        f"{UPSTREAM_HTML} is missing. Several documents used to call this file "
        f"external to the repository; it is tracked at the work-tree root. If "
        f"it has genuinely been removed, this comparison is no longer possible "
        f"and criterion 3 goes back to unanswered — say so rather than deleting "
        f"this file.")
    tracked = subprocess.run(
        ["git", "-C", str(REPO.parent), "ls-files", UPSTREAM_HTML.name],
        capture_output=True, text=True, check=True).stdout.split()
    assert tracked, f"{UPSTREAM_HTML.name} is present but untracked — a fresh clone would not get it"


def test_node_is_available() -> None:
    """Required, not optional. See the module docstring on skips."""
    assert shutil.which("node"), (
        "node is not installed. It runs the upstream kernel, which is the only "
        "check in this package that establishes correctness rather than drift. "
        "Install it rather than skipping this file.")


def _run_upstream(requests: list[dict]) -> list[list[tuple[str, str]]]:
    payload = REPO / "fixtures" / ".upstream_request.json"
    payload.write_text(json.dumps(requests), encoding="utf-8")
    try:
        out = subprocess.run(["node", str(HARNESS), str(payload)],
                             capture_output=True, text=True, check=True).stdout
    finally:
        payload.unlink(missing_ok=True)
    return json.loads(out)


def _q(pair) -> Fraction:
    return Fraction(int(pair[0]), int(pair[1]))


@pytest.fixture(scope="module")
def upstream() -> dict:
    """Every (Ψ, strength, sutra) triple, evaluated once by the real kernel."""
    reqs, keys = [], []
    for label, psi in CASES:
        for s in INTEGER_STRENGTHS:
            for sid in range(1, 30):
                reqs.append({"id": sid, "strength": s,
                             "psi": [[str(x.numerator), str(x.denominator)] for x in psi]})
                keys.append((label, s, sid))
    return dict(zip(keys, _run_upstream(reqs)))


def test_the_comparison_covers_the_whole_corpus(upstream) -> None:
    """Guards the parametrised checks: a short harness run would pass vacuously."""
    expected = len(CASES) * len(INTEGER_STRENGTHS) * 29
    assert len(upstream) == expected, \
        f"upstream returned {len(upstream)} results, expected {expected}"
    assert expected > 6000


@pytest.mark.parametrize("sid", range(1, 30))
def test_operator_matches_upstream_on_every_input(sid: int, upstream) -> None:
    """Exact rational equality — not a tolerance, on either side."""
    wrong = []
    for label, psi in CASES:
        for s in INTEGER_STRENGTHS:
            up = tuple(_q(p) for p in upstream[(label, s, sid)])
            got = tuple(apply_sutra(sid, psi, Fraction(s)))
            if up != got:
                i = next(k for k in range(16) if up[k] != got[k])
                wrong.append(f"{label}@{s}: vertex {i} upstream {up[i]} != port {got[i]}")
    assert not wrong, (
        f"S{sid} disagrees with the upstream STRICT_SUTRA_KERNEL on "
        f"{len(wrong)} input(s):\n  " + "\n  ".join(wrong[:6]))


def test_upstream_rounds_a_fractional_strength_and_this_kernel_does_not() -> None:
    """The one real difference between the two, pinned rather than hidden.

    `ALPHA.computeQ` opens with `BigInt(Math.round(strength))`, so upstream
    cannot represent a fractional strength at all: 7/3 becomes 2. This kernel
    carries it exactly, which is what α(n) = (n/435)·(strength/100) actually
    says. The port is the more faithful of the two to the stated formula, and
    that is why the parametrised comparison above is restricted to integers —
    a restriction with a reason, not a convenience.
    """
    psi = BY_LABEL["monotone_square_over_7"]
    req = [{"id": 1, "strength": 7 / 3,
            "psi": [[str(x.numerator), str(x.denominator)] for x in psi]}]
    up = tuple(_q(p) for p in _run_upstream(req)[0])

    assert up != tuple(apply_sutra(1, psi, Fraction(7, 3))), \
        "upstream now represents a fractional strength; the restriction above can be lifted"
    assert up == tuple(apply_sutra(1, psi, Fraction(2))), \
        "upstream no longer rounds strength to the nearest integer"
