# `.claude/agents/`

Three subagents, each encoding a discipline this repository has needed
repeatedly and that a general-purpose agent does not apply by default.

| agent | use it when |
|---|---|
| `claim-auditor` | a document, README, docstring or comment asserts a count, a line number, a timing, a speedup, a path, or "this command returns N" — and you are about to believe it |
| `regeneration-tester` | you are adding a gate, or you want to know whether an existing one can fail at all |
| `exactness-inspector` | code sits on an exact-arithmetic claim, or a header says "no approximations" |

## Why these three

Each was written from a defect class that actually occurred here, more than
once, and that ordinary review did not catch.

**`claim-auditor`.** `CLAUDE.md` published a verification recipe —
`grep -r "def.*sutra" primarysutra.py | wc -l  # Should be 29` — that returns
**1**. It was written to express the intent "there are 29", not transcribed
from a command anyone had run, so it handed every reader a check that fails
while telling them the answer it was supposed to give. In the same file all
seven line-count figures were wrong when measured (`primarysutra.py` listed at
"3800+" is 2,865; `pcfe_v3_core_engine.py` at "5000+" is 2,116;
`core/operators/base.py` described as "~100 lines" is 513). The rule that falls
out: **run the documented command verbatim, and measure every number.**

**`regeneration-tester`.** `tests/test_documented_paths.py` carried
`vedic_v18.24_full_kernel.html` in its `EXTERNAL` exemption list on the premise
that the file lived only on a developer's machine. The file is tracked at the
work-tree root. Worse, the guard written for exactly that mistake,
`test_no_external_entry_actually_exists_in_the_repo`, checked `REPO / p` where
`REPO` is the *package* directory — asking whether
`vedic_trainer/vedic_v18.24_…` existed rather than whether anything by that
name was tracked at all. A declared exemption short-circuits resolution, so the
one check that would have noticed never looked. The rule: **a gate you have not
seen go red under its own defect is not known to work.**

**`exactness-inspector`.** The Lean library proved 178 obligations with
`native_decide`, which discharges a goal by running the compiled program rather
than by kernel reduction — putting the Lean compiler into the trusted base for
212 of 2,052 theorems. It compiled green throughout. Elsewhere: a synthetic
encoder that asserted a *positive* feature when no marker was found, computed a
normalised tally and discarded it unread; `L_curv` hinged against
`kappa.detach().mean()`, a per-example loss shifted by a statistic of its own
batch; the cudaq probe now tracked as
`pcfe-v3/tests/check_cudaq_environment.py` opened with `importorskip("cudaq")`
so its five checks collapsed to `1 skipped` on every machine including CI. The
rule: **a skip reports as not-run and reads as covered**, and a green build is
not a proof.

## The shared standard

All three refuse the same move: concluding from the absence of evidence. A test
that passes, a build that is green, a document that reads well, and a number
that looks right are each consistent with the thing being broken. Every one of
these agents is required to produce the command it ran.

They are also all required to report what checked out **clean**. A report
containing only failures leaves the reader unable to distinguish audited-clean
from not-looked-at, which is the same defect one level up.

## Scope

`claim-auditor` and `exactness-inspector` have no write tools — they measure
and report, and the caller decides what to change. `regeneration-tester` needs
`Edit`/`Write` because injecting a defect is its method; it is required to
restore the tree and to say how it verified that.
