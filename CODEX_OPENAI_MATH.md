# CODEX — openai/math as the governing standard

**Status: binding on all work in this repository from 2026-10-15.** The
repository owner directed that [openai/math](https://github.com/openai/math)
be "the codex for everything you do". This file states what that means as
rules that can be checked, and links the file in openai/math each rule comes
from, so that the interpretation itself can be audited.

**Reference pin.** openai/math @ `1581d4cbdd69a0faf34b2089614fe2a814316a6a`
(committed 2026-10-12). Every link below is to that commit; the prefix
[`OAI@1581d4cb`](https://github.com/openai/math/tree/1581d4cbdd69a0faf34b2089614fe2a814316a6a)
marks them. If the reference is advanced, this pin changes in the same
commit as any rule that changes with it.

**Not to be confused with** the "CODEX X.Y" sections that `core/` comments
cite. That is a separate specification and no document for it exists in this
repository. This file governs *how results are stated, checked and recorded*;
it does not replace any `core/` invariant.

## What openai/math is

719 manuscripts in 372 families, produced by an internal OpenAI model and
released under Apache-2.0 ([README](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/README.md),
[LICENSE](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/LICENSE)).
About 42% of top-line results carry Lean 4 formalizations (300 / 719 per
[history](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/history.md)).
The repository says of the rest: *"Some of the unformalized results could
have issues"*, and it has withdrawn papers after errors were found (history,
2026-10-07: a sign error invalidated three manuscripts).

Its standing as a codex here is **methodological**: it is a large, public,
versioned body of machine-produced mathematics that separates what has been
formally checked from what has not, and says which is which. That is the
standard adopted. Its unformalized mathematical *claims* are not thereby
adopted as true.

## The rules

### C1. A formal result is a statement checked against a proof by Comparator

Every formalized result has two modules
([Comparator README](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/ComparatorChallenges/README.md);
for one pair, see
[CirculantHadamard](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/ComparatorChallenges/CirculantHadamard.lean)):

- a **challenge module** containing only the theorem *statement*, ending in
  `sorry` — small enough to audit by reading;
- a **solution module** containing the proof.

A JSON configuration names both, the theorem names, and the permitted
axioms. `comparator` builds the solution in a sandbox and passes only if each
named theorem proves *exactly* the challenge's statement, is accepted by the
Lean kernel, and uses no axiom outside the permitted set. A proof that
compiles is not, by itself, a checked result: it may have proved a different
statement.

### C2. Permitted axioms are exactly `propext`, `Quot.sound`, `Classical.choice`

All 416 Comparator configurations in openai/math use this set and no other.
No `sorryAx`. No `Lean.ofReduceBool` (that is, no `native_decide`). No
custom `axiom` declarations.

### C3. Every formalized result has a scope note

openai/math keeps one scope note per family under `lean/docs/`, stating
exactly which statements are formalized and which parts of the paper are
*outside* them — for example the
[circulant Hadamard note](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/docs/179.md)
proves the even-length Barker case and says the odd case is not covered. A
formalization is described by its scope note, never by the title of the
paper it accompanies.

### C4. Formalized results are catalogued in machine-readable form

The [formalization catalogue](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/formalization.yaml),
using the mathlib-initiative `formalization.yaml` schema, lists each
formalized source and its verification configuration.

### C5. Corrections are versions, and withdrawals keep the evidence

Corrections are recorded as new versions with previous editions still
accessible; a withdrawn result keeps its manuscript, gains a notice
explaining the gap, and is logged in the history (README, "Versions and
citations"). Nothing is silently replaced.

### C6. Anything not formally checked is labelled as such

A result without a Comparator-checked formalization is reported as
*unformalized*, with what evidence it does have stated plainly, and is not
described as proved.

### C7. The toolchain and every dependency are pinned

openai/math pins one Lean release, sets `fixedToolchain := true`, pins every
git dependency to a commit
([lakefile](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/lakefile.lean),
[manifest](https://github.com/openai/math/blob/1581d4cbdd69a0faf34b2089614fe2a814316a6a/lean/lake-manifest.json)),
and carries incompatibilities as explicit patch files.

## How this repository applies it

- **Toolchain for new formal work: Lean `v4.34.1` with Mathlib
  `d13f23b723b8a846827a245b89c10fc7d3f11612`** — exactly openai/math's
  pins, so results here can depend on, and be compared with, the reference
  library. The standalone `comparator` is used at `d03acab`, the last commit
  targeting the v4.34 line, with `lean4export` at `076e8e5` and `landrun` at
  `811cfff`; all three are built against Lean v4.34.1. Upstream has since
  merged Comparator into Lean core as `lake comparator` / `lake check`
  (v4.35.0); moving to that is a deliberate future bump of this pin, not
  something to drift into.
- **The gate was seen to fail before it was trusted.** On a control project
  it went GREEN for a correct proof and RED, each for its own reason, for a
  proof of a different statement, a `sorry`, a `native_decide`, and a
  custom `axiom`. On the real challenge below it went RED when the challenge
  claimed `15·W≥2` instead of `16·W≥2` (statement mismatch) and when the
  trusted complement was broken to the identity (kernel rejection).
- The existing root Lean project (`SutraWS`, Lean v4.10.0) predates this
  codex. Its own axiom audit rejects `sorryAx` and `native_decide`, which
  covers C2 but not C1: nothing pins *which statement* it proved. Bringing
  it under C1 is outstanding work, not a claim.
- Empirical results (training runs, benchmark scores) cannot be formalized.
  For them C3–C6 apply in their natural form: a pre-registration committed
  before the run, a scope note saying exactly what was and was not measured,
  versioned corrections, and an explicit label that the result is empirical.

## Formalized under this codex

| Result | Challenge | Scope note |
| --- | --- | --- |
| Residuals and auxiliary losses behind vedic_trainer's experiment | `vedic_trainer/formal/ComparatorChallenges/VedicResiduals.json` | `vedic_trainer/formal/docs/VedicResiduals.md` |
| Closed form, shift invariance and zero set of the per-example `L_cons` | `vedic_trainer/formal/ComparatorChallenges/VedicLCons.json` | `vedic_trainer/formal/docs/VedicResiduals.md` |
