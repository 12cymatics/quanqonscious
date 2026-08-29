# The Lean 4 layer (`SutraWS/`)

`SutraWS/` is the repository's only machine-checked proof artifact: 10,596
tracked lines of Lean 4. This page says what it proves, what it does not
prove, and how to check both. It exists because nothing else in the
repository described the directory at all — a reader arriving at seven files
named `Sutra.lean`, `SutraSemantics.lean`, `Proofs.lean`, `Exhaustive.lean`
had only the filenames to go on, and filenames invite the reading that the 29
Vedic operators have been formalised. Three of the ten modules do work on the
operators; five are a triangular-number identity; `Interval.lean` is a small
rational-interval helper and `AxiomAudit.lean` is the trust-base gate
described at the end.

## How to build it

```bash
lake exe cache get     # mathlib's prebuilt oleans; building it from source takes hours
lake build             # ~9 min on a cold cache, all 2250 targets
```

Pinned by `lean-toolchain` (`leanprover/lean4:v4.10.0`) and `lake-manifest.json`
(mathlib `a719ba5c3115d47b68bf0497a9dd1bcbb21ea663`, the `v4.10.0` tag). CI runs
exactly these two commands in the `sutraws-lean` job of
`.github/workflows/submit-pypi.yml`.

## What is actually proved

### The operator layer — `Vertex`, `VertexProofs`, `Contracts` (293 lines, 18 theorems)

This layer is about the real 29 operators.

`Vertex.lean` transcribes `const VTX` from the v18 kernel into exact ℚ: the
Z₂⁴ tesseract, the ±1 character table, the Hadamard pair `forward`/`inverse`,
and `hw`/`comp`/`neighbors`. `VertexProofs.lean` proves the transform facts.

`Contracts.lean` carries the one theorem here that replaces a numerical check
with a proof. The kernel's §12Y banner claims every operator collapses to the
identity as α → 0, and `CONTRACTS.testIdentity` (`simulation v18:499-510`)
checks it at runtime by zeroing an operator's `strength` and asserting the
field moved by less than `1e-10` in floats. `identity_preserved` proves it
outright:

```lean
theorem identity_preserved (K : OpMap) (u : Sutra) (P : Psi) : step K u 0 P = P
```

for all 29 operators, every field, and *every* choice of the underlying kernel
maps — the guarantee is a property of the coupling shape, so leaving the maps
abstract makes the theorem stronger than any instantiation of it. A tolerance
check on one sampled field becomes a universally quantified identity in ℚ.

The `family` assignment there (`4+5+3+5+4+3+5 = 29`) is the §12Z operator-class
table. It is coarser than the nine distinct maps recorded in
`docs/SUTRA_CATALOGUE.md`: §12Z's REFLECTIVE class holds `S5`, whose target is
`−Ψ_c` rather than the complement average of the other four, and its
PERMUTATIVE class holds `S7`, which permutes axis 0 rather than axis 3. Seven
shapes, nine concrete maps; the two readings agree and neither contradicts the
other.

### The counter layer — `Sutra`, `State`, `SutraSemantics`, `Proofs`, `Exhaustive` (9,940 lines, 1,985 theorems)

This layer is **not** about Vedic mathematics. Its whole content is:

```lean
def delta : Sutra → Rat | Sutra.S1 => 1 | Sutra.S2 => 2 | … | Sutra.S29 => 29

def act (u : Sutra) (s : State) : State :=
  let k := delta u
  { x := s.x + k, y := s.y - k, z := s.z, t := s.t
  , X := s.X + s.x*k, Y := s.Y + s.y*k, Z := s.Z + s.z*k }
```

Operator *n* adds *n*. Everything downstream follows from that and from the
commutativity of addition on ℚ:

* `sumDelta_all : sumDelta Sutra.all = 435` — the triangular identity T(29).
* `applyAll_x : (applyAll s).x = s.x + 435`, and `applyAll_y` its mirror.
* `sum_erase_k`, `x_erase_k`, `all_vs_erase_k_x` for k = 1…29 — "removing
  operator k changes the total", i.e. 435 − k ≠ 435.
* `prefix_x_n` for n = 0…29 — partial sums are triangular numbers.
* `commute_x_i_j` and `commute_y_i_j` for every ordered pair — **1,682 of the
  1,974 theorems in `Exhaustive.lean` are the 29 × 29 instances of
  `a + i + j = a + j + i` on ℚ.**

`SutraWS.lean`'s own header has said this since the module was written ("the
7-rational counter … whose content is the triangular identity Σδ(1..29) =
435"). It is repeated here because the header is invisible to anyone reading
the file list, and because a 9,765-line file named `Exhaustive.lean` full of
theorems named after sutras reads, from outside, like a formalisation of the
sutras.

**A concrete measure of how little the counter layer constrains.** A 130-line
Python module transcribed from `SutraSemantics.lean` alone — `delta(n) = n`,
`act` adds it, no Vedic content of any kind — satisfies every property this
layer establishes. Run against a 446-line property-based suite written to
mirror these theorems (including the full 29 × 29 commutativity sweep and
2,200 randomised Hypothesis examples), it passes 2,007 of 2,007. Any model in
which the 29 operators are the integers 1…29 under addition satisfies the
counter layer exactly.

That is not a defect in the proofs — they are correct, and T(29) = 435 is the
denominator of §12Y's suppression coefficient α(n) = strength · n/435, so the
identity earns its place. It is a statement about what the proofs cover. The
operator semantics live in `vedic_trainer/vedic/kernel/sutras_canonical.py`
and `z2_primitives.py`, and are gated by the tests there, not here.

## Trust base

`lake build` compiles `SutraWS/AxiomAudit.lean`, which enumerates the
environment and **fails the build** if any theorem in the `SutraWS` namespace
depends on:

| axiom | why it is excluded |
|---|---|
| `sorryAx` | an admitted goal — a file full of `sorry` still compiles, and a build that only checks for errors reports it as proved |
| `Lean.ofReduceBool`, `Lean.ofReduceNat` | introduced by `native_decide`, which discharges a goal by running the compiled program instead of by kernel reduction, putting the Lean compiler and runtime into the trusted base; mathlib forbids it |
| `Lean.trustCompiler` | same reason |

The library was originally proved with `native_decide` — 178 sites across
`Exhaustive.lean`, `Proofs.lean`, and `SutraSemantics.lean` — which put
`Lean.ofReduceBool` into the trust base of **212 of its 2,052 theorems**
(measured by running this audit against the pre-change sources; the other
1,840 never touched a `native_decide` proof). It is now 0.

All three obligation shapes — `sumDelta (Sutra.all.erase Sutra.Sk) = K`,
`sumDelta (Sutra.all.take n) = K`, and `Sutra.Sk ∈ Sutra.all` — close under a
single kernel-checked tactic, `norm_num [Sutra.all, sumDelta, delta]`, at no
meaningful cost in build time. `decide` does *not* work here: `Rat`'s
`DecidableEq` reduces through `Nat.gcd`'s well-founded recursion and the
kernel gives up. That is presumably why `native_decide` was reached for in the
first place; `norm_num` is the tactic that was wanted.

The audit enumerates the environment rather than a written-down list, so a
theorem added later is covered without editing it. Both failure modes are
exercised: reintroducing one `native_decide` fails the build naming the
theorem and the axiom, and changing `435` to `436` fails elaboration with
`unsolved goals` — confirming `norm_num` proves the arithmetic rather than
succeeding vacuously.
