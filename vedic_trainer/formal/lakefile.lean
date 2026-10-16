import Lake

open Lake DSL

/-!
Formal statements behind vedic_trainer's experiment, checked under
CODEX_OPENAI_MATH.md (repository root): statement-only challenges in
`ComparatorChallenges/`, proofs in `VedicFormal/`, trusted definitions in
`VedicFormal/Defs.lean`, permitted axioms propext / Quot.sound /
Classical.choice only.

The toolchain and Mathlib revision are exactly openai/math's
(`lean/lean-toolchain`, `lean/lake-manifest.json` @ 1581d4cb), so this
library can be compared with, and depend on, the reference library.
-/

package VedicFormal where
  version := v!"0.1.0"
  fixedToolchain := true
  leanOptions := #[⟨`autoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "d13f23b723b8a846827a245b89c10fc7d3f11612"

lean_lib VedicFormal

lean_lib ComparatorChallenges
