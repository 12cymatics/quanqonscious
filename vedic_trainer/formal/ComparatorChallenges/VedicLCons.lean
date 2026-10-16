import VedicFormal.Defs

/-!
# Challenge: the closed form of the per-example `L_cons` term

**Statements only.** Every theorem here ends in `sorry`; the proofs live in
`VedicFormal/LCons.lean`, and `VedicLCons.json` asks Comparator to check that
each proof there states exactly this and uses only `propext`, `Quot.sound` and
`Classical.choice` (`CODEX_OPENAI_MATH.md`, C1–C2). `lConsRow` and everything
it uses are in `VedicFormal/Defs.lean` and are trusted; audit them there.

What each statement settles, and what it does not, is in the scope note
`../docs/VedicResiduals.md`.
-/

namespace VedicFormal

open Finset

/-- One example of `L_cons` is `9/16` times the sum of the squared energies of
the non-constant symmetric part and of the antisymmetric part:
`(9/16)·(‖S − mean(Ψ)·𝟙‖⁴ + ‖A‖⁴)`. -/
theorem lConsRow_closed_form (ψ : Psi) :
    lConsRow ψ =
      (9 / 16) * ((∑ v, (sym ψ v - mean ψ) ^ 2) ^ 2 + (∑ v, (anti ψ v) ^ 2) ^ 2) := by
  sorry

/-- Adding the same constant to every vertex does not change `L_cons`. -/
theorem lConsRow_shift (ψ : Psi) (c : ℚ) : lConsRow (fun v => ψ v + c) = lConsRow ψ := by
  sorry

/-- `L_cons` vanishes exactly on the constant states. -/
theorem lConsRow_eq_zero_iff (ψ : Psi) : lConsRow ψ = 0 ↔ ∀ v w, ψ v = ψ w := by
  sorry

end VedicFormal
