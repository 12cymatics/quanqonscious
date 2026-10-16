import VedicFormal.Defs

/-!
# Challenge: the residuals and auxiliary losses behind vedic_trainer's experiment

**Statements only.** Every theorem here ends in `sorry`; the proofs live in
`VedicFormal/Residuals.lean`, and `VedicResiduals.json` asks Comparator to
check that each proof there states exactly this and uses only `propext`,
`Quot.sound` and `Classical.choice` (`CODEX_OPENAI_MATH.md`, C1–C2). The
definitions are in `VedicFormal/Defs.lean` and are trusted; audit them there.

What each statement settles, and what it does not, is in the scope note
`../docs/VedicResiduals.md`.
-/

namespace VedicFormal

open Finset

/-- The complement used throughout is `v ⊕ 0b1111`, as `tesseract.COMPLEMENT`
computes it. -/
theorem complement_val (v : Fin 16) : (complement v).val = v.val ^^^ 15 := by
  sorry

/-- R2 vanishes for every Ψ ∈ ℚ¹⁶. -/
theorem r2_eq_zero (ψ : Psi) : r2 ψ = 0 := by
  sorry

/-- R3 vanishes for every Ψ ∈ ℚ¹⁶. -/
theorem r3_eq_zero (ψ : Psi) : r3 ψ = 0 := by
  sorry

/-- R4 vanishes for every Ψ ∈ ℚ¹⁶. -/
theorem r4_eq_zero (ψ : Psi) : r4 ψ = 0 := by
  sorry

/-- Audit closure is decided by the trace counter alone: it holds exactly when
435 divides the trace, whatever Ψ is. -/
theorem auditClosed_iff (ψ : Psi) (t : ℤ) : auditClosed ψ t ↔ (435 : ℤ) ∣ t := by
  sorry

/-- Hence no two states — no two models — can differ on audit closure at the
same trace. -/
theorem auditClosed_indep (ψ φ : Psi) (t : ℤ) : auditClosed ψ t ↔ auditClosed φ t := by
  sorry

/-- The mass term of `L_cons` is identically zero: S29 preserves the sum. -/
theorem dMass_eq_zero (ψ : Psi) : dMass ψ = 0 := by
  sorry

/-- `L_dual` is sixteen times the Walsh–Fourier weight of `S11(S5 Ψ)` above
degree one. -/
theorem lDual_eq_fourier_weight (ψ : Psi) :
    lDual ψ = 16 * ∑ S ∈ univ.filter (fun S => 2 ≤ popcount S), fourier (s11 (s5 ψ)) S ^ 2 := by
  sorry

/-- `L_dual` is not identically zero, so the image of S11∘S5 does *not* lie in
the span of the constant and the four axis characters. -/
theorem lDual_not_identically_zero : ∃ ψ : Psi, lDual ψ ≠ 0 := by
  sorry

end VedicFormal
