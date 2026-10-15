import VedicFormal.Defs

/-!
# Solutions to `ComparatorChallenges/VedicResiduals.lean`

Each theorem below restates a challenge theorem verbatim and proves it.
`ComparatorChallenges/VedicResiduals.json` asks Comparator to confirm that
the statements match and that only `propext`, `Quot.sound` and
`Classical.choice` are used (`CODEX_OPENAI_MATH.md`, C1–C2).

Several proofs expand sums over `Fin 16` term by term and close with `ring`
or `norm_num`. That is brute force over a fixed sixteen-vertex cube, and it
is complete: nothing is sampled, and the kernel checks every term.
-/

namespace VedicFormal

open Finset

theorem complement_val (v : Fin 16) : (complement v).val = v.val ^^^ 15 := by
  revert v; decide

theorem r2_eq_zero (ψ : Psi) : r2 ψ = 0 := by
  have hdisj : Disjoint (univ.filter (fun v : Fin 16 => v < complement v))
      ((univ.filter (fun v : Fin 16 => v < complement v)).image complement) := by decide
  have hunion : univ.filter (fun v : Fin 16 => v < complement v) ∪
      (univ.filter (fun v : Fin 16 => v < complement v)).image complement = univ := by decide
  have hinj : Set.InjOn complement (univ.filter (fun v : Fin 16 => v < complement v)) :=
    fun a _ b _ hab => Fin.rev_injective hab
  have hsum : ∑ v, ψ v = ∑ v ∈ univ.filter (fun v : Fin 16 => v < complement v), ψ v +
      ∑ v ∈ univ.filter (fun v : Fin 16 => v < complement v), ψ (complement v) := by
    rw [← Finset.sum_image hinj, ← Finset.sum_union hdisj, hunion]
  rw [r2, Finset.sum_add_distrib, hsum]; ring

theorem r3_eq_zero (ψ : Psi) : r3 ψ = 0 := by
  simp only [r3, mean, s29, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

theorem r4_eq_zero (ψ : Psi) : r4 ψ = 0 := by
  have h : ∑ v : Fin 16, ψ (complement v) ^ 2 = ∑ v : Fin 16, ψ v ^ 2 :=
    Fintype.sum_equiv Fin.revPerm _ _ (fun v => rfl)
  have hv : ∀ v, sym ψ v * anti ψ v = (ψ v ^ 2 - ψ (complement v) ^ 2) / 4 :=
    fun v => by simp only [sym, anti]; ring
  simp only [r4, hv, ← Finset.sum_div, Finset.sum_sub_distrib, h, sub_self, zero_div]

theorem auditClosed_iff (ψ : Psi) (t : ℤ) : auditClosed ψ t ↔ (435 : ℤ) ∣ t := by
  simp only [auditClosed, r1, r2_eq_zero, r3_eq_zero, r4_eq_zero, and_true,
    Int.dvd_iff_emod_eq_zero]

theorem auditClosed_indep (ψ φ : Psi) (t : ℤ) : auditClosed ψ t ↔ auditClosed φ t := by
  rw [auditClosed_iff, auditClosed_iff]

theorem dMass_eq_zero (ψ : Psi) : dMass ψ = 0 := by
  simp only [dMass, s29, mean, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

/-- What `lDual` computes, with an arbitrary `f` in place of `S11(S5 Ψ)`. -/
def residualSq (f : Psi) : ℚ :=
  let star := mean f
  let coeffs : Fin 4 → ℚ := fun k => (∑ v, f v * axisRow k v) / 16
  let recon : Psi := fun v => star + ∑ k, coeffs k * axisRow k v
  ∑ v, (f v - recon v) ^ 2

/-- Parseval on Z₂⁴ for the degree ≤ 1 projection, for every `f`. -/
theorem residualSq_eq (f : Psi) :
    residualSq f = 16 * ∑ S ∈ univ.filter (fun S => 2 ≤ popcount S), fourier f S ^ 2 := by
  simp only [residualSq, mean, fourier, axisRow, chi, popcount, bit, Finset.sum_filter,
    Fin.sum_univ_succ, Fin.sum_univ_zero]
  norm_num
  ring

theorem lDual_eq_fourier_weight (ψ : Psi) :
    lDual ψ = 16 * ∑ S ∈ univ.filter (fun S => 2 ≤ popcount S), fourier (s11 (s5 ψ)) S ^ 2 :=
  residualSq_eq (s11 (s5 ψ))

/-- The witness is the indicator of vertex 3 (`0b0011`, a popcount-2 vertex). -/
theorem lDual_not_identically_zero : ∃ ψ : Psi, lDual ψ ≠ 0 := by
  refine ⟨fun v => if v = 3 then 1 else 0, ?_⟩
  rw [lDual_eq_fourier_weight]
  simp only [fourier, chi, s11, s5, shellMean, mean, popcount, bit, Finset.sum_filter,
    Finset.card_filter, Fin.sum_univ_succ, Fin.sum_univ_zero]
  norm_num

end VedicFormal
