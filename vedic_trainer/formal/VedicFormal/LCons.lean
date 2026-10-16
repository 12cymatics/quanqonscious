import VedicFormal.Defs

/-!
# Solutions to `ComparatorChallenges/VedicLCons.lean`

Each theorem below restates a challenge theorem verbatim and proves it.
`ComparatorChallenges/VedicLCons.json` asks Comparator to confirm that the
statements match and that only `propext`, `Quot.sound` and `Classical.choice`
are used (`CODEX_OPENAI_MATH.md`, C1–C2).

The closed form is proved structurally, not by expanding all sixteen vertices:
S29 halves `sym` about the mean and halves `anti`, so each energy drifts by
`−3/4` of the corresponding energy, the cross term vanishing because
`Σ_v (sym Ψ v − mean Ψ) = 0`.
-/

namespace VedicFormal

open Finset

/-- The complement is a permutation of the vertices, so it preserves sums. -/
private theorem sum_comp_complement (ψ : Psi) :
    ∑ v : Fin 16, ψ (complement v) = ∑ v : Fin 16, ψ v :=
  Fintype.sum_equiv Fin.revPerm _ _ (fun _ => rfl)

private theorem sum_sym_sub_mean (ψ : Psi) : ∑ v, (sym ψ v - mean ψ) = 0 := by
  have h : ∑ v, sym ψ v = ∑ v, ψ v := by
    simp only [sym, ← Finset.sum_mul, Finset.sum_add_distrib, sum_comp_complement]
    ring
  rw [Finset.sum_sub_distrib, h]
  simp only [mean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  ring

private theorem sym_s29 (ψ : Psi) (v : Fin 16) :
    sym (s29 ψ) v = sym ψ v / 2 + mean ψ / 2 := by
  simp only [sym, s29]
  ring

private theorem anti_s29 (ψ : Psi) (v : Fin 16) : anti (s29 ψ) v = anti ψ v / 2 := by
  simp only [anti, s29]
  ring

/-- The symmetric drift `d_sym`. -/
private theorem d_sym_eq (ψ : Psi) :
    ∑ v, sym (s29 ψ) v * sym (s29 ψ) v - ∑ v, sym ψ v * sym ψ v =
      -(3 / 4) * ∑ v, (sym ψ v - mean ψ) ^ 2 := by
  have hv : ∀ v, sym (s29 ψ) v * sym (s29 ψ) v - sym ψ v * sym ψ v =
      -(3 / 4) * (sym ψ v - mean ψ) ^ 2 - mean ψ * (sym ψ v - mean ψ) :=
    fun v => by rw [sym_s29]; ring
  rw [← Finset.sum_sub_distrib, Finset.sum_congr rfl (fun v _ => hv v),
    Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, sum_sym_sub_mean]
  ring

/-- The antisymmetric drift `d_anti`. -/
private theorem d_anti_eq (ψ : Psi) :
    ∑ v, anti (s29 ψ) v * anti (s29 ψ) v - ∑ v, anti ψ v * anti ψ v =
      -(3 / 4) * ∑ v, (anti ψ v) ^ 2 := by
  rw [← Finset.sum_sub_distrib, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun v _ => by rw [anti_s29]; ring)

theorem lConsRow_closed_form (ψ : Psi) :
    lConsRow ψ =
      (9 / 16) * ((∑ v, (sym ψ v - mean ψ) ^ 2) ^ 2 + (∑ v, (anti ψ v) ^ 2) ^ 2) := by
  simp only [lConsRow]
  rw [d_sym_eq, d_anti_eq]
  ring

theorem lConsRow_shift (ψ : Psi) (c : ℚ) : lConsRow (fun v => ψ v + c) = lConsRow ψ := by
  have hm : mean (fun v => ψ v + c) = mean ψ + c := by
    simp only [mean, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    ring
  have hs : ∀ v, sym (fun v => ψ v + c) v - mean (fun v => ψ v + c) = sym ψ v - mean ψ :=
    fun v => by rw [hm]; simp only [sym]; ring
  have ha : ∀ v, anti (fun v => ψ v + c) v = anti ψ v :=
    fun v => by simp only [anti]; ring
  rw [lConsRow_closed_form, lConsRow_closed_form]
  simp only [hs, ha]

theorem lConsRow_eq_zero_iff (ψ : Psi) : lConsRow ψ = 0 ↔ ∀ v w, ψ v = ψ w := by
  rw [lConsRow_closed_form]
  constructor
  · intro h
    have hA : 0 ≤ ∑ v, (sym ψ v - mean ψ) ^ 2 :=
      Finset.sum_nonneg (fun v _ => sq_nonneg _)
    have hB : 0 ≤ ∑ v, (anti ψ v) ^ 2 :=
      Finset.sum_nonneg (fun v _ => sq_nonneg _)
    have hA2 : (∑ v, (sym ψ v - mean ψ) ^ 2) ^ 2 = 0 := by
      nlinarith [sq_nonneg (∑ v, (sym ψ v - mean ψ) ^ 2), sq_nonneg (∑ v, (anti ψ v) ^ 2)]
    have hB2 : (∑ v, (anti ψ v) ^ 2) ^ 2 = 0 := by
      nlinarith [sq_nonneg (∑ v, (sym ψ v - mean ψ) ^ 2), sq_nonneg (∑ v, (anti ψ v) ^ 2)]
    have hA0 := (Finset.sum_eq_zero_iff_of_nonneg (fun v _ => sq_nonneg (sym ψ v - mean ψ))).1
      (pow_eq_zero_iff two_ne_zero |>.1 hA2)
    have hB0 := (Finset.sum_eq_zero_iff_of_nonneg (fun v _ => sq_nonneg (anti ψ v))).1
      (pow_eq_zero_iff two_ne_zero |>.1 hB2)
    have hconst : ∀ v, ψ v = mean ψ := by
      intro v
      have hs : sym ψ v - mean ψ = 0 :=
        pow_eq_zero_iff two_ne_zero |>.1 (hA0 v (Finset.mem_univ v))
      have ha : anti ψ v = 0 := pow_eq_zero_iff two_ne_zero |>.1 (hB0 v (Finset.mem_univ v))
      have hsa : ψ v = sym ψ v + anti ψ v := by simp only [sym, anti]; ring
      rw [hsa, ha]
      linarith
    intro v w
    rw [hconst v, hconst w]
  · intro h
    obtain ⟨c, rfl⟩ : ∃ c : ℚ, ψ = fun _ => c := ⟨ψ 0, funext fun v => h v 0⟩
    have hs : ∀ v, sym (fun _ : Fin 16 => c) v - mean (fun _ : Fin 16 => c) = 0 := fun v => by
      simp only [sym, mean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      ring
    have ha : ∀ v, anti (fun _ : Fin 16 => c) v = 0 := fun v => by simp only [anti]; ring
    simp only [hs, ha]
    norm_num

end VedicFormal
