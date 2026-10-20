import Mathlib.Tactic

namespace SutraWS.FieldLines

theorem inner_sq_deriv (c r2 r3 : ℝ) :
    HasDerivAt (fun t : ℝ => (c + t) ^ 2 + r2 ^ 2 + r3 ^ 2) (2 * c) 0 := by
  have h := ((((hasDerivAt_id' (0 : ℝ)).const_add c).pow 2).add_const (r2 ^ 2)).add_const (r3 ^ 2)
  convert h using 1; simp

theorem element_div_free (a1 a2 a3 r1 r2 r3 g' : ℝ) (g : ℝ → ℝ)
    (hg : HasDerivAt g g' (r1 ^ 2 + r2 ^ 2 + r3 ^ 2)) :
    HasDerivAt (fun t => (a2 * r3 - a3 * r2) * g ((r1 + t) ^ 2 + r2 ^ 2 + r3 ^ 2)) ((a2 * r3 - a3 * r2) * (g' * (2 * r1))) 0 ∧
    HasDerivAt (fun t => (a3 * r1 - a1 * r3) * g (r1 ^ 2 + (r2 + t) ^ 2 + r3 ^ 2)) ((a3 * r1 - a1 * r3) * (g' * (2 * r2))) 0 ∧
    HasDerivAt (fun t => (a1 * r2 - a2 * r1) * g (r1 ^ 2 + r2 ^ 2 + (r3 + t) ^ 2)) ((a1 * r2 - a2 * r1) * (g' * (2 * r3))) 0 ∧
    (a2 * r3 - a3 * r2) * (g' * (2 * r1)) + (a3 * r1 - a1 * r3) * (g' * (2 * r2)) + (a1 * r2 - a2 * r1) * (g' * (2 * r3)) = 0 := by
  refine ⟨?_, ?_, ?_, by ring⟩
  · have hi := inner_sq_deriv r1 r2 r3
    have hg1 : HasDerivAt g g' ((fun t : ℝ => (r1 + t) ^ 2 + r2 ^ 2 + r3 ^ 2) 0) := by simpa using hg
    exact (hg1.comp 0 hi).const_mul _
  · have hi : HasDerivAt (fun t : ℝ => r1 ^ 2 + (r2 + t) ^ 2 + r3 ^ 2) (2 * r2) 0 := by
      have h := inner_sq_deriv r2 r1 r3
      convert h using 1; funext t; ring
    have hg1 : HasDerivAt g g' ((fun t : ℝ => r1 ^ 2 + (r2 + t) ^ 2 + r3 ^ 2) 0) := by simpa using hg
    exact (hg1.comp 0 hi).const_mul _
  · have hi : HasDerivAt (fun t : ℝ => r1 ^ 2 + r2 ^ 2 + (r3 + t) ^ 2) (2 * r3) 0 := by
      have h := inner_sq_deriv r3 r1 r2
      convert h using 1; funext t; ring
    have hg1 : HasDerivAt g g' ((fun t : ℝ => r1 ^ 2 + r2 ^ 2 + (r3 + t) ^ 2) 0) := by simpa using hg
    exact (hg1.comp 0 hi).const_mul _

theorem element_perp_offset (a1 a2 a3 r1 r2 r3 : ℝ) :
    (a2 * r3 - a3 * r2) * r1 + (a3 * r1 - a1 * r3) * r2 + (a1 * r2 - a2 * r1) * r3 = 0 := by ring

theorem element_perp_current (a1 a2 a3 r1 r2 r3 : ℝ) :
    (a2 * r3 - a3 * r2) * a1 + (a3 * r1 - a1 * r3) * a2 + (a1 * r2 - a2 * r1) * a3 = 0 := by ring

theorem element_right_hand (a1 a2 a3 r1 r2 r3 : ℝ) :
    let b1 := a2 * r3 - a3 * r2; let b2 := a3 * r1 - a1 * r3; let b3 := a1 * r2 - a2 * r1
    a1 * (r2 * b3 - r3 * b2) + a2 * (r3 * b1 - r1 * b3) + a3 * (r1 * b2 - r2 * b1)
      = (a1 ^ 2 + a2 ^ 2 + a3 ^ 2) * (r1 ^ 2 + r2 ^ 2 + r3 ^ 2) - (a1 * r1 + a2 * r2 + a3 * r3) ^ 2 := by
  intro b1 b2 b3; simp only [b1, b2, b3]; ring

theorem right_hand_nonneg (a1 a2 a3 r1 r2 r3 : ℝ) :
    0 ≤ (a1 ^ 2 + a2 ^ 2 + a3 ^ 2) * (r1 ^ 2 + r2 ^ 2 + r3 ^ 2) - (a1 * r1 + a2 * r2 + a3 * r3) ^ 2 := by
  nlinarith [sq_nonneg (a1 * r2 - a2 * r1), sq_nonneg (a2 * r3 - a3 * r2), sq_nonneg (a1 * r3 - a3 * r1)]

theorem seg_kernel_stable {d a b : ℝ} (hd : 0 < d) (hab : 0 < a * b) :
    (a / Real.sqrt (a ^ 2 + d ^ 2) - b / Real.sqrt (b ^ 2 + d ^ 2)) / d ^ 2
      = (a - b) * (a + b) / (Real.sqrt (a ^ 2 + d ^ 2) * Real.sqrt (b ^ 2 + d ^ 2) *
          (a * Real.sqrt (b ^ 2 + d ^ 2) + b * Real.sqrt (a ^ 2 + d ^ 2))) := by
  have hA0 : 0 < a ^ 2 + d ^ 2 := by positivity
  have hB0 : 0 < b ^ 2 + d ^ 2 := by positivity
  set A := Real.sqrt (a ^ 2 + d ^ 2)
  set B := Real.sqrt (b ^ 2 + d ^ 2)
  have hA : 0 < A := Real.sqrt_pos.mpr hA0
  have hB : 0 < B := Real.sqrt_pos.mpr hB0
  have hA2 : A ^ 2 = a ^ 2 + d ^ 2 := Real.sq_sqrt hA0.le
  have hB2 : B ^ 2 = b ^ 2 + d ^ 2 := Real.sq_sqrt hB0.le
  have hD : a * B + b * A ≠ 0 := by
    rcases lt_or_gt_of_ne (show a ≠ 0 by rintro rfl; simp at hab) with ha | ha
    · have hb : b < 0 := by nlinarith
      exact (by nlinarith : a * B + b * A < 0).ne
    · have hb : 0 < b := by nlinarith
      exact (by nlinarith : 0 < a * B + b * A).ne'
  have hd2 : d ^ 2 ≠ 0 := by positivity
  have hden : A * B * (a * B + b * A) ≠ 0 := mul_ne_zero (mul_ne_zero hA.ne' hB.ne') hD
  rw [div_eq_div_iff hd2 hden]
  have e : a / A - b / B = (a * B - b * A) / (A * B) := by field_simp; ring
  rw [e, div_mul_eq_mul_div, div_eq_iff (mul_ne_zero hA.ne' hB.ne')]
  linear_combination (A * B) * (a ^ 2 * hB2 - b ^ 2 * hA2)

end SutraWS.FieldLines
