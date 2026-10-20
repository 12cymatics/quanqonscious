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

end SutraWS.FieldLines
