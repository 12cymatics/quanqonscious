import Mathlib.Tactic

/-!
# Stored precision

`vedic_v18.51.1_exact_phi.html` computes every step of the simulator in exact
rationals, starting from the stored state. Only the state carried into the
next step is rounded: each rational coordinate `x ≠ 0` is scaled by `2^k`,
with `k` chosen so that the scaled value has at least `B` bits, rounded to the
nearest integer (ties away from zero) and scaled back (`storeQ`, with
`B = STORE_BITS = 128`).

This file proves the bound the page states for that rounding:

    2^B ≤ 2 |x| 2^k   →   |x - store x k| ≤ |x| / 2^B.

The hypothesis is the one `storeQ` checks for every value before it rounds:
`num ≥ den * 2^(B-1)` with `num / den = |x| 2^k`.
-/

namespace SutraWS
namespace StoredPrecision

/-- `storeQ` with its scale exponent explicit. JavaScript's
`(2*num + den) / (2*den)` on the magnitude is `⌊|x| 2^k + 1/2⌋`, Mathlib's
`round`, and the sign is restored afterwards. -/
def store (x : ℚ) (k : ℤ) : ℚ :=
  if x < 0 then -((round (-x * 2 ^ k) : ℚ) / 2 ^ k) else (round (x * 2 ^ k) : ℚ) / 2 ^ k

theorem two_zpow_pos (k : ℤ) : (0 : ℚ) < 2 ^ k := zpow_pos_of_pos (by norm_num) k

/-- Half a unit in the last place at scale `k`. -/
theorem store_err_half_ulp (x : ℚ) (k : ℤ) : |x - store x k| ≤ 1 / 2 / 2 ^ k := by
  have hp := two_zpow_pos k
  have key : ∀ y : ℚ, |y - (round (y * 2 ^ k) : ℚ) / 2 ^ k| ≤ 1 / 2 / 2 ^ k := by
    intro y
    have h1 : y - (round (y * 2 ^ k) : ℚ) / 2 ^ k = (y * 2 ^ k - round (y * 2 ^ k)) / 2 ^ k := by
      field_simp
    rw [h1, abs_div, abs_of_pos hp]
    exact div_le_div_of_nonneg_right (abs_sub_round (y * 2 ^ k)) hp.le
  unfold store
  split_ifs with hx
  · have h2 : x - -((round (-x * 2 ^ k) : ℚ) / 2 ^ k) = -((-x) - (round (-x * 2 ^ k) : ℚ) / 2 ^ k) := by
      ring
    rw [h2, abs_neg]
    exact key (-x)
  · exact key x

/-- `store_err_le`: under the scale hypothesis `storeQ` checks, the stored
value is within `|x| / 2^B` of `x`. -/
theorem store_err_le (x : ℚ) (k : ℤ) (B : ℕ) (h : (2 : ℚ) ^ B ≤ 2 * |x| * 2 ^ k) :
    |x - store x k| ≤ |x| / 2 ^ B := by
  have hB : (0 : ℚ) < 2 ^ B := by positivity
  refine le_trans (store_err_half_ulp x k) ?_
  rw [div_div, div_le_div_iff (by positivity) hB]
  nlinarith [h]

/-- Zero is stored exactly. -/
theorem store_zero (k : ℤ) : store 0 k = 0 := by
  simp [store]

/-- Storing a value already on the grid returns it: the store is idempotent on
what it produced. -/
theorem store_of_grid (m : ℤ) (k : ℤ) : store ((m : ℚ) / 2 ^ k) k = (m : ℚ) / 2 ^ k := by
  have h1 : (m : ℚ) / 2 ^ k * 2 ^ k = (m : ℚ) := by field_simp
  have h2 : -((m : ℚ) / 2 ^ k) * 2 ^ k = ((-m : ℤ) : ℚ) := by push_cast; field_simp
  unfold store
  split_ifs with hx
  · rw [h2, round_intCast]; push_cast; ring
  · rw [h1, round_intCast]

/-- `store_err_le_stored`: the same bound in terms of the stored value, which
is what the page has in hand when it sums the budget: `|x − x̃| ≤ 2|x̃| / 2^B`. -/
theorem store_err_le_stored (x : ℚ) (k : ℤ) (B : ℕ) (hB : 1 ≤ B)
    (h : (2 : ℚ) ^ B ≤ 2 * |x| * 2 ^ k) :
    |x - store x k| ≤ 2 * |store x k| / 2 ^ B := by
  have e := store_err_le x k B h
  have hP : (0 : ℚ) < 2 ^ B := by positivity
  have h2 : (2 : ℚ) ≤ 2 ^ B := by
    calc (2 : ℚ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ B := pow_le_pow_right (by norm_num) hB
  -- |x| ≤ |x̃| + |x - x̃| ≤ |x̃| + |x|/2^B, so |x| (1 - 1/2^B) ≤ |x̃|
  have tri : |x| ≤ |store x k| + |x - store x k| := by
    have := abs_add (store x k) (x - store x k)
    simpa using this
  rw [le_div_iff hP] at e ⊢
  nlinarith [abs_nonneg x, abs_nonneg (store x k), abs_nonneg (x - store x k)]

/-- Rounding to a grid that contains 0 and 1 keeps a value of `[0, 1]` in
`[0, 1]`: the R4 factor σ is stored and stays a factor. -/
theorem store_mem_unit (x : ℚ) (k : ℤ) (hk : 0 ≤ k) (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    0 ≤ store x k ∧ store x k ≤ 1 := by
  have hp := two_zpow_pos k
  have hx : ¬ x < 0 := not_lt.mpr h0
  unfold store
  rw [if_neg hx]
  lift k to ℕ using hk
  -- the grid point 2^k is an integer, and round is monotone
  have e : (2 : ℚ) ^ (k : ℤ) = (((2 ^ k : ℕ) : ℤ) : ℚ) := by rw [zpow_natCast]; push_cast; ring
  have rmono : ∀ a b : ℚ, a ≤ b → round a ≤ round b := fun a b h => by
    rw [round_eq, round_eq]; exact Int.floor_mono (by linarith)
  have r0 : (0 : ℤ) ≤ round (x * 2 ^ (k : ℤ)) := by
    have := rmono 0 (x * 2 ^ (k : ℤ)) (by positivity)
    simpa using this
  have r1 : round (x * 2 ^ (k : ℤ)) ≤ ((2 ^ k : ℕ) : ℤ) := by
    have := rmono (x * 2 ^ (k : ℤ)) (((2 ^ k : ℕ) : ℤ) : ℚ) (by rw [← e]; nlinarith)
    rw [round_intCast] at this
    exact this
  constructor
  · exact div_nonneg (by exact_mod_cast r0) hp.le
  · rw [div_le_one hp, e]
    exact_mod_cast r1

/-- The dyadic radical bounds `storeBudget` weights coordinates with, in 256ths:
`√2 < 363/256`, `√5 < 573/256`, `√10 < 810/256`. -/
theorem sqrt2_lt_bound : Real.sqrt 2 < 363 / 256 := by
  rw [Real.sqrt_lt' (by norm_num)]; norm_num

theorem sqrt5_lt_bound : Real.sqrt 5 < 573 / 256 := by
  rw [Real.sqrt_lt' (by norm_num)]; norm_num

theorem sqrt10_lt_bound : Real.sqrt 10 < 810 / 256 := by
  rw [Real.sqrt_lt' (by norm_num)]; norm_num

/-- `real_embed_abs_le`: a coordinate error `(a, b, c, d)` of an element of
`ℚ(√2, √5)` moves its real value by at most the weighted sum `storeBudget`
uses. -/
theorem real_embed_abs_le (a b c d : ℝ) :
    |a + b * Real.sqrt 2 + c * Real.sqrt 5 + d * Real.sqrt 10| ≤
      |a| + |b| * (363 / 256) + |c| * (573 / 256) + |d| * (810 / 256) := by
  have s2 := sqrt2_lt_bound
  have s5 := sqrt5_lt_bound
  have s10 := sqrt10_lt_bound
  have p2 := Real.sqrt_nonneg 2
  have p5 := Real.sqrt_nonneg 5
  have p10 := Real.sqrt_nonneg 10
  have e2 : |b * Real.sqrt 2| ≤ |b| * (363 / 256) := by
    rw [abs_mul, abs_of_nonneg p2]; exact mul_le_mul_of_nonneg_left s2.le (abs_nonneg b)
  have e5 : |c * Real.sqrt 5| ≤ |c| * (573 / 256) := by
    rw [abs_mul, abs_of_nonneg p5]; exact mul_le_mul_of_nonneg_left s5.le (abs_nonneg c)
  have e10 : |d * Real.sqrt 10| ≤ |d| * (810 / 256) := by
    rw [abs_mul, abs_of_nonneg p10]; exact mul_le_mul_of_nonneg_left s10.le (abs_nonneg d)
  calc |a + b * Real.sqrt 2 + c * Real.sqrt 5 + d * Real.sqrt 10|
      ≤ |a + b * Real.sqrt 2 + c * Real.sqrt 5| + |d * Real.sqrt 10| := abs_add _ _
    _ ≤ |a + b * Real.sqrt 2| + |c * Real.sqrt 5| + |d * Real.sqrt 10| := by
        gcongr; exact abs_add _ _
    _ ≤ |a| + |b * Real.sqrt 2| + |c * Real.sqrt 5| + |d * Real.sqrt 10| := by
        gcongr; exact abs_add _ _
    _ ≤ |a| + |b| * (363 / 256) + |c| * (573 / 256) + |d| * (810 / 256) := by
        linarith

end StoredPrecision
end SutraWS
