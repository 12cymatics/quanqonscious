import SutraWS.WheelerSurface
import Mathlib.Tactic

namespace SutraWS.FluxSurface

theorem lerp_abs_le (a b M f : ℝ) (ha : |a| ≤ M) (hb : |b| ≤ M) (h0 : 0 ≤ f) (h1 : f ≤ 1) :
    |(1 - f) * a + f * b| ≤ M := by
  rw [abs_le] at ha hb ⊢
  constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2]

theorem lerp_node (a b : ℝ) : (1 - 0) * a + 0 * b = a := by ring

theorem bilinear_node (a b c d : ℝ) :
    (1 - 0) * (1 - 0) * a + 0 * (1 - 0) * b + (1 - 0) * 0 * c + 0 * 0 * d = a := by ring

theorem share_abs_le (x m : ℝ) (hm : 0 < m) (hx : |x| ≤ m) : |x / m| ≤ 1 := by
  rw [abs_div, abs_of_pos hm, div_le_one hm]; exact hx

theorem share_of_max (x m : ℝ) (hm : m ≠ 0) : x / m * m = x := by field_simp

theorem displacement_le (g s : ℝ) (hs : |s| ≤ 1) : |g * s| ≤ |g| := by
  rw [abs_mul]; exact mul_le_of_le_one_right (abs_nonneg g) hs

theorem cell_share_bilinear_le (a b c d m ft fp : ℝ) (hm : 0 < m) (ha : |a| ≤ m) (hb : |b| ≤ m)
    (hc : |c| ≤ m) (hd : |d| ≤ m) (h0 : 0 ≤ ft) (h1 : ft ≤ 1) (g0 : 0 ≤ fp) (g1 : fp ≤ 1) :
    |(1 - ft) * (1 - fp) * (a / m) + ft * (1 - fp) * (b / m) + (1 - ft) * fp * (c / m) + ft * fp * (d / m)| ≤ 1 :=
  SutraWS.WheelerSurface.bilinear_abs_le _ _ _ _ 1 ft fp (share_abs_le a m hm ha) (share_abs_le b m hm hb)
    (share_abs_le c m hm hc) (share_abs_le d m hm hd) h0 h1 g0 g1

theorem loop_share_lerp_le (a b m f : ℝ) (hm : 0 < m) (ha : |a| ≤ m) (hb : |b| ≤ m) (h0 : 0 ≤ f) (h1 : f ≤ 1) :
    |(1 - f) * (a / m) + f * (b / m)| ≤ 1 :=
  lerp_abs_le _ _ 1 f (share_abs_le a m hm ha) (share_abs_le b m hm hb) h0 h1

theorem tube_radius_pos (r g s : ℝ) (hs : |s| ≤ 1) (hg : |g| < r) : 0 < r + g * s := by
  have := displacement_le g s hs
  rw [abs_le] at this
  linarith [this.1]

theorem soft_ratio_abs_lt (x c : ℝ) (hc : 0 < c) : |x / (1 + c * |x|)| < 1 / c := by
  have hd : 0 < 1 + c * |x| := by positivity
  rw [abs_div, abs_of_pos hd, div_lt_div_iff hd hc]
  nlinarith [abs_nonneg x]

theorem soft_ratio_strictMono (c : ℝ) (hc : 0 < c) : StrictMono (fun x : ℝ => x / (1 + c * |x|)) := by
  intro a b hab
  have ha : 0 < 1 + c * |a| := by positivity
  have hb : 0 < 1 + c * |b| := by positivity
  show a / (1 + c * |a|) < b / (1 + c * |b|)
  rw [div_lt_div_iff ha hb]
  rcases le_or_lt 0 a with h0 | h0 <;> rcases le_or_lt 0 b with h1 | h1
  · rw [abs_of_nonneg h0, abs_of_nonneg h1]; nlinarith
  · linarith
  · rw [abs_of_neg h0, abs_of_nonneg h1]; nlinarith [mul_nonneg (le_of_lt hc) (mul_nonneg h1 (neg_nonneg.mpr (le_of_lt h0)))]
  · rw [abs_of_neg h0, abs_of_neg h1]; nlinarith

end SutraWS.FluxSurface
