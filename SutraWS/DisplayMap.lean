import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# The display map: bounded by construction

`projectTorus` and `renderTorus` in `vedic_v18.51.1_exact_phi.html` turn the
certified channels into a surface and a colour. Nothing there is clamped. What
keeps every drawn quantity in range is the shape of the map itself, and this
file proves it:

* `normalised_abs_le_one`, `gain_mul_normalised_abs_le` — a group drawn
  relative to its own largest magnitude this frame moves the surface by at
  most its gain;
* `quad_avg_abs_le` — a quad's colour input, the mean of its four corners,
  stays in `[-1, 1]`;
* `lerp_mem_unit`, `mul_mem_unit`, `mix_weight_mem`, `half_lambert_mem`,
  `shade_mem`, `spec_mem`, `rim_mem`, `contour_factor_mem` — the colour is a
  sequence of convex mixes and products with factors in `[0, 1]`, so every
  component stays in `[0, 1]`;
* `pinch_mem` — the R4 pinch `1 - (1 - g)·(2/5)` is in `[3/5, 1]` for a mean
  gate `g ∈ [0, 1]`;
* `surface_is_torus` — under the two margin conditions the page decides at
  run time, the tube radius is positive and below the ring radius at every
  point; `default_gains_torus` — the default gains meet both;
* `ndc_depth_mem` — the depth written to the depth buffer lies in `[-1, 1]`.
-/

namespace SutraWS
namespace DisplayMap

theorem normalised_abs_le_one {x M : ℝ} (hM : 0 < M) (hx : |x| ≤ M) : |x / M| ≤ 1 := by
  rw [abs_div, abs_of_pos hM]
  exact (div_le_one hM).mpr hx

theorem gain_mul_normalised_abs_le {x M g : ℝ} (hM : 0 < M) (hx : |x| ≤ M) (hg : 0 ≤ g) :
    |g * (x / M)| ≤ g := by
  rw [abs_mul, abs_of_nonneg hg]
  exact mul_le_of_le_one_right hg (normalised_abs_le_one hM hx)

theorem quad_avg_abs_le {a b c d : ℝ} (ha : |a| ≤ 1) (hb : |b| ≤ 1) (hc : |c| ≤ 1)
    (hd : |d| ≤ 1) : |(a + b + c + d) * (1 / 4)| ≤ 1 := by
  rw [abs_le] at *
  obtain ⟨ha0, ha1⟩ := ha
  obtain ⟨hb0, hb1⟩ := hb
  obtain ⟨hc0, hc1⟩ := hc
  obtain ⟨hd0, hd1⟩ := hd
  constructor <;> linarith

theorem lerp_mem_unit {a b t : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) : 0 ≤ a * (1 - t) + b * t ∧ a * (1 - t) + b * t ≤ 1 := by
  have h1t : 0 ≤ 1 - t := by linarith
  constructor
  · have := mul_nonneg ha0 h1t
    have := mul_nonneg hb0 ht0
    linarith
  · have e1 : a * (1 - t) ≤ 1 * (1 - t) := mul_le_mul_of_nonneg_right ha1 h1t
    have e2 : b * t ≤ 1 * t := mul_le_mul_of_nonneg_right hb1 ht0
    linarith

theorem mul_mem_unit {a k : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hk0 : 0 ≤ k) (hk1 : k ≤ 1) :
    0 ≤ a * k ∧ a * k ≤ 1 :=
  ⟨mul_nonneg ha0 hk0, by nlinarith⟩

/-- Mix weights of the form `c · |w|`, a constant `c ∈ [0, 1]` times a
normalised magnitude. -/
theorem mix_weight_mem {c w : ℝ} (hc0 : 0 ≤ c) (hc1 : c ≤ 1) (hw : |w| ≤ 1) :
    0 ≤ c * |w| ∧ c * |w| ≤ 1 :=
  mul_mem_unit hc0 hc1 (abs_nonneg w) hw

theorem half_lambert_mem {d : ℝ} (h : |d| ≤ 1) : 0 ≤ 1 / 2 + 1 / 2 * d ∧ 1 / 2 + 1 / 2 * d ≤ 1 := by
  rw [abs_le] at h
  constructor <;> linarith [h.1, h.2]

theorem shade_mem {d : ℝ} (h0 : 0 ≤ d) (h1 : d ≤ 1) :
    0 ≤ 4 / 25 + 21 / 25 * (d * d) ∧ 4 / 25 + 21 / 25 * (d * d) ≤ 1 := by
  have := mul_nonneg h0 h0
  have : d * d ≤ 1 := by nlinarith
  constructor <;> linarith

theorem spec_mem {s : ℝ} (h0 : 0 ≤ s) (h1 : s ≤ 1) (n : ℕ) : 0 ≤ s ^ n ∧ s ^ n ≤ 1 :=
  ⟨pow_nonneg h0 n, pow_le_one n h0 h1⟩

theorem rim_mem {nz : ℝ} (h : |nz| ≤ 1) : 0 ≤ (1 - |nz|) ^ 3 ∧ (1 - |nz|) ^ 3 ≤ 1 := by
  have h0 : 0 ≤ 1 - |nz| := by linarith
  have h1 : 1 - |nz| ≤ 1 := by linarith [abs_nonneg nz]
  exact spec_mem h0 h1 3

theorem exp_neg_sq_mem (x : ℝ) : 0 < Real.exp (-(x ^ 2)) ∧ Real.exp (-(x ^ 2)) ≤ 1 :=
  ⟨Real.exp_pos _, Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg x])⟩

theorem contour_factor_mem {w x : ℝ} (hw0 : 0 ≤ w) (hw1 : w ≤ 1) :
    0 ≤ 1 - w * Real.exp (-(x ^ 2)) ∧ 1 - w * Real.exp (-(x ^ 2)) ≤ 1 := by
  obtain ⟨he0, he1⟩ := exp_neg_sq_mem x
  have : w * Real.exp (-(x ^ 2)) ≤ 1 := by nlinarith
  have : 0 ≤ w * Real.exp (-(x ^ 2)) := mul_nonneg hw0 he0.le
  constructor <;> linarith

theorem pinch_mem {g : ℝ} (h0 : 0 ≤ g) (h1 : g ≤ 1) :
    3 / 5 ≤ 1 - (1 - g) * (2 / 5) ∧ 1 - (1 - g) * (2 / 5) ≤ 1 := by
  constructor <;> linarith

/-- The placed surface is a torus. `R = R0 + F·a + B·|ga| + G·wR` and
`r = pinch·(r0 + T·a + P·|gp| + G·wr)`, with `a` the normalised amplitude,
`gp` a difference of two of them, and `wR`, `wr` the normalised Wheeler
components. If the tube margin and the ring margin are positive, the tube
radius is positive and below the ring radius, at every point and for every
pinch in `(0, 1]`. -/
theorem surface_is_torus {R0 r0 F T B P G a ga gp wR wr pinch : ℝ}
    (hT : 0 ≤ T) (hB : 0 ≤ B) (hP : 0 ≤ P) (hG : 0 ≤ G)
    (ha : |a| ≤ 1) (hgp : |gp| ≤ 2) (hwR : |wR| ≤ 1) (hwr : |wr| ≤ 1)
    (hp0 : 0 < pinch) (hp1 : pinch ≤ 1)
    (htube : 0 < r0 - T - G) (hring : 0 < R0 - r0 - |F - T| - 2 * P - 2 * G) :
    0 < pinch * (r0 + T * a + P * |gp| + G * wr) ∧
      pinch * (r0 + T * a + P * |gp| + G * wr) < R0 + F * a + B * |ga| + G * wR := by
  -- each term against its bound, from |x| ≤ c  ⇒  -c ≤ x ≤ c
  have hTa : -T ≤ T * a := by
    have : |T * a| ≤ T := by rw [abs_mul, abs_of_nonneg hT]; exact mul_le_of_le_one_right hT ha
    linarith [neg_abs_le (T * a)]
  have hGwr : -G ≤ G * wr := by
    have : |G * wr| ≤ G := by rw [abs_mul, abs_of_nonneg hG]; exact mul_le_of_le_one_right hG hwr
    linarith [neg_abs_le (G * wr)]
  have hGwr' : G * wr ≤ G := by
    have : |G * wr| ≤ G := by rw [abs_mul, abs_of_nonneg hG]; exact mul_le_of_le_one_right hG hwr
    linarith [le_abs_self (G * wr)]
  have hGwR : -G ≤ G * wR := by
    have : |G * wR| ≤ G := by rw [abs_mul, abs_of_nonneg hG]; exact mul_le_of_le_one_right hG hwR
    linarith [neg_abs_le (G * wR)]
  have hPgp0 : 0 ≤ P * |gp| := mul_nonneg hP (abs_nonneg gp)
  have hPgp : P * |gp| ≤ P * 2 := mul_le_mul_of_nonneg_left hgp hP
  have hBga : 0 ≤ B * |ga| := mul_nonneg hB (abs_nonneg ga)
  have hFT : -|F - T| ≤ (F - T) * a := by
    have : |(F - T) * a| ≤ |F - T| := by
      rw [abs_mul]; exact mul_le_of_le_one_right (abs_nonneg _) ha
    linarith [neg_abs_le ((F - T) * a)]
  have hrpos : 0 < r0 + T * a + P * |gp| + G * wr := by linarith
  have hpr : pinch * (r0 + T * a + P * |gp| + G * wr) ≤ 1 * (r0 + T * a + P * |gp| + G * wr) :=
    mul_le_mul_of_nonneg_right hp1 hrpos.le
  refine ⟨mul_pos hp0 hrpos, ?_⟩
  linarith

/-- The default gains: field 60, tube 35, ripple 8, Wheeler 25, on the bare
torus R = 180, r = 70. -/
theorem default_gains_torus :
    0 < (70 : ℝ) - 35 - 25 ∧ 0 < (180 : ℝ) - 70 - |(60 : ℝ) - 35| - 2 * 8 - 2 * 25 := by
  constructor
  · norm_num
  · rw [abs_of_pos (by norm_num : (0 : ℝ) < 60 - 35)]; norm_num

theorem ndc_depth_mem {z zmin zmax : ℝ} (h0 : zmin ≤ z) (h1 : z ≤ zmax) (hs : zmin < zmax) :
    -1 ≤ (z - zmin) * (1998 / 1000 / (zmax - zmin)) - 999 / 1000 ∧
      (z - zmin) * (1998 / 1000 / (zmax - zmin)) - 999 / 1000 ≤ 1 := by
  have hd : 0 < zmax - zmin := by linarith
  have ht0 : 0 ≤ (z - zmin) / (zmax - zmin) := div_nonneg (by linarith) hd.le
  have ht1 : (z - zmin) / (zmax - zmin) ≤ 1 := (div_le_one hd).mpr (by linarith)
  have e : (z - zmin) * (1998 / 1000 / (zmax - zmin)) = 1998 / 1000 * ((z - zmin) / (zmax - zmin)) := by
    field_simp
    ring
  rw [e]
  constructor <;> linarith

end DisplayMap
end SutraWS
