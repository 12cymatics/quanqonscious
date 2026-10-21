import SutraWS.FluxSurface
import Mathlib.Tactic

namespace SutraWS.PressureHaze

open SutraWS.FluxSurface

theorem haze_pressure_nonneg (bx by' bz : ℝ) : 0 ≤ (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2 := by positivity

theorem haze_stress_trace (bx by' bz : ℝ) :
    (bx * bx - (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) + (by' * by' - (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2)
      + (bz * bz - (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) = -((bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) := by ring

theorem haze_stress_across (bx by' bz nx ny nz : ℝ) (h : bx * nx + by' * ny + bz * nz = 0) :
    (bx * bx - (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) * nx + bx * by' * ny + bx * bz * nz = -((bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) * nx ∧
    by' * bx * nx + (by' * by' - (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) * ny + by' * bz * nz = -((bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) * ny ∧
    bz * bx * nx + bz * by' * ny + (bz * bz - (bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) * nz = -((bx ^ 2 + by' ^ 2 + bz ^ 2) / 2) * nz := by
  refine ⟨?_, ?_, ?_⟩
  · linear_combination bx * h
  · linear_combination by' * h
  · linear_combination bz * h

theorem haze_index_ge_one (κ t : ℝ) (hκ : 0 ≤ κ) (ht : 0 ≤ t) : 1 ≤ 1 + κ * t := by nlinarith

theorem haze_central_le (a b : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    |(b - a) / 2| ≤ 1 / 2 := by
  rw [abs_le]; constructor <;> linarith

theorem haze_side_le (a b : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1) :
    |b - a| ≤ 1 := by
  rw [abs_le]; constructor <;> linarith

theorem lerp_unit (a b f : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (h0 : 0 ≤ f) (h1 : f ≤ 1) : 0 ≤ (1 - f) * a + f * b ∧ (1 - f) * a + f * b ≤ 1 := by
  constructor <;> nlinarith

theorem haze_trilinear_mem (a b c d e f g h M u v w : ℝ)
    (ha : |a| ≤ M) (hb : |b| ≤ M) (hc : |c| ≤ M) (hd : |d| ≤ M)
    (he : |e| ≤ M) (hf : |f| ≤ M) (hg : |g| ≤ M) (hh : |h| ≤ M)
    (u0 : 0 ≤ u) (u1 : u ≤ 1) (v0 : 0 ≤ v) (v1 : v ≤ 1) (w0 : 0 ≤ w) (w1 : w ≤ 1) :
    |(1 - w) * ((1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * d))
      + w * ((1 - v) * ((1 - u) * e + u * f) + v * ((1 - u) * g + u * h))| ≤ M :=
  lerp_abs_le _ _ M w
    (lerp_abs_le _ _ M v (lerp_abs_le a b M u ha hb u0 u1) (lerp_abs_le c d M u hc hd u0 u1) v0 v1)
    (lerp_abs_le _ _ M v (lerp_abs_le e f M u he hf u0 u1) (lerp_abs_le g h M u hg hh u0 u1) v0 v1) w0 w1

theorem haze_trilinear_unit (a b c d e f g h u v w : ℝ)
    (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1)
    (he : 0 ≤ e ∧ e ≤ 1) (hf : 0 ≤ f ∧ f ≤ 1) (hg : 0 ≤ g ∧ g ≤ 1) (hh : 0 ≤ h ∧ h ≤ 1)
    (u0 : 0 ≤ u) (u1 : u ≤ 1) (v0 : 0 ≤ v) (v1 : v ≤ 1) (w0 : 0 ≤ w) (w1 : w ≤ 1) :
    0 ≤ (1 - w) * ((1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * d))
      + w * ((1 - v) * ((1 - u) * e + u * f) + v * ((1 - u) * g + u * h)) ∧
    (1 - w) * ((1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * d))
      + w * ((1 - v) * ((1 - u) * e + u * f) + v * ((1 - u) * g + u * h)) ≤ 1 := by
  have p := lerp_unit a b u ha.1 ha.2 hb.1 hb.2 u0 u1
  have q := lerp_unit c d u hc.1 hc.2 hd.1 hd.2 u0 u1
  have r := lerp_unit e f u he.1 he.2 hf.1 hf.2 u0 u1
  have s := lerp_unit g h u hg.1 hg.2 hh.1 hh.2 u0 u1
  have pq := lerp_unit _ _ v p.1 p.2 q.1 q.2 v0 v1
  have rs := lerp_unit _ _ v r.1 r.2 s.1 s.2 v0 v1
  exact lerp_unit _ _ w pq.1 pq.2 rs.1 rs.2 w0 w1

theorem haze_ray_sum_le {ι : Type*} (s : Finset ι) (w g : ι → ℝ) (M : ℝ)
    (hw : ∀ i ∈ s, 0 ≤ w i) (hg : ∀ i ∈ s, |g i| ≤ M) :
    |∑ i ∈ s, w i * g i| ≤ M * ∑ i ∈ s, w i := by
  calc |∑ i ∈ s, w i * g i| ≤ ∑ i ∈ s, |w i * g i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ s, w i * M := by
        apply Finset.sum_le_sum
        intro i hi
        rw [abs_mul, abs_of_nonneg (hw i hi)]
        exact mul_le_mul_of_nonneg_left (hg i hi) (hw i hi)
    _ = M * ∑ i ∈ s, w i := by rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i _ => mul_comm _ _)

theorem haze_chord_le (px py pz qx qy qz R : ℝ)
    (hp : px ^ 2 + py ^ 2 + pz ^ 2 ≤ R ^ 2) (hq : qx ^ 2 + qy ^ 2 + qz ^ 2 ≤ R ^ 2) :
    (px - qx) ^ 2 + (py - qy) ^ 2 + (pz - qz) ^ 2 ≤ (2 * R) ^ 2 := by
  nlinarith [sq_nonneg (px + qx), sq_nonneg (py + qy), sq_nonneg (pz + qz)]

theorem haze_rescale (g N R0 R1 : ℝ) (h0 : R0 ≠ 0) (h1 : R1 ≠ 0) :
    g * (R1 / R0) * ((N - 1) / (2 * R1)) = g * ((N - 1) / (2 * R0)) := by
  field_simp
  ring

theorem haze_veil_mem {ι : Type*} (s : Finset ι) (w t : ι → ℝ) (L : ℝ) (hL : 0 < L)
    (hw : ∀ i ∈ s, 0 ≤ w i) (ht : ∀ i ∈ s, 0 ≤ t i ∧ t i ≤ 1) (hs : ∑ i ∈ s, w i ≤ L) :
    0 ≤ (∑ i ∈ s, w i * t i) / L ∧ (∑ i ∈ s, w i * t i) / L ≤ 1 := by
  have lo : 0 ≤ ∑ i ∈ s, w i * t i := Finset.sum_nonneg (fun i hi => mul_nonneg (hw i hi) (ht i hi).1)
  have hi : ∑ i ∈ s, w i * t i ≤ ∑ i ∈ s, w i :=
    Finset.sum_le_sum (fun i hi => by have := ht i hi; have := hw i hi; nlinarith)
  exact ⟨div_nonneg lo hL.le, (div_le_one hL).mpr (le_trans hi hs)⟩

theorem haze_turn_norm (x z c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (x * c - z * s) ^ 2 + (x * s + z * c) ^ 2 = x ^ 2 + z ^ 2 := by
  linear_combination (x ^ 2 + z ^ 2) * h

theorem haze_turn_back (x z c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (x * c - z * s) * c + (x * s + z * c) * s = x ∧ -(x * c - z * s) * s + (x * s + z * c) * c = z := by
  constructor
  · linear_combination x * h
  · linear_combination z * h

end SutraWS.PressureHaze
