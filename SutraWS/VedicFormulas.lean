import SutraWS.Vertex
import SutraWS.VertexProofs
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Data.Int.GCD
import Mathlib.Tactic

/-!
# The 29 v18.24 sutra formulas on the real vertex field

The statements the runtime decides, on the live field and the live operands,
for each of the 29 `SUTRAS[].evolve` formulas of `vedic_v18.24_full_kernel.html`,
for the GRVQ coupling into `λ = inverse Ψ`, for the MSTVQ stress tensor and for
the rotation stages of the real field.
-/

namespace SutraWS
namespace Vedic

open Finset

theorem vedic_update_zero (x g : ℚ) : x + 0 * g = x ∧ x * (1 + 0 * g) = x := by
  constructor <;> ring

theorem ekadhikena_factor_bound (x c : ℚ) (hc : 0 < c) : 4 * c * (x / (x ^ 2 + c)) ^ 2 ≤ 1 := by
  rw [div_pow, ← mul_div_assoc, div_le_one (by positivity)]
  nlinarith [sq_nonneg (x ^ 2 - c)]

theorem radix_order_bracket (r : ℕ) (hr : 2 ≤ r) (m : ℝ) (k : ℕ) (hk : 1 ≤ k)
    (h1 : (r : ℝ) ^ (k - 1) ≤ m) (h2 : m < (r : ℝ) ^ k) :
    Int.floor (Real.logb r m) = (k : ℤ) - 1 := by
  have hr1 : (1 : ℝ) < r := by exact_mod_cast lt_of_lt_of_le one_lt_two hr
  have hr0 : (0 : ℝ) < r := lt_trans zero_lt_one hr1
  have hm : 0 < m := lt_of_lt_of_le (pow_pos hr0 _) h1
  rw [Int.floor_eq_iff]
  have hc : ((k : ℤ) - 1 : ℤ) = ((k - 1 : ℕ) : ℤ) := by omega
  constructor
  · rw [hc, Int.cast_natCast, Real.le_logb_iff_rpow_le hr1 hm, Real.rpow_natCast]
    exact h1
  · have hk' : (((k : ℤ) - 1 : ℤ) : ℝ) + 1 = (k : ℝ) := by push_cast; ring
    rw [hk', Real.logb_lt_iff_lt_rpow hr1 hm, Real.rpow_natCast]
    exact h2

theorem nikhilam_affine (x B α : ℚ) : x + ((B - x) - x) * α = (1 - 2 * α) * x + α * B := by ring

theorem yavadunam_magnifies (x α d : ℚ) (hα : 0 ≤ α) : |x| ≤ |x * (1 + α * d ^ 2)| := by
  have h1 : 1 ≤ 1 + α * d ^ 2 := by nlinarith [sq_nonneg d]
  rw [abs_mul, abs_of_pos (lt_of_lt_of_le zero_lt_one h1)]
  nlinarith [abs_nonneg x]

def vxor (i j : Vertex) : Vertex := ⟨(i.val ^^^ j.val) % 16, Nat.mod_lt _ (by norm_num)⟩

theorem vxor_vxor (i j : Vertex) : vxor (vxor i j) j = i := by revert i j; decide

def vxorEquiv (j : Vertex) : Vertex ≃ Vertex :=
  ⟨fun i => vxor i j, fun i => vxor i j, fun i => vxor_vxor i j, fun i => vxor_vxor i j⟩

theorem xor_conv_sum (ψ : Psi) (N : ℚ) :
    (∑ i, (∑ j, ψ j * ψ (vxor i j)) / N) = (∑ i, ψ i) ^ 2 / N := by
  rw [← Finset.sum_div, Finset.sum_comm]
  congr 1
  have h : ∀ j, (∑ i, ψ j * ψ (vxor i j)) = ψ j * ∑ i, ψ i := by
    intro j
    rw [← Finset.mul_sum]
    congr 1
    exact Equiv.sum_comp (vxorEquiv j) ψ
  simp only [h, sq, Finset.sum_mul]

theorem horner_eval (c0 c1 c2 c3 c4 s : ℚ) :
    c0 + s * (c1 + s * (c2 + s * (c3 + s * c4))) = c0 + c1 * s + c2 * s ^ 2 + c3 * s ^ 3 + c4 * s ^ 4 := by
  ring

theorem shunyam_pair_contract (x y α : ℚ) :
    (x + (-y - x) * α) + (y + (-x - y) * α) = (1 - 2 * α) * (x + y) := by ring

theorem anurupye_fixed_point (e α s : ℚ) : -(e * α * |s|) = 0 ↔ e = 0 ∨ α = 0 ∨ s = 0 := by
  rw [neg_eq_zero, mul_eq_zero, mul_eq_zero, abs_eq_zero, or_assoc]

theorem sankalana_lower_fixed (x y δ : ℚ) : ((x + y) / 2 + δ) + ((x - y) / 2 - δ) = x := by ring

def flipEquiv (k : Axis) : Vertex ≃ Vertex :=
  ⟨fun i => flip i k, fun i => flip i k, fun i => by revert i k; decide, fun i => by revert i k; decide⟩

theorem laplacian_sum_zero (ψ : Psi) : (∑ i, ∑ k : Axis, (ψ (flip i k) - ψ i)) = 0 := by
  rw [Finset.sum_comm]
  apply Finset.sum_eq_zero
  intro k _
  rw [Finset.sum_sub_distrib, sub_eq_zero]
  exact Equiv.sum_comp (flipEquiv k) ψ

def layerSum (ψ : Psi) (h : ℕ) : ℚ := ∑ j ∈ univ.filter (fun j => hw j = h), ψ j

def layerCount (h : ℕ) : ℕ := (univ.filter (fun j : Vertex => hw j = h)).card

theorem layer_mean_sum (ψ : Psi) :
    (∑ i, layerSum ψ (hw i) / layerCount (hw i)) = ∑ i, ψ i := by
  have hc : ∀ i : Vertex, (layerCount (hw i) : ℚ) ≠ 0 := by
    intro i; unfold layerCount
    exact_mod_cast (Finset.card_pos.mpr ⟨i, by simp⟩).ne'
  rw [← Finset.sum_fiberwise_of_maps_to (g := hw) (t := univ.image hw) (fun i _ => mem_image_of_mem hw (mem_univ i))]
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (g := hw) (t := univ.image hw)
      (fun i _ => mem_image_of_mem hw (mem_univ i)) (f := ψ)]
  apply Finset.sum_congr rfl
  intro h hh
  obtain ⟨i0, -, rfl⟩ := mem_image.mp hh
  have hfib : ∀ x ∈ univ.filter (fun j => hw j = hw i0),
      layerSum ψ (hw x) / layerCount (hw x) = layerSum ψ (hw i0) / layerCount (hw i0) := by
    intro x hx; rw [(mem_filter.mp hx).2]
  rw [Finset.sum_congr rfl hfib, Finset.sum_const, nsmul_eq_mul]
  unfold layerSum
  have := hc i0
  unfold layerCount at this ⊢
  field_simp

def transfer (ψ : Psi) (i j : Vertex) (δ : ℚ) : Psi :=
  fun v => ψ v - (if v = i then δ else 0) + (if v = j then δ else 0)

theorem transfer_sum (ψ : Psi) (i j : Vertex) (δ : ℚ) : (∑ v, transfer ψ i j δ v) = ∑ v, ψ v := by
  unfold transfer
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_ite_eq', Finset.sum_ite_eq']
  simp

theorem conservation_mean (ψ : Psi) (α : ℚ) :
    mean (fun i => ψ i + (mean ψ - ψ i) * α) = mean ψ := by
  unfold mean
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  push_cast
  ring

theorem digital_root_congr (v m : ℕ) (hm : 0 < m) (hv : 0 < v) :
    1 ≤ 1 + (v - 1) % m ∧ 1 + (v - 1) % m ≤ m ∧ (1 + (v - 1) % m) ≡ v [MOD m] := by
  refine ⟨by omega, by have := Nat.mod_lt (v - 1) hm; omega, ?_⟩
  have h := (Nat.mod_modEq (v - 1) m).add_left 1
  have hv' : 1 + (v - 1) = v := by omega
  rw [hv'] at h
  exact h

def cfStep (s : ℤ × ℤ × ℤ × ℤ) (a : ℤ) : ℤ × ℤ × ℤ × ℤ :=
  (s.2.1, a * s.2.1 + s.1, s.2.2.2, a * s.2.2.2 + s.2.2.1)

def cfDet (s : ℤ × ℤ × ℤ × ℤ) : ℤ := s.2.1 * s.2.2.1 - s.1 * s.2.2.2

theorem cf_det_step (s : ℤ × ℤ × ℤ × ℤ) (a : ℤ) : cfDet (cfStep s a) = -cfDet s := by
  unfold cfDet cfStep; ring

theorem convergent_det (c0 : ℤ) (l : List ℤ) : |cfDet (l.foldl cfStep (1, c0, 0, 1))| = 1 := by
  suffices h : ∀ (l : List ℤ) (s : ℤ × ℤ × ℤ × ℤ), |cfDet s| = 1 → |cfDet (l.foldl cfStep s)| = 1 by
    exact h l _ (by simp [cfDet])
  intro l
  induction l with
  | nil => intro s hs; simpa using hs
  | cons a _ ih => intro s hs; exact ih _ (by rw [cf_det_step, abs_neg]; exact hs)

theorem ekanyunena_contracts (x y α c : ℚ) (hc : 0 < c) (hα : 0 ≤ α) (hαc : α ≤ c) :
    |x * (1 + α * -(1 / (|y| + c)))| ≤ |x| := by
  have hd : 0 < |y| + c := by positivity
  have h1 : α / (|y| + c) ≤ 1 := by
    rw [div_le_one hd]; linarith [abs_nonneg y]
  have h0 : 0 ≤ α / (|y| + c) := div_nonneg hα hd.le
  have hm : 1 + α * -(1 / (|y| + c)) = 1 - α / (|y| + c) := by ring
  rw [hm, abs_mul, abs_of_nonneg (show (0 : ℚ) ≤ 1 - α / (|y| + c) by linarith)]
  nlinarith [abs_nonneg x]

theorem gunita_residual_cross (x y : Fin 8 → ℚ) :
    (∑ b, y b) * (∑ b, x b) - ∑ b, x b * y b = ∑ b, ∑ c ∈ univ.erase b, x b * y c := by
  rw [mul_comm, Finset.sum_mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro b _
  rw [← Finset.add_sum_erase _ _ (mem_univ b)]
  ring

theorem ext_euclid_step (a b r0 r1 s0 s1 t0 t1 q : ℤ) (h0 : s0 * a + t0 * b = r0) (h1 : s1 * a + t1 * b = r1) :
    (s0 - q * s1) * a + (t0 - q * t1) * b = r0 - q * r1 := by
  rw [← h0, ← h1]; ring

theorem bezout_identity (a b s t g : ℤ) (h : s * a + t * b = g) (ha : g ∣ a) (hb : g ∣ b) (hg : 0 ≤ g) :
    g = Int.gcd a b := by
  have h1 : g ∣ (Int.gcd a b : ℤ) := Int.dvd_gcd ha hb
  have h2 : (Int.gcd a b : ℤ) ∣ g := by
    rw [← h]; exact dvd_add (dvd_mul_of_dvd_right Int.gcd_dvd_left _) (dvd_mul_of_dvd_right Int.gcd_dvd_right _)
  exact Int.dvd_antisymm hg (Int.natCast_nonneg _) h1 h2

theorem crt_residue_step (v r m : ℤ) (_hm : 0 < m) (hr0 : 0 ≤ r) (hrm : r < m) :
    (v + (r - v % m + m) % m) % m = r := by
  have h1 : (v + (r - v % m + m) % m) % m = (v + (r - v % m + m)) % m := by
    rw [Int.add_emod, Int.emod_emod_of_dvd _ (dvd_refl m), ← Int.add_emod]
  have h2 : v + (r - v % m + m) = r + m * (v / m + 1) := by
    have := Int.emod_add_ediv v m; linarith
  rw [h1, h2, Int.add_mul_emod_self_left, Int.emod_eq_of_lt hr0 hrm]

theorem vieta_disc_square (x y : ℚ) : (x + y) ^ 2 - 4 * (x * y) = (x - y) ^ 2 ∧ 0 ≤ (x + y) ^ 2 - 4 * (x * y) := by
  constructor
  · ring
  · nlinarith [sq_nonneg (x - y)]

theorem nearest_multiple (v b : ℤ) (hb : 0 < b) :
    let r := v % b
    let adj := if r ≤ b / 2 then -r else b - r
    b ∣ v + adj ∧ 2 * |adj| ≤ b := by
  intro r adj
  have hr0 : 0 ≤ r := Int.emod_nonneg v hb.ne'
  have hrb : r < b := Int.emod_lt_of_pos v hb
  have hdiv : v = b * (v / b) + r := (Int.ediv_add_emod v b).symm
  have hh : 2 * (b / 2) ≤ b ∧ b < 2 * (b / 2) + 2 := by omega
  by_cases hc : r ≤ b / 2
  · have hadj : adj = -r := if_pos hc
    rw [hadj, abs_neg, abs_of_nonneg hr0]
    exact ⟨⟨v / b, by linarith⟩, by omega⟩
  · have hadj : adj = b - r := if_neg hc
    rw [hadj, abs_of_nonneg (by linarith)]
    exact ⟨⟨v / b + 1, by linarith⟩, by omega⟩

theorem osculation_shift (osc m : ℤ) (hm : 0 < m) :
    let d := if osc % m = 0 then 0 else m - osc % m
    0 ≤ d ∧ d < m ∧ m ∣ osc + d := by
  intro d
  have hr0 : 0 ≤ osc % m := Int.emod_nonneg osc hm.ne'
  have hrb : osc % m < m := Int.emod_lt_of_pos osc hm
  have hdiv : osc = m * (osc / m) + osc % m := (Int.ediv_add_emod osc m).symm
  by_cases hc : osc % m = 0
  · have hd : d = 0 := if_pos hc
    rw [hd]; exact ⟨le_refl _, hm, ⟨osc / m, by linarith⟩⟩
  · have hd : d = m - osc % m := if_neg hc
    rw [hd]; exact ⟨by linarith, by omega, ⟨osc / m + 1, by linarith⟩⟩

theorem dvandva_pair_contract (x y α : ℚ) :
    let x' := x + ((x + y) / 2 - x) * α
    let y' := y + ((y + x') / 2 - y) * α
    x' - y' = (1 - α / 2) ^ 2 * (x - y) := by
  intro x' y'; simp only [x', y']; ring

theorem antyayor_pair_difference (x y a : ℚ) : (x + a) - (y + a) = x - y := by ring

theorem gcd_divides_tails (l : List ℕ) (g0 : ℕ) : l.foldl Nat.gcd g0 ∣ g0 ∧ ∀ t ∈ l, l.foldl Nat.gcd g0 ∣ t := by
  induction l generalizing g0 with
  | nil => simp
  | cons a t ih =>
    obtain ⟨h1, h2⟩ := ih (Nat.gcd g0 a)
    refine ⟨h1.trans (Nat.gcd_dvd_left _ _), ?_⟩
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hx
    · exact h1.trans (Nat.gcd_dvd_right _ _)
    · exact h2 x hx

theorem digital_root_nine (n : ℕ) : (Nat.digits 10 n).sum ≡ n [MOD 9] :=
  (Nat.modEq_nine_digits_sum n).symm

theorem log_ratio_sign (S T : ℝ) (hS : S ≠ 0) (hT : T ≠ 0) :
    (0 < Real.log |S| - Real.log |T| ↔ |T| < |S|) ∧ (Real.log |S| - Real.log |T| = 0 ↔ |S| = |T|) := by
  have hs : 0 < |S| := abs_pos.mpr hS
  have ht : 0 < |T| := abs_pos.mpr hT
  constructor
  · rw [sub_pos, Real.log_lt_log_iff ht hs]
  · rw [sub_eq_zero]
    exact ⟨fun h => Real.log_injOn_pos (Set.mem_Ioi.mpr hs) (Set.mem_Ioi.mpr ht) h, fun h => by rw [h]⟩

def flagStep (ψ : ℕ → ℚ) (acc : ℕ × ℚ) (i : ℕ) : ℕ × ℚ := if acc.2 < |ψ i| then (i, |ψ i|) else acc

theorem flag_is_max (ψ : ℕ → ℚ) (l : List ℕ) (acc : ℕ × ℚ) :
    acc.2 ≤ (l.foldl (flagStep ψ) acc).2 ∧ ∀ i ∈ l, |ψ i| ≤ (l.foldl (flagStep ψ) acc).2 := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
    obtain ⟨h1, h2⟩ := ih (flagStep ψ acc a)
    have hstep : acc.2 ≤ (flagStep ψ acc a).2 ∧ |ψ a| ≤ (flagStep ψ acc a).2 := by
      unfold flagStep; split_ifs with h
      · exact ⟨h.le, le_refl _⟩
      · exact ⟨le_refl _, not_lt.mp h⟩
    refine ⟨hstep.1.trans h1, ?_⟩
    intro i hi
    rcases List.mem_cons.mp hi with rfl | hi
    · exact hstep.2.trans h1
    · exact h2 i hi

theorem ap_endpoints (a0 a1 a2 a3 a4 : ℚ) :
    let md := ((a1 - a0) + (a2 - a1) + (a3 - a2) + (a4 - a3)) / 4
    a0 + 0 * md - a0 = 0 ∧ a0 + 4 * md - a4 = 0 := by
  intro md; simp only [md]; constructor <;> ring

theorem purana_uniform_fixed (c w : ℚ) : c + (((c + c) / 2 + (c + c) / 2 + (c + c) / 2 + (c + c) / 2) / 4 - c) * w = c := by
  ring

theorem anurupyena_uniform_fixed (c φ α : ℚ) : c + ((c - c) + (c - c) + (c - c) + (c - c)) * φ / 4 * α = c := by
  ring

theorem walk_measure_total (α wp wn : ℚ) (h : wp + wn ≠ 0) :
    α + (1 - α) * (wp / (wp + wn)) + (1 - α) * (wn / (wp + wn)) = 1 := by
  field_simp; ring

theorem walk_measure_total_split (α : ℚ) :
    α + (1 - α) * (1 / 2) + (1 - α) * (1 / 2) = 1 := by
  ring

theorem median_minimises_l1 (a b c d x : ℝ) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
    |a - b| + |b - b| + |c - b| + |d - b| ≤ |a - x| + |b - x| + |c - x| + |d - x| := by
  have e1 : d - a ≤ |a - x| + |d - x| := by
    have := abs_sub_le d x a; rw [abs_sub_comm x a] at this; rw [abs_of_nonneg (by linarith : 0 ≤ d - a)] at this; linarith
  have e2 : c - b ≤ |b - x| + |c - x| := by
    have := abs_sub_le c x b; rw [abs_sub_comm x b] at this; rw [abs_of_nonneg (by linarith : 0 ≤ c - b)] at this; linarith
  rw [abs_of_nonpos (by linarith : a - b ≤ 0), sub_self, abs_zero, abs_of_nonneg (by linarith : 0 ≤ c - b),
    abs_of_nonneg (by linarith : 0 ≤ d - b)]
  linarith

theorem inverse_inject (ψ : Psi) (k : Axis) (δ : ℚ) :
    inverse (fun i => ψ i + (sgn i k : ℚ) * δ) = fun j => inverse ψ j + if j = k then δ else 0 := by
  funext j
  unfold inverse
  have h := sgn_orthogonal j k
  have hq : (∑ i : Vertex, (sgn i j : ℚ) * (sgn i k : ℚ)) = if j = k then 16 else 0 := by
    have := congrArg (fun z : ℤ => (z : ℚ)) h
    push_cast at this; split_ifs at this ⊢ <;> simpa using this
  simp only [mul_add, Finset.sum_add_distrib, add_div]
  congr 1
  rw [show (∑ i : Vertex, (sgn i j : ℚ) * ((sgn i k : ℚ) * δ)) = (∑ i : Vertex, (sgn i j : ℚ) * (sgn i k : ℚ)) * δ by
    rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun i _ => by ring, hq]
  split_ifs <;> ring

theorem stress_trace (B : Fin 3 → ℚ) :
    (∑ i, (B i * B i - (1 / 2) * ∑ j, B j * B j)) = -(1 / 2) * ∑ j, B j * B j := by
  simp [Fin.sum_univ_succ]; ring

theorem stress_symm (B : Fin 3 → ℚ) (i j : Fin 3) (b2 : ℚ) :
    B i * B j + (if i = j then -(b2 / 2) else 0) = B j * B i + (if j = i then -(b2 / 2) else 0) := by
  rw [mul_comm]; split_ifs with h1 h2 h2 <;> first | rfl | (exact absurd h1.symm h2) | (exact absurd h2.symm h1)

theorem rot_pair_energy (x y t : ℚ) :
    let c := (1 - t ^ 2) / (1 + t ^ 2)
    let s := 2 * t / (1 + t ^ 2)
    (c * x + s * y) ^ 2 + (c * y - s * x) ^ 2 = x ^ 2 + y ^ 2 := by
  intro c s
  have h : (1 + t ^ 2) ≠ 0 := by positivity
  simp only [c, s]
  field_simp
  ring

theorem complement_rotation_swap (x y t : ℚ) :
    let c := (1 - t ^ 2) / (1 + t ^ 2)
    let s := 2 * t / (1 + t ^ 2)
    let c' := (1 - (-t) ^ 2) / (1 + (-t) ^ 2)
    let s' := 2 * (-t) / (1 + (-t) ^ 2)
    c' * y + s' * x = c * y - s * x ∧ c' * x - s' * y = c * x + s * y := by
  intro c s c' s'
  simp only [c, s, c', s']
  constructor <;> ring

theorem pontryagin_density (B01 B02 B03 B12 B13 B23 : ℚ) :
    ((B01 + B23) / 2) ^ 2 + ((B02 - B13) / 2) ^ 2 + ((B03 + B12) / 2) ^ 2
      - (((B01 - B23) / 2) ^ 2 + ((B02 + B13) / 2) ^ 2 + ((B03 - B12) / 2) ^ 2)
      = B01 * B23 - B02 * B13 + B03 * B12 := by ring

theorem hodge_split_norm (B01 B02 B03 B12 B13 B23 : ℚ) :
    ((B01 + B23) / 2) ^ 2 + ((B02 - B13) / 2) ^ 2 + ((B03 + B12) / 2) ^ 2
      + (((B01 - B23) / 2) ^ 2 + ((B02 + B13) / 2) ^ 2 + ((B03 - B12) / 2) ^ 2)
      = (B01 ^ 2 + B02 ^ 2 + B03 ^ 2 + B12 ^ 2 + B13 ^ 2 + B23 ^ 2) / 2 := by ring

end Vedic
end SutraWS
