import SutraWS.WheelerGeometry
import SutraWS.Interpolation
import Mathlib.Tactic

/-!
# The Wheeler surface terms

`projectTorus` and `renderTorus` in `vedic_v18.51.1_exact_phi.html` shape and
colour the Wheeler channel with five helper formulas on `WHEELER`: the
centripetal/centrifugal balance `centBal`, the golden vortex, the field
distortion, the ether pressure and the counterspace intensity. This file
transcribes them and proves what keeps them from blowing up or inverting:

* `centBal_abs_le`, `goldenVortex_abs_le` — both tables are bounded, and the
  vortex is the balance scaled by `α` (`goldenVortex_eq_alpha_centBal`);
* `fieldDistortion_mem` — the distortion is nonnegative and bounded, since
  `α(√2 + √3) < 1` (`alpha_bound_lt_one`);
* `etherPressure_abs_le`, `counterspace_abs_le` — bounded by the inputs they scale;
* `zeta_pow` — `KConsts.powZeta`, the exact `ζ = (√2/2)(1 + i)`, has
  `ζⁿ = cos(nπ/4) + i·sin(nπ/4)`, which is what the tables read off it.

The inertial plane is the kernel's layer `k = hw − 2 = 0`, where magnetism
vanishes (`Wheeler.wheeler_inertial_plane_magnetism_zero`). Distance from it is
`k`:

* `inertiaEnvelope_vertex` — the counterspace envelope `1/(1 + k²)` is
  `1 − magneticWeight`, and it equals 1 exactly on the plane;
* `layer_cos_vertex_eq_one_iff`, `latitude_band_eq_one_iff` — likewise for the
  flux-phase term `cos(kπ/4)` and the colour band `exp(−3(kπ/4)²)`;
* `inertial_locus_nodes` — the surface's `k` channel vanishes at a vertex node
  exactly when that vertex is on the plane, so its zero set is the plane.

`bilinear_abs_le` is why normalising `ω` by the largest grid value keeps every
sampled `wN` in `[−1, 1]`, which `fieldDistortion_mem`, `etherPressure_abs_le` and
`counterspace_abs_le` assume.

`√3`, `π` and `φ³ = 2 + √5` appear as reals. The page holds the tables'
`√2`-and-rational parts exactly and multiplies by `√3` at the float boundary,
as it does for `π`.
-/

namespace SutraWS
namespace WheelerSurface

open Real

noncomputable section

/-- `C.alpha = 1000000 / 137035999`. -/
def alpha : ℚ := 1000000 / 137035999

/-- `KX.PHI_CU = 2 + √5 = φ³`. -/
def phi3 : ℝ := 2 + √5

/-- `centBal`, octants `a = _oct(θ)`, `b = _oct(φ)`: `√2·Re(ζᵃ) − √3·Im(ζᵇ)`. -/
def centBal (a b : ℕ) : ℝ := √2 * Real.cos (a * π / 4) - √3 * Real.sin (b * π / 4)

/-- `goldenVortexQ(i, j) = α(√2·Re(ζⁱ) − √3·Im(ζʲ))`. -/
def goldenVortex (i j : ℕ) : ℝ :=
  (alpha : ℝ) * (√2 * Real.cos (i * π / 4) - √3 * Real.sin (j * π / 4))

/-- `fieldDistortion`: `8πφ³ · rad · (1 + vortex·wN)`. -/
def fieldDistortion (rad gv w : ℝ) : ℝ := 8 * π * phi3 * rad * (1 + gv * w)

/-- `etherPressure`: `φ³ · rad · centBal · (1 + 0.15·wN)`. -/
def etherPressure (rad cb w : ℝ) : ℝ := phi3 * rad * cb * (1 + 3 / 20 * w)

/-- The counterspace envelope `1/(1 + k²)`, `k` the layer distance from the plane. -/
def inertiaEnvelope (k : ℝ) : ℝ := 1 / (1 + k ^ 2)

/-- `counterspaceIntensity`: `(f/α) · 1/(1 + k²) · (1 + 0.3·vortex·wN)`. -/
def counterspaceIntensity (f k gv w : ℝ) : ℝ :=
  f / (alpha : ℝ) * inertiaEnvelope k * (1 + 3 / 10 * gv * w)

theorem zeta_pow (n : ℕ) :
    ((↑(√2 / 2) + ↑(√2 / 2) * Complex.I : ℂ) ^ n).re = Real.cos (n * π / 4) ∧
    ((↑(√2 / 2) + ↑(√2 / 2) * Complex.I : ℂ) ^ n).im = Real.sin (n * π / 4) := by
  rw [← Interp.exp_pi_div_four_mul_I, ← Complex.exp_nat_mul,
    show ((n : ℂ) * (↑(π / 4) * Complex.I)) = ↑((n : ℝ) * π / 4) * Complex.I by push_cast; ring]
  exact ⟨Complex.exp_ofReal_mul_I_re _, Complex.exp_ofReal_mul_I_im _⟩

theorem centBal_abs_le (a b : ℕ) : |centBal a b| ≤ √2 + √3 := by
  unfold centBal
  have h2 : 0 ≤ √2 := Real.sqrt_nonneg 2
  have h3 : 0 ≤ √3 := Real.sqrt_nonneg 3
  calc |√2 * Real.cos (a * π / 4) - √3 * Real.sin (b * π / 4)|
      ≤ |√2 * Real.cos (a * π / 4)| + |√3 * Real.sin (b * π / 4)| := abs_sub _ _
    _ = √2 * |Real.cos (a * π / 4)| + √3 * |Real.sin (b * π / 4)| := by
        rw [abs_mul, abs_mul, abs_of_nonneg h2, abs_of_nonneg h3]
    _ ≤ √2 * 1 + √3 * 1 := by
        gcongr
        · exact Real.abs_cos_le_one _
        · exact Real.abs_sin_le_one _
    _ = √2 + √3 := by ring

theorem goldenVortex_eq_alpha_centBal (i j : ℕ) :
    goldenVortex i j = (alpha : ℝ) * centBal i j := rfl

theorem alpha_pos : (0 : ℝ) < alpha := by unfold alpha; norm_num

theorem alpha_bound_lt_one : (alpha : ℝ) * (√2 + √3) < 1 := by
  have h2 : √2 < 2 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have h3 : √3 < 2 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ha : (alpha : ℝ) < 1 / 4 := by unfold alpha; norm_num
  nlinarith [alpha_pos]

theorem goldenVortex_abs_le (i j : ℕ) : |goldenVortex i j| ≤ (alpha : ℝ) * (√2 + √3) := by
  rw [goldenVortex_eq_alpha_centBal, abs_mul, abs_of_pos alpha_pos]
  exact mul_le_mul_of_nonneg_left (centBal_abs_le i j) alpha_pos.le

theorem phi3_pos : 0 < phi3 := by unfold phi3; positivity

theorem fieldDistortion_mem (rad gv w : ℝ) (h0 : 0 ≤ rad) (h1 : rad ≤ phi3)
    (hg : |gv| ≤ (alpha : ℝ) * (√2 + √3)) (hw : |w| ≤ 1) :
    0 ≤ fieldDistortion rad gv w ∧
      fieldDistortion rad gv w ≤ 8 * π * phi3 ^ 2 * (1 + (alpha : ℝ) * (√2 + √3)) := by
  have hgw : |gv * w| ≤ (alpha : ℝ) * (√2 + √3) := by
    rw [abs_mul]
    calc |gv| * |w| ≤ (alpha : ℝ) * (√2 + √3) * 1 :=
          mul_le_mul hg hw (abs_nonneg _) (le_trans (abs_nonneg _) hg)
      _ = _ := mul_one _
  have hlo : 0 ≤ 1 + gv * w := by
    have := neg_abs_le (gv * w); linarith [alpha_bound_lt_one]
  have hhi : 1 + gv * w ≤ 1 + (alpha : ℝ) * (√2 + √3) := by
    have := le_abs_self (gv * w); linarith
  have hk : 0 ≤ 8 * π * phi3 := by have := phi3_pos; positivity
  unfold fieldDistortion
  constructor
  · exact mul_nonneg (mul_nonneg hk h0) hlo
  · calc 8 * π * phi3 * rad * (1 + gv * w)
        ≤ 8 * π * phi3 * phi3 * (1 + (alpha : ℝ) * (√2 + √3)) :=
          mul_le_mul (mul_le_mul_of_nonneg_left h1 hk) hhi hlo (by
            have := phi3_pos; positivity)
      _ = _ := by ring

theorem etherPressure_abs_le (rad cb w : ℝ) (h0 : 0 ≤ rad) (h1 : rad ≤ phi3)
    (hc : |cb| ≤ √2 + √3) (hw : |w| ≤ 1) :
    |etherPressure rad cb w| ≤ phi3 ^ 2 * (√2 + √3) * (23 / 20) := by
  unfold etherPressure
  have hp : |1 + 3 / 20 * w| ≤ 23 / 20 := by
    rw [abs_le] at hw ⊢; constructor <;> linarith
  rw [abs_mul, abs_mul, abs_mul, abs_of_pos phi3_pos, abs_of_nonneg h0]
  have := phi3_pos
  calc phi3 * rad * |cb| * |1 + 3 / 20 * w| ≤ phi3 * phi3 * (√2 + √3) * (23 / 20) := by
        gcongr
    _ = _ := by ring

theorem inertiaEnvelope_eq_one_sub_magneticWeight (k : ℝ) :
    inertiaEnvelope k = 1 - k ^ 2 / (1 + k ^ 2) := by
  unfold inertiaEnvelope
  have : (1 : ℝ) + k ^ 2 ≠ 0 := by positivity
  field_simp

theorem inertiaEnvelope_mem (k : ℝ) : 0 < inertiaEnvelope k ∧ inertiaEnvelope k ≤ 1 := by
  unfold inertiaEnvelope
  have hp : (0 : ℝ) < 1 + k ^ 2 := by positivity
  exact ⟨by positivity, by rw [div_le_one hp]; nlinarith [sq_nonneg k]⟩

theorem inertiaEnvelope_eq_one_iff (k : ℝ) : inertiaEnvelope k = 1 ↔ k = 0 := by
  unfold inertiaEnvelope
  have hp : (1 : ℝ) + k ^ 2 ≠ 0 := by positivity
  rw [div_eq_one_iff_eq hp]
  constructor
  · intro h; have : k ^ 2 = 0 := by linarith
    exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
  · intro h; subst h; norm_num

theorem inertiaEnvelope_vertex (v : Fin 16) :
    inertiaEnvelope (Wheeler.k v) = 1 - (Wheeler.magneticWeight v : ℝ) := by
  rw [inertiaEnvelope_eq_one_sub_magneticWeight]
  simp only [Wheeler.magneticWeight]
  push_cast
  ring

theorem inertiaEnvelope_vertex_eq_one_iff (v : Fin 16) :
    inertiaEnvelope (Wheeler.k v) = 1 ↔ Wheeler.hw v = 2 := by
  rw [inertiaEnvelope_eq_one_iff]
  have : ((Wheeler.k v : ℤ) : ℝ) = 0 ↔ Wheeler.k v = 0 := by exact_mod_cast Iff.rfl
  rw [this]
  unfold Wheeler.k
  omega

theorem counterspace_abs_le (f k gv w : ℝ) (hg : |gv| ≤ (alpha : ℝ) * (√2 + √3))
    (hw : |w| ≤ 1) :
    |counterspaceIntensity f k gv w|
      ≤ |f| / (alpha : ℝ) * (1 + 3 / 10 * ((alpha : ℝ) * (√2 + √3))) := by
  unfold counterspaceIntensity
  obtain ⟨he0, he1⟩ := inertiaEnvelope_mem k
  have hgw : |gv * w| ≤ (alpha : ℝ) * (√2 + √3) := by
    rw [abs_mul]
    calc |gv| * |w| ≤ (alpha : ℝ) * (√2 + √3) * 1 :=
          mul_le_mul hg hw (abs_nonneg _) (le_trans (abs_nonneg _) hg)
      _ = _ := mul_one _
  have ht : |1 + 3 / 10 * gv * w| ≤ 1 + 3 / 10 * ((alpha : ℝ) * (√2 + √3)) := by
    have := abs_le.mp hgw
    rw [abs_le]; constructor <;> nlinarith [alpha_bound_lt_one]
  rw [abs_mul, abs_mul, abs_div, abs_of_pos (show (0 : ℝ) < alpha from alpha_pos),
    abs_of_pos he0]
  have := alpha_pos
  calc |f| / (alpha : ℝ) * inertiaEnvelope k * |1 + 3 / 10 * gv * w|
      ≤ |f| / (alpha : ℝ) * 1 * (1 + 3 / 10 * ((alpha : ℝ) * (√2 + √3))) := by
        gcongr
    _ = _ := by ring

theorem inertial_locus_nodes (xw yz : ℝ) (v : Fin 16) :
    Interp.realEvaluator (Interp.coeffs (Interp.scatter fun u => ((Wheeler.k u : ℝ) : ℂ))) xw yz
        (Interp.tessVertexToTorus xw yz 0 v).1 (Interp.tessVertexToTorus xw yz 0 v).2 = 0
      ↔ Wheeler.hw v = 2 := by
  rw [Interp.real_node_reproduction]
  have : ((Wheeler.k v : ℤ) : ℝ) = 0 ↔ Wheeler.k v = 0 := by exact_mod_cast Iff.rfl
  rw [this]
  unfold Wheeler.k
  omega

theorem bilinear_abs_le (a b c d M ft fp : ℝ) (ha : |a| ≤ M) (hb : |b| ≤ M) (hc : |c| ≤ M)
    (hd : |d| ≤ M) (h0 : 0 ≤ ft) (h1 : ft ≤ 1) (g0 : 0 ≤ fp) (g1 : fp ≤ 1) :
    |(1 - ft) * (1 - fp) * a + ft * (1 - fp) * b + (1 - ft) * fp * c + ft * fp * d| ≤ M := by
  have w1 : 0 ≤ (1 - ft) * (1 - fp) := mul_nonneg (by linarith) (by linarith)
  have w2 : 0 ≤ ft * (1 - fp) := mul_nonneg h0 (by linarith)
  have w3 : 0 ≤ (1 - ft) * fp := mul_nonneg (by linarith) g0
  have w4 : 0 ≤ ft * fp := mul_nonneg h0 g0
  rw [abs_le] at ha hb hc hd ⊢
  constructor <;> nlinarith [ha.1, ha.2, hb.1, hb.2, hc.1, hc.2, hd.1, hd.2]

theorem latitude_band_eq_one_iff (k : ℝ) :
    Real.exp (-3 * (k * π / 4) ^ 2) = 1 ↔ k = 0 := by
  rw [Real.exp_eq_one_iff]
  constructor
  · intro h
    have h0 : (k * π / 4) ^ 2 = 0 := by linarith
    have h1 : k * π / 4 = 0 := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp h0
    rcases mul_eq_zero.mp ((div_eq_zero_iff.mp h1).resolve_right (by norm_num)) with h | h
    · exact h
    · exact absurd h Real.pi_ne_zero
  · intro h; subst h; ring

theorem layer_cos_eq_one_iff (k : ℤ) (hk : -2 ≤ k ∧ k ≤ 2) :
    Real.cos (k * π / 4) = 1 ↔ k = 0 := by
  have h2 : (0 : ℝ) < √2 / 2 ∧ √2 / 2 < 1 := by
    constructor
    · positivity
    · have : √2 < 2 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
      linarith
  obtain ⟨lo, hi⟩ := hk
  interval_cases k
  · simp only [show ((-2 : ℤ) : ℝ) * π / 4 = -(π / 2) by push_cast; ring, Real.cos_neg,
      Real.cos_pi_div_two]
    norm_num
  · simp only [show ((-1 : ℤ) : ℝ) * π / 4 = -(π / 4) by push_cast; ring, Real.cos_neg,
      Real.cos_pi_div_four]
    constructor
    · intro h; linarith [h2.2]
    · intro h; norm_num at h
  · simp
  · simp only [show ((1 : ℤ) : ℝ) * π / 4 = π / 4 by push_cast; ring, Real.cos_pi_div_four]
    constructor
    · intro h; linarith [h2.2]
    · intro h; norm_num at h
  · simp only [show ((2 : ℤ) : ℝ) * π / 4 = π / 2 by push_cast; ring, Real.cos_pi_div_two]
    norm_num

theorem layer_cos_vertex_eq_one_iff (v : Fin 16) :
    Real.cos (Wheeler.k v * π / 4) = 1 ↔ Wheeler.hw v = 2 := by
  have hk : -2 ≤ Wheeler.k v ∧ Wheeler.k v ≤ 2 := by revert v; decide
  rw [layer_cos_eq_one_iff _ hk]
  unfold Wheeler.k
  omega

end

end WheelerSurface
end SutraWS
