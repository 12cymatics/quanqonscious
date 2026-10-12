import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Tactic

/-!
# Maya: the coherence-layer flow

The Maya illusion sutra of v18.24 (§7, "Maya quantum state predictor"):

    dφ/dμ = β(φ) / μ,     β(φ) = (φ_g / e) · sin φ − (e / π) · φ³,

`φ_g` the golden ratio. Its fixed points are the coherence layers; the flow
carries the Maya phase onto the stable one. The Maya factor
`M = 1 + β / (20 (1 + |β|))` is v18.24's `M = observed / true`: it scales the
field where it is shown and never the field itself. The kernel takes the flow
as v18.24 stepped it: `d(log μ) = 1/100` per kernel step, in five Euler
sub-steps of `1/500`.

`e`, `π`, `√5` and `sin` are not in the kernel's number field, so the runtime
evaluates `β` at a stored rational phase inside rational enclosures. This file
proves that those enclosures hold and what the flow does:

* `sin_enclose`, `cos_enclose` — the Taylor sum of `exp (q i)` to `n` terms is
  within `|q|^n (n+1)/(n! n)` of `cos q + i sin q` (from `Complex.exp_bound`);
* `e_enclose` — the same for `e = exp 1`;
* `pi_gt_of_sin_lt_cos`, `pi_lt_of_cos_lt_sin` — a rational `q ∈ [0, 1]` with
  `sin q < cos q` lies below `π/4`, one with `cos q < sin q` above it, so two
  sign checks bracket `π`;
* `sqrt_lower`, `sqrt_upper` — integer square roots bracket `√m`;
* `mul_mem_hull`, `div_mem_pos`, `floor_grid_le`, `le_ceil_grid` — the interval
  products, quotients and outward rounding the enclosure is assembled from;
* `beta_lipschitz`, `euler_err` — how far a computed phase can be from the
  phase of the exact Euler map, step by step;
* `beta_pos_small`, `beta_strict_anti`, `layer` — `β` is positive from `0` to a
  single root `φ*` in any bracket `[lo, hi] ⊆ [3/4, 1]` where it changes sign,
  and negative after it: one stable coherence layer;
* `toward_layer`, `layer_attracts_above` — an Euler step moves the phase toward
  `φ*` and never past it;
* `factor_mem`, `factor_eq_one_iff`, `factor_gt_one_iff`, `factor_lipschitz` —
  the Maya factor stays in `(19/20, 21/20)`, is `1` exactly on a layer, and
  moves by at most `1/20` of a change in `β`.
-/

namespace SutraWS
namespace MayaFlow

open Real Finset

/-! ## Taylor enclosures of `cos` and `sin` -/

/-- Coefficient of `q^m / m!` in the real part of `exp (q i)`. -/
noncomputable def reCoef (m : ℕ) : ℝ :=
  if m % 4 = 0 then 1 else if m % 4 = 2 then -1 else 0

/-- Coefficient of `q^m / m!` in the imaginary part of `exp (q i)`. -/
noncomputable def imCoef (m : ℕ) : ℝ :=
  if m % 4 = 1 then 1 else if m % 4 = 3 then -1 else 0

/-- The runtime's cosine sum. -/
noncomputable def cosSum (q : ℝ) (n : ℕ) : ℝ :=
  ∑ m ∈ range n, reCoef m * (q ^ m / m.factorial)

/-- The runtime's sine sum. -/
noncomputable def sinSum (q : ℝ) (n : ℕ) : ℝ :=
  ∑ m ∈ range n, imCoef m * (q ^ m / m.factorial)

/-- The tail bound of `Complex.exp_bound`. -/
noncomputable def tail (q : ℝ) (n : ℕ) : ℝ :=
  |q| ^ n * ((n.succ : ℝ) * ((n.factorial : ℝ) * n)⁻¹)

theorem I_pow_mod (m : ℕ) : Complex.I ^ m = Complex.I ^ (m % 4) := by
  conv_lhs => rw [← Nat.div_add_mod m 4]
  rw [pow_add, pow_mul, Complex.I_pow_four, one_pow, one_mul]

theorem I_pow_eq (m : ℕ) : Complex.I ^ m = ⟨reCoef m, imCoef m⟩ := by
  rw [I_pow_mod]
  have h : m % 4 < 4 := Nat.mod_lt _ (by norm_num)
  unfold reCoef imCoef
  rcases (by omega : m % 4 = 0 ∨ m % 4 = 1 ∨ m % 4 = 2 ∨ m % 4 = 3) with h0 | h1 | h2 | h3
  · rw [h0]; apply Complex.ext <;> simp
  · rw [h1]; apply Complex.ext <;> simp
  · rw [h2]; apply Complex.ext <;> simp [pow_two]
  · rw [h3]; apply Complex.ext <;> simp [pow_succ, pow_two]

theorem expSum_re_im (q : ℝ) (n : ℕ) :
    (∑ m ∈ range n, ((q : ℂ) * Complex.I) ^ m / (m.factorial : ℂ)) =
      ⟨cosSum q n, sinSum q n⟩ := by
  apply Complex.ext
  · rw [Complex.re_sum]; unfold cosSum
    apply Finset.sum_congr rfl; intro m _
    rw [mul_pow, I_pow_eq, ← Complex.ofReal_pow, ← Complex.ofReal_natCast]
    simp only [Complex.div_ofReal_re, Complex.re_ofReal_mul]
    ring
  · rw [Complex.im_sum]; unfold sinSum
    apply Finset.sum_congr rfl; intro m _
    rw [mul_pow, I_pow_eq, ← Complex.ofReal_pow, ← Complex.ofReal_natCast]
    simp only [Complex.div_ofReal_im, Complex.im_ofReal_mul]
    ring

theorem exp_mul_I_err (q : ℝ) (hq : |q| ≤ 1) {n : ℕ} (hn : 0 < n) :
    Complex.abs (Complex.exp ((q : ℂ) * Complex.I) - ⟨cosSum q n, sinSum q n⟩) ≤ tail q n := by
  have habs : Complex.abs ((q : ℂ) * Complex.I) = |q| := by
    rw [map_mul, Complex.abs_I, mul_one, Complex.abs_ofReal]
  have h := Complex.exp_bound (x := (q : ℂ) * Complex.I) (by rw [habs]; exact hq) hn
  rw [expSum_re_im, habs] at h
  exact h

/-- **`cos_enclose`**: the runtime's cosine sum is within the tail of `cos q`. -/
theorem cos_enclose (q : ℝ) (hq : |q| ≤ 1) {n : ℕ} (hn : 0 < n) :
    |Real.cos q - cosSum q n| ≤ tail q n := by
  have h := exp_mul_I_err q hq hn
  have hre : (Complex.exp ((q : ℂ) * Complex.I) - ⟨cosSum q n, sinSum q n⟩).re =
      Real.cos q - cosSum q n := by
    rw [Complex.sub_re, Complex.exp_ofReal_mul_I_re]
  rw [← hre]
  exact le_trans (Complex.abs_re_le_abs _) h

/-- **`sin_enclose`**: the runtime's sine sum is within the tail of `sin q`. -/
theorem sin_enclose (q : ℝ) (hq : |q| ≤ 1) {n : ℕ} (hn : 0 < n) :
    |Real.sin q - sinSum q n| ≤ tail q n := by
  have h := exp_mul_I_err q hq hn
  have him : (Complex.exp ((q : ℂ) * Complex.I) - ⟨cosSum q n, sinSum q n⟩).im =
      Real.sin q - sinSum q n := by
    rw [Complex.sub_im, Complex.exp_ofReal_mul_I_im]
  rw [← him]
  exact le_trans (Complex.abs_im_le_abs _) h

/-- **`e_enclose`**: the partial sums of `1/m!` are within `(n+1)/(n! n)` of `e`. -/
theorem e_enclose {n : ℕ} (hn : 0 < n) :
    |Real.exp 1 - ∑ m ∈ range n, (1 : ℝ) / m.factorial| ≤
      (n.succ : ℝ) / ((n.factorial : ℝ) * n) := by
  have h := Real.exp_bound (x := 1) (by norm_num) hn
  simpa using h

/-! ## Bracketing `π` -/

/-- **`pi_gt_of_sin_lt_cos`**: a `q ∈ [0, 1]` with `sin q < cos q` is below `π/4`. -/
theorem pi_gt_of_sin_lt_cos {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (h : Real.sin q < Real.cos q) :
    4 * q < π := by
  by_contra hc
  push_neg at hc
  have h2 := Real.two_le_pi
  have hq : π / 4 ≤ q := by linarith
  have hcos : Real.cos q ≤ Real.cos (π / 4) :=
    Real.strictAntiOn_cos.antitoneOn ⟨by positivity, by linarith⟩ ⟨h0, by linarith⟩ hq
  have hsin : Real.sin (π / 4) ≤ Real.sin q :=
    Real.strictMonoOn_sin.monotoneOn ⟨by linarith, by linarith⟩ ⟨by linarith, by linarith⟩ hq
  rw [Real.cos_pi_div_four] at hcos
  rw [Real.sin_pi_div_four] at hsin
  linarith

/-- **`pi_lt_of_cos_lt_sin`**: a `q ∈ [0, 1]` with `cos q < sin q` is above `π/4`. -/
theorem pi_lt_of_cos_lt_sin {q : ℝ} (h0 : 0 ≤ q) (h : Real.cos q < Real.sin q) :
    π < 4 * q := by
  by_contra hc
  push_neg at hc
  have h2 := Real.two_le_pi
  have hq : q ≤ π / 4 := by linarith
  have hcos : Real.cos (π / 4) ≤ Real.cos q :=
    Real.strictAntiOn_cos.antitoneOn ⟨h0, by linarith⟩ ⟨by positivity, by linarith⟩ hq
  have hsin : Real.sin q ≤ Real.sin (π / 4) :=
    Real.strictMonoOn_sin.monotoneOn ⟨by linarith, by linarith⟩ ⟨by linarith, by linarith⟩ hq
  rw [Real.cos_pi_div_four] at hcos
  rw [Real.sin_pi_div_four] at hsin
  linarith

/-! ## Square roots, products, quotients and outward rounding -/

/-- **`sqrt_lower`**: `s² ≤ m 4^k` puts `s / 2^k` at or below `√m`. -/
theorem sqrt_lower {m s : ℝ} {k : ℕ} (h : s ^ 2 ≤ m * 4 ^ k) :
    s / 2 ^ k ≤ Real.sqrt m := by
  have h4 : (4 : ℝ) ^ k = (2 ^ k) ^ 2 := by rw [← pow_mul, mul_comm, pow_mul]; norm_num
  apply Real.le_sqrt_of_sq_le
  rw [div_pow, div_le_iff (by positivity), ← h4]
  exact h

/-- **`sqrt_upper`**: `m 4^k < (s+1)²` puts `√m` below `(s+1) / 2^k`. -/
theorem sqrt_upper {m s : ℝ} {k : ℕ} (hs : 0 ≤ s) (h : m * 4 ^ k < (s + 1) ^ 2) :
    Real.sqrt m < (s + 1) / 2 ^ k := by
  have h4 : (4 : ℝ) ^ k = (2 ^ k) ^ 2 := by rw [← pow_mul, mul_comm, pow_mul]; norm_num
  rw [Real.sqrt_lt' (by positivity), div_pow, lt_div_iff (by positivity), ← h4]
  exact h

/-- **`mul_mem_hull`**: a product lies between the least and greatest corner
products of the two intervals, whatever their signs. -/
theorem mul_mem_hull {x y a b c d : ℝ} (hxa : a ≤ x) (hxb : x ≤ b) (hyc : c ≤ y) (hyd : y ≤ d) :
    min (min (a * c) (a * d)) (min (b * c) (b * d)) ≤ x * y ∧
      x * y ≤ max (max (a * c) (a * d)) (max (b * c) (b * d)) := by
  constructor
  · rcases le_total 0 y with hy | hy
    · have h1 : a * y ≤ x * y := mul_le_mul_of_nonneg_right hxa hy
      rcases le_total 0 a with ha | ha
      · have h2 : a * c ≤ a * y := mul_le_mul_of_nonneg_left hyc ha
        exact le_trans (le_trans (min_le_left _ _) (min_le_left _ _)) (by linarith)
      · have h2 : a * d ≤ a * y := mul_le_mul_of_nonpos_left hyd ha
        exact le_trans (le_trans (min_le_left _ _) (min_le_right _ _)) (by linarith)
    · have h1 : b * y ≤ x * y := mul_le_mul_of_nonpos_right hxb hy
      rcases le_total 0 b with hb | hb
      · have h2 : b * c ≤ b * y := mul_le_mul_of_nonneg_left hyc hb
        exact le_trans (le_trans (min_le_right _ _) (min_le_left _ _)) (by linarith)
      · have h2 : b * d ≤ b * y := mul_le_mul_of_nonpos_left hyd hb
        exact le_trans (le_trans (min_le_right _ _) (min_le_right _ _)) (by linarith)
  · rcases le_total 0 y with hy | hy
    · have h1 : x * y ≤ b * y := mul_le_mul_of_nonneg_right hxb hy
      rcases le_total 0 b with hb | hb
      · have h2 : b * y ≤ b * d := mul_le_mul_of_nonneg_left hyd hb
        exact le_trans (by linarith) (le_trans (le_max_right _ _) (le_max_right _ _))
      · have h2 : b * y ≤ b * c := mul_le_mul_of_nonpos_left hyc hb
        exact le_trans (by linarith) (le_trans (le_max_left _ _) (le_max_right _ _))
    · have h1 : x * y ≤ a * y := mul_le_mul_of_nonpos_right hxa hy
      rcases le_total 0 a with ha | ha
      · have h2 : a * y ≤ a * d := mul_le_mul_of_nonneg_left hyd ha
        exact le_trans (by linarith) (le_trans (le_max_right _ _) (le_max_left _ _))
      · have h2 : a * y ≤ a * c := mul_le_mul_of_nonpos_left hyc ha
        exact le_trans (by linarith) (le_trans (le_max_left _ _) (le_max_left _ _))

/-- **`div_mem_pos`**: the quotient of two positive enclosed values. -/
theorem div_mem_pos {x y a b c d : ℝ} (ha : 0 < a) (hc : 0 < c) (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d) : a / d ≤ x / y ∧ x / y ≤ b / c :=
  ⟨div_le_div (by linarith) hxa (by linarith) hyd, div_le_div (by linarith) hxb hc hyc⟩

/-- **`floor_grid_le`**: rounding down onto the `2^-K` grid. -/
theorem floor_grid_le (r : ℝ) (K : ℕ) : (⌊r * 2 ^ K⌋ : ℝ) / 2 ^ K ≤ r := by
  rw [div_le_iff (by positivity)]
  exact Int.floor_le _

/-- **`le_floor_grid_succ`**: one grid step above the floor is above the value. -/
theorem le_floor_grid_succ (r : ℝ) (K : ℕ) : r ≤ ((⌊r * 2 ^ K⌋ : ℝ) + 1) / 2 ^ K := by
  rw [le_div_iff (by positivity)]
  exact (Int.lt_floor_add_one _).le

/-- **`le_ceil_grid`**: rounding up onto the `2^-K` grid. -/
theorem le_ceil_grid (r : ℝ) (K : ℕ) : r ≤ (⌈r * 2 ^ K⌉ : ℝ) / 2 ^ K := by
  rw [le_div_iff (by positivity)]
  exact Int.le_ceil _

/-! ## `β` and the Euler step -/

/-- The golden ratio, `(1 + √5) / 2`. -/
noncomputable def goldenPhi : ℝ := (1 + Real.sqrt 5) / 2

/-- `phaseRotation = φ_g / e`. -/
noncomputable def mayaA : ℝ := goldenPhi / Real.exp 1

/-- `consciousnessDamping = e / π`. -/
noncomputable def mayaB : ℝ := Real.exp 1 / π

/-- `β(φ) = a sin φ − b φ³`. -/
noncomputable def beta (a b x : ℝ) : ℝ := a * Real.sin x - b * x ^ 3

/-- One Euler sub-step of `d(log μ) = h`. -/
noncomputable def euler (a b h x : ℝ) : ℝ := x + h * beta a b x

/-- **`beta_mem`**: `β` from enclosures of `a`, `b` and `sin x`. -/
theorem beta_mem {a b x aL aU bL bU sL sU : ℝ} (ha : aL ≤ a ∧ a ≤ aU) (hb : bL ≤ b ∧ b ≤ bU)
    (hs : sL ≤ Real.sin x ∧ Real.sin x ≤ sU) :
    min (min (aL * sL) (aL * sU)) (min (aU * sL) (aU * sU)) -
        max (max (bL * x ^ 3) (bL * x ^ 3)) (max (bU * x ^ 3) (bU * x ^ 3)) ≤ beta a b x ∧
      beta a b x ≤ max (max (aL * sL) (aL * sU)) (max (aU * sL) (aU * sU)) -
        min (min (bL * x ^ 3) (bL * x ^ 3)) (min (bU * x ^ 3) (bU * x ^ 3)) := by
  have h1 := mul_mem_hull ha.1 ha.2 hs.1 hs.2
  have h2 := mul_mem_hull hb.1 hb.2 (le_refl (x ^ 3)) (le_refl (x ^ 3))
  unfold beta
  constructor <;> linarith [h1.1, h1.2, h2.1, h2.2]

theorem abs_sin_le_abs (t : ℝ) : |Real.sin t| ≤ |t| := by
  rcases le_or_lt |t| 1 with h | h
  · have hb := Real.sin_bound h
    have hu : 0 ≤ |t| := abs_nonneg t
    have ht2 : t ^ 2 = |t| ^ 2 := (sq_abs t).symm
    have ht21 : t ^ 2 ≤ 1 := by nlinarith [sq_abs t, abs_nonneg t]
    have hpoly : |t - t ^ 3 / 6| = |t| - |t| ^ 3 / 6 := by
      have : t - t ^ 3 / 6 = t * (1 - t ^ 2 / 6) := by ring
      rw [this, abs_mul, abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 - t ^ 2 / 6), ht2]
      ring
    have htri : |Real.sin t| ≤ |Real.sin t - (t - t ^ 3 / 6)| + |t - t ^ 3 / 6| := by
      have := abs_add (Real.sin t - (t - t ^ 3 / 6)) (t - t ^ 3 / 6)
      simpa using this
    have h4 : |t| ^ 4 * (5 / 96) ≤ |t| ^ 3 / 6 := by
      have : |t| ^ 4 = |t| ^ 3 * |t| := by ring
      rw [this]; nlinarith [pow_nonneg hu 3]
    linarith
  · exact le_trans (Real.abs_sin_le_one t) h.le

/-- **`sin_lipschitz`**: `|sin x − sin y| ≤ |x − y|`. -/
theorem sin_lipschitz (x y : ℝ) : |Real.sin x - Real.sin y| ≤ |x - y| := by
  rw [Real.sin_sub_sin]
  have h1 : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := abs_sin_le_abs _
  have h2 : |Real.cos ((x + y) / 2)| ≤ 1 := Real.abs_cos_le_one _
  have h3 : |(x - y) / 2| = |x - y| / 2 := by rw [abs_div, abs_two]
  rw [abs_mul, abs_mul, abs_two]
  nlinarith [mul_le_mul_of_nonneg_left h2 (abs_nonneg (Real.sin ((x - y) / 2))),
             abs_nonneg (Real.cos ((x + y) / 2))]

/-- **`beta_lipschitz`**: on `[-1, 1]`, `β` moves by at most `(a + 3b)` times the phase. -/
theorem beta_lipschitz {a b x y : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : |x| ≤ 1) (hy : |y| ≤ 1) :
    |beta a b x - beta a b y| ≤ (a + 3 * b) * |x - y| := by
  have hs := sin_lipschitz x y
  have hx2 : x ^ 2 ≤ 1 := by nlinarith [sq_abs x, abs_nonneg x]
  have hy2 : y ^ 2 ≤ 1 := by nlinarith [sq_abs y, abs_nonneg y]
  have hxy : |x * y| ≤ 1 := by rw [abs_mul]; nlinarith [abs_nonneg x, abs_nonneg y]
  have hq : |x ^ 2 + x * y + y ^ 2| ≤ 3 := by
    rw [abs_le]; constructor <;> nlinarith [abs_le.mp hxy, sq_nonneg x, sq_nonneg y]
  have hcube : |x ^ 3 - y ^ 3| ≤ 3 * |x - y| := by
    have : x ^ 3 - y ^ 3 = (x - y) * (x ^ 2 + x * y + y ^ 2) := by ring
    rw [this, abs_mul]; nlinarith [abs_nonneg (x - y)]
  have : beta a b x - beta a b y = a * (Real.sin x - Real.sin y) - b * (x ^ 3 - y ^ 3) := by
    unfold beta; ring
  rw [this]
  calc |a * (Real.sin x - Real.sin y) - b * (x ^ 3 - y ^ 3)|
      ≤ |a * (Real.sin x - Real.sin y)| + |b * (x ^ 3 - y ^ 3)| := abs_sub _ _
    _ = a * |Real.sin x - Real.sin y| + b * |x ^ 3 - y ^ 3| := by
        rw [abs_mul, abs_mul, abs_of_nonneg ha, abs_of_nonneg hb]
    _ ≤ a * |x - y| + b * (3 * |x - y|) := by gcongr
    _ = (a + 3 * b) * |x - y| := by ring

/-- **`euler_err`**: a computed sub-step `xt + h bt + r`, with `bt` within `w`
of `β(xt)` and `r` the storage rounding, is within
`(1 + h(a+3b)) |xt − x| + h w + |r|` of the exact Euler step from `x`. -/
theorem euler_err {a b h w r x xt bt : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hh : 0 ≤ h)
    (hx : |x| ≤ 1) (hxt : |xt| ≤ 1) (hbt : |bt - beta a b xt| ≤ w) :
    |(xt + h * bt + r) - euler a b h x| ≤ (1 + h * (a + 3 * b)) * |xt - x| + h * w + |r| := by
  have hL := beta_lipschitz ha hb hxt hx
  have : (xt + h * bt + r) - euler a b h x =
      (xt - x) + h * (beta a b xt - beta a b x) + h * (bt - beta a b xt) + r := by
    unfold euler; ring
  rw [this]
  calc |(xt - x) + h * (beta a b xt - beta a b x) + h * (bt - beta a b xt) + r|
      ≤ |xt - x| + |h * (beta a b xt - beta a b x)| + |h * (bt - beta a b xt)| + |r| := by
        have e1 := abs_add ((xt - x) + h * (beta a b xt - beta a b x) + h * (bt - beta a b xt)) r
        have e2 := abs_add ((xt - x) + h * (beta a b xt - beta a b x)) (h * (bt - beta a b xt))
        have e3 := abs_add (xt - x) (h * (beta a b xt - beta a b x))
        linarith
    _ = |xt - x| + h * |beta a b xt - beta a b x| + h * |bt - beta a b xt| + |r| := by
        rw [abs_mul, abs_mul, abs_of_nonneg hh]
    _ ≤ |xt - x| + h * ((a + 3 * b) * |xt - x|) + h * w + |r| := by gcongr
    _ = (1 + h * (a + 3 * b)) * |xt - x| + h * w + |r| := by ring

/-- **`euler_mono`**: with `h (a + 3b) ≤ 1` an Euler sub-step keeps phases in order. -/
theorem euler_mono {a b h x y : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hh : 0 ≤ h)
    (hL : h * (a + 3 * b) ≤ 1) (hx : |x| ≤ 1) (hy : |y| ≤ 1) (hxy : x ≤ y) :
    euler a b h x ≤ euler a b h y := by
  have hlip := beta_lipschitz ha hb hy hx
  rw [abs_of_nonneg (by linarith : 0 ≤ y - x)] at hlip
  have hd : -((a + 3 * b) * (y - x)) ≤ beta a b y - beta a b x := (abs_le.mp hlip).1
  unfold euler
  nlinarith [mul_le_mul_of_nonneg_left hd hh]

/-! ## One stable coherence layer -/

/-- **`beta_pos_small`**: `β > 0` on `(0, 3/4]`. -/
theorem beta_pos_small {a b x : ℝ} (ha : 59 / 100 ≤ a) (hb0 : 0 ≤ b) (hb : b ≤ 87 / 100)
    (hx0 : 0 < x) (hx1 : x ≤ 3 / 4) : 0 < beta a b x := by
  have habs : |x| = x := abs_of_pos hx0
  have hbnd := Real.sin_bound (by rw [habs]; linarith : |x| ≤ 1)
  rw [habs] at hbnd
  have hs : x - x ^ 3 / 6 - x ^ 4 * (5 / 96) ≤ Real.sin x := by
    have := (abs_le.mp hbnd).1; linarith
  have hx2 : x ^ 2 ≤ 9 / 16 := by nlinarith
  have hx3 : x ^ 3 ≤ 27 / 64 := by nlinarith
  have hinner : 0 ≤ 1 - x ^ 2 / 6 - 5 * x ^ 3 / 96 := by nlinarith
  have key : 0 < a * (1 - x ^ 2 / 6 - 5 * x ^ 3 / 96) - b * x ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_right ha hinner, mul_le_mul_of_nonneg_left hx2 hb0,
               mul_le_mul_of_nonneg_right hb (sq_nonneg x)]
  have hlow : a * (x - x ^ 3 / 6 - x ^ 4 * (5 / 96)) - b * x ^ 3 ≤ beta a b x := by
    unfold beta; nlinarith [mul_le_mul_of_nonneg_left hs (by linarith : (0 : ℝ) ≤ a)]
  have heq : a * (x - x ^ 3 / 6 - x ^ 4 * (5 / 96)) - b * x ^ 3 =
      x * (a * (1 - x ^ 2 / 6 - 5 * x ^ 3 / 96) - b * x ^ 2) := by ring
  rw [heq] at hlow
  exact lt_of_lt_of_le (mul_pos hx0 key) hlow

/-- **`beta_strict_anti`**: `β` is strictly decreasing on `[3/4, 1]`. -/
theorem beta_strict_anti {a b x y : ℝ} (ha0 : 0 ≤ a) (ha : a ≤ 3 / 5) (hb : 86 / 100 ≤ b)
    (hy : 3 / 4 ≤ y) (hxy : y < x) (hx : x ≤ 1) : beta a b x < beta a b y := by
  have hs : Real.sin x - Real.sin y ≤ x - y := by
    have := sin_lipschitz x y
    rw [abs_of_pos (by linarith : 0 < x - y)] at this
    exact le_trans (le_abs_self _) this
  have hq : 27 / 16 ≤ x ^ 2 + x * y + y ^ 2 := by nlinarith
  have hd : 0 < x - y := by linarith
  have hcube : (x - y) * (27 / 16) ≤ x ^ 3 - y ^ 3 := by
    have : x ^ 3 - y ^ 3 = (x - y) * (x ^ 2 + x * y + y ^ 2) := by ring
    rw [this]; exact mul_le_mul_of_nonneg_left hq hd.le
  have : beta a b x - beta a b y = a * (Real.sin x - Real.sin y) - b * (x ^ 3 - y ^ 3) := by
    unfold beta; ring
  have h1 : a * (Real.sin x - Real.sin y) ≤ a * (x - y) := mul_le_mul_of_nonneg_left hs ha0
  have h2 : b * ((x - y) * (27 / 16)) ≤ b * (x ^ 3 - y ^ 3) :=
    mul_le_mul_of_nonneg_left hcube (by linarith)
  nlinarith

theorem beta_continuous (a b : ℝ) : Continuous (beta a b) := by
  unfold beta
  exact (continuous_const.mul Real.continuous_sin).sub (continuous_const.mul (continuous_pow 3))

/-- **`layer`**: when `β` changes sign on a bracket `[lo, hi] ⊆ [3/4, 1]`, it has
exactly one root `φ*` there, is positive from `0` up to it and negative after:
one stable coherence layer. -/
theorem layer {a b lo hi : ℝ} (ha : 59 / 100 ≤ a) (ha' : a ≤ 3 / 5) (hb : 86 / 100 ≤ b)
    (hb' : b ≤ 87 / 100) (h34 : 3 / 4 ≤ lo) (hlh : lo < hi) (hhi : hi ≤ 1)
    (hpos : 0 < beta a b lo) (hneg : beta a b hi < 0) :
    ∃ xs, lo < xs ∧ xs < hi ∧ beta a b xs = 0 ∧
      (∀ x, 0 < x → x < xs → 0 < beta a b x) ∧ (∀ x, xs < x → x ≤ 1 → beta a b x < 0) := by
  have hcont := (beta_continuous a b).continuousOn (s := Set.Icc lo hi)
  obtain ⟨xs, ⟨hxs1, hxs2⟩, hxs⟩ :=
    intermediate_value_Ioo' hlh.le hcont (show (0 : ℝ) ∈ Set.Ioo (beta a b hi) (beta a b lo) from ⟨hneg, hpos⟩)
  refine ⟨xs, hxs1, hxs2, hxs, ?_, ?_⟩
  · intro x hx0 hx
    rcases le_or_lt x (3 / 4) with h | h
    · exact beta_pos_small ha (by linarith) hb' hx0 h
    · have := beta_strict_anti (by linarith) ha' hb h.le hx (by linarith)
      rw [hxs] at this; exact this
  · intro x hx hx1
    have := beta_strict_anti (by linarith) ha' hb (by linarith) hx hx1
    rw [hxs] at this; exact this

/-- **`toward_layer`**: below the layer an Euler sub-step raises the phase and
does not pass the layer. -/
theorem toward_layer {a b h x xs : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hh : 0 < h)
    (hL : h * (a + 3 * b) ≤ 1) (hx0 : 0 < x) (hx : x < xs) (hxs1 : xs ≤ 1)
    (hxs : beta a b xs = 0) (hpos : 0 < beta a b x) :
    x < euler a b h x ∧ euler a b h x ≤ xs := by
  constructor
  · unfold euler; nlinarith [mul_pos hh hpos]
  · have hm := euler_mono ha hb hh.le hL (by rw [abs_le]; constructor <;> linarith)
      (by rw [abs_le]; constructor <;> linarith) hx.le
    unfold euler at hm ⊢
    rw [hxs, mul_zero, add_zero] at hm
    exact hm

/-- **`layer_attracts_above`**: above the layer an Euler sub-step lowers the
phase and does not pass the layer. -/
theorem layer_attracts_above {a b h x xs : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hh : 0 < h)
    (hL : h * (a + 3 * b) ≤ 1) (hxs0 : 0 ≤ xs) (hx : xs < x) (hx1 : x ≤ 1)
    (hxs : beta a b xs = 0) (hneg : beta a b x < 0) :
    xs ≤ euler a b h x ∧ euler a b h x < x := by
  constructor
  · have hm := euler_mono ha hb hh.le hL (by rw [abs_le]; constructor <;> linarith)
      (by rw [abs_le]; constructor <;> linarith) hx.le
    unfold euler at hm ⊢
    rw [hxs, mul_zero, add_zero] at hm
    exact hm
  · unfold euler; nlinarith [mul_neg_of_pos_of_neg hh hneg]

/-! ## The Maya factor -/

/-- `M = 1 + β / (20 (1 + |β|))`, v18.24's `modulationFactor`. -/
noncomputable def factor (β : ℝ) : ℝ := 1 + β / (20 * (1 + |β|))

theorem factor_frac_abs_lt (β : ℝ) : |β / (20 * (1 + |β|))| < 1 / 20 := by
  have hpos : 0 < 20 * (1 + |β|) := by positivity
  rw [abs_div, abs_of_pos hpos, div_lt_iff hpos]
  linarith [abs_nonneg β]

/-- **`factor_mem`**: the Maya factor lies in `(19/20, 21/20)`. -/
theorem factor_mem (β : ℝ) : 19 / 20 < factor β ∧ factor β < 21 / 20 := by
  have h := abs_lt.mp (factor_frac_abs_lt β)
  unfold factor
  constructor <;> linarith [h.1, h.2]

/-- **`factor_eq_one_iff`**: the factor is `1` exactly where `β = 0`, on a layer. -/
theorem factor_eq_one_iff (β : ℝ) : factor β = 1 ↔ β = 0 := by
  have hpos : 0 < 20 * (1 + |β|) := by positivity
  unfold factor
  constructor
  · intro h
    have : β / (20 * (1 + |β|)) = 0 := by linarith
    rcases div_eq_zero_iff.mp this with h0 | h0
    · exact h0
    · linarith
  · intro h; rw [h]; simp

/-- **`factor_gt_one_iff`**: the factor amplifies exactly while `β > 0`. -/
theorem factor_gt_one_iff (β : ℝ) : 1 < factor β ↔ 0 < β := by
  have hpos : 0 < 20 * (1 + |β|) := by positivity
  unfold factor
  constructor
  · intro h
    have : 0 < β / (20 * (1 + |β|)) := by linarith
    exact (div_pos_iff_of_pos_right hpos).mp this
  · intro h
    have : 0 < β / (20 * (1 + |β|)) := div_pos h hpos
    linarith

theorem frac_lipschitz (x y : ℝ) : |x / (1 + |x|) - y / (1 + |y|)| ≤ |x - y| := by
  have hx1 : 0 < 1 + |x| := by positivity
  have hy1 : 0 < 1 + |y| := by positivity
  rw [div_sub_div _ _ hx1.ne' hy1.ne', abs_div, abs_of_pos (mul_pos hx1 hy1),
    div_le_iff (mul_pos hx1 hy1)]
  rcases le_total 0 x with hx | hx <;> rcases le_total 0 y with hy | hy
  · rw [abs_of_nonneg hx, abs_of_nonneg hy]
    have : x * (1 + y) - (1 + x) * y = x - y := by ring
    rw [this]; nlinarith [abs_nonneg (x - y), mul_nonneg hx hy]
  · rw [abs_of_nonneg hx, abs_of_nonpos hy, abs_of_nonneg (by linarith : 0 ≤ x - y)]
    rw [abs_of_nonneg (by nlinarith)]
    nlinarith [mul_nonneg hx (neg_nonneg.mpr hy)]
  · rw [abs_of_nonpos hx, abs_of_nonneg hy, abs_of_nonpos (by linarith : x - y ≤ 0)]
    rw [abs_of_nonpos (by nlinarith)]
    nlinarith [mul_nonneg (neg_nonneg.mpr hx) hy]
  · rw [abs_of_nonpos hx, abs_of_nonpos hy]
    have : x * (1 + -y) - (1 + -x) * y = x - y := by ring
    rw [this]; nlinarith [abs_nonneg (x - y), mul_nonneg (neg_nonneg.mpr hx) (neg_nonneg.mpr hy)]

/-- **`factor_lipschitz`**: the factor moves by at most `1/20` of a change in `β`,
which is what the runtime's `β` enclosure costs the state. -/
theorem factor_lipschitz (x y : ℝ) : |factor x - factor y| ≤ |x - y| / 20 := by
  have h := frac_lipschitz x y
  have : factor x - factor y = (x / (1 + |x|) - y / (1 + |y|)) / 20 := by
    unfold factor; field_simp; ring
  rw [this, abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 20)]
  exact div_le_div_of_nonneg_right h (by norm_num)

/-- **`observe_order`**: the observed field `M · v` keeps the order and the sign
of the true field `v`, so the illusion rescales the picture and never
reorders it. -/
theorem observe_order (β v w : ℝ) : v < w ↔ factor β * v < factor β * w := by
  have h := (factor_mem β).1
  have hpos : 0 < factor β := by linarith
  exact (mul_lt_mul_left hpos).symm

end MayaFlow
end SutraWS
