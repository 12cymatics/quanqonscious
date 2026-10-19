import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
import Mathlib.MeasureTheory.Integral.FundThmCalculus
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# The fields around the construct

The space around the torus shows what the magnetism memory and the dielectric
field make there. On the hypercube:

* the **current** is Ampère's law on the lattice, `J = (d¹)ᵀ B`: each edge
  carries the signed sum of the flux of the three faces round it. It is
  conserved at every vertex (`kirchhoff`), because `d¹ d⁰ = 0` on the page's
  own incidence tables (`incidence_dd`);
* the **charge** is Gauss's law on the dielectric field `q`, `ρ = (d⁰)ᵀ d⁰ q`,
  the lattice Laplacian of `q`. It sums to zero (`charge_neutral`): the
  construct is neutral, so its field has no monopole part.

In space, each edge carries its current along its path on the drawn surface,
straight piece by straight piece, and each vertex carries its charge:

* `segment_field_integral` — a straight piece's Biot–Savart integral has the
  closed form the page evaluates, `∫ d / (t² + d²)^{3/2} dt = [t / (d √(t² + d²))]`;
* `poynting_orth`, `poynting_le_energy` — the energy flow `S = E × B` is
  perpendicular to both fields and never exceeds the energy density
  `½(|E|² + |B|²)` (with the unit of each field its own);
* `rk4_step_le` — a field line is traced by steps of `h` along unit field
  directions, so a step never moves more than `h`.

The page evaluates the fields in space in IEEE doubles; the lattice current
and charge, and the two conservation laws, are decided exactly in the kernel.
-/

namespace SutraWS
namespace Fields

/-! ### The page's incidence tables

`MAG_EDGES`: edge `e` runs along axis `e / 8` from the `e mod 8`-th vertex with
that bit clear (`eTail`) to the vertex with it set (`eHead`). `MAG_FACES`: face
`f` is bounded by four edges with signs `+ + − −` (`fEdge`, `fSign`). -/

def eTail : Fin 32 → Fin 16 :=
  ![0, 2, 4, 6, 8, 10, 12, 14, 0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 2, 3, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]
def eAxis (e : Fin 32) : Fin 4 := ⟨e.val / 8, by omega⟩
def eHead (e : Fin 32) : Fin 16 := ⟨(eTail e).val ||| (1 <<< (eAxis e).val), by revert e; decide⟩

def fEdge : Fin 24 → Fin 4 → Fin 32 :=
  ![![0, 9, 1, 8], ![2, 11, 3, 10], ![4, 13, 5, 12], ![6, 15, 7, 14],
    ![0, 17, 2, 16], ![1, 19, 3, 18], ![4, 21, 6, 20], ![5, 23, 7, 22],
    ![0, 25, 4, 24], ![1, 27, 5, 26], ![2, 29, 6, 28], ![3, 31, 7, 30],
    ![8, 18, 10, 16], ![9, 19, 11, 17], ![12, 22, 14, 20], ![13, 23, 15, 21],
    ![8, 26, 12, 24], ![9, 27, 13, 25], ![10, 30, 14, 28], ![11, 31, 15, 29],
    ![16, 28, 20, 24], ![17, 29, 21, 25], ![18, 30, 22, 26], ![19, 31, 23, 27]]
def fSign : Fin 4 → ℤ := ![1, 1, -1, -1]

/-- `d⁰`: the difference across edge `e`, head minus tail. -/
def D0 (e : Fin 32) (v : Fin 16) : ℤ := if v = eHead e then 1 else if v = eTail e then -1 else 0

/-- `d¹`: the signed edges of face `f`. -/
def D1 (f : Fin 24) (e : Fin 32) : ℤ := ∑ k : Fin 4, if fEdge f k = e then fSign k else 0

/-- Every edge has its tail clear and its head set on its own axis: the table
is the 32 edges of the 4-cube. -/
theorem edges_are_edges : ∀ e : Fin 32, eTail e ≠ eHead e ∧
    (eTail e).val &&& (1 <<< (eAxis e).val) = 0 := by decide

/-- **`incidence_dd`**: on the page's own tables, `d¹ d⁰ = 0`: going round any
face, every vertex is entered once and left once. -/
theorem incidence_dd : ∀ f : Fin 24, ∀ v : Fin 16, ∑ e, D1 f e * D0 e v = 0 := by decide

/-- Each edge row of `d⁰` sums to zero: a constant has no gradient. -/
theorem d0_rows : ∀ e : Fin 32, ∑ v, D0 e v = 0 := by decide

/-- The lattice current, Ampère's law: `J = (d¹)ᵀ B`. -/
def current (B : Fin 24 → ℚ) (e : Fin 32) : ℚ := ∑ f, (D1 f e : ℚ) * B f

/-- Its divergence at a vertex, `(d⁰)ᵀ J`: what flows in less what flows out. -/
def divergence (J : Fin 32 → ℚ) (v : Fin 16) : ℚ := ∑ e, (D0 e v : ℚ) * J e

/-- **`kirchhoff`**: the lattice current is conserved at every vertex, for every
flux `B`. -/
theorem kirchhoff (B : Fin 24 → ℚ) (v : Fin 16) : divergence (current B) v = 0 := by
  unfold divergence current
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  have h : ∀ f : Fin 24, ∑ e, (D0 e v : ℚ) * ((D1 f e : ℚ) * B f) = 0 := by
    intro f
    have hz : ((∑ e, D1 f e * D0 e v : ℤ) : ℚ) = 0 := by rw [incidence_dd f v]; simp
    push_cast at hz
    calc ∑ e, (D0 e v : ℚ) * ((D1 f e : ℚ) * B f) = (∑ e, (D1 f e : ℚ) * (D0 e v : ℚ)) * B f := by
          rw [Finset.sum_mul]; refine Finset.sum_congr rfl fun e _ => by ring
      _ = 0 := by rw [hz, zero_mul]
  exact Finset.sum_eq_zero fun f _ => h f

/-- The lattice charge, Gauss's law on the dielectric field: `ρ = (d⁰)ᵀ d⁰ q`. -/
def charge (q : Fin 16 → ℚ) (v : Fin 16) : ℚ := ∑ e, (D0 e v : ℚ) * ∑ w, (D0 e w : ℚ) * q w

/-- **`charge_neutral`**: the construct's charges sum to zero, whatever the
dielectric field. -/
theorem charge_neutral (q : Fin 16 → ℚ) : ∑ v, charge q v = 0 := by
  unfold charge
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun e _ => ?_
  rw [← Finset.sum_mul]
  have hz : ((∑ v, D0 e v : ℤ) : ℚ) = 0 := by rw [d0_rows e]; simp
  push_cast at hz
  rw [hz, zero_mul]

/-- Two vertices are neighbours when they differ in exactly one bit. -/
def adjacent (v w : Fin 16) : Bool := (List.finRange 4).any fun a => w.val = v.val ^^^ (1 <<< a.val)

/-- **`charge_laplacian`**: the charge is the graph Laplacian of the dielectric
field, the kernel's `laplacianV`: weight 4 at the vertex itself and −1 at each
of its four neighbours. -/
theorem charge_laplacian : ∀ v w : Fin 16,
    ∑ e, D0 e v * D0 e w = if v = w then 4 else if adjacent v w then -1 else 0 := by decide

/-! ### In space -/

/-- The antiderivative of the Biot–Savart kernel along a straight piece. -/
noncomputable def segPrim (d t : ℝ) : ℝ := t / (d * Real.sqrt (t ^ 2 + d ^ 2))

theorem segPrim_hasDeriv {d : ℝ} (hd : 0 < d) (t : ℝ) :
    HasDerivAt (segPrim d) (d / (Real.sqrt (t ^ 2 + d ^ 2)) ^ 3) t := by
  have hpos : 0 < t ^ 2 + d ^ 2 := by positivity
  have hs : 0 < Real.sqrt (t ^ 2 + d ^ 2) := Real.sqrt_pos.mpr hpos
  have hsq : Real.sqrt (t ^ 2 + d ^ 2) ^ 2 = t ^ 2 + d ^ 2 := Real.sq_sqrt hpos.le
  have h1 : HasDerivAt (fun t : ℝ => t ^ 2 + d ^ 2) (2 * t) t := by
    simpa using (hasDerivAt_pow 2 t).add_const (d ^ 2)
  have h2 := h1.sqrt hpos.ne'
  have h3 := h2.const_mul d
  have h4 := (hasDerivAt_id t).div h3 (by positivity)
  convert h4 using 1
  set s := Real.sqrt (t ^ 2 + d ^ 2)
  simp only [id]
  field_simp
  linear_combination (-2 * d * s ^ 3) * hsq

/-- **`segment_field_integral`**: the Biot–Savart integral along a straight
piece, at perpendicular distance `d > 0`, between positions `a` and `b` along it
measured from the foot of the perpendicular, is the closed form the page
evaluates. -/
theorem segment_field_integral {d : ℝ} (hd : 0 < d) (a b : ℝ) :
    ∫ t in a..b, d / (Real.sqrt (t ^ 2 + d ^ 2)) ^ 3 = segPrim d b - segPrim d a := by
  apply intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => segPrim_hasDeriv hd t)
  apply Continuous.intervalIntegrable
  have : ∀ t : ℝ, Real.sqrt (t ^ 2 + d ^ 2) ≠ 0 := fun t => (Real.sqrt_pos.mpr (by positivity)).ne'
  exact continuous_const.div ((Real.continuous_sqrt.comp (by continuity)).pow 3) (fun t => pow_ne_zero 3 (this t))

/-- The energy flow `S = E × B`, by components. -/
def cross (ex ey ez bx by_ bz : ℝ) : ℝ × ℝ × ℝ := (ey * bz - ez * by_, ez * bx - ex * bz, ex * by_ - ey * bx)

/-- **`poynting_orth`**: the energy flow is perpendicular to both fields. -/
theorem poynting_orth (ex ey ez bx by_ bz : ℝ) :
    let s := cross ex ey ez bx by_ bz
    ex * s.1 + ey * s.2.1 + ez * s.2.2 = 0 ∧ bx * s.1 + by_ * s.2.1 + bz * s.2.2 = 0 := by
  simp only [cross]; constructor <;> ring

/-- **`poynting_le_energy`**: the energy flow never exceeds the energy density
`½(|E|² + |B|²)`: `|E × B|² ≤ (½(|E|² + |B|²))²`. -/
theorem poynting_le_energy (ex ey ez bx by_ bz : ℝ) :
    let s := cross ex ey ez bx by_ bz
    s.1 ^ 2 + s.2.1 ^ 2 + s.2.2 ^ 2 ≤ ((ex ^ 2 + ey ^ 2 + ez ^ 2 + (bx ^ 2 + by_ ^ 2 + bz ^ 2)) / 2) ^ 2 := by
  simp only [cross]
  have lagrange : (ey * bz - ez * by_) ^ 2 + (ez * bx - ex * bz) ^ 2 + (ex * by_ - ey * bx) ^ 2 =
      (ex ^ 2 + ey ^ 2 + ez ^ 2) * (bx ^ 2 + by_ ^ 2 + bz ^ 2) - (ex * bx + ey * by_ + ez * bz) ^ 2 := by ring
  rw [lagrange]
  nlinarith [sq_nonneg (ex * bx + ey * by_ + ez * bz), sq_nonneg (ex ^ 2 + ey ^ 2 + ez ^ 2 - (bx ^ 2 + by_ ^ 2 + bz ^ 2))]

/-- **`rk4_step_le`**: a field line moves by `h (k₁ + 2k₂ + 2k₃ + k₄) / 6` with
each `kᵢ` a unit field direction (or shorter), so one step never moves more
than `h`. -/
theorem rk4_step_le {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {h : ℝ} (hh : 0 ≤ h)
    (k₁ k₂ k₃ k₄ : V) (h₁ : ‖k₁‖ ≤ 1) (h₂ : ‖k₂‖ ≤ 1) (h₃ : ‖k₃‖ ≤ 1) (h₄ : ‖k₄‖ ≤ 1) :
    ‖(h / 6) • (k₁ + (2 : ℝ) • k₂ + (2 : ℝ) • k₃ + k₄)‖ ≤ h := by
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  have : ‖k₁ + (2 : ℝ) • k₂ + (2 : ℝ) • k₃ + k₄‖ ≤ 6 := by
    calc ‖k₁ + (2 : ℝ) • k₂ + (2 : ℝ) • k₃ + k₄‖
        ≤ ‖k₁‖ + ‖(2 : ℝ) • k₂‖ + ‖(2 : ℝ) • k₃‖ + ‖k₄‖ := by
          refine (norm_add_le _ _).trans ?_
          gcongr
          refine (norm_add_le _ _).trans ?_
          gcongr
          exact norm_add_le _ _
      _ = ‖k₁‖ + 2 * ‖k₂‖ + 2 * ‖k₃‖ + ‖k₄‖ := by
          rw [norm_smul, norm_smul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
      _ ≤ 6 := by linarith
  calc h / 6 * ‖k₁ + (2 : ℝ) • k₂ + (2 : ℝ) • k₃ + k₄‖ ≤ h / 6 * 6 := by gcongr
    _ = h := by ring

/-! ## The energy plane's colour -/

/-- **`share_mem`**: two non-negative amounts, not both zero, split into shares
in `[0, 1]` that sum to one. The plane's hue is the magnetic and electric
energies' shares, each energy first placed in its own range (`plot_mem`). -/
theorem share_mem {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hs : 0 < a + b) :
    0 ≤ a / (a + b) ∧ a / (a + b) ≤ 1 ∧ a / (a + b) + b / (a + b) = 1 := by
  refine ⟨div_nonneg ha hs.le, (div_le_one hs).mpr (by linarith), ?_⟩
  rw [← add_div, div_self hs.ne']

/-- **`mix_mem`**: a colour channel mixed from two channels in `[0, 1]` by a
share `f ∈ [0, 1]` stays in `[0, 1]`, between the two. -/
theorem mix_mem {p q f : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hf0 : 0 ≤ f) (hf1 : f ≤ 1) :
    0 ≤ p * f + q * (1 - f) ∧ p * f + q * (1 - f) ≤ 1 ∧
      min p q ≤ p * f + q * (1 - f) ∧ p * f + q * (1 - f) ≤ max p q := by
  have h1f : 0 ≤ 1 - f := by linarith
  refine ⟨by positivity, by nlinarith, ?_, ?_⟩
  · have := min_le_left p q; have := min_le_right p q; nlinarith
  · have := le_max_left p q; have := le_max_right p q; nlinarith

/-- **`level_mem`**: a level `w₀ + w₁ t` with `t ∈ [0, 1]` (a place in a range,
`plot_mem`, or the larger of two) stays in `[w₀, w₀ + w₁]`: the plane's glow,
the arrows' and lines' widths and opacities. -/
theorem level_mem {t w₀ w₁ : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) (hw : 0 ≤ w₁) :
    w₀ ≤ w₀ + w₁ * t ∧ w₀ + w₁ * t ≤ w₀ + w₁ :=
  ⟨by nlinarith, by nlinarith⟩

end Fields
end SutraWS
