import Mathlib.Tactic
import Mathlib.Data.Rat.Defs

/-!
# Wheeler geometry certificates

The seven statements that gate the Wheeler render channels in
`vedic_v18.51.1_exact_phi.html`.  Each one is about `computeWheeler` and
`exactGeometry` in the `STRICT_V5` kernel.  Most are identities or inequalities
over `Rat`; `Rat`'s `DecidableEq` reduces through `Nat.gcd`'s well-founded
recursion and the kernel gives up, so they are closed by `norm_num` over an
enumeration of the sixteen vertices rather than by `decide`.  `native_decide`
is forbidden here — see `SutraWS.AxiomAudit`.

The page carries an executable decision procedure for the same seven statements
(`STRICT_V5.wheelerAudit`).  A channel is drawn only when its statement holds at
runtime, so the two are kept in step: this file says what is true, `wheelerAudit`
checks it against the arithmetic that actually places the pixels.

Vertices are the sixteen points of the 4-cube, indexed `0..15`; `hw v` is the
Hamming weight of that index and `k v = hw v - 2` is the signed distance from
the inertial plane.
-/

namespace SutraWS
namespace Wheeler

/-- Hamming weight of a 4-bit vertex index. -/
def hw (v : Fin 16) : Nat :=
  (v.val % 2) + (v.val / 2 % 2) + (v.val / 4 % 2) + (v.val / 8 % 2)

/-- Signed distance from the inertial plane, `k = hw v - 2`. -/
def k (v : Fin 16) : Int := (hw v : Int) - 2

/-- Magnetic weight `k^2 / (1 + k^2)`. -/
def magneticWeight (v : Fin 16) : Rat :=
  let kk : Rat := ((k v) * (k v) : Int)
  kk / (1 + kk)

/-- Dielectric magnitude `D = 2 * phi * s * field v`, carried as its rational
part `s * field v`; the `2 * phi` factor is a unit of the ambient field and
scales out of every statement below. -/
def dielectricTrace (s : Rat) (field : Fin 16 → Rat) (v : Fin 16) : Rat :=
  s * field v

/-- Magnetic magnitude `M = s * field v * magneticWeight v * H`. -/
def magnetic (s H : Rat) (field : Fin 16 → Rat) (v : Fin 16) : Rat :=
  dielectricTrace s field v * magneticWeight v * H

/-- Larmor precession `omega = gamma_W * M`. -/
def omega (s H gammaW : Rat) (field : Fin 16 → Rat) (v : Fin 16) : Rat :=
  gammaW * magnetic s H field v

/-- Wheeler's radial rarefaction profile `rho = eps^2 / (r^2 + eps^2)`.
`radialFactor = phi^3 * rho`, so bounding `rho` bounds it by `phi^3`. -/
def rho (r eps : Rat) : Rat := eps ^ 2 / (r ^ 2 + eps ^ 2)

/-- **1. The inertial plane carries no magnetism.**  This is Wheeler's central
claim, and here it is forced by the weight `k^2/(1+k^2)` vanishing at `k = 0`. -/
theorem wheeler_inertial_plane_magnetism_zero
    (s H : Rat) (field : Fin 16 → Rat) (v : Fin 16) (h : hw v = 2) :
    magnetic s H field v = 0 := by
  have hk : k v = 0 := by simp [k, h]
  simp [magnetic, magneticWeight, hk]

/-- The inertial plane is exactly the six weight-two vertices. -/
theorem wheeler_inertial_plane_card :
    (Finset.univ.filter (fun v : Fin 16 => hw v = 2)).card = 6 := by decide

/-- **2. The magnetic weight lies in `[0, 1)`, and vanishes only on the plane.** -/
theorem wheeler_magnetic_weight_in_unit_interval (v : Fin 16) :
    0 ≤ magneticWeight v ∧ magneticWeight v < 1 := by
  fin_cases v <;> norm_num [magneticWeight, k, hw]

theorem wheeler_magnetic_weight_zero_iff (v : Fin 16) :
    magneticWeight v = 0 ↔ hw v = 2 := by
  fin_cases v <;> norm_num [magneticWeight, k, hw]

/-- **3. The dielectric is exactly phi-scaled**: `D / field` is the same constant
at every vertex, so the dielectric channel is a pure rescaling of the field and
introduces no structure of its own. -/
theorem wheeler_dielectric_is_phi_scaled
    (s : Rat) (field : Fin 16 → Rat) (u v : Fin 16)
    (hu : field u ≠ 0) (hv : field v ≠ 0) :
    dielectricTrace s field u / field u = dielectricTrace s field v / field v := by
  field_simp [dielectricTrace]

/-- **4. Precession is exactly linear in the magnetic field.** -/
theorem wheeler_omega_linear_in_magnetic
    (s H gammaW : Rat) (field : Fin 16 → Rat) (v : Fin 16) :
    omega s H gammaW field v = gammaW * magnetic s H field v := rfl

theorem wheeler_omega_additive
    (s H gammaW : Rat) (f g : Fin 16 → Rat) (v : Fin 16) :
    omega s H gammaW (fun w => f w + g w) v
      = omega s H gammaW f v + omega s H gammaW g v := by
  simp only [omega, magnetic, dielectricTrace]; ring

/-- **5. The radial factor is bounded by `phi^3`**, via `0 ≤ rho ≤ 1`. -/
theorem wheeler_radial_factor_bounded_by_phi_cubed
    (r eps : Rat) (h : eps ≠ 0) : 0 ≤ rho r eps ∧ rho r eps ≤ 1 := by
  have he : 0 < eps ^ 2 := by positivity
  have hpos : 0 < r ^ 2 + eps ^ 2 := by nlinarith [sq_nonneg r]
  constructor
  · exact div_nonneg (sq_nonneg eps) hpos.le
  · rw [rho, div_le_one hpos]; nlinarith [sq_nonneg r]

theorem wheeler_rho_zero_eps (eps : Rat) (h : eps ≠ 0) : rho 0 eps = 1 := by
  have h2 : eps ^ 2 ≠ 0 := pow_ne_zero 2 h
  field_simp [rho]

/-- **6. The Cayley transform of `omega` is a genuine rotation**: the induced
`(c, s)` sits on the unit circle exactly, so the omega channel rotates the
geometry without dilating it. -/
theorem wheeler_cayley_rotation_is_orthogonal (tau : Rat) :
    ((1 - tau ^ 2) / (1 + tau ^ 2)) ^ 2 + ((2 * tau) / (1 + tau ^ 2)) ^ 2 = 1 := by
  have _h : (1 : Rat) + tau ^ 2 ≠ 0 := by positivity
  field_simp
  ring

/-- **7. The compression factor is strictly positive**, so `exactGeometry`'s
z-compression `1 / (1 + geoD * D)` never inverts the geometry or divides by
zero, provided the dielectric stays above `-1/geoD`. -/
theorem wheeler_compression_positive
    (geoD D : Rat) (hg : 0 < geoD) (hD : -(1 / geoD) < D) : 0 < 1 + geoD * D := by
  have h1 : geoD * (-(1 / geoD)) < geoD * D := (mul_lt_mul_left hg).mpr hD
  have h2 : geoD * (-(1 / geoD)) = -1 := by field_simp
  rw [h2] at h1
  linarith

theorem wheeler_compression_positive_of_nonneg
    (geoD D : Rat) (hg : 0 < geoD) (hD : 0 ≤ D) : 0 < 1 + geoD * D := by
  nlinarith

/-!
## The golden-Pythagorean identities

`GOLDENPYTHAGOREAN` states `φ + φ + 1 = φ³ = 4.23606` and divides the line into
four segments `φ : 1 : 1 : 1/φ` whose total is `φ³`.  Both reduce to the same
algebraic fact, and both are exact in `ℚ(√5)` — which is why the render can
carry them without a float anywhere.

`φ` itself is irrational, so these are stated over an abstract `φ` pinned by its
defining equation `φ² = φ + 1` rather than over `Rat`.
-/

section GoldenPythagorean

variable {K : Type*} [Field K] (φ : K)

/-- The golden ratio's defining relation. -/
def IsGolden (φ : K) : Prop := φ ^ 2 = φ + 1

variable (h : IsGolden φ)

/-- `φ ≠ 0`, so `1/φ` is available. -/
theorem golden_ne_zero : φ ≠ 0 := by
  intro h0
  have h2 : φ ^ 2 = φ + 1 := h
  rw [h0] at h2
  norm_num at h2

/-- **`φ³ = 2φ + 1`** — the cube in terms of the ratio itself. -/
theorem phi_cubed_eq : φ ^ 3 = 2 * φ + 1 := by
  have h2 : φ ^ 2 = φ + 1 := h
  calc φ ^ 3 = φ * φ ^ 2 := by ring
    _ = φ * (φ + 1) := by rw [h2]
    _ = φ ^ 2 + φ := by ring
    _ = φ + 1 + φ := by rw [h2]
    _ = 2 * φ + 1 := by ring

/-- **`φ + φ + 1 = φ³`** — the identity written on the chart. -/
theorem phi_plus_phi_plus_one : φ + φ + 1 = φ ^ 3 := by
  rw [phi_cubed_eq φ h]; ring

/-- **`φ + 1/φ = √5`**, in the form that avoids naming `√5`: `φ - 1/φ = 1`. -/
theorem phi_sub_inv : φ - 1 / φ = 1 := by
  have hne := golden_ne_zero φ h
  have h2 : φ ^ 2 = φ + 1 := h
  field_simp
  linear_combination h2

/-- **The divided line sums to `φ³`**: `φ + 1 + 1 + 1/φ = φ³`.
This is the GOLDENPYTHAGOREAN divided line `Θ : Α : Υ : Γ`, and it is what
`C.dividedLine` is checked against at load time. -/
theorem divided_line_sums_to_phi_cubed : φ + 1 + 1 + 1 / φ = φ ^ 3 := by
  have hne := golden_ne_zero φ h
  have h2 : φ ^ 2 = φ + 1 := h
  rw [phi_cubed_eq φ h]
  field_simp
  linear_combination -h2

end GoldenPythagorean

/-- Over `ℚ(√5)` the same statement is a concrete identity: `φ³ = 2 + √5`.
Represented here on the pair `(a, b) ↦ a + b√5`, where `φ = (1,1)/2`. -/
def sqrt5Mul (x y : Rat × Rat) : Rat × Rat :=
  (x.1 * y.1 + 5 * x.2 * y.2, x.1 * y.2 + x.2 * y.1)

/-- `φ³ = 2 + √5` in the `(a, b) ↦ a + b√5` representation. -/
theorem phi_cubed_is_two_plus_sqrt5 :
    sqrt5Mul (sqrt5Mul (1/2, 1/2) (1/2, 1/2)) (1/2, 1/2) = (2, 1) := by
  norm_num [sqrt5Mul, Prod.ext_iff]

end Wheeler

/-!
# Execution modes

The simulator runs a set of sutras three ways, and the point of having three is
that they are different operators.  Over a commutative ring the difference is
already visible on two operators:

* `SERIES` composes — `S₂ ∘ S₁`
* `PARALLEL` superposes displacements — `ψ + Σᵢ (Sᵢ ψ − ψ)`
* `SYMMETRIC_CONCURRENT` is the half-strength forward/reverse split

`PARALLEL` drops every cross term, which is exactly why it does not preserve
norm when `SERIES` does, and is what the mode-comparison panel shows.
-/

namespace Modes

variable {V : Type*} [AddCommGroup V]

/-- Series: apply each operator in turn. -/
def series (fs : List (V → V)) (ψ : V) : V := fs.foldl (fun x f => f x) ψ

/-- Parallel: sum the independent displacements about the same input. -/
def parallel (fs : List (V → V)) (ψ : V) : V :=
  ψ + (fs.map (fun f => f ψ - ψ)).sum

/-- On the empty set both modes are the identity. -/
theorem series_nil (ψ : V) : series ([] : List (V → V)) ψ = ψ := rfl

theorem parallel_nil (ψ : V) : parallel ([] : List (V → V)) ψ = ψ := by
  simp [parallel]

/-- **On a single operator the two modes agree** — there is nothing to order and
nothing to superpose.  This is what the panel reports for a one-sutra set. -/
theorem series_eq_parallel_singleton (f : V → V) (ψ : V) :
    series [f] ψ = parallel [f] ψ := by
  simp [series, parallel]

/-- The displacement an operator applies at a point — the quantity `PARALLEL`
superposes. -/
def disp (f : V → V) (ψ : V) : V := f ψ - ψ

/-- **On two operators they differ by exactly the cross term.**  `SERIES` feeds
`f₁ ψ` into `f₂`; `PARALLEL` evaluates both at `ψ`.  The gap is the change in
`f₂`'s *displacement* between those two points.

It is not `f₂ (f₁ ψ) − f₂ ψ`: that drops `f₁`'s own displacement, and the two
agree only when `f₁ ψ = ψ`. `parallel [f₁, f₂] ψ` is `f₁ ψ + f₂ ψ − ψ`, so the
honest gap carries the `− (f₁ ψ − ψ)` term as well, which is what `disp`
collects. -/
theorem series_sub_parallel_pair (f₁ f₂ : V → V) (ψ : V) :
    series [f₁, f₂] ψ - parallel [f₁, f₂] ψ = disp f₂ (f₁ ψ) - disp f₂ ψ := by
  simp only [series, parallel, disp, List.foldl, List.map, List.sum_cons,
             List.sum_nil]
  abel

/-- If `f₂` displaces `f₁ ψ` exactly as it displaces `ψ`, the modes coincide —
the precise condition under which the comparison is degenerate. -/
theorem series_eq_parallel_of_affine (f₁ f₂ : V → V) (ψ : V)
    (h : disp f₂ (f₁ ψ) = disp f₂ ψ) : series [f₁, f₂] ψ = parallel [f₁, f₂] ψ := by
  have hd := series_sub_parallel_pair f₁ f₂ ψ
  rw [h, sub_self] at hd
  exact sub_eq_zero.mp hd

end Modes
end SutraWS
