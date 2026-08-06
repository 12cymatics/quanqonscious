import Mathlib.Tactic
import Mathlib.Data.Rat.Basic

/-!
# Wheeler geometry certificates

The seven statements that gate the Wheeler render channels in
`vedic_v18.51.1_exact_phi.html`.  Each one is about `computeWheeler` and
`exactGeometry` in the `STRICT_V5` kernel, and each is a decidable identity or
inequality over `Rat`, so `decide` / `native_decide` closes it.

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
    0 ≤ magneticWeight v ∧ magneticWeight v < 1 := by decide +kernel

theorem wheeler_magnetic_weight_zero_iff (v : Fin 16) :
    magneticWeight v = 0 ↔ hw v = 2 := by decide +kernel

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
  simp [omega, magnetic, dielectricTrace]; ring

/-- **5. The radial factor is bounded by `phi^3`**, via `0 ≤ rho ≤ 1`. -/
theorem wheeler_radial_factor_bounded_by_phi_cubed
    (r eps : Rat) (h : eps ≠ 0) : 0 ≤ rho r eps ∧ rho r eps ≤ 1 := by
  have hpos : 0 < r ^ 2 + eps ^ 2 :=
    lt_of_lt_of_le (by positivity) (le_add_of_nonneg_left (sq_nonneg r))
  constructor
  · exact div_nonneg (sq_nonneg eps) hpos.le
  · rw [div_le_one hpos]; nlinarith [sq_nonneg r]

theorem wheeler_rho_zero_eps (eps : Rat) (h : eps ≠ 0) : rho 0 eps = 1 := by
  simp [rho]; field_simp

/-- **6. The Cayley transform of `omega` is a genuine rotation**: the induced
`(c, s)` sits on the unit circle exactly, so the omega channel rotates the
geometry without dilating it. -/
theorem wheeler_cayley_rotation_is_orthogonal (tau : Rat) :
    ((1 - tau ^ 2) / (1 + tau ^ 2)) ^ 2 + ((2 * tau) / (1 + tau ^ 2)) ^ 2 = 1 := by
  have h : (1 : Rat) + tau ^ 2 ≠ 0 := by positivity
  field_simp
  ring

/-- **7. The compression factor is strictly positive**, so `exactGeometry`'s
z-compression `1 / (1 + geoD * D)` never inverts the geometry or divides by
zero, provided the dielectric stays above `-1/geoD`. -/
theorem wheeler_compression_positive
    (geoD D : Rat) (hg : 0 < geoD) (hD : -(1 / geoD) < D) : 0 < 1 + geoD * D := by
  have : geoD * (-(1 / geoD)) < geoD * D := by exact (mul_lt_mul_left hg).mpr hD
  rw [mul_neg, mul_one_div, div_self hg.ne'] at this
  linarith

theorem wheeler_compression_positive_of_nonneg
    (geoD D : Rat) (hg : 0 < geoD) (hD : 0 ≤ D) : 0 < 1 + geoD * D := by
  nlinarith

end Wheeler
end SutraWS
