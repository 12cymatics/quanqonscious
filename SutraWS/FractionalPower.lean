import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import SutraWS.NonExpansive

/-!
# Strength on a fixed transform: the principal fractional power

Nine sūtras of `vedic_v18.51.1_exact_phi.html` act through a transform with no
angle for the strength to scale: a vertex permutation (S2, S6, S7, S12, S18,
S26), the Walsh–Hadamard transform (S3) and the twisted complement swap (S23).
Each has order dividing 4. At strength `s` the kernel applies its principal
fractional power `A^α`, `α = (4/π) arctan s`, exactly:

    u = cayley(s) = ((1 − s²) + 2s i) / (1 + s²),
    φ = (1, u, u², conj u),
    c_k = (1/4) Σ_j φ_j (−i)^{jk},
    U = c₀ + c₁ A + c₂ A² + c₃ A³        (`fracCoeffs`, `fracPower`).

This file proves what the page relies on:

* `frac_energy` — for any map `A` that preserves the inner sum `Σ conj(x_v) y_v`
  and has `A⁴ = 1`, and any coefficients with the autocorrelation of a unit
  spectrum, `U` preserves the energy `Σ |ψ_v|²` (`NonExpansive.energy`);
* `cay_normSq`, `coeff_autocorr` — the kernel's coefficients have that
  autocorrelation at every real strength;
* `coeff_zero`, `coeff_one` — strength 0 is the identity, strength 1 is `A`;
* `frac_eigen_one`, `frac_eigen_I`, `frac_eigen_neg_one`, `frac_eigen_neg_I` —
  on the eigenvalue-`1, i, −1, −i` eigenvectors `U` multiplies by `1, u, u²,
  conj u`;
* `cay_eq_exp` — `u = e^{2i arctan s}`, so those are `λ^α` with the principal
  branch and `α = (4/π) arctan s`; `alpha_one`: strength 1 is `α = 1`;
* `perm_ip`, `perm_order_four` — vertex permutations preserve the inner sum,
  and the kernel's six base maps have order dividing 4 (decided);
* `wht_ip`, `wht_wht` — the normalised Walsh transform preserves it and is an
  involution;
* `twist_ip`, `twist_twist` — so is the twisted complement swap.

The correspondence between these statements and the page is by transcription.
-/

namespace SutraWS
namespace FractionalPower

open Complex ComplexConjugate
open NonExpansive (energy wsignZ wht16 wsign_orth normSq_eq_conj_mul_re)

/-- The inner sum `Σ_v conj(x_v) y_v`. -/
noncomputable def ip {n : ℕ} (x y : Fin n → ℂ) : ℂ := ∑ v, conj (x v) * y v

theorem ip_add_left {n : ℕ} (x y z : Fin n → ℂ) : ip (x + y) z = ip x z + ip y z := by
  simp only [ip, Pi.add_apply, map_add, add_mul, Finset.sum_add_distrib]

theorem ip_add_right {n : ℕ} (x y z : Fin n → ℂ) : ip x (y + z) = ip x y + ip x z := by
  simp only [ip, Pi.add_apply, mul_add, Finset.sum_add_distrib]

theorem ip_smul_left {n : ℕ} (c : ℂ) (x y : Fin n → ℂ) : ip (c • x) y = conj c * ip x y := by
  simp only [ip, Pi.smul_apply, smul_eq_mul, map_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun v _ => by ring

theorem ip_smul_right {n : ℕ} (c : ℂ) (x y : Fin n → ℂ) : ip x (c • y) = c * ip x y := by
  simp only [ip, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun v _ => by ring

theorem energy_eq_ip {n : ℕ} (x : Fin n → ℂ) : energy x = (ip x x).re := by
  simp only [energy, ip, re_sum, normSq_eq_conj_mul_re]

/-! ## The operator: unitary whenever the spectrum is unit -/

/-- **`frac_energy`.** If `A` preserves the inner sum and `A⁴ = 1`, then
`c₀ + c₁A + c₂A² + c₃A³` preserves the energy whenever the coefficients have the
autocorrelation of a unit spectrum (`h0`–`h3`). The Gram entries
`⟨Aᵏx, Aˡx⟩` depend on `l − k mod 4` only, so the energy is
`Σ_m (Σ_k conj c_k c_{k+m}) ⟨x, Aᵐx⟩ = ⟨x, x⟩`. -/
theorem frac_energy {n : ℕ} (A : (Fin n → ℂ) → (Fin n → ℂ))
    (hA : ∀ x y, ip (A x) (A y) = ip x y) (h4 : ∀ x, A (A (A (A x))) = x)
    (c0 c1 c2 c3 : ℂ)
    (h0 : conj c0 * c0 + conj c1 * c1 + conj c2 * c2 + conj c3 * c3 = 1)
    (h1 : conj c0 * c1 + conj c1 * c2 + conj c2 * c3 + conj c3 * c0 = 0)
    (h2 : conj c0 * c2 + conj c1 * c3 + conj c2 * c0 + conj c3 * c1 = 0)
    (h3 : conj c0 * c3 + conj c1 * c0 + conj c2 * c1 + conj c3 * c2 = 0)
    (x : Fin n → ℂ) :
    energy (c0 • x + c1 • A x + c2 • A (A x) + c3 • A (A (A x))) = energy x := by
  have hx : A (A (A (A x))) = x := h4 x
  have e11 : ip (A x) (A x) = ip x x := hA x x
  have e22 : ip (A (A x)) (A (A x)) = ip x x := (hA _ _).trans e11
  have e33 : ip (A (A (A x))) (A (A (A x))) = ip x x := (hA _ _).trans e22
  have e12 : ip (A x) (A (A x)) = ip x (A x) := hA x (A x)
  have e23 : ip (A (A x)) (A (A (A x))) = ip x (A x) := (hA _ _).trans e12
  have e13 : ip (A x) (A (A (A x))) = ip x (A (A x)) := hA x (A (A x))
  have e10 : ip (A x) x = ip x (A (A (A x))) := by
    have h := hA x (A (A (A x))); rwa [hx] at h
  have e20 : ip (A (A x)) x = ip x (A (A x)) := by
    have h := hA (A x) (A (A (A x))); rw [hx] at h; exact h.trans e13
  have e30 : ip (A (A (A x))) x = ip x (A x) := by
    have h := hA (A (A x)) (A (A (A x))); rw [hx] at h; exact h.trans e23
  have e21 : ip (A (A x)) (A x) = ip x (A (A (A x))) := (hA (A x) x).trans e10
  have e31 : ip (A (A (A x))) (A x) = ip x (A (A x)) := (hA (A (A x)) x).trans e20
  have e32 : ip (A (A (A x))) (A (A x)) = ip x (A (A (A x))) := (hA (A (A x)) (A x)).trans e21
  have key : ip (c0 • x + c1 • A x + c2 • A (A x) + c3 • A (A (A x)))
      (c0 • x + c1 • A x + c2 • A (A x) + c3 • A (A (A x))) = ip x x := by
    simp only [ip_add_left, ip_add_right, ip_smul_left, ip_smul_right]
    rw [e11, e22, e33, e12, e23, e13, e10, e20, e30, e21, e31, e32]
    linear_combination (ip x x) * h0 + (ip x (A x)) * h1 + (ip x (A (A x))) * h2 +
      (ip x (A (A (A x)))) * h3
  rw [energy_eq_ip, energy_eq_ip, key]

/-! ## The coefficients -/

/-- The kernel's `cayley(s)` as a complex number: `((1 − s²) + 2s i)/(1 + s²)`. -/
noncomputable def cay (s : ℝ) : ℂ := ⟨(1 - s ^ 2) / (1 + s ^ 2), 2 * s / (1 + s ^ 2)⟩

@[simp] theorem cay_re (s : ℝ) : (cay s).re = (1 - s ^ 2) / (1 + s ^ 2) := rfl
@[simp] theorem cay_im (s : ℝ) : (cay s).im = 2 * s / (1 + s ^ 2) := rfl
theorem cay_eq_mk (s : ℝ) : cay s = ⟨(1 - s ^ 2) / (1 + s ^ 2), 2 * s / (1 + s ^ 2)⟩ := rfl

private theorem hD (s : ℝ) : (1 + s ^ 2 : ℝ) ≠ 0 := by positivity

theorem cay_normSq (s : ℝ) : normSq (cay s) = 1 := by
  rw [normSq_apply, cay_re, cay_im]
  field_simp
  ring

/-- `fracCoeffs`: `c_k = (1/4) Σ_j φ_j (−i)^{jk}` with `φ = (1, u, u², conj u)`,
written out from the kernel's table `(−i)^{(jk) mod 4} = [1, −i, −1, i]`. -/
noncomputable def c0 (s : ℝ) : ℂ := (1 + cay s + cay s ^ 2 + conj (cay s)) / 4
noncomputable def c1 (s : ℝ) : ℂ := (1 + cay s * (-I) + cay s ^ 2 * (-1) + conj (cay s) * I) / 4
noncomputable def c2 (s : ℝ) : ℂ := (1 + cay s * (-1) + cay s ^ 2 * 1 + conj (cay s) * (-1)) / 4
noncomputable def c3 (s : ℝ) : ℂ := (1 + cay s * I + cay s ^ 2 * (-1) + conj (cay s) * (-I)) / 4

theorem mk_add_mk (a b c d : ℝ) : (⟨a, b⟩ : ℂ) + ⟨c, d⟩ = ⟨a + c, b + d⟩ :=
  Complex.ext (by simp) (by simp)
theorem mk_mul_mk (a b c d : ℝ) : (⟨a, b⟩ : ℂ) * ⟨c, d⟩ = ⟨a * c - b * d, a * d + b * c⟩ :=
  Complex.ext (by simp) (by simp)
theorem conj_mk (a b : ℝ) : conj (⟨a, b⟩ : ℂ) = ⟨a, -b⟩ := Complex.ext (by simp) (by simp)
theorem neg_mk (a b : ℝ) : -(⟨a, b⟩ : ℂ) = ⟨-a, -b⟩ := Complex.ext (by simp) (by simp)
theorem mk_div_four (a b : ℝ) : (⟨a, b⟩ : ℂ) / 4 = ⟨a / 4, b / 4⟩ := Complex.ext (by simp) (by simp)
theorem one_eq_mk : (1 : ℂ) = ⟨1, 0⟩ := Complex.ext (by simp) (by simp)
theorem zero_eq_mk : (0 : ℂ) = ⟨0, 0⟩ := Complex.ext (by simp) (by simp)
theorem I_eq_mk : I = ⟨0, 1⟩ := Complex.ext (by simp) (by simp)
theorem mk_sq (a b : ℝ) : (⟨a, b⟩ : ℂ) ^ 2 = ⟨a * a - b * b, a * b + b * a⟩ := by
  rw [pow_two, mk_mul_mk]

/-- The coefficients in closed form, `D = 1 + s²`:
`c₀ = (1 − s²)(1 + is)/D²`, `c₁ = s(1 + s)((1 + s) − i(1 − s))/D²`,
`c₂ = s(1 − s²)(i − s)/D²`, `c₃ = −s(1 − s)((1 − s) + i(1 + s))/D²`. -/
theorem c0_closed (s : ℝ) : c0 s = ⟨(1 - s ^ 2) / (1 + s ^ 2) ^ 2, s * (1 - s ^ 2) / (1 + s ^ 2) ^ 2⟩ := by
  have h := hD s
  rw [c0]
  simp only [cay_eq_mk, mk_sq, conj_mk, I_eq_mk, one_eq_mk, neg_mk, mk_mul_mk, mk_add_mk, mk_div_four,
    Complex.mk.injEq]
  constructor <;> field_simp <;> ring

theorem c1_closed (s : ℝ) :
    c1 s = ⟨s * (1 + s) ^ 2 / (1 + s ^ 2) ^ 2, -(s * (1 - s ^ 2)) / (1 + s ^ 2) ^ 2⟩ := by
  have h := hD s
  rw [c1]
  simp only [cay_eq_mk, mk_sq, conj_mk, I_eq_mk, one_eq_mk, neg_mk, mk_mul_mk, mk_add_mk, mk_div_four,
    Complex.mk.injEq]
  constructor <;> field_simp <;> ring

theorem c2_closed (s : ℝ) :
    c2 s = ⟨-(s ^ 2 * (1 - s ^ 2)) / (1 + s ^ 2) ^ 2, s * (1 - s ^ 2) / (1 + s ^ 2) ^ 2⟩ := by
  have h := hD s
  rw [c2]
  simp only [cay_eq_mk, mk_sq, conj_mk, I_eq_mk, one_eq_mk, neg_mk, mk_mul_mk, mk_add_mk, mk_div_four,
    Complex.mk.injEq]
  constructor <;> field_simp <;> ring

theorem c3_closed (s : ℝ) :
    c3 s = ⟨-(s * (1 - s) ^ 2) / (1 + s ^ 2) ^ 2, -(s * (1 - s ^ 2)) / (1 + s ^ 2) ^ 2⟩ := by
  have h := hD s
  rw [c3]
  simp only [cay_eq_mk, mk_sq, conj_mk, I_eq_mk, one_eq_mk, neg_mk, mk_mul_mk, mk_add_mk, mk_div_four,
    Complex.mk.injEq]
  constructor <;> field_simp <;> ring

/-- **`coeff_autocorr`.** The four coefficients have the autocorrelation of the
unit spectrum `(1, u, u², conj u)`: `Σ_k conj c_k c_{k+m} = δ_{m0}`. -/
theorem coeff_autocorr (s : ℝ) :
    conj (c0 s) * c0 s + conj (c1 s) * c1 s + conj (c2 s) * c2 s + conj (c3 s) * c3 s = 1 ∧
    conj (c0 s) * c1 s + conj (c1 s) * c2 s + conj (c2 s) * c3 s + conj (c3 s) * c0 s = 0 ∧
    conj (c0 s) * c2 s + conj (c1 s) * c3 s + conj (c2 s) * c0 s + conj (c3 s) * c1 s = 0 ∧
    conj (c0 s) * c3 s + conj (c1 s) * c0 s + conj (c2 s) * c1 s + conj (c3 s) * c2 s = 0 := by
  rw [c0_closed, c1_closed, c2_closed, c3_closed]
  simp only [conj_mk, mk_mul_mk, mk_add_mk]
  rw [one_eq_mk, zero_eq_mk]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [Complex.mk.injEq] <;> constructor <;> field_simp <;> ring

/-- **`coeff_zero`.** At strength 0, `u = 1` and `U` is the identity. -/
theorem coeff_zero : c0 0 = 1 ∧ c1 0 = 0 ∧ c2 0 = 0 ∧ c3 0 = 0 := by
  rw [c0_closed, c1_closed, c2_closed, c3_closed, one_eq_mk, zero_eq_mk]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [Complex.mk.injEq] <;> constructor <;> norm_num

/-- **`coeff_one`.** At strength 1, `u = i` and `U = A`. -/
theorem coeff_one : c0 1 = 0 ∧ c1 1 = 1 ∧ c2 1 = 0 ∧ c3 1 = 0 := by
  rw [c0_closed, c1_closed, c2_closed, c3_closed, one_eq_mk, zero_eq_mk]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [Complex.mk.injEq] <;> constructor <;> norm_num

/-- On an eigenvector, `U` is the scalar `c₀ + c₁μ + c₂μ² + c₃μ³`. -/
theorem frac_apply_eigen {n : ℕ} (A : (Fin n → ℂ) → (Fin n → ℂ))
    (hlin : ∀ (c : ℂ) (x : Fin n → ℂ), A (c • x) = c • A x)
    (μ : ℂ) (y : Fin n → ℂ) (hy : A y = μ • y) (k0 k1 k2 k3 : ℂ) :
    k0 • y + k1 • A y + k2 • A (A y) + k3 • A (A (A y)) =
      (k0 + k1 * μ + k2 * μ ^ 2 + k3 * μ ^ 3) • y := by
  rw [hy, hlin, hy, hlin, hlin, hy]
  funext v
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- **`frac_eigen`.** The scalar on the eigenvalue-`1, i, −1, −i` eigenspaces is
`1, u, u², conj u`: the principal power `λ^α` with `u = e^{iπα/2}`
(`cay_eq_exp`). -/
theorem frac_eigen (s : ℝ) :
    c0 s + c1 s + c2 s + c3 s = 1 ∧
    c0 s + c1 s * I + c2 s * (I * I) + c3 s * (I * I * I) = cay s ∧
    c0 s + c1 s * (-1) + c2 s * ((-1) * (-1)) + c3 s * ((-1) * (-1) * (-1)) = cay s * cay s ∧
    c0 s + c1 s * (-I) + c2 s * ((-I) * (-I)) + c3 s * ((-I) * (-I) * (-I)) = conj (cay s) := by
  have h := hD s
  rw [c0_closed, c1_closed, c2_closed, c3_closed, cay_eq_mk]
  simp only [I_eq_mk, one_eq_mk, neg_mk, conj_mk, mk_mul_mk, mk_add_mk]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [Complex.mk.injEq] <;> constructor <;> field_simp <;> ring

/-- **`cay_eq_exp`.** `u = e^{2i·arg(1 + s i)}`; `arg(1 + s i) = arctan s`, so with
`α = (4/π) arg(1 + s i)` this is `i^α` on the principal branch, `u² = (−1)^α` and
`conj u = (−i)^α`. -/
theorem cay_eq_exp (s : ℝ) : cay s = exp (((2 * arg (1 + (s : ℂ) * I) : ℝ) : ℂ) * I) := by
  set z : ℂ := 1 + (s : ℂ) * I with hz
  have hz0 : z ≠ 0 := by
    intro h0
    have := congrArg re h0
    simp [hz] at this
  have habs : (abs z : ℂ) ≠ 0 := by exact_mod_cast (map_ne_zero Complex.abs).mpr hz0
  have he : exp ((arg z : ℂ) * I) = z / (abs z : ℂ) := by
    rw [eq_div_iff habs, mul_comm]
    exact abs_mul_exp_arg_mul_I z
  have h2 : exp ((((2 * arg z : ℝ)) : ℂ) * I) = exp ((arg z : ℂ) * I) ^ 2 := by
    rw [← exp_nat_mul]
    push_cast
    ring_nf
  have hn : (abs z : ℂ) ^ 2 = ((1 + s ^ 2 : ℝ) : ℂ) := by
    rw [← ofReal_pow, Complex.sq_abs, hz, normSq_apply]
    simp
    ring
  rw [h2, he, div_pow, hn, hz]
  have e : (1 + (s : ℂ) * I) ^ 2 = ⟨1 - s ^ 2, 2 * s⟩ :=
    Complex.ext (by simp [pow_two]) (by simp [pow_two]; ring)
  rw [e, cay_eq_mk]
  apply Complex.ext <;> simp only [div_ofReal_re, div_ofReal_im]

/-! ## The transforms the kernel raises to a fractional power -/

/-- A permutation of the vertices preserves the inner sum. -/
theorem perm_ip {n : ℕ} (σ : Equiv.Perm (Fin n)) (x y : Fin n → ℂ) :
    ip (fun v => x (σ v)) (fun v => y (σ v)) = ip x y :=
  Equiv.sum_comp σ (fun v => conj (x v) * y v)

/-- A map of the vertices with `f⁴ = id` is a permutation, with inverse `f³`. -/
def permOfOrderFour {n : ℕ} (f : Fin n → Fin n) (h : ∀ v, f (f (f (f v))) = v) :
    Equiv.Perm (Fin n) where
  toFun := f
  invFun := fun v => f (f (f v))
  left_inv := fun v => h v
  right_inv := fun v => h v

/-- **`perm_frac_energy`.** Raising a vertex permutation of order dividing 4 to
the fractional power at any strength preserves the energy. -/
theorem perm_frac_energy {n : ℕ} (f : Fin n → Fin n) (h : ∀ v, f (f (f (f v))) = v)
    (s : ℝ) (x : Fin n → ℂ) :
    energy (c0 s • x + c1 s • (fun v => x (f v)) +
      c2 s • (fun v => x (f (f v))) + c3 s • (fun v => x (f (f (f v))))) =
      energy x := by
  obtain ⟨h0, h1, h2, h3⟩ := coeff_autocorr s
  have := frac_energy (fun y v => y (f v))
    (fun a b => perm_ip (permOfOrderFour f h) a b)
    (fun y => by funext v; simp only [h]) _ _ _ _ h0 h1 h2 h3 x
  simpa using this

/-! The kernel's vertex index maps (`comp`, `rotateBits`, `bitReverse`,
`swap03`, `affine18`, and `v ^ 2^b` for S26), transcribed on `ℕ`. -/

def compN (v : ℕ) : ℕ := v ^^^ 15
def rotN (v : ℕ) : ℕ := ((v <<< 1) &&& 15) ||| ((v >>> 3) &&& 1)
def revN (v : ℕ) : ℕ := ((v &&& 1) <<< 3) ||| ((v &&& 2) <<< 1) ||| ((v &&& 4) >>> 1) ||| ((v &&& 8) >>> 3)
def swap03N (v : ℕ) : ℕ := ((v &&& 1) <<< 3) ||| (v &&& 6) ||| ((v &&& 8) >>> 3)
def affine18N (v : ℕ) : ℕ :=
  let x0 := v &&& 1; let x1 := (v >>> 1) &&& 1; let x2 := (v >>> 2) &&& 1; let x3 := (v >>> 3) &&& 1
  (x0 ^^^ x1) ||| ((x1 ^^^ x2) <<< 1) ||| ((x2 ^^^ x3) <<< 2) ||| (x3 <<< 3)
def xorN (b v : ℕ) : ℕ := v ^^^ (1 <<< b)

/-- **`perm_order_four`.** Each of the kernel's base vertex maps sends `0..15`
into `0..15` and has order dividing 4 there (decided over all 16 vertices), so
every iterate the sutras use does too. -/
theorem perm_order_four : ∀ v : Fin 16,
    compN v.val < 16 ∧ compN (compN (compN (compN v.val))) = v.val ∧
    rotN v.val < 16 ∧ rotN (rotN (rotN (rotN v.val))) = v.val ∧
    revN v.val < 16 ∧ revN (revN (revN (revN v.val))) = v.val ∧
    swap03N v.val < 16 ∧ swap03N (swap03N (swap03N (swap03N v.val))) = v.val ∧
    affine18N v.val < 16 ∧ affine18N (affine18N (affine18N (affine18N v.val))) = v.val ∧
    (∀ b : Fin 4, xorN b.val v.val < 16 ∧
      xorN b.val (xorN b.val (xorN b.val (xorN b.val v.val))) = v.val) := by
  decide

/-- The Walsh signs are symmetric: `popcount(u ∧ v) = popcount(v ∧ u)`. -/
theorem wsign_symm : ∀ u v : Fin 16, wsignZ u v = wsignZ v u := by decide

theorem wht_orth_complex (v w : Fin 16) :
    (∑ u : Fin 16, ((wsignZ u v : ℤ) : ℂ) * ((wsignZ u w : ℤ) : ℂ)) = if v = w then 16 else 0 := by
  have := wsign_orth v w
  have h2 : (∑ u : Fin 16, ((wsignZ u v : ℤ) : ℂ) * ((wsignZ u w : ℤ) : ℂ)) =
      (((∑ u : Fin 16, wsignZ u v * wsignZ u w : ℤ)) : ℂ) := by push_cast; rfl
  rw [h2, this]
  split_ifs <;> simp

/-- **`wht_wht`.** The normalised Walsh transform is an involution. -/
theorem wht_wht (x : Fin 16 → ℂ) : wht16 (wht16 x) = x := by
  funext u
  calc wht16 (wht16 x) u
      = (1 / 16 : ℂ) * ∑ w, (∑ v, ((wsignZ v u : ℤ) : ℂ) * ((wsignZ v w : ℤ) : ℂ)) * x w := by
        simp only [wht16, Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun w _ => Finset.sum_congr rfl fun v _ => ?_
        rw [wsign_symm u v]
        ring
    _ = x u := by
        simp only [wht_orth_complex, ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, if_true]
        ring

/-- **`wht_ip`.** The normalised Walsh transform preserves the inner sum. -/
theorem wht_ip (x y : Fin 16 → ℂ) : ip (wht16 x) (wht16 y) = ip x y := by
  have hc : ∀ u, conj (wht16 x u) = (1 / 4 : ℂ) * ∑ w, ((wsignZ u w : ℤ) : ℂ) * conj (x w) := by
    intro u
    simp only [wht16, map_mul, map_sum, map_div₀, map_one, map_ofNat, map_intCast]
  unfold ip
  calc (∑ u : Fin 16, conj (wht16 x u) * wht16 y u)
      = ∑ u : Fin 16, (1 / 16 : ℂ) * ∑ w, ∑ v,
          ((wsignZ u w : ℤ) : ℂ) * ((wsignZ u v : ℤ) : ℂ) * (conj (x w) * y v) := by
        refine Finset.sum_congr rfl fun u _ => ?_
        rw [hc u, wht16, mul_mul_mul_comm, Finset.sum_mul_sum]
        congr 1
        · norm_num
        · refine Finset.sum_congr rfl fun w _ => Finset.sum_congr rfl fun v _ => ?_
          ring
    _ = (1 / 16 : ℂ) * ∑ w, ∑ v,
          (∑ u : Fin 16, ((wsignZ u w : ℤ) : ℂ) * ((wsignZ u v : ℤ) : ℂ)) * (conj (x w) * y v) := by
        rw [← Finset.mul_sum]
        congr 1
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun w _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun v _ => ?_
        rw [Finset.sum_mul]
    _ = ∑ v : Fin 16, conj (x v) * y v := by
        simp only [wht_orth_complex, ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ,
          if_true, Finset.mul_sum]
        refine Finset.sum_congr rfl fun v _ => ?_
        ring

/-- **`wht_frac_energy`.** S3's Walsh transform at its fractional power
preserves the energy. -/
theorem wht_frac_energy (s : ℝ) (x : Fin 16 → ℂ) :
    energy (c0 s • x + c1 s • wht16 x + c2 s • wht16 (wht16 x) +
      c3 s • wht16 (wht16 (wht16 x))) = energy x := by
  obtain ⟨h0, h1, h2, h3⟩ := coeff_autocorr s
  exact frac_energy wht16 wht_ip (fun y => by rw [wht_wht, wht_wht]) _ _ _ _ h0 h1 h2 h3 x

/-- S23's twisted complement swap: `(Jx)_v = p·x_{c(v)}` on the lower half and
`conj(p)·x_{c(v)}` on the upper, `c(v) = 15 − v`, `|p| = 1`. -/
noncomputable def twist (p : ℂ) (x : Fin 16 → ℂ) (v : Fin 16) : ℂ :=
  (if v.val < 8 then p else conj p) * x (Fin.rev v)

theorem twist_twist (p : ℂ) (hp : normSq p = 1) (x : Fin 16 → ℂ) : twist p (twist p x) = x := by
  have hpp : p * conj p = 1 := by rw [mul_conj, hp]; simp
  funext v
  simp only [twist, Fin.rev_rev]
  have hv : (Fin.rev v).val < 8 ↔ ¬ v.val < 8 := by
    rw [Fin.val_rev]; omega
  by_cases h : v.val < 8
  · rw [if_pos h, if_neg (by rw [hv]; exact not_not.mpr h), ← mul_assoc, hpp, one_mul]
  · rw [if_neg h, if_pos (hv.mpr h), ← mul_assoc, mul_comm (conj p), hpp, one_mul]

theorem twist_ip (p : ℂ) (hp : normSq p = 1) (x y : Fin 16 → ℂ) :
    ip (twist p x) (twist p y) = ip x y := by
  have hpp : conj p * p = 1 := by rw [mul_comm, mul_conj, hp]; simp
  unfold ip
  have : ∀ v : Fin 16, conj (twist p x v) * twist p y v = conj (x (Fin.rev v)) * y (Fin.rev v) := by
    intro v
    simp only [twist, map_mul]
    split_ifs
    · rw [mul_mul_mul_comm, hpp, one_mul]
    · rw [conj_conj, mul_mul_mul_comm, mul_comm p, hpp, one_mul]
  simp only [this]
  exact Equiv.sum_comp Fin.revPerm (fun v => conj (x v) * y v)

/-- **`twist_frac_energy`.** S23's twisted swap at its fractional power
preserves the energy. -/
theorem twist_frac_energy (p : ℂ) (hp : normSq p = 1) (s : ℝ) (x : Fin 16 → ℂ) :
    energy (c0 s • x + c1 s • twist p x + c2 s • twist p (twist p x) +
      c3 s • twist p (twist p (twist p x))) = energy x := by
  obtain ⟨h0, h1, h2, h3⟩ := coeff_autocorr s
  exact frac_energy (twist p) (twist_ip p hp)
    (fun y => by rw [twist_twist p hp, twist_twist p hp]) _ _ _ _ h0 h1 h2 h3 x

end FractionalPower
end SutraWS
