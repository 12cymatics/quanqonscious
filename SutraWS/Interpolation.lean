import SutraWS.Vertex
import Mathlib.Analysis.SpecialFunctions.Complex.Arg
import Mathlib.Algebra.GeomSum
import Mathlib.Tactic

/-!
# The FAITHFUL interpolation

`FAITHFUL` in `vedic_v18.51.1_exact_phi.html` lifts the sixteen vertex amplitudes
onto the torus the page draws: it pairs each vertex's bits into a ring position
(`ringTheta`, `ringPhi`), takes a 4×4 discrete Fourier transform (`coeffs`), and
evaluates the trigonometric polynomial over `MODES = [0, 1, 2, -1]`
(`evaluator`, `multiRealEvaluator`). `audit` then decides two residuals on the
live state: `node`, the interpolant against the state at each vertex's
`tessVertexToTorus` position, and `rel`, the 64×64 grid mean of `|Ψ|²` times 16
against `Σ|ψₖ|²`.

Each of those functions is transcribed below and the identities `audit` measures
are proved for every state:

* `cell_bijective`, `cube_edge_iff_torus_edge` — the Gray pairing writes all
  sixteen grid cells once each, and carries 4-cube edges exactly onto torus edges;
* `node_theta`, `node_phi` — `tessVertexToTorus` puts vertex `k` at
  `π/4 + ringTheta(k)·π/2` (and likewise `φ`), which is what `OFFSET` assumes;
* `node_reproduction` — exactly, `node = 0`;
* `interpolant_unique` — no other coefficients over `MODES` reproduce the nodes;
* `parseval_coeffs`, `parseval_grid` — exactly, `rel = 0`, on any grid of at
  least 4×4. This rests on `MODES` being four distinct residues mod 4
  (`MI_mode`) whose pairwise differences stay below 4 (`mode_diff_bound`);
* `real_node_reproduction`, `phase_at_node` — what the Wheeler and phase channels
  read off the same interpolant.

The statements are over `ℝ` and `ℂ`. The page evaluates them in IEEE doubles, so
the runtime residuals are the distance of the float computation from these exact
values, which is why `audit` compares them against `1e-9` rather than `0`. The
node theorems take `TESS_ROT.xy = 0`: the page never changes it, and an `xy`
rotation is not a rigid shift of `θ` and `φ`, so no such theorem holds for it.
-/

namespace SutraWS
namespace Interp

/-! ### Ring mapping -/

/-- `GRAY_POS = { 0: 0, 1: 1, 3: 2, 2: 3 }`, indexed by the 2-bit pair value. -/
def grayPos : Fin 4 → Fin 4 := ![0, 1, 3, 2]

/-- The `(w,x)` pair inside `ringTheta`: `(((k >> 3) & 1) << 1) | (k & 1)`. -/
def pairTheta (k : Vertex) : ℕ := (((k.val >>> 3) &&& 1) <<< 1) ||| (k.val &&& 1)
/-- The `(z,y)` pair inside `ringPhi`: `(((k >> 2) & 1) << 1) | ((k >> 1) & 1)`. -/
def pairPhi (k : Vertex) : ℕ := (((k.val >>> 2) &&& 1) <<< 1) ||| ((k.val >>> 1) &&& 1)

theorem pairTheta_lt (k : Vertex) : pairTheta k < 4 := by revert k; decide
theorem pairPhi_lt (k : Vertex) : pairPhi k < 4 := by revert k; decide

/-- `ringTheta = k => GRAY_POS[pairTheta(k)]`. -/
def ringTheta (k : Vertex) : Fin 4 := grayPos ⟨pairTheta k, pairTheta_lt k⟩
/-- `ringPhi = k => GRAY_POS[pairPhi(k)]`. -/
def ringPhi (k : Vertex) : Fin 4 := grayPos ⟨pairPhi k, pairPhi_lt k⟩

/-- The page's flat index `row * 4 + col`. -/
def idx (row col : Fin 4) : Fin 16 := ⟨row.val * 4 + col.val, by omega⟩

/-- Where `coeffs` stores vertex `k`: `gr[ringPhi(k) * 4 + ringTheta(k)]`. -/
def cell (k : Vertex) : Fin 16 := idx (ringPhi k) (ringTheta k)

/-- The scatter writes each of the sixteen grid cells exactly once: no cell is
left at the `Float64Array`'s zero and none is overwritten. -/
theorem cell_bijective : Function.Bijective cell := by
  refine (Fintype.bijective_iff_injective_and_card cell).mpr ⟨?_, rfl⟩
  decide

/-- Nearest neighbours on the 4×4 torus, `Fin 4` addition wrapping. -/
def torusAdj (a b : Fin 4 × Fin 4) : Prop :=
  (a.2 = b.2 ∧ (b.1 = a.1 + 1 ∨ a.1 = b.1 + 1)) ∨
  (a.1 = b.1 ∧ (b.2 = a.2 + 1 ∨ a.2 = b.2 + 1))

instance (a b : Fin 4 × Fin 4) : Decidable (torusAdj a b) := by unfold torusAdj; infer_instance

/-- `Q₄ = C₄ □ C₄` under the Gray pairing: two vertices share a 4-cube edge
(`VTX.neighbors`) exactly when their ring indices are torus neighbours. -/
theorem cube_edge_iff_torus_edge (k l : Vertex) :
    (∃ a, l = flip k a) ↔ torusAdj (ringTheta k, ringPhi k) (ringTheta l, ringPhi l) := by
  revert k l; decide

/-- `MI = f => ((f % 4) + 4) % 4`, with ECMAScript's truncating `%` (`Int.mod`). -/
def MI (f : ℤ) : Fin 4 :=
  ⟨(Int.mod (Int.mod f 4 + 4) 4).toNat, by
    have := Int.mod_lt_of_pos (Int.mod f 4 + 4) (show (0 : ℤ) < 4 by decide)
    exact Nat.lt_succ_of_le (Int.toNat_le.mpr (by omega))⟩

/-- `MODES = [0, 1, 2, -1]`: the Nyquist frequency is carried once, unsplit, at `+2`. -/
def MODES : List ℤ := [0, 1, 2, -1]

/-- `MODES[p]`. -/
def mode : Fin 4 → ℤ := ![0, 1, 2, -1]

/-- `MODES` is a complete residue system mod 4, and `MI` reads each mode back to
its own slot -- the property the unsplit Nyquist term exists to keep. -/
theorem MI_mode (p : Fin 4) : MI (mode p) = p := by revert p; decide


/-! ### Node placement: `tessVertexToTorus` -/

open Complex

/-- `Math.atan2(y, x)`. ECMAScript specifies it as the angle of the point
`(x, y)`, which is `Complex.arg (x + y i)`; the engine's float result
approximates this value. -/
noncomputable def atan2 (y x : ℝ) : ℝ := Complex.arg ⟨x, y⟩

/-- Rounding toward zero, as ECMAScript's `%` uses. -/
noncomputable def jsTrunc (x : ℝ) : ℤ := if 0 ≤ x then ⌊x⌋ else ⌈x⌉

/-- ECMAScript `x % m` on Numbers: `x - m * trunc(x / m)`. -/
noncomputable def jsRem (x m : ℝ) : ℝ := x - m * jsTrunc (x / m)

/-- `TESS_V[k]`: coordinate `b` of vertex `k` (x, y, z, w = bits 0..3), `±1`. -/
def tv (k : Vertex) (b : Fin 4) : ℝ := (sgn k b : ℝ)

/-- `tessVertexToTorus(TESS_V[k])` with `TESS_ROT = {xw, yz, xy}`, returning
`(theta, phi)`; `TAU = 2π`. -/
noncomputable def tessVertexToTorus (xw yz xy : ℝ) (k : Vertex) : ℝ × ℝ :=
  let x := tv k 0
  let y := tv k 1
  let z := tv k 2
  let w := tv k 3
  let x₁ := x * Real.cos xw - w * Real.sin xw
  let w₁ := x * Real.sin xw + w * Real.cos xw
  let y₁ := y * Real.cos yz - z * Real.sin yz
  let z₁ := y * Real.sin yz + z * Real.cos yz
  let x₂ := x₁ * Real.cos xy - y₁ * Real.sin xy
  let y₂ := x₁ * Real.sin xy + y₁ * Real.cos xy
  (jsRem (atan2 w₁ x₂ + Real.pi) (2 * Real.pi), jsRem (atan2 z₁ y₂ + Real.pi) (2 * Real.pi))

/-- The `(±1, ±1)` pair at each ring position. -/
def signPair : Fin 4 → ℤ × ℤ := ![(-1, -1), (1, -1), (1, 1), (-1, 1)]

theorem theta_signs (k : Vertex) : (sgn k 0, sgn k 3) = signPair (ringTheta k) := by
  revert k; decide

theorem phi_signs (k : Vertex) : (sgn k 1, sgn k 2) = signPair (ringPhi k) := by
  revert k; decide

theorem exp_pi_div_four_mul_I :
    exp (↑(Real.pi / 4) * I) = ↑(Real.sqrt 2 / 2) + ↑(Real.sqrt 2 / 2) * I := by
  rw [exp_mul_I, ← ofReal_cos, ← ofReal_sin, Real.cos_pi_div_four, Real.sin_pi_div_four]

theorem exp_pi_div_two_mul_I' : exp (↑(Real.pi / 2) * I) = I := by
  rw [exp_mul_I, ← ofReal_cos, ← ofReal_sin, Real.cos_pi_div_two, Real.sin_pi_div_two]
  simp

theorem signPair_polar (r : Fin 4) :
    ((signPair r).1 : ℂ) + ((signPair r).2 : ℂ) * I
      = ↑(Real.sqrt 2) * exp (↑(Real.pi / 4 + (r : ℝ) * (Real.pi / 2) - Real.pi) * I) := by
  have hsplit : (↑(Real.pi / 4 + (r : ℝ) * (Real.pi / 2) - Real.pi) * I : ℂ)
      = ↑(Real.pi / 4) * I + ((r : ℕ) : ℂ) * (↑(Real.pi / 2) * I) + (-(↑Real.pi * I)) := by
    push_cast; ring
  have h2 : (↑(Real.sqrt 2) : ℂ) ^ 2 = 2 := by
    rw [← ofReal_pow, Real.sq_sqrt (by norm_num)]; norm_num
  have hI3 : I ^ 3 = -I := by rw [pow_succ, I_sq]; ring
  have hI4 : I ^ 4 = 1 := by rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, I_sq]; norm_num
  rw [hsplit, exp_add, exp_add, exp_nat_mul, exp_pi_div_four_mul_I, exp_pi_div_two_mul_I',
    exp_neg, exp_pi_mul_I]
  fin_cases r <;> simp [signPair] <;> ring_nf <;> simp only [h2, hI3, hI4, I_sq] <;> ring


theorem angle_of_pair (a : ℝ) (r : Fin 4) : ∃ n : ℤ,
    jsRem (atan2 (((signPair r).1 : ℝ) * Real.sin a + ((signPair r).2 : ℝ) * Real.cos a)
                 (((signPair r).1 : ℝ) * Real.cos a - ((signPair r).2 : ℝ) * Real.sin a)
            + Real.pi) (2 * Real.pi)
      = Real.pi / 4 + a + (r : ℝ) * (Real.pi / 2) + n * (2 * Real.pi) := by
  set ψ := Real.pi / 4 + (r : ℝ) * (Real.pi / 2) - Real.pi with hψ
  set Y := ((signPair r).1 : ℝ) * Real.sin a + ((signPair r).2 : ℝ) * Real.cos a
  set X := ((signPair r).1 : ℝ) * Real.cos a - ((signPair r).2 : ℝ) * Real.sin a
  have hz : (⟨X, Y⟩ : ℂ) = ↑(Real.sqrt 2) * exp (↑(ψ + a) * I) := by
    have : (⟨X, Y⟩ : ℂ) = (((signPair r).1 : ℂ) + ((signPair r).2 : ℂ) * I) * exp (↑a * I) := by
      apply Complex.ext <;> simp [X, Y, exp_ofReal_mul_I_re, exp_ofReal_mul_I_im]
    rw [this, signPair_polar, mul_assoc, ← exp_add, hψ]
    congr 2
    push_cast
    ring
  obtain ⟨t, ht⟩ : ∃ t : ℤ, atan2 Y X = ψ + a - t * (2 * Real.pi) := by
    refine ⟨toIocDiv (mul_pos two_pos Real.pi_pos) (-Real.pi) (ψ + a), ?_⟩
    unfold atan2
    rw [hz, arg_real_mul _ (by positivity), arg_exp_mul_I, ← self_sub_toIocDiv_zsmul,
      zsmul_eq_mul]
  refine ⟨-t - jsTrunc ((atan2 Y X + Real.pi) / (2 * Real.pi)), ?_⟩
  unfold jsRem
  rw [ht, hψ]
  push_cast
  ring

/-- With `xy = 0`, vertex `k` lands at `θ = π/4 + xw + ringTheta(k)·π/2` (mod 2π):
`OFFSET` is where vertex 0 sits, and the Gray ring index counts quarter turns. -/
theorem node_theta (xw yz : ℝ) (k : Vertex) : ∃ n : ℤ,
    (tessVertexToTorus xw yz 0 k).1
      = Real.pi / 4 + xw + (ringTheta k : ℝ) * (Real.pi / 2) + n * (2 * Real.pi) := by
  obtain ⟨n, hn⟩ := angle_of_pair xw (ringTheta k)
  refine ⟨n, ?_⟩
  have h1 : sgn k 0 = (signPair (ringTheta k)).1 := congrArg Prod.fst (theta_signs k)
  have h2 : sgn k 3 = (signPair (ringTheta k)).2 := congrArg Prod.snd (theta_signs k)
  simp only [tessVertexToTorus, tv, Real.cos_zero, Real.sin_zero, mul_one, mul_zero, sub_zero]
  rw [h1, h2]
  exact hn

/-- The same for `φ` with the `(y,z)` pair and `ringPhi`. -/
theorem node_phi (xw yz : ℝ) (k : Vertex) : ∃ n : ℤ,
    (tessVertexToTorus xw yz 0 k).2
      = Real.pi / 4 + yz + (ringPhi k : ℝ) * (Real.pi / 2) + n * (2 * Real.pi) := by
  obtain ⟨n, hn⟩ := angle_of_pair yz (ringPhi k)
  refine ⟨n, ?_⟩
  have h1 : sgn k 1 = (signPair (ringPhi k)).1 := congrArg Prod.fst (phi_signs k)
  have h2 : sgn k 2 = (signPair (ringPhi k)).2 := congrArg Prod.snd (phi_signs k)
  simp only [tessVertexToTorus, tv, Real.cos_zero, Real.sin_zero, mul_one, mul_zero, sub_zero,
    zero_add]
  rw [h1, h2]
  exact hn


/-! ### Forward transform and evaluator -/

/-- `cos t + i sin t`, the factor `coeffs` multiplies each sample by. -/
noncomputable def cis (t : ℝ) : ℂ := ↑(Real.cos t) + ↑(Real.sin t) * I

theorem cis_eq_exp (t : ℝ) : cis t = exp (↑t * I) := by
  rw [cis, exp_mul_I, ← ofReal_cos, ← ofReal_sin]

/-- `OFFSET = Math.PI / 4`. -/
noncomputable def OFFSET : ℝ := Real.pi / 4

/-- The scatter at the top of `coeffs`: a zero array, then
`gr[ringPhi(k)*4+ringTheta(k)] = re[k]` (and `gi` likewise) for `k = 0..15`. -/
noncomputable def scatter (ψ : Vertex → ℂ) : Fin 16 → ℂ :=
  (List.finRange 16).foldl (fun g k => Function.update g (cell k) (ψ k)) 0

/-- `c[n*4+m]` from the loop body of `coeffs`:
`t = -2π(m·i + n·j)/4`, `(sr,si) += g[j*4+i] · (cos t + i sin t)`, then `/16`. -/
noncomputable def coeffAt (gr : Fin 16 → ℂ) (m n : Fin 4) : ℂ :=
  (∑ i : Fin 4, ∑ j : Fin 4,
      gr (idx j i) * cis (-2 * Real.pi * ((m : ℝ) * (i : ℝ) + (n : ℝ) * (j : ℝ)) / 4)) / 16

/-- The flat array `coeffs` returns: slot `q` holds `c[n*4+m]` with `m = q % 4`, `n = q / 4`. -/
noncomputable def coeffs (gr : Fin 16 → ℂ) : Fin 16 → ℂ := fun q =>
  coeffAt gr ⟨q.val % 4, Nat.mod_lt _ (by norm_num)⟩ ⟨q.val / 4, by omega⟩

theorem coeffs_idx (gr : Fin 16 → ℂ) (m n : Fin 4) : coeffs gr (idx n m) = coeffAt gr m n := by
  unfold coeffs idx
  congr 1 <;> ext <;> simp <;> omega

/-- `evaluator(c)(theta, phi)` with `TESS_ROT.xw = xw`, `TESS_ROT.yz = yz`:
`u = θ - OFFSET - xw`, `v = φ - OFFSET - yz`, and for `fm, fn` in `MODES`,
`c[MI(fn)*4+MI(fm)] · (er + i·ei)` with the page's `er`, `ei`. -/
noncomputable def evaluator (c : Fin 16 → ℂ) (xw yz : ℝ) (θ φ : ℝ) : ℂ :=
  let u := θ - OFFSET - xw
  let v := φ - OFFSET - yz
  (MODES.map fun fm => (MODES.map fun fn =>
      c (idx (MI fn) (MI fm)) *
        (↑(Real.cos (fm * u) * Real.cos (fn * v) - Real.sin (fm * u) * Real.sin (fn * v)) +
         ↑(Real.cos (fm * u) * Real.sin (fn * v) + Real.sin (fm * u) * Real.cos (fn * v)) * I)
    ).sum).sum

theorem sum_MODES (f : ℤ → ℂ) : (MODES.map f).sum = ∑ p : Fin 4, f (mode p) := by
  simp [mode, MODES, Fin.sum_univ_four, add_assoc]

theorem rotor_eq_exp (A B : ℝ) :
    (↑(Real.cos A * Real.cos B - Real.sin A * Real.sin B) +
      ↑(Real.cos A * Real.sin B + Real.sin A * Real.cos B) * I : ℂ) = exp (↑(A + B) * I) := by
  rw [← cis_eq_exp, cis, Real.cos_add, Real.sin_add]
  push_cast
  ring

theorem evaluator_exp (c : Fin 16 → ℂ) (xw yz θ φ : ℝ) :
    evaluator c xw yz θ φ
      = ∑ p : Fin 4, ∑ q : Fin 4, c (idx q p) *
          exp (↑((mode p : ℝ) * (θ - OFFSET - xw) + (mode q : ℝ) * (φ - OFFSET - yz)) * I) := by
  unfold evaluator
  simp only [rotor_eq_exp, sum_MODES, MI_mode]


/-! ### Node reproduction and uniqueness -/

theorem cell_injective : Function.Injective cell := cell_bijective.1

private theorem foldl_update_off (ψ : Vertex → ℂ) (q : Fin 16) :
    ∀ (xs : List Vertex) (g : Fin 16 → ℂ), (∀ x ∈ xs, cell x ≠ q) →
      xs.foldl (fun g k => Function.update g (cell k) (ψ k)) g q = g q := by
  intro xs
  induction xs with
  | nil => intro g _; rfl
  | cons x xs ih =>
    intro g h
    rw [List.foldl_cons, ih _ (fun y hy => h y (List.mem_cons_of_mem _ hy)),
      Function.update_noteq (h x (List.mem_cons_self _ _)).symm]

private theorem foldl_update_on (ψ : Vertex → ℂ) (k : Vertex) :
    ∀ (xs : List Vertex) (g : Fin 16 → ℂ), k ∈ xs →
      xs.foldl (fun g k => Function.update g (cell k) (ψ k)) g (cell k) = ψ k := by
  intro xs
  induction xs with
  | nil => intro g h; cases h
  | cons x xs ih =>
    intro g h
    rw [List.foldl_cons]
    by_cases hk : k ∈ xs
    · exact ih _ hk
    · have hx : x = k := by
        rcases List.mem_cons.mp h with h | h
        · exact h.symm
        · exact absurd h hk
      subst hx
      rw [foldl_update_off ψ (cell x) xs _ (fun y hy hyx => hk (cell_injective hyx ▸ hy)),
        Function.update_same]

theorem scatter_cell (ψ : Vertex → ℂ) (k : Vertex) : scatter ψ (cell k) = ψ k :=
  foldl_update_on ψ k _ _ (List.mem_finRange k)

theorem I_zpow_emod (z : ℤ) : I ^ z = I ^ (z % 4).toNat := by
  have h4 : I ^ (4 : ℤ) = 1 := by
    rw [show (4 : ℤ) = ((2 * 2 : ℕ) : ℤ) by norm_num, zpow_natCast, pow_mul, I_sq]; norm_num
  conv_lhs => rw [← Int.emod_add_ediv z 4]
  rw [zpow_add₀ I_ne_zero, zpow_mul, h4, one_zpow, mul_one, ← zpow_natCast,
    Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]

theorem I_zpow_congr {a b : ℤ} (h : a ≡ b [ZMOD 4]) : I ^ a = I ^ b := by
  rw [I_zpow_emod a, I_zpow_emod b, h]

theorem quarter_ortho (r i : Fin 4) :
    ∑ p : Fin 4, I ^ ((p : ℤ) * ((r : ℤ) - (i : ℤ))) = if i = r then 4 else 0 := by
  have hI3 : I ^ 3 = -I := by rw [pow_succ, I_sq]; ring
  have h3 : ((3 : Fin 4) : ℕ) = 3 := rfl
  fin_cases r <;> fin_cases i <;>
    simp only [Fin.sum_univ_four, I_zpow_emod, Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two,
      h3, Fin.zero_eta, Fin.mk_one] <;>
    norm_num [Fin.ext_iff] <;> ring_nf <;> simp only [I_sq, hI3] <;> ring

theorem exp_quarter (z : ℤ) : exp (↑((z : ℝ) * (Real.pi / 2)) * I) = I ^ z := by
  rw [show (↑((z : ℝ) * (Real.pi / 2)) * I : ℂ) = (z : ℂ) * (↑(Real.pi / 2) * I) by push_cast; ring,
    exp_int_mul, exp_pi_div_two_mul_I']

theorem cis_page (m n i j : Fin 4) :
    cis (-2 * Real.pi * ((m : ℝ) * (i : ℝ) + (n : ℝ) * (j : ℝ)) / 4)
      = I ^ (-((m : ℤ) * i + (n : ℤ) * j)) := by
  rw [cis_eq_exp, ← exp_quarter]
  congr 2
  push_cast
  ring

theorem mode_modEq (p : Fin 4) : mode p ≡ (p : ℤ) [ZMOD 4] := by
  revert p; decide


theorem sum_swap4 (F : Fin 4 → Fin 4 → Fin 4 → Fin 4 → ℂ) :
    ∑ p, ∑ q, ∑ i, ∑ j, F p q i j = ∑ i, ∑ j, ∑ p, ∑ q, F p q i j :=
  calc _ = ∑ p, ∑ i, ∑ q, ∑ j, F p q i j := Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ p, ∑ i, ∑ j, ∑ q, F p q i j :=
        Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => Finset.sum_comm
    _ = ∑ i, ∑ p, ∑ j, ∑ q, F p q i j := Finset.sum_comm
    _ = _ := Finset.sum_congr rfl fun _ _ => Finset.sum_comm

theorem idft (G : Fin 4 → Fin 4 → ℂ) (r s : Fin 4) :
    ∑ p : Fin 4, ∑ q : Fin 4,
      (∑ i : Fin 4, ∑ j : Fin 4, G i j * I ^ (-((p : ℤ) * i + (q : ℤ) * j))) / 16
        * I ^ ((p : ℤ) * r + (q : ℤ) * s) = G r s := by
  have hterm : ∀ p q i j : Fin 4,
      I ^ (-((p : ℤ) * i + (q : ℤ) * j)) * I ^ ((p : ℤ) * r + (q : ℤ) * s)
        = I ^ ((p : ℤ) * ((r : ℤ) - i)) * I ^ ((q : ℤ) * ((s : ℤ) - j)) := by
    intro p q i j
    rw [← zpow_add₀ I_ne_zero, ← zpow_add₀ I_ne_zero]
    ring_nf
  calc _ = ∑ p : Fin 4, ∑ q : Fin 4, ∑ i : Fin 4, ∑ j : Fin 4,
        G i j * (I ^ ((p : ℤ) * ((r : ℤ) - i)) * I ^ ((q : ℤ) * ((s : ℤ) - j))) / 16 := by
          refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
          rw [Finset.sum_div, Finset.sum_mul]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_div, Finset.sum_mul]
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [← hterm]
          ring
    _ = ∑ i : Fin 4, ∑ j : Fin 4, ∑ p : Fin 4, ∑ q : Fin 4,
        G i j * (I ^ ((p : ℤ) * ((r : ℤ) - i)) * I ^ ((q : ℤ) * ((s : ℤ) - j))) / 16 :=
          sum_swap4 _
    _ = ∑ i : Fin 4, ∑ j : Fin 4, G i j *
        ((∑ p : Fin 4, I ^ ((p : ℤ) * ((r : ℤ) - i))) *
          (∑ q : Fin 4, I ^ ((q : ℤ) * ((s : ℤ) - j)))) / 16 := by
          refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.sum_mul_sum, Finset.mul_sum, Finset.sum_div]
          refine Finset.sum_congr rfl fun p _ => ?_
          rw [Finset.mul_sum, Finset.sum_div]
    _ = G r s := by
          simp only [quarter_ortho, mul_ite, ite_mul, mul_zero, zero_mul, div_eq_mul_inv,
            Finset.sum_ite_eq', Finset.mem_univ, if_true]
          ring

/-- Any coefficient array, evaluated at the node `tessVertexToTorus` gives vertex
`k`, is the inverse transform read at that vertex's ring position. -/
theorem evaluator_at_node (c : Fin 16 → ℂ) (xw yz : ℝ) (k : Vertex) :
    evaluator c xw yz (tessVertexToTorus xw yz 0 k).1 (tessVertexToTorus xw yz 0 k).2
      = ∑ p : Fin 4, ∑ q : Fin 4,
          c (idx q p) * I ^ ((p : ℤ) * (ringTheta k : ℤ) + (q : ℤ) * (ringPhi k : ℤ)) := by
  obtain ⟨n, hn⟩ := node_theta xw yz k
  obtain ⟨n', hn'⟩ := node_phi xw yz k
  rw [evaluator_exp, hn, hn']
  refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
  congr 1
  rw [show (↑((mode p : ℝ) * (Real.pi / 4 + xw + (ringTheta k : ℝ) * (Real.pi / 2)
            + n * (2 * Real.pi) - OFFSET - xw)
          + (mode q : ℝ) * (Real.pi / 4 + yz + (ringPhi k : ℝ) * (Real.pi / 2)
            + n' * (2 * Real.pi) - OFFSET - yz)) * I : ℂ)
        = ((mode p * (ringTheta k : ℤ) + mode q * (ringPhi k : ℤ) : ℤ) : ℂ)
            * (↑(Real.pi / 2) * I)
          + ((mode p * n + mode q * n' : ℤ) : ℂ) * (2 * ↑Real.pi * I) by
      unfold OFFSET; push_cast; ring]
  rw [exp_add, exp_int_mul, exp_pi_div_two_mul_I', exp_int_mul_two_pi_mul_I, mul_one]
  exact I_zpow_congr (Int.ModEq.add ((mode_modEq p).mul_right _) ((mode_modEq q).mul_right _))

/-- **Node reproduction.** The interpolant `FAITHFUL` builds from a state takes,
at the position `tessVertexToTorus` gives each vertex, exactly that vertex's
value -- for every state and every `xw`, `yz` rotation. This is the identity
`audit` measures as `node`; exactly, `node = 0`. -/
theorem node_reproduction (ψ : Vertex → ℂ) (xw yz : ℝ) (k : Vertex) :
    evaluator (coeffs (scatter ψ)) xw yz
      (tessVertexToTorus xw yz 0 k).1 (tessVertexToTorus xw yz 0 k).2 = ψ k := by
  rw [evaluator_at_node]
  simp only [coeffs_idx, coeffAt, cis_page]
  exact (idft (fun i j => scatter ψ (idx j i)) _ _).trans (scatter_cell ψ k)

theorem dft_idft (C : Fin 4 → Fin 4 → ℂ) (m n : Fin 4) :
    (∑ i : Fin 4, ∑ j : Fin 4,
      (∑ p : Fin 4, ∑ q : Fin 4, C p q * I ^ ((p : ℤ) * i + (q : ℤ) * j))
        * I ^ (-((m : ℤ) * i + (n : ℤ) * j))) / 16 = C m n := by
  have hterm : ∀ p q i j : Fin 4,
      I ^ ((p : ℤ) * i + (q : ℤ) * j) * I ^ (-((m : ℤ) * i + (n : ℤ) * j))
        = I ^ ((i : ℤ) * ((p : ℤ) - m)) * I ^ ((j : ℤ) * ((q : ℤ) - n)) := by
    intro p q i j
    rw [← zpow_add₀ I_ne_zero, ← zpow_add₀ I_ne_zero]
    ring_nf
  calc _ = (∑ i : Fin 4, ∑ j : Fin 4, ∑ p : Fin 4, ∑ q : Fin 4,
        C p q * (I ^ ((i : ℤ) * ((p : ℤ) - m)) * I ^ ((j : ℤ) * ((q : ℤ) - n)))) / 16 := by
          congr 1
          refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun p _ => ?_
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun q _ => ?_
          rw [← hterm]
          ring
    _ = (∑ p : Fin 4, ∑ q : Fin 4, C p q *
        ((∑ i : Fin 4, I ^ ((i : ℤ) * ((p : ℤ) - m))) *
          (∑ j : Fin 4, I ^ ((j : ℤ) * ((q : ℤ) - n))))) / 16 := by
          rw [← sum_swap4]
          congr 1
          refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
          rw [Finset.sum_mul_sum, Finset.mul_sum]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.mul_sum]
    _ = C m n := by
          simp only [quarter_ortho, mul_ite, ite_mul, mul_zero, zero_mul,
            Finset.sum_ite_eq, Finset.mem_univ, if_true]
          ring

theorem idx_surjective (q : Fin 16) : ∃ i j : Fin 4, idx j i = q := by
  revert q; decide

/-- **Uniqueness.** A coefficient array over `MODES × MODES` that reproduces the
sixteen vertex values at their nodes *is* `coeffs` of those values: the
interpolant `FAITHFUL` draws is the only one with this mode set. -/
theorem interpolant_unique (ψ : Vertex → ℂ) (xw yz : ℝ) (c : Fin 16 → ℂ)
    (h : ∀ k, evaluator c xw yz
      (tessVertexToTorus xw yz 0 k).1 (tessVertexToTorus xw yz 0 k).2 = ψ k) :
    c = coeffs (scatter ψ) := by
  have hgrid : ∀ i j : Fin 4, scatter ψ (idx j i)
      = ∑ p : Fin 4, ∑ q : Fin 4, c (idx q p) * I ^ ((p : ℤ) * i + (q : ℤ) * j) := by
    intro i j
    obtain ⟨k, hk⟩ := cell_bijective.2 (idx j i)
    have hpos : ringTheta k = i ∧ ringPhi k = j := by
      have hk' : idx (ringPhi k) (ringTheta k) = idx j i := hk
      unfold idx at hk'
      simp only [Fin.mk.injEq] at hk'
      constructor <;> ext <;> omega
    rw [← hk, scatter_cell, ← h k, evaluator_at_node, hpos.1, hpos.2]
  funext q
  obtain ⟨m, n, rfl⟩ := idx_surjective q
  rw [coeffs_idx]
  unfold coeffAt
  simp only [cis_page, hgrid]
  exact (dft_idft (fun p q => c (idx q p)) m n).symm

/-! ### Parseval -/

theorem conj_I_zpow (z : ℤ) : (starRingEnd ℂ) (I ^ z) = I ^ (-z) := by
  rw [map_zpow₀, conj_I, zpow_neg, ← inv_zpow, inv_I]

theorem mode_injective : Function.Injective mode := by decide

theorem mode_diff_bound (p p' : Fin 4) : mode p - mode p' ≤ 3 ∧ -3 ≤ mode p - mode p' := by
  revert p p'; decide

theorem conj_exp_ofReal_mul_I (x : ℝ) :
    (starRingEnd ℂ) (exp (↑x * I)) = exp (↑(-x) * I) := by
  rw [← exp_conj, map_mul, conj_ofReal, conj_I]
  push_cast
  ring_nf

theorem char_sum (N : ℕ) (hN : 4 ≤ N) (s : ℝ) (p p' : Fin 4) :
    ∑ a in Finset.range N,
      exp (↑((mode p : ℝ) * (2 * Real.pi * (a : ℝ) / N - OFFSET - s)) * I) *
        (starRingEnd ℂ) (exp (↑((mode p' : ℝ) * (2 * Real.pi * (a : ℝ) / N - OFFSET - s)) * I))
      = if p = p' then (N : ℂ) else 0 := by
  have hN0 : (N : ℝ) ≠ 0 := by positivity
  set d : ℤ := mode p - mode p' with hd
  have hterm : ∀ a : ℕ,
      exp (↑((mode p : ℝ) * (2 * Real.pi * (a : ℝ) / N - OFFSET - s)) * I) *
        (starRingEnd ℂ) (exp (↑((mode p' : ℝ) * (2 * Real.pi * (a : ℝ) / N - OFFSET - s)) * I))
      = exp (↑(-(d : ℝ) * (OFFSET + s)) * I) *
          exp (↑(2 * Real.pi * (d : ℝ) / N) * I) ^ a := by
    intro a
    rw [conj_exp_ofReal_mul_I, ← exp_add, ← exp_nat_mul, ← exp_add]
    congr 1
    rw [hd]
    push_cast
    ring
  simp only [hterm, ← Finset.mul_sum]
  split_ifs with hpp
  · subst hpp
    have : d = 0 := by rw [hd]; ring
    simp [this]
  · have hd0 : d ≠ 0 := fun h => hpp (mode_injective (by omega))
    have hζ1 : exp (↑(2 * Real.pi * (d : ℝ) / N) * I) ≠ 1 := by
      intro h
      obtain ⟨n, hn⟩ := exp_eq_one_iff.mp h
      have hr : 2 * Real.pi * (d : ℝ) / N = n * (2 * Real.pi) := by
        have := congrArg Complex.im hn
        simpa using this
      have hdn : (d : ℝ) = n * N := by
        field_simp at hr
        nlinarith [Real.pi_pos]
      have hdn' : d = n * N := by exact_mod_cast hdn
      obtain ⟨hb1, hb2⟩ := mode_diff_bound p p'
      rw [← hd] at hb1 hb2
      rcases lt_trichotomy n 0 with h | h | h
      · nlinarith
      · exact hd0 (by rw [hdn', h]; ring)
      · nlinarith
    have hζN : exp (↑(2 * Real.pi * (d : ℝ) / N) * I) ^ N = 1 := by
      have hNc : (N : ℂ) ≠ 0 := by exact_mod_cast (show N ≠ 0 by omega)
      rw [← exp_nat_mul]
      rw [show ((N : ℕ) : ℂ) * (↑(2 * Real.pi * (d : ℝ) / N) * I) = (d : ℂ) * (2 * Real.pi * I) by
        push_cast; field_simp [hNc]; ring]
      exact exp_int_mul_two_pi_mul_I d
    rw [geom_sum_eq hζ1, hζN]
    simp


theorem normSq_sum_mul (X : Fin 4 → ℂ) (E : Fin 4 → ℂ) :
    ((Complex.normSq (∑ p, X p * E p) : ℝ) : ℂ)
      = ∑ p, ∑ p', X p * (starRingEnd ℂ) (X p') * (E p * (starRingEnd ℂ) (E p')) := by
  rw [← mul_conj, map_sum, Finset.sum_mul_sum]
  refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun p' _ => ?_
  rw [map_mul]
  ring

theorem parseval_of_orthogonal (N : ℕ) (E : Fin 4 → ℕ → ℂ)
    (hE : ∀ p p', ∑ a in Finset.range N, E p a * (starRingEnd ℂ) (E p' a)
      = if p = p' then (N : ℂ) else 0) (X : Fin 4 → ℂ) :
    ∑ a in Finset.range N, Complex.normSq (∑ p, X p * E p a)
      = N * ∑ p, Complex.normSq (X p) := by
  apply Complex.ofReal_injective
  push_cast
  simp only [normSq_sum_mul]
  rw [Finset.sum_comm]
  calc _ = ∑ p : Fin 4, ∑ p' : Fin 4, X p * (starRingEnd ℂ) (X p') *
        ∑ a in Finset.range N, E p a * (starRingEnd ℂ) (E p' a) := by
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun p' _ => ?_
        rw [Finset.mul_sum]
    _ = _ := by
        simp only [hE, mul_ite, mul_zero, Finset.sum_ite_eq, Finset.mem_univ, if_true,
          Finset.mul_sum]
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [← mul_conj]
        ring

theorem parseval_line (N : ℕ) (hN : 4 ≤ N) (s : ℝ) (X : Fin 4 → ℂ) :
    ∑ a in Finset.range N, Complex.normSq (∑ p, X p *
        exp (↑((mode p : ℝ) * (2 * Real.pi * (a : ℝ) / N - OFFSET - s)) * I))
      = N * ∑ p, Complex.normSq (X p) := by
  have hc := char_sum N hN s
  have h := parseval_of_orthogonal N
    (fun p a => exp (↑((mode p : ℝ) * (2 * Real.pi * (a : ℝ) / N - OFFSET - s)) * I))
    (by intro p p'; simp only []; exact hc p p') X
  exact h


theorem sum_grid (f : Fin 16 → ℂ) : ∑ i : Fin 4, ∑ j : Fin 4, f (idx j i) = ∑ q, f q :=
  calc _ = ∑ x : Fin 4 × Fin 4, f (idx x.2 x.1) :=
        (Fintype.sum_prod_type' (f := fun (i : Fin 4) (j : Fin 4) => f (idx j i))).symm
    _ = _ := Fintype.sum_bijective (fun x : Fin 4 × Fin 4 => idx x.2 x.1) (by decide) _ _
        (fun _ => rfl)

theorem sum_scatter (ψ : Vertex → ℂ) (h : ℂ → ℂ) :
    ∑ q, h (scatter ψ q) = ∑ k, h (ψ k) :=
  (Fintype.sum_bijective cell cell_bijective (fun k => h (ψ k)) (fun q => h (scatter ψ q))
    (fun k => by show h (ψ k) = h (scatter ψ (cell k)); rw [scatter_cell])).symm

theorem conj_coeffAt (gr : Fin 16 → ℂ) (m n : Fin 4) :
    (starRingEnd ℂ) (coeffAt gr m n)
      = (∑ i : Fin 4, ∑ j : Fin 4,
          (starRingEnd ℂ) (gr (idx j i)) * I ^ ((m : ℤ) * i + (n : ℤ) * j)) / 16 := by
  unfold coeffAt
  simp only [cis_page, map_div₀, map_sum, map_mul, conj_I_zpow, neg_neg, map_ofNat]

theorem parseval_coeffs (ψ : Vertex → ℂ) :
    (∑ m : Fin 4, ∑ n : Fin 4, Complex.normSq (coeffAt (scatter ψ) m n)) * 16
      = ∑ k, Complex.normSq (ψ k) := by
  have hcc : ∀ m n : Fin 4,
      coeffAt (scatter ψ) m n * (starRingEnd ℂ) (coeffAt (scatter ψ) m n)
        = ∑ i : Fin 4, ∑ j : Fin 4, (starRingEnd ℂ) (scatter ψ (idx j i)) *
            (coeffAt (scatter ψ) m n * I ^ ((m : ℤ) * i + (n : ℤ) * j)) / 16 := by
    intro m n
    rw [conj_coeffAt, Finset.sum_div, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.sum_div, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  have hinv : ∀ i j : Fin 4,
      ∑ m : Fin 4, ∑ n : Fin 4, coeffAt (scatter ψ) m n * I ^ ((m : ℤ) * i + (n : ℤ) * j)
        = scatter ψ (idx j i) := by
    intro i j
    simp only [coeffAt, cis_page]
    exact idft (fun i j => scatter ψ (idx j i)) i j
  have hgrid := sum_grid (fun q => (starRingEnd ℂ) (scatter ψ q) * scatter ψ q)
  have hscat := sum_scatter ψ (fun z => (starRingEnd ℂ) z * z)
  simp only [] at hgrid hscat
  apply Complex.ofReal_injective
  push_cast
  simp only [← mul_conj, hcc]
  rw [sum_swap4]
  simp only [← Finset.sum_div, ← Finset.mul_sum, hinv]
  rw [div_mul_cancel₀ _ (by norm_num), hgrid, hscat]
  exact Finset.sum_congr rfl fun k _ => mul_comm _ _


theorem evaluator_split (c : Fin 16 → ℂ) (xw yz θ φ : ℝ) :
    evaluator c xw yz θ φ
      = ∑ p : Fin 4, (∑ q : Fin 4, c (idx q p) *
            exp (↑((mode q : ℝ) * (φ - OFFSET - yz)) * I)) *
          exp (↑((mode p : ℝ) * (θ - OFFSET - xw)) * I) := by
  rw [evaluator_exp]
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [ofReal_add, add_mul, exp_add]
  ring

theorem parseval_grid (ψ : Vertex → ℂ) (xw yz : ℝ) (NT NP : ℕ) (hT : 4 ≤ NT) (hP : 4 ≤ NP) :
    (∑ a in Finset.range NT, ∑ b in Finset.range NP,
        Complex.normSq (evaluator (coeffs (scatter ψ)) xw yz
          (2 * Real.pi * (a : ℝ) / NT) (2 * Real.pi * (b : ℝ) / NP)))
        / (NT * NP) * 16
      = ∑ k, Complex.normSq (ψ k) := by
  have hsum : ∑ a in Finset.range NT, ∑ b in Finset.range NP,
        Complex.normSq (evaluator (coeffs (scatter ψ)) xw yz
          (2 * Real.pi * (a : ℝ) / NT) (2 * Real.pi * (b : ℝ) / NP))
      = NT * NP * ∑ m : Fin 4, ∑ n : Fin 4, Complex.normSq (coeffAt (scatter ψ) m n) := by
    simp only [evaluator_split, coeffs_idx]
    rw [Finset.sum_comm]
    simp only [parseval_line NT hT xw]
    rw [← Finset.mul_sum, Finset.sum_comm]
    simp only [parseval_line NP hP yz]
    rw [← Finset.mul_sum, mul_assoc]
  rw [hsum, ← parseval_coeffs ψ]
  field_simp


/-! ### What the channels read off the interpolant -/

theorem sum_MODES_real (f : ℤ → ℝ) : (MODES.map f).sum = ∑ p : Fin 4, f (mode p) := by
  simp [mode, MODES, Fin.sum_univ_four, add_assoc]

/-- `multiRealEvaluator`, one channel: the same pairing, `MODES` and offsets,
accumulating `c.cr[idx] * er - c.ci[idx] * ei`. -/
noncomputable def realEvaluator (c : Fin 16 → ℂ) (xw yz : ℝ) (θ φ : ℝ) : ℝ :=
  let u := θ - OFFSET - xw
  let v := φ - OFFSET - yz
  (MODES.map fun fm => (MODES.map fun fn =>
      (c (idx (MI fn) (MI fm))).re *
          (Real.cos (fm * u) * Real.cos (fn * v) - Real.sin (fm * u) * Real.sin (fn * v)) -
        (c (idx (MI fn) (MI fm))).im *
          (Real.cos (fm * u) * Real.sin (fn * v) + Real.sin (fm * u) * Real.cos (fn * v))
    ).sum).sum

theorem realEvaluator_eq_re (c : Fin 16 → ℂ) (xw yz θ φ : ℝ) :
    realEvaluator c xw yz θ φ = (evaluator c xw yz θ φ).re := by
  unfold realEvaluator evaluator
  simp only [sum_MODES, sum_MODES_real, Complex.re_sum, mul_re, add_re, add_im, mul_im,
    ofReal_re, ofReal_im, I_re, I_im, mul_zero, mul_one, sub_zero, zero_add, add_zero]

/-- A real vertex quantity carried through `multiRealEvaluator` (each Wheeler
channel) takes its own value at every vertex node. -/
theorem real_node_reproduction (vals : Vertex → ℝ) (xw yz : ℝ) (k : Vertex) :
    realEvaluator (coeffs (scatter fun k => (vals k : ℂ))) xw yz
      (tessVertexToTorus xw yz 0 k).1 (tessVertexToTorus xw yz 0 k).2 = vals k := by
  rw [realEvaluator_eq_re, node_reproduction]
  simp

/-- The phase channel's `Math.atan2(s.im, s.re)`, at a vertex node, is the
argument of that vertex's own amplitude. -/
theorem phase_at_node (ψ : Vertex → ℂ) (xw yz : ℝ) (k : Vertex) :
    atan2 (evaluator (coeffs (scatter ψ)) xw yz
            (tessVertexToTorus xw yz 0 k).1 (tessVertexToTorus xw yz 0 k).2).im
          (evaluator (coeffs (scatter ψ)) xw yz
            (tessVertexToTorus xw yz 0 k).1 (tessVertexToTorus xw yz 0 k).2).re
      = Complex.arg (ψ k) := by
  rw [node_reproduction]
  rfl

end Interp
end SutraWS
