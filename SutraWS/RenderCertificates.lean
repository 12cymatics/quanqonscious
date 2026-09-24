import SutraWS.Vertex
import Mathlib.Tactic

/-!
# Render certificates

`VISUAL_CERTIFICATE` in `vedic_v18.51.1_exact_phi.html` names, for each drawn
channel, the theorems that channel rests on.  This file carries those theorems.

The statements are transcribed from the arithmetic that places the pixels:

* the `hw_*` stencil facts from `S17_STENCIL` (`simulation:541`),
* the `rho_*` gate facts from `rhoOf` (`simulation:1273`),
* the `tau_*` facts from `_tauMax` (`simulation:1612`),
* the cell counts from `LEAN_PROVED.cellCounts`,
* the class/defect facts from `OP_CLASS_LAW` and `refreshSutraCertificate`
  (`simulation:2109,2140`).

`Rat`'s `DecidableEq` reduces through `Nat.gcd`'s well-founded recursion and the
kernel gives up, so rational goals here are closed by `norm_num`, not `decide`.
-/

namespace SutraWS
namespace Render

/-! ### Amplitude channel — the hw stencil -/

/-- `S17_STENCIL`: the five Hamming-weight coefficients the amplitude channel
reads, in the order the page stores them. -/
def stencil : List ℚ := [9/128, 7/256, 3/128, 7/512, 9/640]

theorem hw_count : stencil.length = 5 := rfl

theorem hw_all_positive : ∀ q ∈ stencil, 0 < q := by
  intro q hq
  simp only [stencil, List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with h | h | h | h | h <;> subst h <;> norm_num

/-- The denominators `S17_STENCIL` stores, in order.  They are carried
separately rather than read off with `Rat.den`, because `Rat`'s normalisation
runs through `Nat.gcd`'s well-founded recursion and the kernel will not reduce
it; `stencil_times_den` ties the two together without it. -/
def stencilDen : List ℕ := [128, 256, 128, 512, 640]

/-- Each entry really is the stored numerator over the stored denominator. -/
theorem stencil_times_den :
    List.zipWith (fun (q : ℚ) (d : ℕ) => q * (d : ℚ)) stencil stencilDen
      = [9, 7, 3, 7, 9] := by
  simp only [stencil, stencilDen, List.zipWith]
  norm_num

/-- Each stencil denominator is a power of two times an odd number. -/
theorem hw_denominators_are_powers_of_two_times_odd :
    ∀ d ∈ stencilDen, ∃ a m, Odd m ∧ d = 2 ^ a * m := by
  intro d hd
  simp only [stencilDen, List.mem_cons, List.not_mem_nil, or_false] at hd
  rcases hd with h | h | h | h | h <;> subst h
  · exact ⟨7, 1, by decide, by norm_num⟩
  · exact ⟨8, 1, by decide, by norm_num⟩
  · exact ⟨7, 1, by decide, by norm_num⟩
  · exact ⟨9, 1, by decide, by norm_num⟩
  · exact ⟨7, 5, by decide, by norm_num⟩

/-! ### R4 gate channel

The pinch the channel draws is `VTX_DISPLAY.r4F`, which `r4ReinforcedOf`
(`simulation:1661`) builds as a product of one factor per axis -- not the
Wheeler radial profile `rho` below, which is a different quantity the page also
computes.  The `holds` predicate for this channel asks that `r4F` be finite and
in `[0,1]`, so that is what is proved here. -/

/-- One axis factor of `r4ReinforcedOf`: `lam^4 / (R^4 + lam^4)`. -/
def r4Factor (lam R : ℚ) : ℚ := lam ^ 4 / (R ^ 4 + lam ^ 4)

/-- `r4ReinforcedOf` over the four axes. -/
def r4Reinforced (lam : Fin 4 → ℚ) (R : ℚ) : ℚ := ∏ k, r4Factor (lam k) R

theorem r4_factor_nonneg (lam R : ℚ) : 0 ≤ r4Factor lam R := by
  unfold r4Factor
  apply div_nonneg <;> positivity

/-- The page throws when `R^4 + lam^4` is exactly zero, which happens only at
`R = 0` and `lam = 0`; that is the hypothesis here. -/
theorem r4_factor_le_one (lam R : ℚ) (h : 0 < R ^ 4 + lam ^ 4) :
    r4Factor lam R ≤ 1 := by
  rw [r4Factor, div_le_one h]
  nlinarith [sq_nonneg (R ^ 2)]

/-- **The R4 reinforcement lies in `[0,1]`** -- which is exactly the predicate
the channel is gated on. -/
theorem r4_reinforced_mem_unit_interval (lam : Fin 4 → ℚ) (R : ℚ)
    (h : ∀ k, 0 < R ^ 4 + (lam k) ^ 4) :
    0 ≤ r4Reinforced lam R ∧ r4Reinforced lam R ≤ 1 := by
  constructor
  · exact Finset.prod_nonneg (fun k _ => r4_factor_nonneg (lam k) R)
  · exact Finset.prod_le_one (fun k _ => r4_factor_nonneg (lam k) R)
      (fun k _ => r4_factor_le_one (lam k) R (h k))

/-! #### Wheeler's radial profile

Kept because the page computes it too (`rhoOf`, `simulation:1273`), but it is
not what the R4 channel draws. -/

/-- `rhoOf(r, eps) = eps^2 / (r^2 + eps^2)`, the R4 tube pinch. -/
def rho (r eps : ℚ) : ℚ := eps ^ 2 / (r ^ 2 + eps ^ 2)

theorem rho_den_pos_of_hw_pos (r eps : ℚ) (h : eps ≠ 0) : 0 < r ^ 2 + eps ^ 2 := by
  have h2 : 0 < eps ^ 2 := by positivity
  nlinarith [sq_nonneg r]

theorem rho_num_nonneg (r eps : ℚ) (h : eps ≠ 0) : 0 ≤ rho r eps := by
  have hd := rho_den_pos_of_hw_pos r eps h
  have hn : 0 ≤ eps ^ 2 := sq_nonneg eps
  exact div_nonneg hn hd.le

theorem rho_num_le_den (r eps : ℚ) (h : eps ≠ 0) : rho r eps ≤ 1 := by
  have hd := rho_den_pos_of_hw_pos r eps h
  rw [rho, div_le_one hd]
  nlinarith [sq_nonneg r]

theorem rho_zero_eps (eps : ℚ) (h : eps ≠ 0) : rho 0 eps = 1 := by
  have h2 : eps ^ 2 ≠ 0 := pow_ne_zero 2 h
  field_simp [rho]

/-- `_tauMax = 3/4 + 1/16 + 1/64 + 1/64 + 1/128`. -/
def tauMax : ℚ := 3/4 + 1/16 + 1/64 + 1/64 + 1/128

def tauBase : ℚ := 3/4

theorem tau_max_num_eq : tauMax = 109/128 := by norm_num [tauMax]

theorem tau_max_below_removed_ceiling : tauMax < 15/16 := by norm_num [tauMax]

theorem tau_base_is_three_quarters : tauBase = 96/128 := by norm_num [tauBase]

/-! ### Cell counts of the 4-cube -/

theorem masks_card_0 : (Finset.univ.filter (fun i : Vertex => hw i = 0)).card = 1 := by decide
theorem masks_card_1 : (Finset.univ.filter (fun i : Vertex => hw i = 1)).card = 4 := by decide
theorem masks_card_2 : (Finset.univ.filter (fun i : Vertex => hw i = 2)).card = 6 := by decide
theorem masks_card_3 : (Finset.univ.filter (fun i : Vertex => hw i = 3)).card = 4 := by decide
theorem masks_card_4 : (Finset.univ.filter (fun i : Vertex => hw i = 4)).card = 1 := by decide

/-- `LEAN_PROVED.cellCounts` is **not** the f-vector of the tesseract, despite
the name the page gives it. It counts `(vertex, k-mask)` pairs, `16 * choose 4 k`,
and since a genuine k-cell has `2^k` vertices it counts each one `2^k` times. -/
def vertexMaskIncidences : List ℕ := [16, 64, 96, 64, 16]

theorem vertex_mask_incidences_eq :
    vertexMaskIncidences = (List.range 5).map (fun k => 16 * Nat.choose 4 k) := by decide

/-- The tesseract's actual f-vector: `choose 4 k * 2^(4-k)` k-cells. -/
def cellCounts : List ℕ := [16, 32, 24, 8, 1]

theorem cellCounts_eq :
    cellCounts = (List.range 5).map (fun k => Nat.choose 4 k * 2 ^ (4 - k)) := by decide

/-- The two tables differ by exactly the `2^k` over-count. -/
theorem incidences_eq_cells_times_two_pow :
    vertexMaskIncidences = (List.range 5).map (fun k => cellCounts[k]! * 2 ^ k) := by decide

/-- The incidence table's alternating sum vanishes -- it is `16 * (1-1)^4`. -/
theorem euler_characteristic_zero :
    ((16 : ℤ) - 64 + 96 - 64 + 16) = 0 := by decide

/-- The solid 4-cube's Euler characteristic is 1, not 0; its boundary's is 0
(`SutraWS.DEC.dec_euler_characteristic_zero`). Neither is the number above. -/
theorem cell_counts_euler_characteristic_one :
    ((16 : ℤ) - 32 + 24 - 8 + 1) = 1 := by decide

/-! ### Per-sutra channel — declared class versus measured behaviour -/

inductive OpClass
  | unitary | isometric | contractive | projective | dissipative | invertibleNonUnitary
  deriving DecidableEq, Repr

/-- `OP_CLASS_LAW[c].preservesNorm`. -/
def preservesNorm : OpClass → Bool
  | .unitary => true | .isometric => true
  | .contractive => false | .projective => false
  | .dissipative => false | .invertibleNonUnitary => false

/-- `OP_CLASS_LAW[c].invertible`. -/
def invertible : OpClass → Bool
  | .unitary => true | .isometric => true
  | .contractive => true | .projective => false
  | .dissipative => false | .invertibleNonUnitary => true

/-- `refreshSutraCertificate`: a sutra is defective when either measured
property disagrees with the class it declares. -/
def isDefect (c : OpClass) (measuredNorm measuredInv : Bool) : Bool :=
  (preservesNorm c != measuredNorm) || (invertible c != measuredInv)

/-- No class in the table preserves norm while reducing rank. -/
theorem norm_preserving_not_reducing (c : OpClass) :
    preservesNorm c = true → invertible c = true := by
  cases c <;> decide

theorem projective_not_invertible : invertible .projective = false := by decide

theorem unitary_measured_unitary_no_defect : isDefect .unitary true true = false := by decide

theorem projective_measured_projective_no_defect :
    isDefect .projective false false = false := by decide

theorem declared_unitary_but_rank_deficient_is_defect :
    isDefect .unitary true false = true := by decide

/-- For a projective operator the norm measurement contributes nothing to the
defect verdict: only the rank measurement can make it defective. -/
theorem projective_failing_unitarity_is_not_defect (measuredInv : Bool) :
    isDefect .projective false measuredInv = (invertible .projective != measuredInv) := by
  cases measuredInv <;> decide

end Render
end SutraWS
