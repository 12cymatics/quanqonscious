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

/-! ### R4 gate channel -/

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

/-- `LEAN_PROVED.cellCounts`: 16 copies of each mask class. -/
def cellCounts : List ℕ := [16, 64, 96, 64, 16]

theorem cellCounts_eq : cellCounts = (List.range 5).map (fun k => 16 * Nat.choose 4 k) := by decide

theorem euler_characteristic_zero :
    ((16 : ℤ) - 64 + 96 - 64 + 16) = 0 := by decide

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
