import Mathlib.Tactic

/-!
# Error budget of a stored step

Every operator and stage of a simulator step is evaluated exactly from stored
values and its result stored (`StoredPrecision.lean`) before the next acts.
This file proves how those storage errors combine, in any normed space — the
state space is ℂ^16 under the real embedding of its coordinates:

* `chain_err_le`: through a chain of non-expansive maps, the computed result is
  within the sum of the per-stage storage errors of the result computed without
  storage;
* `parallel_err_le`: a superposition `x + Σ (f_i x - x)` whose terms were each
  stored is within the sum of their errors;
* `energy_allowance`: if the result computed without storage has energy at most
  `T`, the stored one has energy at most `T + d (T+1) + d^2`, `d` the budget.
  R4's trigger allows exactly this much before it calls a step a singularity;
* `l2_le_l1`: the l² error is at most the l¹ sum the budget accumulates.
-/

namespace SutraWS
namespace ErrorBudget

variable {E : Type*} [NormedAddCommGroup E]

/-- A map that does not increase distances. -/
def NonExpansive (f : E → E) : Prop := ∀ x y, ‖f x - f y‖ ≤ ‖x - y‖

theorem nonExpansive_comp {f g : E → E} (hf : NonExpansive f) (hg : NonExpansive g) :
    NonExpansive (f ∘ g) := fun x y => le_trans (hf (g x) (g y)) (hg x y)

/-- `exact k x` is `x` after the first `k` stages without storage;
`stored k` after the first `k` stages with storage error `e (k+1)` at stage `k+1`. -/
def exactChain (F : ℕ → E → E) (x : E) : ℕ → E
  | 0 => x
  | k + 1 => F k (exactChain F x k)

def storedChain (F : ℕ → E → E) (x : E) (e : ℕ → E) : ℕ → E
  | 0 => x
  | k + 1 => F k (storedChain F x e k) + e k

/-- `chain_err_le`: through non-expansive stages the storage errors add up and
nothing amplifies them. -/
theorem chain_err_le (F : ℕ → E → E) (hF : ∀ k, NonExpansive (F k)) (x : E) (e : ℕ → E) :
    ∀ n, ‖storedChain F x e n - exactChain F x n‖ ≤ ∑ k in Finset.range n, ‖e k‖ := by
  intro n
  induction n with
  | zero => simp [storedChain, exactChain]
  | succ n ih =>
    simp only [storedChain, exactChain, Finset.sum_range_succ]
    calc ‖F n (storedChain F x e n) + e n - F n (exactChain F x n)‖
        = ‖(F n (storedChain F x e n) - F n (exactChain F x n)) + e n‖ := by congr 1; abel
      _ ≤ ‖F n (storedChain F x e n) - F n (exactChain F x n)‖ + ‖e n‖ := norm_add_le _ _
      _ ≤ ‖storedChain F x e n - exactChain F x n‖ + ‖e n‖ := by gcongr; exact hF n _ _
      _ ≤ (∑ k in Finset.range n, ‖e k‖) + ‖e n‖ := by gcongr

/-- `parallel_err_le`: the stored superposition `x + Σ (y_i - x)`, with each
`y_i` within `ε_i` of `f_i x`, is within `Σ ε_i` of the exact one. -/
theorem parallel_err_le {n : ℕ} (x : E) (f y : Fin n → E) (ε : Fin n → ℝ)
    (h : ∀ i, ‖y i - f i‖ ≤ ε i) :
    ‖(x + ∑ i, (y i - x)) - (x + ∑ i, (f i - x))‖ ≤ ∑ i, ε i := by
  have : (x + ∑ i, (y i - x)) - (x + ∑ i, (f i - x)) = ∑ i, (y i - f i) := by
    rw [add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun i _ => by abel)
  rw [this]
  exact le_trans (norm_sum_le _ _) (Finset.sum_le_sum (fun i _ => h i))

/-- `energy_allowance`: storage that moved the state by at most `d` lifts an
energy of at most `T` to at most `T + d (T+1) + d^2`. -/
theorem energy_allowance (x xs : E) (T d : ℝ) (hT : 0 ≤ T) (hd : 0 ≤ d)
    (hx : ‖x‖ ^ 2 ≤ T) (hdist : ‖xs - x‖ ≤ d) : ‖xs‖ ^ 2 ≤ T + d * (T + 1) + d ^ 2 := by
  have h1 : ‖xs‖ ≤ ‖x‖ + d := by
    calc ‖xs‖ = ‖(xs - x) + x‖ := by congr 1; abel
      _ ≤ ‖xs - x‖ + ‖x‖ := norm_add_le _ _
      _ ≤ d + ‖x‖ := by gcongr
      _ = ‖x‖ + d := by ring
  have h2 : ‖x‖ ≤ Real.sqrt T := Real.le_sqrt_of_sq_le hx
  have hs : Real.sqrt T ^ 2 = T := Real.sq_sqrt hT
  have h3 : 2 * Real.sqrt T ≤ T + 1 := by nlinarith [sq_nonneg (Real.sqrt T - 1)]
  have h4 : ‖xs‖ ^ 2 ≤ (Real.sqrt T + d) ^ 2 :=
    pow_le_pow_left (norm_nonneg _) (le_trans h1 (by linarith)) 2
  nlinarith [h4, hs, h3, mul_le_mul_of_nonneg_left h3 hd]

/-- The l² norm of a vector of reals is at most its l¹ norm. -/
theorem l2_le_l1 {n : ℕ} (v : Fin n → ℝ) :
    Real.sqrt (∑ i, v i ^ 2) ≤ ∑ i, |v i| := by
  rw [Real.sqrt_le_left (Finset.sum_nonneg (fun i _ => abs_nonneg _))]
  calc ∑ i, v i ^ 2 = ∑ i, |v i| ^ 2 := by simp [sq_abs]
    _ ≤ (∑ i, |v i|) ^ 2 := by
      rw [sq, Finset.sum_mul_sum]
      calc ∑ i, |v i| ^ 2 = ∑ i, |v i| * |v i| := by simp [sq]
        _ ≤ ∑ i, ∑ j, |v i| * |v j| := Finset.sum_le_sum (fun i _ =>
            Finset.single_le_sum (f := fun j => |v i| * |v j|)
              (fun j _ => mul_nonneg (abs_nonneg _) (abs_nonneg _)) (Finset.mem_univ i))

end ErrorBudget
end SutraWS
