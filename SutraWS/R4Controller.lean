import Mathlib.Tactic

/-!
# R4 singularity controller

`step` in `vedic_v18.51.1_exact_phi.html` ends with the R4 controller. It is
dormant unless the step is a blow-up: a state it stores rises above
`L = max(2 E_in, C)`, more than double its start and above the seed energy
`C`, past what storage alone can add (`energy_allowance`). Then it recomputes
the step with every strength multiplied by `σ ∈ [0,1]`
(`r4_reinforced_mem_unit_interval`), stops every stored state that rises above
`L` with the R4 gate

    g_v = a_v + b_v ε,   a_v = s4/(s4+q_v^4),   b_v = q_v^4/(s4+q_v^4),

on the dielectric field `q`, averaged across complement pairs, with the largest
`ε` on a `2^-32` grid whose exact gated energy is at most `T = max(E_in, C)`,
and gates the step's end to `T` the same way.

The weights are non-negative and sum to one, so every gate factor and their
mean lie in `[0,1]`; the gated energy is the quadratic the controller bisects,
non-decreasing in `ε ≥ 0`. A dormant step ends at most `L`, a stopped one at
most `T ≤ L`, so a run ends step `n` at most `2^n max(E_0, C)`
(`run_le_pow_two`): the energy can at most double per step, and no step can
take it further.
-/

namespace SutraWS
namespace R4Controller

/-- `gateABField` weight `a` for one vertex, before pair averaging. -/
def gateA (s4 q : ℚ) : ℚ := s4 / (s4 + q ^ 4)

/-- `gateABField` weight `b` for one vertex, before pair averaging. -/
def gateB (s4 q : ℚ) : ℚ := q ^ 4 / (s4 + q ^ 4)

theorem gate_den_pos (s4 q : ℚ) (hs : 0 < s4) : 0 < s4 + q ^ 4 := by positivity

theorem gate_weights_nonneg (s4 q : ℚ) (hs : 0 < s4) : 0 ≤ gateA s4 q ∧ 0 ≤ gateB s4 q :=
  ⟨div_nonneg hs.le (gate_den_pos s4 q hs).le, div_nonneg (by positivity) (gate_den_pos s4 q hs).le⟩

theorem gate_weights_sum (s4 q : ℚ) (hs : 0 < s4) : gateA s4 q + gateB s4 q = 1 := by
  unfold gateA gateB
  rw [div_add_div_same, div_self (gate_den_pos s4 q hs).ne']

/-- Averaging a vertex's weights with its complement's keeps both properties. -/
theorem pair_avg_weights (a1 b1 a2 b2 : ℚ) (h1 : 0 ≤ a1) (h2 : 0 ≤ b1) (h3 : 0 ≤ a2) (h4 : 0 ≤ b2)
    (hs1 : a1 + b1 = 1) (hs2 : a2 + b2 = 1) :
    0 ≤ (a1 + a2) / 2 ∧ 0 ≤ (b1 + b2) / 2 ∧ (a1 + a2) / 2 + (b1 + b2) / 2 = 1 :=
  ⟨by linarith, by linarith, by linarith⟩

theorem gate_factor_mem (a b ε : ℚ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1)
    (h0 : 0 ≤ ε) (h1 : ε ≤ 1) : 0 ≤ a + b * ε ∧ a + b * ε ≤ 1 := by
  constructor
  · have : 0 ≤ b * ε := mul_nonneg hb h0
    linarith
  · nlinarith

/-- The mean of sixteen gate factors in `[0,1]` is in `[0,1]`. -/
theorem mean_gate_mem (g : Fin 16 → ℚ) (h : ∀ i, 0 ≤ g i ∧ g i ≤ 1) :
    0 ≤ (∑ i, g i) / 16 ∧ (∑ i, g i) / 16 ≤ 1 := by
  have h0 : 0 ≤ ∑ i, g i := Finset.sum_nonneg (fun i _ => (h i).1)
  have h1 : ∑ i, g i ≤ ∑ _i : Fin 16, (1 : ℚ) := Finset.sum_le_sum (fun i _ => (h i).2)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at h1
  constructor
  · positivity
  · rw [div_le_one (by norm_num)]
    exact_mod_cast h1

/-- The gated energy, summed over complement pairs with pair energy `P i`, is
the quadratic `A ε² + B ε + C` the controller bisects. -/
theorem gate_energy_poly {n : ℕ} (a b P : Fin n → ℝ) (ε : ℝ) :
    ∑ i, (a i + b i * ε) ^ 2 * P i
      = (∑ i, b i ^ 2 * P i) * ε ^ 2 + (∑ i, 2 * a i * b i * P i) * ε + ∑ i, a i ^ 2 * P i := by
  simp only [Finset.sum_mul, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun i _ => by ring)

theorem gate_poly_coeffs_nonneg {n : ℕ} (a b P : Fin n → ℝ) (ha : ∀ i, 0 ≤ a i)
    (hb : ∀ i, 0 ≤ b i) (hP : ∀ i, 0 ≤ P i) :
    0 ≤ ∑ i, b i ^ 2 * P i ∧ 0 ≤ ∑ i, 2 * a i * b i * P i ∧ 0 ≤ ∑ i, a i ^ 2 * P i := by
  refine ⟨Finset.sum_nonneg (fun i _ => ?_), Finset.sum_nonneg (fun i _ => ?_),
    Finset.sum_nonneg (fun i _ => ?_)⟩
  · exact mul_nonneg (sq_nonneg _) (hP i)
  · exact mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) (ha i)) (hb i)) (hP i)
  · exact mul_nonneg (sq_nonneg _) (hP i)

/-- `gate_energy_mono`: with non-negative coefficients the gated energy does
not decrease as `ε` grows from `0`, which is what makes bisection sound. -/
theorem gate_energy_mono (A B C e1 e2 : ℝ) (hA : 0 ≤ A) (hB : 0 ≤ B) (h0 : 0 ≤ e1)
    (h : e1 ≤ e2) : A * e1 ^ 2 + B * e1 + C ≤ A * e2 ^ 2 + B * e2 + C := by
  have hsq : e1 ^ 2 ≤ e2 ^ 2 := pow_le_pow_left h0 h 2
  nlinarith [mul_le_mul_of_nonneg_left hsq hA, mul_le_mul_of_nonneg_left h hB]

/-- Every grid point at or below an accepted one is accepted too. -/
theorem bisection_below (A B C lo x T : ℝ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hx0 : 0 ≤ x)
    (hx : x ≤ lo) (hlo : A * lo ^ 2 + B * lo + C ≤ T) : A * x ^ 2 + B * x + C ≤ T :=
  le_trans (gate_energy_mono A B C x lo hA hB hx0 hx) hlo

noncomputable def controlled (eU eR eG L T : ℝ) : ℝ := if eU ≤ L then eU else if eR ≤ T then eR else eG

theorem controlled_dormant (eU eR eG L T : ℝ) (h : eU ≤ L) : controlled eU eR eG L T = eU := by
  simp [controlled, h]

theorem controlled_stopped_le (eU eR eG L T : ℝ) (h : ¬ eU ≤ L) (hG : eG ≤ T) :
    controlled eU eR eG L T ≤ T := by
  unfold controlled
  rw [if_neg h]
  split_ifs with h2
  · exact h2
  · exact hG

theorem controlled_le (eU eR eG L T : ℝ) (hG : eG ≤ T) (hTL : T ≤ L) : controlled eU eR eG L T ≤ L := by
  by_cases h : eU ≤ L
  · rw [controlled_dormant eU eR eG L T h]; exact h
  · exact le_trans (controlled_stopped_le eU eR eG L T h hG) hTL

theorem target_ge (eIn C : ℝ) : eIn ≤ max eIn C ∧ C ≤ max eIn C :=
  ⟨le_max_left _ _, le_max_right _ _⟩

theorem target_le_level (eIn C : ℝ) (h : 0 ≤ eIn) : max eIn C ≤ max (2 * eIn) C :=
  max_le_max (by linarith) le_rfl

theorem level_le_double (eIn C : ℝ) (h : 0 ≤ C) : max (2 * eIn) C ≤ 2 * max eIn C := by
  rcases le_total eIn C with hc | hc
  · rw [max_eq_right hc]
    exact max_le (by linarith) (by linarith)
  · rw [max_eq_left hc]
    exact max_le le_rfl (by linarith)

theorem run_le_pow_two (E : ℕ → ℝ) (C : ℝ) (hC : 0 ≤ C)
    (h : ∀ n, E (n + 1) ≤ max (2 * E n) C) : ∀ n, E n ≤ 2 ^ n * max (E 0) C := by
  intro n
  induction n with
  | zero => simp
  | succ k ih =>
    have hM : C ≤ max (E 0) C := le_max_right _ _
    have hp : (1 : ℝ) ≤ 2 ^ k := one_le_pow_of_one_le (by norm_num) k
    calc E (k + 1) ≤ max (2 * E k) C := h k
      _ ≤ max (2 * (2 ^ k * max (E 0) C)) C := max_le_max (by linarith) le_rfl
      _ ≤ 2 ^ (k + 1) * max (E 0) C := by
          apply max_le
          · rw [pow_succ]; linarith
          · rw [pow_succ]; nlinarith

end R4Controller
end SutraWS
