import Mathlib.Tactic

/-!
# The kernel's operators do not increase energy

A simulator step in `vedic_v18.51.1_exact_phi.html` is a chain of operators on
the sixteen complex vertex amplitudes, each evaluated exactly and its result
stored. `ErrorBudget.lean` shows that storage errors add without growing
through non-expansive maps. This file proves, for each kind of operator the
kernel uses, that it does not increase the energy `Σ |ψ_v|²`; for a linear map
that is non-expansiveness, since `‖L x − L y‖ = ‖L (x − y)‖`. A translation
(the ZPE source term) is an isometry outright.

* `perm_energy` — a permutation of the vertices (Nikhilam's complement swap,
  the axis-bit rotations);
* `phase_energy` — a unit phase per vertex;
* `gate_energy_le` — a real factor in `[0, 1]` per vertex (the R4 gate);
* `pair_rotation_energy` — a rotation `(c, s)`, `c² + s² = 1`, of a vertex
  pair (the Cayley pair rotations);
* `butterfly_energy` — the parallelogram law behind the transform below;
* `wht_energy` — the normalised 16-point Walsh–Hadamard transform
  `(1/4) Σ_v (−1)^{popcount(u ∧ v)} ψ_v`, whose sign table is orthogonal
  (`wsign_orth`, decided);
* `mean_contract_energy`, `mean_free_energy` — removing `s ∈ [0, 2]` times
  the mean (Śūnyam at strength `s`), and the mean-free projection;
* `complement_average_energy` — `(1 − h) ψ_v + h ψ_{c(v)}`, `h ∈ [0, 1]`,
  for any permutation `c` (the complement average, `h = strength / 2`).
-/

namespace SutraWS
namespace NonExpansive

open Complex ComplexConjugate

/-- The energy of a state, `Σ |ψ_v|²`. -/
def energy {n : ℕ} (x : Fin n → ℂ) : ℝ := ∑ v, normSq (x v)

theorem energy_nonneg {n : ℕ} (x : Fin n → ℂ) : 0 ≤ energy x :=
  Finset.sum_nonneg fun v _ => normSq_nonneg (x v)

theorem perm_energy {n : ℕ} (σ : Equiv.Perm (Fin n)) (x : Fin n → ℂ) :
    energy (fun v => x (σ v)) = energy x :=
  Equiv.sum_comp σ (fun v => normSq (x v))

theorem phase_energy {n : ℕ} (u x : Fin n → ℂ) (hu : ∀ v, normSq (u v) = 1) :
    energy (fun v => u v * x v) = energy x := by
  unfold energy
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [normSq_mul, hu v, one_mul]

theorem gate_energy_le {n : ℕ} (g : Fin n → ℝ) (x : Fin n → ℂ)
    (hg : ∀ v, 0 ≤ g v ∧ g v ≤ 1) : energy (fun v => (g v : ℂ) * x v) ≤ energy x := by
  unfold energy
  refine Finset.sum_le_sum fun v _ => ?_
  rw [normSq_mul, normSq_ofReal]
  obtain ⟨h0, h1⟩ := hg v
  have hgg : g v * g v ≤ 1 := by nlinarith
  nlinarith [normSq_nonneg (x v)]

theorem pair_rotation_energy (c s : ℝ) (a b : ℂ) (h : c ^ 2 + s ^ 2 = 1) :
    normSq ((c : ℂ) * a + (s : ℂ) * b) + normSq (-(s : ℂ) * a + (c : ℂ) * b) =
      normSq a + normSq b := by
  simp only [normSq_apply, add_re, add_im, mul_re, mul_im, ofReal_re, ofReal_im, neg_re, neg_im]
  linear_combination (a.re ^ 2 + a.im ^ 2 + b.re ^ 2 + b.im ^ 2) * h

theorem butterfly_energy (a b : ℂ) : normSq (a + b) + normSq (a - b) = 2 * (normSq a + normSq b) := by
  simp only [normSq_apply, add_re, add_im, sub_re, sub_im]
  ring

/-- Parity of the number of set bits of `k < 16`. -/
def par4 (k : ℕ) : ℕ := (k % 2 + k / 2 % 2 + k / 4 % 2 + k / 8 % 2) % 2

/-- `(−1)^{popcount(u ∧ v)}`: the kernel's `parityDot(u, v) ? −1 : 1`. -/
def wsignZ (u v : Fin 16) : ℤ := if par4 (u.val &&& v.val) = 1 then -1 else 1

/-- The sign table is orthogonal: `H Hᵀ = 16 I`. Decided over all 256 pairs. -/
theorem wsign_orth : ∀ v w : Fin 16,
    (∑ u : Fin 16, wsignZ u v * wsignZ u w) = if v = w then 16 else 0 := by
  decide

/-- The kernel's `wht`: `out[u] = (1/4) Σ_v (−1)^{popcount(u ∧ v)} ψ_v`. -/
noncomputable def wht16 (x : Fin 16 → ℂ) (u : Fin 16) : ℂ :=
  (1 / 4 : ℂ) * ∑ v, ((wsignZ u v : ℤ) : ℂ) * x v

theorem normSq_eq_conj_mul_re (z : ℂ) : normSq z = (conj z * z).re := by
  rw [← normSq_eq_conj_mul_self]; simp

theorem wht_energy (x : Fin 16 → ℂ) : energy (wht16 x) = energy x := by
  have orth : ∀ v w : Fin 16,
      (∑ u : Fin 16, ((wsignZ u w : ℤ) : ℂ) * ((wsignZ u v : ℤ) : ℂ)) = if v = w then 16 else 0 := by
    intro v w
    have := wsign_orth w v
    have h2 : (∑ u : Fin 16, ((wsignZ u w : ℤ) : ℂ) * ((wsignZ u v : ℤ) : ℂ)) =
        (((∑ u : Fin 16, wsignZ u w * wsignZ u v : ℤ)) : ℂ) := by push_cast; rfl
    rw [h2, this]
    by_cases h : v = w
    · subst h; simp
    · rw [if_neg (Ne.symm h), if_neg h]; simp
  -- work in ℂ: the energy is the real part of Σ conj(y) y
  have key : (∑ u : Fin 16, conj (wht16 x u) * wht16 x u) = ∑ v : Fin 16, conj (x v) * x v := by
    have hc : ∀ u, conj (wht16 x u) = (1 / 4 : ℂ) * ∑ w, ((wsignZ u w : ℤ) : ℂ) * conj (x w) := by
      intro u
      simp only [wht16, map_mul, map_sum, map_div₀, map_one, map_ofNat, map_intCast]
    calc (∑ u : Fin 16, conj (wht16 x u) * wht16 x u)
        = ∑ u : Fin 16, (1 / 16 : ℂ) * ∑ w, ∑ v,
            ((wsignZ u w : ℤ) : ℂ) * ((wsignZ u v : ℤ) : ℂ) * (conj (x w) * x v) := by
          refine Finset.sum_congr rfl fun u _ => ?_
          rw [hc u, wht16, mul_mul_mul_comm, Finset.sum_mul_sum]
          congr 1
          · norm_num
          · refine Finset.sum_congr rfl fun w _ => Finset.sum_congr rfl fun v _ => ?_
            ring
      _ = (1 / 16 : ℂ) * ∑ w, ∑ v,
            (∑ u : Fin 16, ((wsignZ u w : ℤ) : ℂ) * ((wsignZ u v : ℤ) : ℂ)) * (conj (x w) * x v) := by
          rw [← Finset.mul_sum]
          congr 1
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun w _ => ?_
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun v _ => ?_
          rw [Finset.sum_mul]
      _ = (1 / 16 : ℂ) * ∑ w, ∑ v, (if v = w then (16 : ℂ) else 0) * (conj (x w) * x v) := by
          simp only [orth]
      _ = (1 / 16 : ℂ) * ∑ w, 16 * (conj (x w) * x w) := by
          congr 1
          refine Finset.sum_congr rfl fun w _ => ?_
          simp only [ite_mul, zero_mul]
          rw [Finset.sum_ite_eq']
          simp
      _ = ∑ v : Fin 16, conj (x v) * x v := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun v _ => ?_
          ring
  unfold energy
  have hre := congrArg Complex.re key
  simp only [re_sum] at hre
  simpa [normSq_eq_conj_mul_re] using hre

theorem mean_contract_energy {n : ℕ} (hn : 0 < n) (s : ℝ) (hs0 : 0 ≤ s) (hs2 : s ≤ 2)
    (x : Fin n → ℂ) :
    energy (fun v => x v - (s : ℂ) * ((∑ w, x w) / n)) ≤ energy x := by
  set m : ℂ := (∑ w, x w) / n with hm
  have hnC : (n : ℂ) ≠ 0 := by exact_mod_cast hn.ne'
  have hS : (∑ w, x w) = (n : ℂ) * m := by rw [hm]; field_simp
  -- |x_v - s m|^2 = |x_v|^2 + s^2 |m|^2 - 2 s Re(x_v conj m)
  have term : ∀ v, normSq (x v - (s : ℂ) * m) =
      normSq (x v) + s ^ 2 * normSq m - 2 * s * (x v * conj m).re := by
    intro v
    rw [normSq_sub, normSq_mul, normSq_ofReal]
    simp only [map_mul, conj_ofReal]
    have : (x v * ((s : ℂ) * conj m)).re = s * (x v * conj m).re := by
      rw [show x v * ((s : ℂ) * conj m) = (s : ℂ) * (x v * conj m) by ring, re_ofReal_mul]
    rw [this]; ring
  have hsum : (∑ v, (x v * conj m).re) = n * normSq m := by
    rw [← re_sum, ← Finset.sum_mul, hS, mul_assoc, mul_conj]
    simp
  unfold energy
  simp only [term, Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum, hsum]
  have hm0 : 0 ≤ normSq m := normSq_nonneg m
  have hn0 : (0 : ℝ) ≤ n := by exact_mod_cast hn.le
  have : 0 ≤ (n : ℝ) * normSq m * (s * (2 - s)) :=
    mul_nonneg (mul_nonneg hn0 hm0) (mul_nonneg hs0 (by linarith))
  nlinarith

theorem mean_free_energy {n : ℕ} (hn : 0 < n) (x : Fin n → ℂ) :
    energy (fun v => x v - (∑ w, x w) / n) ≤ energy x := by
  have := mean_contract_energy hn 1 (by norm_num) (by norm_num) x
  simpa using this

theorem complement_average_energy {n : ℕ} (c : Equiv.Perm (Fin n)) (h : ℝ) (h0 : 0 ≤ h)
    (h1 : h ≤ 1) (x : Fin n → ℂ) :
    energy (fun v => ((1 - h : ℝ) : ℂ) * x v + (h : ℂ) * x (c v)) ≤ energy x := by
  -- convexity: |(1-h) a + h b|^2 = (1-h)|a|^2 + h|b|^2 - h(1-h)|a-b|^2
  have conv : ∀ a b : ℂ, normSq (((1 - h : ℝ) : ℂ) * a + (h : ℂ) * b) ≤
      (1 - h) * normSq a + h * normSq b := by
    intro a b
    have e : (1 - h) * normSq a + h * normSq b - normSq (((1 - h : ℝ) : ℂ) * a + (h : ℂ) * b) =
        h * (1 - h) * normSq (a - b) := by
      simp only [normSq_apply, add_re, add_im, mul_re, mul_im, ofReal_re, ofReal_im, sub_re, sub_im]
      ring
    have : 0 ≤ h * (1 - h) * normSq (a - b) :=
      mul_nonneg (mul_nonneg h0 (by linarith)) (normSq_nonneg _)
    linarith
  unfold energy
  calc (∑ v, normSq (((1 - h : ℝ) : ℂ) * x v + (h : ℂ) * x (c v)))
      ≤ ∑ v, ((1 - h) * normSq (x v) + h * normSq (x (c v))) :=
        Finset.sum_le_sum fun v _ => conv _ _
    _ = (1 - h) * (∑ v, normSq (x v)) + h * (∑ v, normSq (x (c v))) := by
        rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]
    _ = ∑ v, normSq (x v) := by
        rw [Equiv.sum_comp c (fun v => normSq (x v))]; ring

end NonExpansive
end SutraWS
