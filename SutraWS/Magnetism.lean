import SutraWS.DECComplex
import Mathlib.Tactic

/-!
# The magnetism memory on the tesseract

v18.24's magnetic engine (§4b), made exact and given memory that feeds back:

* the vector potential lives on the 32 edges and the flux `B = d¹A` on the 24
  faces (`DECComplex.lean`);
* each step `A` keeps `499/500` of itself and gains Faraday induction from the
  change of the field's energy density, `−(1/10) d⁰(Δρ)`, and the TGCR vortex,
  `ω (6/100 or η·3/100) m_e`, steered along each edge by that edge's share of
  the field's energy;
* the field's phases then turn by `dt (∇·A + p_B)`, the potential and the
  magnetic pressure at each vertex.

The runtime keeps the Faraday part as a vertex potential `χ` (`A_F = −c d⁰χ`),
so its zero flux is exact rather than a cancellation of stored values.

* `faraday_flux_free` — Faraday induction adds no flux (`d¹ ∘ d⁰ = 0`);
* `flux_closed` — the flux through the boundary of every cube is zero;
* `helicity_of_gradient` — a potential with no flux has no helicity, so
  helicity counts only the linked part;
* `retention_bound` — with retention `r < 1` and sources at most `s` the
  memory never exceeds `max |A₀| (s / (1 − r))`: it cannot run away;
* `pressure_nonneg` — the magnetic pressure at a vertex is nonnegative;
* `reconnect_iff` — v18.24's reconnection test `cos∠(B_i, B_j) < −0.7` holds
  exactly when `B_i·B_j < 0` and `(B_i·B_j)² > 0.49 |B_i|² |B_j|²`, which is how
  the runtime decides it, without square roots;
* `edge_share_total` — the edge shares `8 (ρ_t + ρ_h) / E` sum to the number of
  edges, so steering moves the vortex's injection, it does not change its total;
* `polarised_nonneg`, `polarised_le` — the dielectric update with the field's
  polarisation `χ ρ / 8` stays nonnegative and bounded.
-/

namespace SutraWS
namespace Magnetism

open DEC Finset

/-- **`faraday_flux_free`**: `d¹` of any multiple of a gradient is zero. -/
theorem faraday_flux_free (c : ℚ) (χ : Vertex → ℚ) (a b : Fin 4) (i : Vertex) :
    d1 (fun a i => -c * d0 χ a i) a b i = 0 := by
  have h := d1_d0_zero χ a b i
  simp only [d1, d0] at h ⊢
  linear_combination (-c) * h

/-- **`flux_closed`**: the flux of `B = d¹A` out of every cube is zero. -/
theorem flux_closed (A : Fin 4 → Vertex → ℚ) (a b q : Fin 4) (i : Vertex) :
    d2 (d1 A) a b q i = 0 :=
  d2_d1_zero A a b q i

/-- Helicity `Σ_e A_e · (1/3) Σ_{faces f ∋ e} B_f`, over any list of edges and,
for each, any list of the faces it averages over. -/
def helicity (edges : List (Fin 4 × Vertex)) (faces : Fin 4 × Vertex → List (Fin 4 × Fin 4 × Vertex))
    (A : Fin 4 → Vertex → ℚ) : ℚ :=
  (edges.map fun e => A e.1 e.2 * ((faces e).map fun f => d1 A f.1 f.2.1 f.2.2).sum / 3).sum

/-- **`helicity_of_gradient`**: a gradient potential carries no helicity. -/
theorem helicity_of_gradient (edges : List (Fin 4 × Vertex))
    (faces : Fin 4 × Vertex → List (Fin 4 × Fin 4 × Vertex)) (f : Vertex → ℚ) :
    helicity edges faces (d0 f) = 0 := by
  unfold helicity
  have hz : ∀ e : Fin 4 × Vertex,
      ((faces e).map fun g => d1 (d0 f) g.1 g.2.1 g.2.2).sum = 0 := by
    intro e
    rw [List.sum_eq_zero]
    intro x hx
    obtain ⟨g, _, rfl⟩ := List.mem_map.mp hx
    exact d1_d0_zero f g.1 g.2.1 g.2.2
  rw [List.sum_eq_zero]
  intro x hx
  obtain ⟨e, _, rfl⟩ := List.mem_map.mp hx
  rw [hz e]; ring

/-- **`retention_bound`**: memory with retention `0 ≤ r < 1` and sources of size
at most `s` stays within `max |x₀| (s / (1 − r))` for ever. -/
theorem retention_bound (x : ℕ → ℝ) (r s : ℝ) (hr0 : 0 ≤ r) (hr1 : r < 1)
    (hstep : ∀ n, |x (n + 1)| ≤ r * |x n| + s) :
    ∀ n, |x n| ≤ max |x 0| (s / (1 - r)) := by
  have h1r : 0 < 1 - r := by linarith
  intro n
  induction n with
  | zero => exact le_max_left _ _
  | succ n ih =>
    set M := max |x 0| (s / (1 - r))
    have hM : s / (1 - r) ≤ M := le_max_right _ _
    have hsM : s ≤ (1 - r) * M := by
      rw [div_le_iff h1r] at hM; linarith
    calc |x (n + 1)| ≤ r * |x n| + s := hstep n
      _ ≤ r * M + s := by gcongr
      _ ≤ M := by linarith

/-- **`pressure_nonneg`**: the mean squared flux of the faces at a vertex. -/
theorem pressure_nonneg (B : Fin 6 → ℝ) : 0 ≤ (∑ p, B p ^ 2) / 6 := by
  apply div_nonneg _ (by norm_num)
  exact Finset.sum_nonneg (fun p _ => sq_nonneg (B p))

/-- **`reconnect_iff`**: the cosine test without square roots. -/
theorem reconnect_iff {dot na nb : ℝ} (ha : 0 < na) (hb : 0 < nb) :
    dot / (Real.sqrt na * Real.sqrt nb) < -(7 / 10) ↔
      dot < 0 ∧ 49 / 100 * (na * nb) < dot ^ 2 := by
  have hsa := Real.sqrt_pos.mpr ha
  have hsb := Real.sqrt_pos.mpr hb
  set s := Real.sqrt na * Real.sqrt nb with hs_def
  have hs : 0 < s := mul_pos hsa hsb
  have hs2 : s ^ 2 = na * nb := by
    rw [hs_def, mul_pow, Real.sq_sqrt ha.le, Real.sq_sqrt hb.le]
  rw [div_lt_iff hs, ← hs2]
  constructor
  · intro h
    have hneg : dot < 0 := by nlinarith
    refine ⟨hneg, ?_⟩
    nlinarith
  · rintro ⟨hneg, hsq⟩
    by_contra hc
    push_neg at hc
    have : dot ^ 2 ≤ (7 / 10 * s) ^ 2 := by nlinarith
    nlinarith

theorem flipA_flipA (i : Vertex) (a : Fin 4) : flipA (flipA i a) a = i := by
  revert i a; decide

theorem clear_flipA (i : Vertex) (a : Fin 4) :
    ((flipA i a).val &&& (1 <<< a.val) = 0) ↔ ¬ (i.val &&& (1 <<< a.val) = 0) := by
  revert i a; decide

/-- Along one axis, the edges' two ends together visit every vertex once. -/
theorem axis_share (ρ : Vertex → ℚ) (a : Fin 4) :
    (∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ i + ρ (flipA i a) else 0) =
      ∑ i : Vertex, ρ i := by
  have h1 : (∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ (flipA i a) else 0) =
      ∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then 0 else ρ i := by
    let e : Vertex ≃ Vertex :=
      ⟨fun i => flipA i a, fun i => flipA i a, fun i => flipA_flipA i a, fun i => flipA_flipA i a⟩
    rw [← e.sum_comp (fun j => if j.val &&& (1 <<< a.val) = 0 then 0 else ρ j)]
    apply Finset.sum_congr rfl
    intro i _
    show (if i.val &&& (1 <<< a.val) = 0 then ρ (flipA i a) else 0) =
      (if (flipA i a).val &&& (1 <<< a.val) = 0 then 0 else ρ (flipA i a))
    by_cases h : i.val &&& (1 <<< a.val) = 0
    · have h' : ¬ ((flipA i a).val &&& (1 <<< a.val) = 0) := fun hc => (clear_flipA i a).mp hc h
      rw [if_pos h, if_neg h']
    · have h' : (flipA i a).val &&& (1 <<< a.val) = 0 := (clear_flipA i a).mpr h
      rw [if_neg h, if_pos h']
  calc (∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ i + ρ (flipA i a) else 0)
      = ∑ i : Vertex, ((if i.val &&& (1 <<< a.val) = 0 then ρ i else 0) +
          (if i.val &&& (1 <<< a.val) = 0 then ρ (flipA i a) else 0)) := by
        apply Finset.sum_congr rfl; intro i _; split_ifs <;> ring
    _ = (∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ i else 0) +
          ∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ (flipA i a) else 0 :=
        Finset.sum_add_distrib
    _ = (∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ i else 0) +
          ∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then 0 else ρ i := by rw [h1]
    _ = ∑ i : Vertex, ρ i := by
        rw [← Finset.sum_add_distrib]; apply Finset.sum_congr rfl; intro i _; split_ifs <;> ring

/-- **`edge_share_total`**: over the 32 edges, `Σ (ρ_t + ρ_h) = 4 Σ_v ρ_v` —
every vertex ends four edges — so the shares `8 (ρ_t + ρ_h) / E` sum to `32`. -/
theorem edge_share_total (ρ : Vertex → ℚ) :
    (∑ a : Fin 4, ∑ i : Vertex, if i.val &&& (1 <<< a.val) = 0 then ρ i + ρ (flipA i a) else 0) =
      4 * ∑ i : Vertex, ρ i := by
  simp only [axis_share, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  norm_num

/-- The dielectric update with polarisation:
`d' = d/2 + pairMean/4 + neighbourMean/8 + planeSource/8 + χ ρ/8`. -/
def polarised (d pairMean nbrMean planeSource χ ρ : ℚ) : ℚ :=
  d / 2 + pairMean / 4 + nbrMean / 8 + planeSource / 8 + χ * ρ / 8

/-- **`polarised_nonneg`**: nonnegative inputs give a nonnegative dielectric. -/
theorem polarised_nonneg {d pm nm ps χ ρ : ℚ} (hd : 0 ≤ d) (hpm : 0 ≤ pm) (hnm : 0 ≤ nm)
    (hps : 0 ≤ ps) (hχ : 0 ≤ χ) (hρ : 0 ≤ ρ) : 0 ≤ polarised d pm nm ps χ ρ := by
  unfold polarised; positivity

/-- **`polarised_le`**: with the dielectric at most `D`, the plane source at most
`P` and the field's energy density at most `R`, the next dielectric is at most
`7D/8 + (P + χR)/8`; so it stays at most `P + χR` once it is. -/
theorem polarised_le {d pm nm ps χ ρ D P R : ℚ} (hd : d ≤ D) (hpm : pm ≤ D) (hnm : nm ≤ D)
    (hps : ps ≤ P) (hχ : 0 ≤ χ) (hρ : ρ ≤ R) :
    polarised d pm nm ps χ ρ ≤ 7 * D / 8 + (P + χ * R) / 8 := by
  unfold polarised
  have : χ * ρ ≤ χ * R := mul_le_mul_of_nonneg_left hρ hχ
  linarith

end Magnetism
end SutraWS
