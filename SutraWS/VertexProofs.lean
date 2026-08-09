import SutraWS.Vertex

/-!
# Invariants of the v18 vertex substrate

The v18 banner (`vedic_v18.24_full_kernel.html:5479`, `simulation v18:232`) advertises the
substrate as a Hadamard duality `Ψ[16] ↔ λ[4]` on Z₂⁴ in exact ℚ, and the runtime harness
`CONTRACTS` (`simulation v18:486-540`) checks energy and complement behaviour numerically, in
floats, every 120 cycles. The theorems below are the exact algebraic facts those float checks
are sampling.

The load-bearing fact is character orthogonality on Z₂⁴, `sgn_orthogonal`: the 16 sign vectors
are pairwise orthogonal with norm² = 16, which is precisely why `VTX.inverse()` divides by 16.

Note what is *not* claimed: `forward ∘ inverse = id` is false, and deliberately not stated —
`forward` maps a 4-dimensional space into a 16-dimensional one, so that composite is a
projection onto its range, not the identity. Only `inverse ∘ forward = id` holds.
-/

namespace SutraWS

/-- Character orthogonality on Z₂⁴: the columns of the sign matrix are orthogonal with norm² 16.
This is what makes the `/16` in `VTX.inverse()` correct. -/
theorem sgn_orthogonal (k j : Axis) :
    (∑ i : Vertex, sgn i k * sgn i j) = if k = j then 16 else 0 := by
  fin_cases k <;> fin_cases j <;> decide

/-- Every non-trivial character sums to zero over the cube. -/
theorem sgn_sum_zero (k : Axis) : (∑ i : Vertex, sgn i k) = 0 := by
  fin_cases k <;> decide

/-- The antipodal vertex flips all four bits, hence negates every sign. -/
theorem sgn_comp (i : Vertex) (k : Axis) : sgn (comp i) k = -sgn i k := by revert i k; decide

/-- Complementing a vertex complements its Hamming weight. -/
theorem hw_comp (i : Vertex) : hw (comp i) = 4 - hw i := by revert i; decide

/-- **Hadamard involution.** `VTX.inverse()` undoes `VTX.forward()` exactly, in ℚ, with no
rounding — the substrate's round-trip guarantee. -/
theorem inverse_forward (L : Lambda) : inverse (forward L) = L := by
  funext k
  fin_cases k <;>
    simp (config := { decide := true }) [inverse, forward, sgn, Fin.sum_univ_succ, Fin.succ] <;>
    ring

/-- **Parseval / energy identity.** The exact form of what `CONTRACTS.testEnergy`
(`simulation v18:512-519`) measures in floats: the vertex energy is 16× the axis energy. -/
theorem parseval (L : Lambda) : normSq (forward L) = 16 * ∑ k, (L k) ^ 2 := by
  simp (config := { decide := true }) [normSq, forward, sgn, Fin.sum_univ_succ]
  ring

/-- **Complement antisymmetry.** Antipodal vertices carry exactly opposite amplitudes. -/
theorem forward_comp (L : Lambda) (i : Vertex) :
    forward L (comp i) = -(forward L i) := by
  fin_cases i <;>
    simp (config := { decide := true }) [forward, comp, sgn, Fin.sum_univ_succ] <;>
    ring

/-- The exact statement behind the float heuristic `CONTRACTS.complementCoherence`
(`simulation v18:492-496`), which counts vertices with `ψ[i]·ψ[i^15] < 0`: that product is
*never* positive, and is negative exactly when the amplitude is non-zero. -/
theorem complement_product_nonpos (L : Lambda) (i : Vertex) :
    forward L (comp i) * forward L i ≤ 0 := by
  rw [forward_comp, neg_mul]
  exact neg_nonpos.mpr (mul_self_nonneg _)

/-- A forward-transformed field always has zero mean: the substrate carries no DC component,
because every character sums to zero over the cube. -/
theorem mean_forward_eq_zero (L : Lambda) : mean (forward L) = 0 := by
  simp (config := { decide := true }) [mean, forward, sgn, Fin.sum_univ_succ]
  ring

end SutraWS
