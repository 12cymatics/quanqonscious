import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic

/-!
# The v18 vertex substrate

Exact-ℚ transcription of `const VTX` from the v18 kernel
(`vedic_v18.24_full_kernel.html:5479-5512`, also `simulation v18:3718` and `version18.5:3623`):

```js
const VTX = {
  psi: Array.from({length:16}, () => Q.ZERO),
  signs: Array.from({length:16}, (_, i) => [
    (i & 1) ? 1n : -1n, (i & 2) ? 1n : -1n,
    (i & 4) ? 1n : -1n, (i & 8) ? 1n : -1n
  ]),
  forward() { ψ[i] = Σ_k signs[i][k] · λ[k] },
  inverse() { λ[k] = (Σ_i signs[i][k] · ψ[i]) / 16 },
  hw(i), comp(i) { return i ^ 15 }, neighbors(i) { return [i^1,i^2,i^4,i^8] },
  mean(), normSq()
};
```

The banner calls this "Hadamard transform Ψ[16] ↔ λ[4] on the Z₂⁴ tesseract graph" with
"exact rational arithmetic (BigInt num/den), zero IEEE-754 in the core". The `Q` of the kernel
is modelled here by `ℚ`, which is exactly that.

The sign table is kept over `ℤ` rather than `ℚ` on purpose: the kernel has fast literal support
for `Int`, so the finite sign facts close by `decide`, whereas the same `decide` over `Rat`
reduces through `Nat.gcd`'s well-founded recursion. Signs are cast to `ℚ` only at use sites.
-/

namespace SutraWS

/-- Vertices of the 4-cube Z₂⁴, indexed as the JS array index `i ∈ [0,16)`. -/
abbrev Vertex := Fin 16

/-- The four Hadamard/`λ` axes. -/
abbrev Axis := Fin 4

/-- `λ[4]`: the four axis amplitudes. -/
abbrev Lambda := Axis → ℚ

/-- `Ψ[16]`: the vertex field. -/
abbrev Psi := Vertex → ℚ

/-- The ±1 sign matrix: `signs[i][k] = (i & 2^k) ? 1 : -1`, i.e. bit `k` of the vertex index.
This is the character `χ_k` of Z₂⁴ evaluated at vertex `i`. -/
def sgn (i : Vertex) (k : Axis) : ℤ :=
  if i.val / 2 ^ k.val % 2 = 1 then 1 else -1

/-- `VTX.forward()` — Ψᵢ = Σₖ signs[i][k]·λₖ. -/
def forward (L : Lambda) : Psi := fun i => ∑ k, (sgn i k : ℚ) * L k

/-- `VTX.inverse()` — λₖ = (Σᵢ signs[i][k]·Ψᵢ)/16. -/
def inverse (P : Psi) : Lambda := fun k => (∑ i, (sgn i k : ℚ) * P i) / 16

/-- `VTX.normSq()` — Σᵢ Ψᵢ². -/
def normSq (P : Psi) : ℚ := ∑ i, P i * P i

/-- `VTX.mean()` — (Σᵢ Ψᵢ)/16. -/
def mean (P : Psi) : ℚ := (∑ i, P i) / 16

/-- `VTX.comp(i)` — the antipodal vertex, `i ^ 15`.
Written as `15 - i` because for `i < 16` truncated subtraction from `1111₂` borrows nowhere and
therefore coincides with the bitwise complement; `comp_val` below records that equality. -/
def comp (i : Vertex) : Vertex := ⟨15 - i.val, by omega⟩

/-- `comp` really is the JS `i ^ 15`. -/
theorem comp_val (i : Vertex) : (comp i).val = i.val ^^^ 15 := by decide

/-- Toggle bit `k` of a vertex index. The `% 16` is a no-op (an xor of two values below 16 is
below 16) and is present only to make the `Fin` bound hold definitionally. -/
def flip (i : Vertex) (k : Axis) : Vertex :=
  ⟨(i.val ^^^ 2 ^ k.val) % 16, Nat.mod_lt _ (by norm_num)⟩

/-- `VTX.neighbors(i)` — the four Hamming-distance-1 vertices `[i^1, i^2, i^4, i^8]`. -/
def neighbors (i : Vertex) : List Vertex :=
  [flip i 0, flip i 1, flip i 2, flip i 3]

/-- `VTX.hw(i)` — Hamming weight of the vertex index, over the four bits. -/
def hw (i : Vertex) : ℕ :=
  (if i.val % 2 = 1 then 1 else 0)
    + (if i.val / 2 % 2 = 1 then 1 else 0)
    + (if i.val / 4 % 2 = 1 then 1 else 0)
    + (if i.val / 8 % 2 = 1 then 1 else 0)

end SutraWS
