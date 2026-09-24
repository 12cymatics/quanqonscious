import SutraWS.Vertex
import Mathlib.Tactic

/-!
# The cubical cochain complex of the tesseract boundary

`DEC_LGT_REGGE` in `vedic_v18.51.1_exact_phi.html` builds the cochain complex of
the tesseract — 16 vertices, 32 edges, 24 faces, 8 cubes — and checks `d ∘ d = 0`
at runtime on one sample cochain (`verifyD2zero`, `simulation:3417`).  A sample
is not a proof, so the same two statements are proved here for every cochain.

The coboundaries are transcribed from `applyD0`/`applyD1`/`applyD2`.  A `k`-cell
is named by the axes it spans and the vertex at its base, which is how the page
enumerates them, so an edge is `(a, i)`, a face `(a, b, i)` and a cube
`(a, b, q, i)`.
-/

namespace SutraWS
namespace DEC

/-- Flip bit `a` of a vertex index — the page's `i ^ (1 <<< a)`. -/
def flipA (i : Vertex) (a : Fin 4) : Vertex :=
  ⟨(i.val ^^^ (1 <<< a.val)) % 16, Nat.mod_lt _ (by norm_num)⟩

/-- Flipping two axes commutes, because `xor` does. -/
theorem flipA_comm (i : Vertex) (a b : Fin 4) :
    flipA (flipA i a) b = flipA (flipA i b) a := by
  revert i a b
  decide

/-- `applyD0`: the difference across an edge. -/
def d0 (f : Vertex → ℚ) : Fin 4 → Vertex → ℚ :=
  fun a i => f (flipA i a) - f i

/-- `applyD1`: the circulation around a face. -/
def d1 (ω : Fin 4 → Vertex → ℚ) : Fin 4 → Fin 4 → Vertex → ℚ :=
  fun a b i => ω a i + ω b (flipA i a) - ω a (flipA i b) - ω b i

/-- `applyD2`: the flux through the boundary of a cube, with the page's six
signed faces. -/
def d2 (η : Fin 4 → Fin 4 → Vertex → ℚ) : Fin 4 → Fin 4 → Fin 4 → Vertex → ℚ :=
  fun a b q i =>
    -η a b i + η a b (flipA i q)
      + η a q i - η a q (flipA i b)
      - η b q i + η b q (flipA i a)

/-- **`d1 ∘ d0 = 0`** — every face circulation of a gradient telescopes away. -/
theorem d1_d0_zero (f : Vertex → ℚ) (a b : Fin 4) (i : Vertex) :
    d1 (d0 f) a b i = 0 := by
  simp only [d1, d0]
  rw [flipA_comm i b a]
  ring

/-- **`d2 ∘ d1 = 0`** — every cube flux of a circulation cancels in pairs. -/
theorem d2_d1_zero (ω : Fin 4 → Vertex → ℚ) (a b q : Fin 4) (i : Vertex) :
    d2 (d1 ω) a b q i = 0 := by
  simp only [d2, d1]
  rw [flipA_comm i q a, flipA_comm i q b, flipA_comm i b a]
  ring

/-! ### The dimensions the page reports

`enumerateCubes` stops at the eight 3-dimensional facets, so this is the
*boundary* complex of the tesseract, not the solid 4-cube: the alternating sum
below is the Euler characteristic of that boundary 3-sphere, which is 0.  The
solid 4-cube would carry a ninth cell and sum to 1.  The page's `eulerChar`
(`simulation:3438`) computes the same boundary figure, so this matches what it
reports rather than what "4-cube" might suggest. -/

def dimC0 : ℕ := 16
def dimC1 : ℕ := 32
def dimC2 : ℕ := 24
def dimC3 : ℕ := 8

/-- `cochainDims` — an edge per axis per vertex with that bit clear, and so on. -/
theorem cochain_dims :
    dimC0 = 16 ∧ dimC1 = 4 * 8 ∧ dimC2 = 6 * 4 ∧ dimC3 = 4 * 2 := by
  refine ⟨rfl, rfl, rfl, rfl⟩

/-- `eulerChar` of the boundary complex is zero -- it is chi(S^3). -/
theorem dec_euler_characteristic_zero :
    (dimC0 : ℤ) - dimC1 + dimC2 - dimC3 = 0 := by
  norm_num [dimC0, dimC1, dimC2, dimC3]

end DEC
end SutraWS
