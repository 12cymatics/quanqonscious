import SutraWS.Interpolation
import SutraWS.Magnetism
import Mathlib.Tactic

/-!
# The magnetism engine drawn on the torus

The page lifts the sixteen vertices onto the torus it draws with the Gray
pairing (`Interpolation.lean`): vertex `k` sits at ring position
`(ringTheta k, ringPhi k)` of a 4×4 lattice, and every 4-cube edge is a
nearest-neighbour edge of that lattice (`cube_edge_iff_torus_edge`). The
magnetism memory lives on the same cells: `A` on the 32 edges, `B = d¹A` on the
24 faces. This file says where each of those cells is on the torus, which is
what the page needs to draw them there.

* `edge_step` — an edge along axis 0 or 3 is one quarter turn of `θ` and none
  of `φ`; along axis 1 or 2, one quarter turn of `φ` and none of `θ`. Its
  direction, `±1`, is `edgeDir`.
* `hole_loop` — a face spanning axes 0 and 3 is not a cell of the surface: its
  Stokes boundary visits `θ = 0, 1, 2, 3` in order at one `φ`, once round the
  torus the long way. Its flux is the flux through the hole.
* `tube_loop` — a face spanning axes 1 and 2 goes once round the tube, `φ = 0,
  1, 2, 3` at one `θ`. Its flux is the flux along the tube.
* `cell_corners`, `cells_tile` — every other face is one unit cell of the
  lattice, and the sixteen of them are the sixteen cells, once each.
* `cell_orient_area` — `cellOrient` is the signed area of a cell's Stokes loop
  in `(θ, φ)`: `+1` counter-clockwise, `−1` clockwise. A cell's flux times its
  orientation is its circulation in one frame for every cell.
* `cube_strip` — each of the 8 cubes is a strip of the torus: two loops one
  quarter turn apart and the four cells between them.
* `hole_flux_step_y`, `hole_flux_step_z`, `tube_flux_step_x`,
  `tube_flux_step_w` — so (from `d² ∘ d¹ = 0`) the flux through the hole
  changes from one loop to the next by exactly the net flux through the four
  cells between them, and likewise along the tube.
* `node_grid_theta`, `node_grid_phi`, `cell_centre_theta`, `cell_centre_phi` —
  with 200 × 104 surface samples, every vertex is a sample point and every cell
  centre is one, so what is drawn lies on the drawn surface.
* `glyph_mem` — each glyph's size `w₀ + w₁ |x| / m` stays in `[w₀, w₀ + w₁]`.
-/

namespace SutraWS
namespace TorusLattice

open Interp DEC

/-- Bit `a` of `i` is clear: `i` is the base of the edge along `a`. -/
def clear (i : Vertex) (a : Fin 4) : Prop := i.val &&& (1 <<< a.val) = 0

instance (i : Vertex) (a : Fin 4) : Decidable (clear i a) := by unfold clear; infer_instance

/-- Axes 0 and 3 are the `(x, w)` pair `tessVertexToTorus` reads as `θ`. -/
def thetaAxis (a : Fin 4) : Prop := a = 0 ∨ a = 3

instance (a : Fin 4) : Decidable (thetaAxis a) := by unfold thetaAxis; infer_instance

/-- The quarter turns an edge makes along its own ring, as a residue mod 4. -/
def edgeDir (a : Fin 4) (i : Vertex) : Fin 4 :=
  if thetaAxis a then ringTheta (flipA i a) - ringTheta i else ringPhi (flipA i a) - ringPhi i

/-- **`edge_step`**: each of the 32 edges is one quarter turn, forward or back,
along exactly one of the two rings. -/
theorem edge_step (a : Fin 4) (i : Vertex) (h : clear i a) :
    (edgeDir a i = 1 ∨ edgeDir a i = 3) ∧
      (thetaAxis a → ringPhi (flipA i a) = ringPhi i) ∧
      (¬ thetaAxis a → ringTheta (flipA i a) = ringTheta i) := by
  revert a i h; decide

/-- A residue `1` or `3` read as `+1` or `−1`. -/
def toZ (d : Fin 4) : ℤ := if d = 1 then 1 else -1

/-- **`hole_loop`**: the boundary of a `(0,3)` face, in its Stokes order
`i → i^x → i^x^w → i^w`, is `θ = 0, 1, 2, 3` at one `φ`. -/
theorem hole_loop (i : Vertex) (h0 : clear i 0) (h3 : clear i 3) :
    ringTheta i = 0 ∧ ringTheta (flipA i 0) = 1 ∧ ringTheta (flipA (flipA i 0) 3) = 2 ∧
      ringTheta (flipA i 3) = 3 ∧
      ringPhi (flipA i 0) = ringPhi i ∧ ringPhi (flipA (flipA i 0) 3) = ringPhi i ∧
      ringPhi (flipA i 3) = ringPhi i := by
  revert i h0 h3; decide

/-- **`tube_loop`**: the boundary of a `(1,2)` face is `φ = 0, 1, 2, 3` at one `θ`. -/
theorem tube_loop (i : Vertex) (h1 : clear i 1) (h2 : clear i 2) :
    ringPhi i = 0 ∧ ringPhi (flipA i 1) = 1 ∧ ringPhi (flipA (flipA i 1) 2) = 2 ∧
      ringPhi (flipA i 2) = 3 ∧
      ringTheta (flipA i 1) = ringTheta i ∧ ringTheta (flipA (flipA i 1) 2) = ringTheta i ∧
      ringTheta (flipA i 2) = ringTheta i := by
  revert i h1 h2; decide

/-- The four axis pairs whose faces lie on the surface: one `θ` axis and one `φ`
axis, in the kernel's `a < b` order. -/
def cellPair : Fin 4 → Fin 4 × Fin 4 := ![(0, 1), (0, 2), (1, 3), (2, 3)]

/-- The `θ` axis and the `φ` axis of a surface pair. -/
def thAx (a b : Fin 4) : Fin 4 := if thetaAxis a then a else b
def phAx (a b : Fin 4) : Fin 4 := if thetaAxis a then b else a

/-- The lattice cell a surface face covers, named by its corner of least `θ`
and least `φ` (cyclically): the face spans `θ ∈ {r, r+1}`, `φ ∈ {s, s+1}`. -/
def cellOf (a b : Fin 4) (i : Vertex) : Fin 4 × Fin 4 :=
  ( if ringTheta (flipA i (thAx a b)) = ringTheta i + 1 then ringTheta i else ringTheta (flipA i (thAx a b)),
    if ringPhi (flipA i (phAx a b)) = ringPhi i + 1 then ringPhi i else ringPhi (flipA i (phAx a b)) )

/-- The four corners of a face in its Stokes order. -/
def corners (a b : Fin 4) (i : Vertex) : List Vertex :=
  [i, flipA i a, flipA (flipA i a) b, flipA i b]

/-- Lattice position of a vertex. -/
def pos (k : Vertex) : Fin 4 × Fin 4 := (ringTheta k, ringPhi k)

/-- **`cell_corners`**: a surface face's four corners are the four corners of
its lattice cell, each once. -/
theorem cell_corners (p : Fin 4) (i : Vertex) (ha : clear i (cellPair p).1) (hb : clear i (cellPair p).2) :
    ((corners (cellPair p).1 (cellPair p).2 i).map pos).Nodup ∧
      ∀ k ∈ corners (cellPair p).1 (cellPair p).2 i,
        (pos k).1 ∈ [(cellOf (cellPair p).1 (cellPair p).2 i).1, (cellOf (cellPair p).1 (cellPair p).2 i).1 + 1] ∧
        (pos k).2 ∈ [(cellOf (cellPair p).1 (cellPair p).2 i).2, (cellOf (cellPair p).1 (cellPair p).2 i).2 + 1] := by
  revert p i ha hb; decide

/-- Each surface pair has four faces: a base vertex for each setting of the
other two bits. -/
theorem cell_count (p : Fin 4) :
    ((List.finRange 16).filter fun i => clear i (cellPair p).1 ∧ clear i (cellPair p).2).length = 4 := by
  revert p; decide

/-- **`cells_tile`**: every one of the sixteen lattice cells is the cell of a
surface face. With `cell_count` (four pairs, four faces each) that is sixteen
faces onto sixteen cells: each cell exactly once. -/
theorem cells_tile (r s : Fin 4) : ∃ p : Fin 4, ∃ i : Vertex,
    clear i (cellPair p).1 ∧ clear i (cellPair p).2 ∧ cellOf (cellPair p).1 (cellPair p).2 i = (r, s) := by
  revert r s; decide

/-- The Stokes loop of a cell as unwrapped integer steps: first along `a`,
then along `b`, then back. -/
def stepVec (a : Fin 4) (i : Vertex) : ℤ × ℤ :=
  if thetaAxis a then (toZ (edgeDir a i), 0) else (0, toZ (edgeDir a i))

/-- Signed area of the loop `0 → s₁ → s₁ + s₂ → s₂ → 0` (shoelace). -/
def loopArea (s₁ s₂ : ℤ × ℤ) : ℤ := s₁.1 * s₂.2 - s₁.2 * s₂.1

/-- The orientation the kernel tabulates: the product of the two steps, negated
when the loop starts along `φ`. -/
def cellOrient (a b : Fin 4) (i : Vertex) : ℤ :=
  if thetaAxis a then toZ (edgeDir a i) * toZ (edgeDir b (flipA i a))
  else -(toZ (edgeDir a i) * toZ (edgeDir b (flipA i a)))

/-- **`cell_orient_area`**: `cellOrient` is the signed area of the cell's Stokes
loop in `(θ, φ)`, and it is `±1`: positive flux through a cell with orientation
`+1` circulates counter-clockwise in `(θ, φ)`, with `−1` clockwise. -/
theorem cell_orient_area (p : Fin 4) (i : Vertex) (ha : clear i (cellPair p).1) (hb : clear i (cellPair p).2) :
    cellOrient (cellPair p).1 (cellPair p).2 i =
        loopArea (stepVec (cellPair p).1 i) (stepVec (cellPair p).2 (flipA i (cellPair p).1)) ∧
      (cellOrient (cellPair p).1 (cellPair p).2 i = 1 ∨ cellOrient (cellPair p).1 (cellPair p).2 i = -1) := by
  revert p i ha hb; decide

/-- The four axis triples of the cubes, in the kernel's order. -/
def cubeAx : Fin 4 → Fin 4 × Fin 4 × Fin 4 := ![(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

/-- The six faces of the cube `(a, b, q)` at base `i`, the six `DEC.d2` sums over. -/
def cubeFaces (a b q : Fin 4) (i : Vertex) : List (Fin 4 × Fin 4 × Vertex) :=
  [(a, b, i), (a, b, flipA i q), (a, q, i), (a, q, flipA i b), (b, q, i), (b, q, flipA i a)]

/-- A face that goes round the torus: through the hole `(0,3)` or along the tube `(1,2)`. -/
def isHole (f : Fin 4 × Fin 4 × Vertex) : Bool := f.1 = 0 && f.2.1 = 3
def isTube (f : Fin 4 × Fin 4 × Vertex) : Bool := f.1 = 1 && f.2.1 = 2
def isLoop (f : Fin 4 × Fin 4 × Vertex) : Bool := isHole f || isTube f

/-- The ring position a loop sits at: `φ` for a hole loop (`hole_loop`), `θ`
for a tube loop (`tube_loop`). -/
def ringOf (f : Fin 4 × Fin 4 × Vertex) : Fin 4 :=
  if isHole f then ringPhi f.2.2 else ringTheta f.2.2

/-- Of two ring positions a quarter turn apart, the one the other follows. -/
def lower (s t : Fin 4) : Fin 4 := if t = s + 1 then s else t

/-- Four ring positions that are all four. -/
def covers4 (l : List (Fin 4)) : Bool := l.length == 4 && [0, 1, 2, 3].all fun x => l.contains x

/-- A cube is a strip of the torus: exactly two of its faces are loops, of the
same kind and one quarter turn apart, and the other four are the four cells
between those loops, one at each position round the other ring. -/
def isStrip (a b q : Fin 4) (i : Vertex) : Bool :=
  match (cubeFaces a b q i).filter isLoop, (cubeFaces a b q i).filter (fun f => !isLoop f) with
  | [f, g], C =>
      (isHole f == isHole g) && (ringOf g == ringOf f + 1 || ringOf f == ringOf g + 1) &&
      (if isHole f then
        covers4 (C.map fun h => (cellOf h.1 h.2.1 h.2.2).1) &&
          C.all fun h => (cellOf h.1 h.2.1 h.2.2).2 == lower (ringOf f) (ringOf g)
      else
        covers4 (C.map fun h => (cellOf h.1 h.2.1 h.2.2).2) &&
          C.all fun h => (cellOf h.1 h.2.1 h.2.2).1 == lower (ringOf f) (ringOf g))
  | _, _ => false

/-- **`cube_strip`**: each of the 8 cubes (two per axis triple, one for each
setting of the remaining bit) is a strip: the four whose axes include `0` and
`3` run round the torus between two loops through the hole, the four whose
axes include `1` and `2` run round the tube between two loops along it. -/
theorem cube_strip (t : Fin 4) (i : Vertex) (ha : clear i (cubeAx t).1) (hb : clear i (cubeAx t).2.1)
    (hq : clear i (cubeAx t).2.2) : isStrip (cubeAx t).1 (cubeAx t).2.1 (cubeAx t).2.2 i = true := by
  revert t i ha hb hq; decide

/-- **`hole_flux_step_y`**: across axis 1, the flux through the hole changes by
the net flux of the four cells between the two loops (`d² ∘ d¹ = 0` on cube
`(0,1,3)`). -/
theorem hole_flux_step_y (A : Fin 4 → Vertex → ℚ) (i : Vertex) :
    d1 A 0 3 (flipA i 1) - d1 A 0 3 i =
      (d1 A 0 1 (flipA i 3) - d1 A 0 1 i) + (d1 A 1 3 (flipA i 0) - d1 A 1 3 i) := by
  have h := d2_d1_zero A 0 1 3 i
  simp only [d2] at h
  linear_combination -h

/-- **`hole_flux_step_z`**: the same across axis 2 (cube `(0,2,3)`). -/
theorem hole_flux_step_z (A : Fin 4 → Vertex → ℚ) (i : Vertex) :
    d1 A 0 3 (flipA i 2) - d1 A 0 3 i =
      (d1 A 0 2 (flipA i 3) - d1 A 0 2 i) + (d1 A 2 3 (flipA i 0) - d1 A 2 3 i) := by
  have h := d2_d1_zero A 0 2 3 i
  simp only [d2] at h
  linear_combination -h

/-- **`tube_flux_step_x`**: across axis 0, the flux along the tube changes by
the net flux of the four cells between the two loops (cube `(0,1,2)`). -/
theorem tube_flux_step_x (A : Fin 4 → Vertex → ℚ) (i : Vertex) :
    d1 A 1 2 (flipA i 0) - d1 A 1 2 i =
      (d1 A 0 1 i - d1 A 0 1 (flipA i 2)) + (d1 A 0 2 (flipA i 1) - d1 A 0 2 i) := by
  have h := d2_d1_zero A 0 1 2 i
  simp only [d2] at h
  linear_combination h

/-- **`tube_flux_step_w`**: the same across axis 3 (cube `(1,2,3)`). -/
theorem tube_flux_step_w (A : Fin 4 → Vertex → ℚ) (i : Vertex) :
    d1 A 1 2 (flipA i 3) - d1 A 1 2 i =
      (d1 A 1 3 (flipA i 2) - d1 A 1 3 i) + (d1 A 2 3 i - d1 A 2 3 (flipA i 1)) := by
  have h := d2_d1_zero A 1 2 3 i
  simp only [d2] at h
  linear_combination h

/-! ### On the drawn surface

`TH_SEG = 200`, `PH_SEG = 104`: sample `i` of a ring is at angle `i · 2π/N`. -/

/-- **`node_grid_theta`**: vertex `k`'s `θ` (`node_theta`) is sample `25 + 50·r`. -/
theorem node_grid_theta (r : ℕ) :
    ((25 + 50 * r : ℕ) : ℝ) * (2 * Real.pi / 200) = Real.pi / 4 + (r : ℝ) * (Real.pi / 2) := by
  push_cast; ring

/-- **`node_grid_phi`**: vertex `k`'s `φ` (`node_phi`) is sample `13 + 26·s`. -/
theorem node_grid_phi (s : ℕ) :
    ((13 + 26 * s : ℕ) : ℝ) * (2 * Real.pi / 104) = Real.pi / 4 + (s : ℝ) * (Real.pi / 2) := by
  push_cast; ring

/-- **`cell_centre_theta`**: sample `50·(r+1)` is midway between nodes `r` and `r+1`. -/
theorem cell_centre_theta (r : ℕ) :
    ((50 * (r + 1) : ℕ) : ℝ) * (2 * Real.pi / 200) =
      ((Real.pi / 4 + (r : ℝ) * (Real.pi / 2)) + (Real.pi / 4 + ((r : ℝ) + 1) * (Real.pi / 2))) / 2 := by
  push_cast; ring

/-- **`cell_centre_phi`**: sample `26·(s+1)` is midway between nodes `s` and `s+1`. -/
theorem cell_centre_phi (s : ℕ) :
    ((26 * (s + 1) : ℕ) : ℝ) * (2 * Real.pi / 104) =
      ((Real.pi / 4 + (s : ℝ) * (Real.pi / 2)) + (Real.pi / 4 + ((s : ℝ) + 1) * (Real.pi / 2))) / 2 := by
  push_cast; ring

/-- **`glyph_mem`**: a glyph sized `w₀ + w₁ |x| / m`, with `m` the largest `|x|`
among what is drawn, stays in `[w₀, w₀ + w₁]`; `|x| / m` itself is in `[0, 1]`. -/
theorem glyph_mem {x m w₀ w₁ : ℝ} (hm : 0 < m) (hx : |x| ≤ m) (hw : 0 ≤ w₁) :
    0 ≤ |x| / m ∧ |x| / m ≤ 1 ∧ w₀ ≤ w₀ + w₁ * (|x| / m) ∧ w₀ + w₁ * (|x| / m) ≤ w₀ + w₁ := by
  have h0 : 0 ≤ |x| / m := div_nonneg (abs_nonneg x) hm.le
  have h1 : |x| / m ≤ 1 := (div_le_one hm).mpr hx
  refine ⟨h0, h1, ?_, ?_⟩
  · nlinarith
  · nlinarith

end TorusLattice
end SutraWS
