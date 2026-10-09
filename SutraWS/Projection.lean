import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

/-!
# The torus camera

`projectTorus` in `vedic_v18.51.1_exact_phi.html` rotates each surface point by
`camRY` in the `(x, z)` plane and then by `camRX` in the `(y, z)` plane, and
divides by `D + z`. A point with `D + z ≤ 0` is behind the camera: the division
flips or explodes it, and the surface turns to shards. The page now places the
camera at `camDist`, beyond the largest point norm of the frame, and these
statements are why that is enough for every point and every camera angle.
-/

namespace SutraWS
namespace Projection

open Real

/-- The depth `projectTorus` divides by: `z` after the `camRY` turn of `(x, z)`
and the `camRX` turn of `(y, z)`. -/
noncomputable def camDepth (aY aX x y z : ℝ) : ℝ :=
  y * Real.sin aX + (x * Real.sin aY + z * Real.cos aY) * Real.cos aX

/-- Both turns are rotations, so depth never exceeds the point's distance from
the origin. -/
theorem camDepth_sq_le (aY aX x y z : ℝ) :
    camDepth aY aX x y z ^ 2 ≤ x ^ 2 + y ^ 2 + z ^ 2 := by
  unfold camDepth
  have hY := Real.sin_sq_add_cos_sq aY
  have hX := Real.sin_sq_add_cos_sq aX
  have i1 : (x * Real.sin aY + z * Real.cos aY) ^ 2 + (x * Real.cos aY - z * Real.sin aY) ^ 2
      = (x ^ 2 + z ^ 2) * (Real.sin aY ^ 2 + Real.cos aY ^ 2) := by ring
  rw [hY, mul_one] at i1
  set z1 := x * Real.sin aY + z * Real.cos aY
  have i2 : (y * Real.sin aX + z1 * Real.cos aX) ^ 2 + (y * Real.cos aX - z1 * Real.sin aX) ^ 2
      = (y ^ 2 + z1 ^ 2) * (Real.sin aX ^ 2 + Real.cos aX ^ 2) := by ring
  rw [hX, mul_one] at i2
  nlinarith [sq_nonneg (x * Real.cos aY - z * Real.sin aY),
    sq_nonneg (y * Real.cos aX - z1 * Real.sin aX)]

/-- A point strictly inside the camera's distance is in front of it, whatever
the two camera angles. -/
theorem camera_in_front (aY aX x y z D : ℝ) (hD : 0 < D)
    (hn : x ^ 2 + y ^ 2 + z ^ 2 < D ^ 2) : 0 < D + camDepth aY aX x y z := by
  have h := lt_of_le_of_lt (camDepth_sq_le aY aX x y z) hn
  by_contra hc
  push_neg at hc
  nlinarith

/-- The camera distance the page uses: `CAM_DIST`, pushed back in proportion
once the frame's largest point norm exceeds the bare torus `TORUS_R + TORUS_r`,
so the perspective ratio of the bare torus is kept. -/
noncomputable def camDist (base bare maxNorm : ℝ) : ℝ := max base (base * maxNorm / bare)

/-- With `CAM_DIST = 700` beyond `TORUS_R + TORUS_r = 250`, the camera is always
farther than the largest point. -/
theorem camDist_gt (base bare maxNorm : ℝ) (hb : 0 < bare) (hbb : bare < base) :
    maxNorm < camDist base bare maxNorm := by
  unfold camDist
  rcases lt_or_le maxNorm bare with h | h
  · exact lt_of_lt_of_le (lt_trans h hbb) (le_max_left _ _)
  · have : maxNorm < base * maxNorm / bare := by
      rw [lt_div_iff hb]
      nlinarith
    exact lt_of_lt_of_le this (le_max_right _ _)

/-- Every point of the frame is in front of the camera. -/
theorem frame_in_front (aY aX x y z maxNorm : ℝ) (hm : x ^ 2 + y ^ 2 + z ^ 2 ≤ maxNorm ^ 2)
    (hm0 : 0 ≤ maxNorm) :
    0 < camDist 700 250 maxNorm + camDepth aY aX x y z := by
  have hgt := camDist_gt 700 250 maxNorm (by norm_num) (by norm_num)
  have hD : 0 < camDist 700 250 maxNorm := lt_of_le_of_lt hm0 hgt
  apply camera_in_front aY aX x y z _ hD
  calc x ^ 2 + y ^ 2 + z ^ 2 ≤ maxNorm ^ 2 := hm
    _ < camDist 700 250 maxNorm ^ 2 := by
        exact pow_lt_pow_left hgt hm0 (by norm_num)

end Projection
end SutraWS
