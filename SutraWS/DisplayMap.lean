import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# The display map

`projectTorus` and `renderTorus` in `vedic_v18.51.1_exact_phi.html` turn the
certified channels into a surface and a colour. The surface is drawn at the
scale the formulas give it: nothing there is normalised or clamped. This file
proves what keeps the rest in range:

* `pinch_mem` — the R4 pinch `1 - (1 - g)·(2/5)` is in `[3/5, 1]` for a mean
  gate `g ∈ [0, 1]`;
* `ndc_depth_mem` — the depth written to the depth buffer lies in `[-1, 1]`;
* `tone_mem`, `tone_mono`, `tone_mid` — the colour formula runs past 255 and
  below 0, and each channel goes through one tone curve that is the identity
  on `[24, 231]` and approaches 0 and 255 without reaching them, keeping the
  order of every value;
* `seam_mem` — the quad seams darken a toned colour by a factor in `(0, 1]`
  and leave it in `(0, 1)`.
-/

namespace SutraWS
namespace DisplayMap

theorem pinch_mem {g : ℝ} (h0 : 0 ≤ g) (h1 : g ≤ 1) :
    3 / 5 ≤ 1 - (1 - g) * (2 / 5) ∧ 1 - (1 - g) * (2 / 5) ≤ 1 := by
  constructor <;> linarith

theorem ndc_depth_mem {z zmin zmax : ℝ} (h0 : zmin ≤ z) (h1 : z ≤ zmax) (hs : zmin < zmax) :
    -1 ≤ (z - zmin) * (1998 / 1000 / (zmax - zmin)) - 999 / 1000 ∧
      (z - zmin) * (1998 / 1000 / (zmax - zmin)) - 999 / 1000 ≤ 1 := by
  have hd : 0 < zmax - zmin := by linarith
  have ht0 : 0 ≤ (z - zmin) / (zmax - zmin) := div_nonneg (by linarith) hd.le
  have ht1 : (z - zmin) / (zmax - zmin) ≤ 1 := (div_le_one hd).mpr (by linarith)
  have e : (z - zmin) * (1998 / 1000 / (zmax - zmin)) = 1998 / 1000 * ((z - zmin) / (zmax - zmin)) := by
    field_simp
    ring
  rw [e]
  constructor <;> linarith

/-! ## The tone curve

`renderTorus` keeps its colour formula and lets it run past 255 and below 0.
Each channel then goes through `tone`: the identity on `[24, 231]`, and an
exponential knee either side that approaches 0 and 255 without reaching them.
Where a clamp would make every value past 255 the same, `tone` keeps their
order. -/

/-- The page's `tone(v)`, with `TONE_LO = 24`, `TONE_HI = 231`. -/
noncomputable def tone (v : ℝ) : ℝ :=
  if v < 24 then 24 * Real.exp ((v - 24) / 24)
  else if v ≤ 231 then v
  else 231 + 24 * (1 - Real.exp (-(v - 231) / 24))

theorem tone_low {v : ℝ} (h : v < 24) : 0 < tone v ∧ tone v < 24 := by
  unfold tone; rw [if_pos h]
  have e1 : Real.exp ((v - 24) / 24) < 1 := by
    rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by linarith)
  have e0 : 0 < Real.exp ((v - 24) / 24) := Real.exp_pos _
  constructor <;> nlinarith

theorem tone_high {v : ℝ} (h : 231 < v) : 231 < tone v ∧ tone v < 255 := by
  unfold tone; rw [if_neg (by linarith), if_neg (by linarith)]
  have e1 : Real.exp (-(v - 231) / 24) < 1 := by
    rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by
      have : 0 < v - 231 := by linarith
      have : -(v - 231) / 24 < 0 := by linarith
      exact this)
  have e0 : 0 < Real.exp (-(v - 231) / 24) := Real.exp_pos _
  constructor <;> nlinarith

/-- `tone_mid`: the colour formula's own value wherever it is in range. -/
theorem tone_mid {v : ℝ} (h0 : 24 ≤ v) (h1 : v ≤ 231) : tone v = v := by
  unfold tone; rw [if_neg (by linarith), if_pos h1]

/-- `tone_mem`: every channel is strictly inside `(0, 255)`. -/
theorem tone_mem (v : ℝ) : 0 < tone v ∧ tone v < 255 := by
  rcases lt_or_le v 24 with h | h
  · obtain ⟨a, b⟩ := tone_low h; exact ⟨a, by linarith⟩
  rcases le_or_lt v 231 with h' | h'
  · rw [tone_mid h h']; constructor <;> linarith
  · obtain ⟨a, b⟩ := tone_high h'; exact ⟨by linarith, b⟩

/-- `tone_mono`: a larger value always comes out brighter. -/
theorem tone_mono : StrictMono tone := by
  intro a b hab
  rcases lt_or_le a 24 with ha | ha
  · rcases lt_or_le b 24 with hb | hb
    · unfold tone; rw [if_pos ha, if_pos hb]
      have : Real.exp ((a - 24) / 24) < Real.exp ((b - 24) / 24) := Real.exp_lt_exp.mpr (by linarith)
      linarith
    · have l := (tone_low ha).2
      rcases le_or_lt b 231 with hb' | hb'
      · rw [tone_mid hb hb']; linarith
      · linarith [(tone_high hb').1]
  · rcases le_or_lt a 231 with ha' | ha'
    · rw [tone_mid ha ha']
      rcases le_or_lt b 231 with hb' | hb'
      · rw [tone_mid (by linarith) hb']; exact hab
      · linarith [(tone_high hb').1]
    · unfold tone
      rw [if_neg (by linarith), if_neg (by linarith), if_neg (by linarith), if_neg (by linarith)]
      have : Real.exp (-(b - 231) / 24) < Real.exp (-(a - 231) / 24) := Real.exp_lt_exp.mpr (by linarith)
      linarith

/-- The seam shade: inside a seam the toned colour is multiplied by `39/50`,
outside by `1`. Either way it stays in `(0, 1)` after dividing by 255. -/
theorem seam_mem (v : ℝ) (k : ℝ) (hk0 : 0 < k) (hk1 : k ≤ 1) :
    0 < tone v / 255 * k ∧ tone v / 255 * k < 1 := by
  obtain ⟨a, b⟩ := tone_mem v
  constructor
  · positivity
  · have : tone v / 255 < 1 := by rw [div_lt_one (by norm_num)]; exact b
    have : 0 < tone v / 255 := by positivity
    nlinarith

end DisplayMap
end SutraWS
