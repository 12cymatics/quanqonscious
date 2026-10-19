import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The display map

`projectTorus` and `renderTorus` in `vedic_v18.51.1_exact_phi.html` turn the
certified channels into a surface and a colour. The surface is drawn at the
scale the formulas give it: nothing there is normalised or clamped. This file
proves what keeps the rest in range:

* `pinch_mem` — the R4 pinch `1 - (1 - g)·(2/5)` is in `[3/5, 1]` for a mean
  gate `g ∈ [0, 1]`;
* `ndc_depth_mem` — the depth written to the depth buffer lies in `[-1, 1]`;
* `ndc_depth_lift_mem` — and so does a glyph's, lifted toward the camera;
* `frame_hold` — the framing zooms out past a growing surface, in once it is
  well inside, holds between, and never passes the surface's size;
* `tone_mem`, `tone_mono`, `tone_mid` — the colour formula runs past 255 and
  below 0, and each channel goes through one tone curve that is the identity
  on `[24, 231]` and approaches 0 and 255 without reaching them, keeping the
  order of every value;
* `seam_mem` — the quad seams darken a toned colour by a factor in `(0, 1]`
  and leave it in `(0, 1)`;
* `blend_zero`, `blend_one`, `blend_mem` — between two kernel steps the drawn
  field moves in a straight line, `(1 − τ) a + τ b`, `τ ∈ [0, 1]`: it starts at
  what was on screen, ends exactly at the new state, and never leaves the
  segment between them;
* `ema_factor_mem`, `ema_pos` — the framing follows the surface through a lag
  `F + (b − F) k`, `k = 1 − e^{−dt/T} ∈ [0, 1)`, so it stays positive and moves
  only part of the way each frame.
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

/-- `ndc_depth_lift_mem`: the page maps the frame's depth range onto
`[-0.995, 0.999]`, leaving room below for the lift `b ∈ [0, 0.004]` that puts
glyphs lying on the surface just in front of it; lifted or not, every depth
written lies in `[-1, 1]`. (Mapped onto `[-0.999, 0.999]`, `ndc_depth_mem`, the
lifted depth of the nearest glyph was `-1.003`, outside, and the GPU clipped
it.) -/
theorem ndc_depth_lift_mem {z zmin zmax b : ℝ} (h0 : zmin ≤ z) (h1 : z ≤ zmax) (hs : zmin < zmax)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 4 / 1000) :
    -1 ≤ (z - zmin) * (1994 / 1000 / (zmax - zmin)) - 995 / 1000 - b ∧
      (z - zmin) * (1994 / 1000 / (zmax - zmin)) - 995 / 1000 - b ≤ 1 := by
  have hd : 0 < zmax - zmin := by linarith
  have ht0 : 0 ≤ (z - zmin) / (zmax - zmin) := div_nonneg (by linarith) hd.le
  have ht1 : (z - zmin) / (zmax - zmin) ≤ 1 := (div_le_one hd).mpr (by linarith)
  have e : (z - zmin) * (1994 / 1000 / (zmax - zmin)) = 1994 / 1000 * ((z - zmin) / (zmax - zmin)) := by
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

/-! ## Between two steps

The page draws each kernel step's exact field once its step interval has
passed. In between, every cell moves in a straight line from the value on
screen to the new state's value, and the framing eases rather than jumps. -/

/-- At `τ = 0` the blend is what was on screen. -/
theorem blend_zero (a b : ℝ) : (1 - 0) * a + 0 * b = a := by ring

/-- At `τ = 1` the blend is the new state's value, exactly. -/
theorem blend_one (a b : ℝ) : (1 - 1) * a + 1 * b = b := by ring

/-- `blend_mem`: for `τ ∈ [0, 1]` the blend lies in any interval holding both
ends -- in particular between them, so nothing jumps past either state. -/
theorem blend_mem {lo hi a b t : ℝ} (ha0 : lo ≤ a) (ha1 : a ≤ hi) (hb0 : lo ≤ b) (hb1 : b ≤ hi)
    (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    lo ≤ (1 - t) * a + t * b ∧ (1 - t) * a + t * b ≤ hi := by
  constructor <;> nlinarith

/-- `ema_factor_mem`: the framing lag's factor `1 − e^{−dt/T}` is in `[0, 1)`
for a frame time `dt ≥ 0` and a lag `T > 0`. -/
theorem ema_factor_mem {dt T : ℝ} (hdt : 0 ≤ dt) (hT : 0 < T) :
    0 ≤ 1 - Real.exp (-(dt / T)) ∧ 1 - Real.exp (-(dt / T)) < 1 := by
  have hq : 0 ≤ dt / T := div_nonneg hdt hT.le
  have hle : Real.exp (-(dt / T)) ≤ 1 := by
    rw [← Real.exp_zero]; exact Real.exp_le_exp.mpr (by linarith)
  have hpos : 0 < Real.exp (-(dt / T)) := Real.exp_pos _
  constructor <;> linarith

/-- `ema_pos`: easing a positive framing toward a positive target by a factor
in `[0, 1]` keeps it positive. -/
theorem ema_pos {f b k : ℝ} (hf : 0 < f) (hb : 0 < b) (h0 : 0 ≤ k) (h1 : k ≤ 1) :
    0 < f + (b - f) * k := by
  have e : f + (b - f) * k = (1 - k) * f + k * b := by ring
  rw [e]
  rcases h0.lt_or_eq with hk | hk
  · have := mul_pos hk hb
    nlinarith [mul_nonneg (sub_nonneg.mpr h1) hf.le]
  · subst hk
    simpa using hf

/-- `plot_mem`: a timeline places a value between the lowest and highest it has
shown at `(v − lo) / (hi − lo) ∈ [0, 1]`. -/
theorem plot_mem {v lo hi : ℝ} (h0 : lo ≤ v) (h1 : v ≤ hi) (hs : lo < hi) :
    0 ≤ (v - lo) / (hi - lo) ∧ (v - lo) / (hi - lo) ≤ 1 := by
  have hd : 0 < hi - lo := by linarith
  constructor
  · exact div_nonneg (by linarith) hd.le
  · rw [div_le_one hd]; linarith

/-- `plot_mono`: the timeline keeps the order of the values it plots. -/
theorem plot_mono {v w lo hi : ℝ} (hs : lo < hi) (hvw : v ≤ w) :
    (v - lo) / (hi - lo) ≤ (w - lo) / (hi - lo) :=
  div_le_div_of_nonneg_right (by linarith) (by linarith)

/-- `plot_log_mono`: energies are plotted on a log scale, which keeps their order. -/
theorem plot_log_mono {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) : Real.log x ≤ Real.log y :=
  Real.log_le_log hx hxy

/-! ## The colour, hue kept

Per channel, `tone` sent every very bright colour towards white: once all
three channels pass 231 they all approach 255. The surface now keeps hue:
each channel goes through the low knee only (`lowKnee`, positive and the
identity from 24 up), and then all three are scaled by one factor so that the
largest goes through the high knee (`highKnee`, below 255, the identity up to
231). On a grey (three equal channels) this is `tone` itself
(`toneRGB_grey`). -/

/-- The low knee alone: `24 e^{(v−24)/24}` below 24, the identity from 24 up. -/
noncomputable def lowKnee (v : ℝ) : ℝ := if v < 24 then 24 * Real.exp ((v - 24) / 24) else v

/-- The high knee alone: the identity up to 231, `231 + 24 (1 − e^{−(m−231)/24})` past it. -/
noncomputable def highKnee (m : ℝ) : ℝ := if m ≤ 231 then m else 231 + 24 * (1 - Real.exp (-(m - 231) / 24))

/-- `lowKnee_pos`: every channel is positive after the low knee. -/
theorem lowKnee_pos (v : ℝ) : 0 < lowKnee v := by
  unfold lowKnee; split_ifs with h
  · positivity
  · linarith

/-- `highKnee_lt`: the largest channel ends below 255. -/
theorem highKnee_lt (m : ℝ) : highKnee m < 255 := by
  unfold highKnee; split_ifs with h
  · linarith
  · have : 0 < Real.exp (-(m - 231) / 24) := Real.exp_pos _
    linarith

/-- `highKnee_le`: the high knee only ever compresses, `0 < highKnee m ≤ m` for `m > 0`. -/
theorem highKnee_le {m : ℝ} (hm : 0 < m) : 0 < highKnee m ∧ highKnee m ≤ m := by
  unfold highKnee; split_ifs with h
  · exact ⟨hm, le_refl m⟩
  · have e1 : Real.exp (-(m - 231) / 24) < 1 := by
      rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by rw [neg_div]; linarith)
    have e2 : 1 - (m - 231) / 24 ≤ Real.exp (-(m - 231) / 24) := by
      have := Real.add_one_le_exp (-(m - 231) / 24); rw [neg_div] at this ⊢; linarith
    constructor <;> nlinarith

/-- `highKnee_mono`: a brighter largest channel comes out brighter. -/
theorem highKnee_mono : StrictMono highKnee := by
  intro a b hab
  unfold highKnee
  by_cases ha : a ≤ 231
  · rw [if_pos ha]
    by_cases hb : b ≤ 231
    · rw [if_pos hb]; exact hab
    · rw [if_neg hb]
      have : Real.exp (-(b - 231) / 24) < 1 := by
        rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by push_neg at hb; rw [neg_div]; linarith)
      linarith
  · rw [if_neg ha, if_neg (by push_neg at ha ⊢; linarith)]
    have : Real.exp (-(b - 231) / 24) < Real.exp (-(a - 231) / 24) := Real.exp_lt_exp.mpr (by linarith)
    linarith

/-- `toneRGB_mem`: with `m` the largest of the low-kneed channels, each channel
`c ∈ (0, m]` comes out as `c · highKnee m / m ∈ (0, 255)`. -/
theorem toneRGB_mem {c m : ℝ} (hc : 0 < c) (hcm : c ≤ m) :
    0 < c * (highKnee m / m) ∧ c * (highKnee m / m) < 255 := by
  have hm : 0 < m := lt_of_lt_of_le hc hcm
  obtain ⟨h0, _⟩ := highKnee_le hm
  have hk : 0 < highKnee m / m := div_pos h0 hm
  constructor
  · exact mul_pos hc hk
  · have : c * (highKnee m / m) ≤ highKnee m := by
      calc c * (highKnee m / m) ≤ m * (highKnee m / m) := mul_le_mul_of_nonneg_right hcm hk.le
        _ = highKnee m := by field_simp
    linarith [highKnee_lt m]

/-- `toneRGB_ratio`: the three channels keep their ratios, so the hue is kept. -/
theorem toneRGB_ratio {a b k : ℝ} (hk : k ≠ 0) (hb : b ≠ 0) : (a * k) / (b * k) = a / b := by
  field_simp; ring

/-- `toneRGB_grey`: on a grey, three equal channels, it is the per-channel `tone`. -/
theorem toneRGB_grey (v : ℝ) : lowKnee v * (highKnee (lowKnee v) / lowKnee v) = tone v := by
  have hp := lowKnee_pos v
  rw [mul_div_cancel₀ _ hp.ne']
  unfold tone highKnee lowKnee
  by_cases h1 : v < 24
  · rw [if_pos h1, if_pos h1]
    have : 24 * Real.exp ((v - 24) / 24) ≤ 231 := by
      have : Real.exp ((v - 24) / 24) < 1 := by rw [← Real.exp_zero]; exact Real.exp_lt_exp.mpr (by linarith)
      linarith
    rw [if_pos this]
  · rw [if_neg h1, if_neg h1]

/-- `bary_mem`: the GPU colours a triangle by barycentric weights of its three
vertices' colours, so a colour in `[lo, hi]` at every vertex stays there. -/
theorem bary_mem {a b c lo hi u v w : ℝ} (ha : lo ≤ a ∧ a ≤ hi) (hb : lo ≤ b ∧ b ≤ hi) (hc : lo ≤ c ∧ c ≤ hi)
    (hu : 0 ≤ u) (hv : 0 ≤ v) (hw : 0 ≤ w) (hs : u + v + w = 1) :
    lo ≤ u * a + v * b + w * c ∧ u * a + v * b + w * c ≤ hi := by
  have e1 : u * lo + v * lo + w * lo = lo := by rw [← add_mul, ← add_mul, hs, one_mul]
  have e2 : u * hi + v * hi + w * hi = hi := by rw [← add_mul, ← add_mul, hs, one_mul]
  have l1 := mul_le_mul_of_nonneg_left ha.1 hu
  have l2 := mul_le_mul_of_nonneg_left hb.1 hv
  have l3 := mul_le_mul_of_nonneg_left hc.1 hw
  have u1 := mul_le_mul_of_nonneg_left ha.2 hu
  have u2 := mul_le_mul_of_nonneg_left hb.2 hv
  have u3 := mul_le_mul_of_nonneg_left hc.2 hw
  constructor <;> linarith

/-- `frame_out_in`: the framing's two lags, both positive, each keep its
factor in `[0, 1)`, so it zooms out fast and in slowly without ever
overshooting. -/
theorem frame_out_in {dt Tout Tin : ℝ} (hdt : 0 ≤ dt) (ho : 0 < Tout) (hi : 0 < Tin) :
    (0 ≤ 1 - Real.exp (-(dt / Tout)) ∧ 1 - Real.exp (-(dt / Tout)) < 1) ∧
    (0 ≤ 1 - Real.exp (-(dt / Tin)) ∧ 1 - Real.exp (-(dt / Tin)) < 1) :=
  ⟨ema_factor_mem hdt ho, ema_factor_mem hdt hi⟩

/-- The page's framing factor: zoom out (lag `Tout`) when the surface `b` is
past the frame `F`, zoom in (lag `Tin`) once it is inside `h F`, hold between. -/
noncomputable def frameFactor (dt Tout Tin h F b : ℝ) : ℝ :=
  if F < b then 1 - Real.exp (-(dt / Tout)) else if b < h * F then 1 - Real.exp (-(dt / Tin)) else 0

/-- **`frame_hold`**: the factor is in `[0, 1)`, so the framing moves from `F`
toward `b` without passing it, and while the surface is within the band
`[h F, F]` the frame holds still. -/
theorem frame_hold {dt Tout Tin h F b : ℝ} (hdt : 0 ≤ dt) (ho : 0 < Tout) (hi : 0 < Tin) :
    let k := frameFactor dt Tout Tin h F b
    (0 ≤ k ∧ k < 1) ∧ min F b ≤ F + (b - F) * k ∧ F + (b - F) * k ≤ max F b ∧
      (h * F ≤ b → b ≤ F → F + (b - F) * k = F) := by
  intro k
  have hk : 0 ≤ k ∧ k < 1 := by
    simp only [k, frameFactor]
    split_ifs
    · exact ema_factor_mem hdt ho
    · exact ema_factor_mem hdt hi
    · exact ⟨le_refl 0, one_pos⟩
  refine ⟨hk, ?_, ?_, ?_⟩
  · have := min_le_left F b; have := min_le_right F b
    have e : F + (b - F) * k = (1 - k) * F + k * b := by ring
    rw [e]; nlinarith [hk.1, hk.2]
  · have := le_max_left F b; have := le_max_right F b
    have e : F + (b - F) * k = (1 - k) * F + k * b := by ring
    rw [e]; nlinarith [hk.1, hk.2]
  · intro hl hu
    have : k = 0 := by
      simp only [k, frameFactor]
      rw [if_neg (not_lt.mpr hu), if_neg (not_lt.mpr hl)]
    rw [this]; ring

end DisplayMap
end SutraWS
