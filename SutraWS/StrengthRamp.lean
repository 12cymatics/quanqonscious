import Mathlib.Tactic

namespace SutraWS.StrengthRamp

def ramp (a b : ℚ) (n j : ℕ) : ℚ := a + (b - a) * ((min j n : ℕ) : ℚ) / n

theorem ramp_zero (a b : ℚ) (n : ℕ) : ramp a b n 0 = a := by simp [ramp]

theorem ramp_hold (a b : ℚ) {n j : ℕ} (hn : 0 < n) (h : n ≤ j) : ramp a b n j = b := by
  have hn' : (n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [ramp, min_eq_right h]
  field_simp

theorem ramp_end (a b : ℚ) {n : ℕ} (hn : 0 < n) : ramp a b n n = b := ramp_hold a b hn le_rfl

theorem ramp_steady (a b : ℚ) {n j : ℕ} (hn : 0 < n) (h : j < n) :
    ramp a b n (j + 1) - ramp a b n j = (b - a) / n := by
  have hn' : (n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [ramp, min_eq_left (Nat.succ_le_of_lt h), min_eq_left h.le]
  push_cast
  field_simp
  ring

theorem ramp_share {n : ℕ} (hn : 0 < n) (j : ℕ) :
    0 ≤ ((min j n : ℕ) : ℚ) / n ∧ ((min j n : ℕ) : ℚ) / n ≤ 1 := by
  have hn' : (0 : ℚ) < n := by exact_mod_cast hn
  refine ⟨by positivity, ?_⟩
  rw [div_le_one hn']
  exact_mod_cast min_le_right j n

theorem ramp_eq_lerp (a b : ℚ) (n j : ℕ) :
    ramp a b n j = (1 - ((min j n : ℕ) : ℚ) / n) * a + ((min j n : ℕ) : ℚ) / n * b := by
  simp only [ramp]; ring

theorem ramp_between (a b : ℚ) {n : ℕ} (hn : 0 < n) (j : ℕ) :
    min a b ≤ ramp a b n j ∧ ramp a b n j ≤ max a b := by
  obtain ⟨h0, h1⟩ := ramp_share hn j
  rw [ramp_eq_lerp]
  set f := ((min j n : ℕ) : ℚ) / n
  constructor
  · rcases le_total a b with h | h
    · rw [min_eq_left h]; nlinarith
    · rw [min_eq_right h]; nlinarith
  · rcases le_total a b with h | h
    · rw [max_eq_right h]; nlinarith
    · rw [max_eq_left h]; nlinarith

theorem ramp_unit (a b : ℚ) {n : ℕ} (hn : 0 < n) (j : ℕ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (hb0 : 0 ≤ b) (hb1 : b ≤ 1) : 0 ≤ ramp a b n j ∧ ramp a b n j ≤ 1 := by
  obtain ⟨l, u⟩ := ramp_between a b hn j
  exact ⟨le_trans (le_min ha0 hb0) l, le_trans u (max_le ha1 hb1)⟩

theorem ramp_mono (a b : ℚ) {n : ℕ} (hn : 0 < n) (hab : a ≤ b) : Monotone (ramp a b n) := by
  intro i j hij
  have hn' : (0 : ℚ) < n := by exact_mod_cast hn
  have hm : ((min i n : ℕ) : ℚ) ≤ ((min j n : ℕ) : ℚ) := by exact_mod_cast min_le_min_right n hij
  have hd := (div_le_div_right hn').mpr hm
  rw [ramp_eq_lerp, ramp_eq_lerp]
  nlinarith

theorem ramp_pos (a b : ℚ) {n j : ℕ} (hn : 0 < n) (hj : 0 < j) (ha : 0 ≤ a) (hb : 0 < b) :
    0 < ramp a b n j := by
  have hn' : (0 : ℚ) < n := by exact_mod_cast hn
  have hm : (0 : ℚ) < ((min j n : ℕ) : ℚ) := by exact_mod_cast lt_min hj hn
  have hf : 0 < ((min j n : ℕ) : ℚ) / n := div_pos hm hn'
  have h1 := (ramp_share hn j).2
  rw [ramp_eq_lerp]
  nlinarith

theorem ramp_restart (a b c : ℚ) (n j : ℕ) : ramp (ramp a b n j) c n 0 = ramp a b n j := ramp_zero _ _ _

end SutraWS.StrengthRamp
