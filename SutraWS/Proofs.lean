import Mathlib.Tactic
import SutraWS.Sutra
import SutraWS.State
import SutraWS.SutraSemantics

namespace SutraWS

theorem all_length : Sutra.all.length = 29 := by decide
theorem all_sum : sumDelta Sutra.all = (435 : Rat) := by native_decide
theorem all_x (s : State) : (applyAll s).x = s.x + (435 : Rat) := by simp [applyAll_x]
theorem all_y (s : State) : (applyAll s).y = s.y - (435 : Rat) := by simp [applyAll_y]

end SutraWS
