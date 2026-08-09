import Mathlib.Tactic
import SutraWS.Sutra
import SutraWS.State
import SutraWS.SutraSemantics

namespace SutraWS


theorem delta_1 : delta Sutra.S1 = (1 : Rat) := by rfl


theorem act_1_x (st : State) : (act Sutra.S1 st).x = st.x + (1 : Rat) := by rfl


theorem act_1_y (st : State) : (act Sutra.S1 st).y = st.y - (1 : Rat) := by rfl


theorem mem_all_1 : Sutra.S1 ∈ Sutra.all := by native_decide


theorem sum_erase_1 : sumDelta (Sutra.all.erase Sutra.S1) = (434 : Rat) := by native_decide


theorem x_erase_1 (st : State) : (applyList (Sutra.all.erase Sutra.S1) st).x = st.x + (434 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S1) = (434 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_1 (st : State) : (applyList (Sutra.all.erase Sutra.S1) st).y = st.y - (434 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S1) = (434 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_1_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S1) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S1) st).x = st.x + (434 : Rat) := x_erase_1 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (434 : Rat) := by linarith
  norm_num at h


theorem delta_2 : delta Sutra.S2 = (2 : Rat) := by rfl


theorem act_2_x (st : State) : (act Sutra.S2 st).x = st.x + (2 : Rat) := by rfl


theorem act_2_y (st : State) : (act Sutra.S2 st).y = st.y - (2 : Rat) := by rfl


theorem mem_all_2 : Sutra.S2 ∈ Sutra.all := by native_decide


theorem sum_erase_2 : sumDelta (Sutra.all.erase Sutra.S2) = (433 : Rat) := by native_decide


theorem x_erase_2 (st : State) : (applyList (Sutra.all.erase Sutra.S2) st).x = st.x + (433 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S2) = (433 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_2 (st : State) : (applyList (Sutra.all.erase Sutra.S2) st).y = st.y - (433 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S2) = (433 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_2_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S2) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S2) st).x = st.x + (433 : Rat) := x_erase_2 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (433 : Rat) := by linarith
  norm_num at h


theorem delta_3 : delta Sutra.S3 = (3 : Rat) := by rfl


theorem act_3_x (st : State) : (act Sutra.S3 st).x = st.x + (3 : Rat) := by rfl


theorem act_3_y (st : State) : (act Sutra.S3 st).y = st.y - (3 : Rat) := by rfl


theorem mem_all_3 : Sutra.S3 ∈ Sutra.all := by native_decide


theorem sum_erase_3 : sumDelta (Sutra.all.erase Sutra.S3) = (432 : Rat) := by native_decide


theorem x_erase_3 (st : State) : (applyList (Sutra.all.erase Sutra.S3) st).x = st.x + (432 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S3) = (432 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_3 (st : State) : (applyList (Sutra.all.erase Sutra.S3) st).y = st.y - (432 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S3) = (432 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_3_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S3) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S3) st).x = st.x + (432 : Rat) := x_erase_3 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (432 : Rat) := by linarith
  norm_num at h


theorem delta_4 : delta Sutra.S4 = (4 : Rat) := by rfl


theorem act_4_x (st : State) : (act Sutra.S4 st).x = st.x + (4 : Rat) := by rfl


theorem act_4_y (st : State) : (act Sutra.S4 st).y = st.y - (4 : Rat) := by rfl


theorem mem_all_4 : Sutra.S4 ∈ Sutra.all := by native_decide


theorem sum_erase_4 : sumDelta (Sutra.all.erase Sutra.S4) = (431 : Rat) := by native_decide


theorem x_erase_4 (st : State) : (applyList (Sutra.all.erase Sutra.S4) st).x = st.x + (431 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S4) = (431 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_4 (st : State) : (applyList (Sutra.all.erase Sutra.S4) st).y = st.y - (431 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S4) = (431 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_4_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S4) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S4) st).x = st.x + (431 : Rat) := x_erase_4 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (431 : Rat) := by linarith
  norm_num at h


theorem delta_5 : delta Sutra.S5 = (5 : Rat) := by rfl


theorem act_5_x (st : State) : (act Sutra.S5 st).x = st.x + (5 : Rat) := by rfl


theorem act_5_y (st : State) : (act Sutra.S5 st).y = st.y - (5 : Rat) := by rfl


theorem mem_all_5 : Sutra.S5 ∈ Sutra.all := by native_decide


theorem sum_erase_5 : sumDelta (Sutra.all.erase Sutra.S5) = (430 : Rat) := by native_decide


theorem x_erase_5 (st : State) : (applyList (Sutra.all.erase Sutra.S5) st).x = st.x + (430 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S5) = (430 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_5 (st : State) : (applyList (Sutra.all.erase Sutra.S5) st).y = st.y - (430 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S5) = (430 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_5_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S5) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S5) st).x = st.x + (430 : Rat) := x_erase_5 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (430 : Rat) := by linarith
  norm_num at h


theorem delta_6 : delta Sutra.S6 = (6 : Rat) := by rfl


theorem act_6_x (st : State) : (act Sutra.S6 st).x = st.x + (6 : Rat) := by rfl


theorem act_6_y (st : State) : (act Sutra.S6 st).y = st.y - (6 : Rat) := by rfl


theorem mem_all_6 : Sutra.S6 ∈ Sutra.all := by native_decide


theorem sum_erase_6 : sumDelta (Sutra.all.erase Sutra.S6) = (429 : Rat) := by native_decide


theorem x_erase_6 (st : State) : (applyList (Sutra.all.erase Sutra.S6) st).x = st.x + (429 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S6) = (429 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_6 (st : State) : (applyList (Sutra.all.erase Sutra.S6) st).y = st.y - (429 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S6) = (429 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_6_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S6) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S6) st).x = st.x + (429 : Rat) := x_erase_6 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (429 : Rat) := by linarith
  norm_num at h


theorem delta_7 : delta Sutra.S7 = (7 : Rat) := by rfl


theorem act_7_x (st : State) : (act Sutra.S7 st).x = st.x + (7 : Rat) := by rfl


theorem act_7_y (st : State) : (act Sutra.S7 st).y = st.y - (7 : Rat) := by rfl


theorem mem_all_7 : Sutra.S7 ∈ Sutra.all := by native_decide


theorem sum_erase_7 : sumDelta (Sutra.all.erase Sutra.S7) = (428 : Rat) := by native_decide


theorem x_erase_7 (st : State) : (applyList (Sutra.all.erase Sutra.S7) st).x = st.x + (428 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S7) = (428 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_7 (st : State) : (applyList (Sutra.all.erase Sutra.S7) st).y = st.y - (428 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S7) = (428 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_7_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S7) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S7) st).x = st.x + (428 : Rat) := x_erase_7 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (428 : Rat) := by linarith
  norm_num at h


theorem delta_8 : delta Sutra.S8 = (8 : Rat) := by rfl


theorem act_8_x (st : State) : (act Sutra.S8 st).x = st.x + (8 : Rat) := by rfl


theorem act_8_y (st : State) : (act Sutra.S8 st).y = st.y - (8 : Rat) := by rfl


theorem mem_all_8 : Sutra.S8 ∈ Sutra.all := by native_decide


theorem sum_erase_8 : sumDelta (Sutra.all.erase Sutra.S8) = (427 : Rat) := by native_decide


theorem x_erase_8 (st : State) : (applyList (Sutra.all.erase Sutra.S8) st).x = st.x + (427 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S8) = (427 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_8 (st : State) : (applyList (Sutra.all.erase Sutra.S8) st).y = st.y - (427 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S8) = (427 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_8_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S8) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S8) st).x = st.x + (427 : Rat) := x_erase_8 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (427 : Rat) := by linarith
  norm_num at h


theorem delta_9 : delta Sutra.S9 = (9 : Rat) := by rfl


theorem act_9_x (st : State) : (act Sutra.S9 st).x = st.x + (9 : Rat) := by rfl


theorem act_9_y (st : State) : (act Sutra.S9 st).y = st.y - (9 : Rat) := by rfl


theorem mem_all_9 : Sutra.S9 ∈ Sutra.all := by native_decide


theorem sum_erase_9 : sumDelta (Sutra.all.erase Sutra.S9) = (426 : Rat) := by native_decide


theorem x_erase_9 (st : State) : (applyList (Sutra.all.erase Sutra.S9) st).x = st.x + (426 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S9) = (426 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_9 (st : State) : (applyList (Sutra.all.erase Sutra.S9) st).y = st.y - (426 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S9) = (426 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_9_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S9) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S9) st).x = st.x + (426 : Rat) := x_erase_9 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (426 : Rat) := by linarith
  norm_num at h


theorem delta_10 : delta Sutra.S10 = (10 : Rat) := by rfl


theorem act_10_x (st : State) : (act Sutra.S10 st).x = st.x + (10 : Rat) := by rfl


theorem act_10_y (st : State) : (act Sutra.S10 st).y = st.y - (10 : Rat) := by rfl


theorem mem_all_10 : Sutra.S10 ∈ Sutra.all := by native_decide


theorem sum_erase_10 : sumDelta (Sutra.all.erase Sutra.S10) = (425 : Rat) := by native_decide


theorem x_erase_10 (st : State) : (applyList (Sutra.all.erase Sutra.S10) st).x = st.x + (425 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S10) = (425 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_10 (st : State) : (applyList (Sutra.all.erase Sutra.S10) st).y = st.y - (425 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S10) = (425 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_10_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S10) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S10) st).x = st.x + (425 : Rat) := x_erase_10 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (425 : Rat) := by linarith
  norm_num at h


theorem delta_11 : delta Sutra.S11 = (11 : Rat) := by rfl


theorem act_11_x (st : State) : (act Sutra.S11 st).x = st.x + (11 : Rat) := by rfl


theorem act_11_y (st : State) : (act Sutra.S11 st).y = st.y - (11 : Rat) := by rfl


theorem mem_all_11 : Sutra.S11 ∈ Sutra.all := by native_decide


theorem sum_erase_11 : sumDelta (Sutra.all.erase Sutra.S11) = (424 : Rat) := by native_decide


theorem x_erase_11 (st : State) : (applyList (Sutra.all.erase Sutra.S11) st).x = st.x + (424 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S11) = (424 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_11 (st : State) : (applyList (Sutra.all.erase Sutra.S11) st).y = st.y - (424 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S11) = (424 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_11_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S11) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S11) st).x = st.x + (424 : Rat) := x_erase_11 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (424 : Rat) := by linarith
  norm_num at h


theorem delta_12 : delta Sutra.S12 = (12 : Rat) := by rfl


theorem act_12_x (st : State) : (act Sutra.S12 st).x = st.x + (12 : Rat) := by rfl


theorem act_12_y (st : State) : (act Sutra.S12 st).y = st.y - (12 : Rat) := by rfl


theorem mem_all_12 : Sutra.S12 ∈ Sutra.all := by native_decide


theorem sum_erase_12 : sumDelta (Sutra.all.erase Sutra.S12) = (423 : Rat) := by native_decide


theorem x_erase_12 (st : State) : (applyList (Sutra.all.erase Sutra.S12) st).x = st.x + (423 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S12) = (423 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_12 (st : State) : (applyList (Sutra.all.erase Sutra.S12) st).y = st.y - (423 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S12) = (423 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_12_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S12) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S12) st).x = st.x + (423 : Rat) := x_erase_12 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (423 : Rat) := by linarith
  norm_num at h


theorem delta_13 : delta Sutra.S13 = (13 : Rat) := by rfl


theorem act_13_x (st : State) : (act Sutra.S13 st).x = st.x + (13 : Rat) := by rfl


theorem act_13_y (st : State) : (act Sutra.S13 st).y = st.y - (13 : Rat) := by rfl


theorem mem_all_13 : Sutra.S13 ∈ Sutra.all := by native_decide


theorem sum_erase_13 : sumDelta (Sutra.all.erase Sutra.S13) = (422 : Rat) := by native_decide


theorem x_erase_13 (st : State) : (applyList (Sutra.all.erase Sutra.S13) st).x = st.x + (422 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S13) = (422 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_13 (st : State) : (applyList (Sutra.all.erase Sutra.S13) st).y = st.y - (422 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S13) = (422 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_13_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S13) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S13) st).x = st.x + (422 : Rat) := x_erase_13 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (422 : Rat) := by linarith
  norm_num at h


theorem delta_14 : delta Sutra.S14 = (14 : Rat) := by rfl


theorem act_14_x (st : State) : (act Sutra.S14 st).x = st.x + (14 : Rat) := by rfl


theorem act_14_y (st : State) : (act Sutra.S14 st).y = st.y - (14 : Rat) := by rfl


theorem mem_all_14 : Sutra.S14 ∈ Sutra.all := by native_decide


theorem sum_erase_14 : sumDelta (Sutra.all.erase Sutra.S14) = (421 : Rat) := by native_decide


theorem x_erase_14 (st : State) : (applyList (Sutra.all.erase Sutra.S14) st).x = st.x + (421 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S14) = (421 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_14 (st : State) : (applyList (Sutra.all.erase Sutra.S14) st).y = st.y - (421 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S14) = (421 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_14_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S14) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S14) st).x = st.x + (421 : Rat) := x_erase_14 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (421 : Rat) := by linarith
  norm_num at h


theorem delta_15 : delta Sutra.S15 = (15 : Rat) := by rfl


theorem act_15_x (st : State) : (act Sutra.S15 st).x = st.x + (15 : Rat) := by rfl


theorem act_15_y (st : State) : (act Sutra.S15 st).y = st.y - (15 : Rat) := by rfl


theorem mem_all_15 : Sutra.S15 ∈ Sutra.all := by native_decide


theorem sum_erase_15 : sumDelta (Sutra.all.erase Sutra.S15) = (420 : Rat) := by native_decide


theorem x_erase_15 (st : State) : (applyList (Sutra.all.erase Sutra.S15) st).x = st.x + (420 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S15) = (420 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_15 (st : State) : (applyList (Sutra.all.erase Sutra.S15) st).y = st.y - (420 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S15) = (420 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_15_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S15) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S15) st).x = st.x + (420 : Rat) := x_erase_15 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (420 : Rat) := by linarith
  norm_num at h


theorem delta_16 : delta Sutra.S16 = (16 : Rat) := by rfl


theorem act_16_x (st : State) : (act Sutra.S16 st).x = st.x + (16 : Rat) := by rfl


theorem act_16_y (st : State) : (act Sutra.S16 st).y = st.y - (16 : Rat) := by rfl


theorem mem_all_16 : Sutra.S16 ∈ Sutra.all := by native_decide


theorem sum_erase_16 : sumDelta (Sutra.all.erase Sutra.S16) = (419 : Rat) := by native_decide


theorem x_erase_16 (st : State) : (applyList (Sutra.all.erase Sutra.S16) st).x = st.x + (419 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S16) = (419 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_16 (st : State) : (applyList (Sutra.all.erase Sutra.S16) st).y = st.y - (419 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S16) = (419 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_16_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S16) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S16) st).x = st.x + (419 : Rat) := x_erase_16 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (419 : Rat) := by linarith
  norm_num at h


theorem delta_17 : delta Sutra.S17 = (17 : Rat) := by rfl


theorem act_17_x (st : State) : (act Sutra.S17 st).x = st.x + (17 : Rat) := by rfl


theorem act_17_y (st : State) : (act Sutra.S17 st).y = st.y - (17 : Rat) := by rfl


theorem mem_all_17 : Sutra.S17 ∈ Sutra.all := by native_decide


theorem sum_erase_17 : sumDelta (Sutra.all.erase Sutra.S17) = (418 : Rat) := by native_decide


theorem x_erase_17 (st : State) : (applyList (Sutra.all.erase Sutra.S17) st).x = st.x + (418 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S17) = (418 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_17 (st : State) : (applyList (Sutra.all.erase Sutra.S17) st).y = st.y - (418 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S17) = (418 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_17_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S17) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S17) st).x = st.x + (418 : Rat) := x_erase_17 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (418 : Rat) := by linarith
  norm_num at h


theorem delta_18 : delta Sutra.S18 = (18 : Rat) := by rfl


theorem act_18_x (st : State) : (act Sutra.S18 st).x = st.x + (18 : Rat) := by rfl


theorem act_18_y (st : State) : (act Sutra.S18 st).y = st.y - (18 : Rat) := by rfl


theorem mem_all_18 : Sutra.S18 ∈ Sutra.all := by native_decide


theorem sum_erase_18 : sumDelta (Sutra.all.erase Sutra.S18) = (417 : Rat) := by native_decide


theorem x_erase_18 (st : State) : (applyList (Sutra.all.erase Sutra.S18) st).x = st.x + (417 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S18) = (417 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_18 (st : State) : (applyList (Sutra.all.erase Sutra.S18) st).y = st.y - (417 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S18) = (417 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_18_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S18) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S18) st).x = st.x + (417 : Rat) := x_erase_18 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (417 : Rat) := by linarith
  norm_num at h


theorem delta_19 : delta Sutra.S19 = (19 : Rat) := by rfl


theorem act_19_x (st : State) : (act Sutra.S19 st).x = st.x + (19 : Rat) := by rfl


theorem act_19_y (st : State) : (act Sutra.S19 st).y = st.y - (19 : Rat) := by rfl


theorem mem_all_19 : Sutra.S19 ∈ Sutra.all := by native_decide


theorem sum_erase_19 : sumDelta (Sutra.all.erase Sutra.S19) = (416 : Rat) := by native_decide


theorem x_erase_19 (st : State) : (applyList (Sutra.all.erase Sutra.S19) st).x = st.x + (416 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S19) = (416 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_19 (st : State) : (applyList (Sutra.all.erase Sutra.S19) st).y = st.y - (416 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S19) = (416 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_19_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S19) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S19) st).x = st.x + (416 : Rat) := x_erase_19 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (416 : Rat) := by linarith
  norm_num at h


theorem delta_20 : delta Sutra.S20 = (20 : Rat) := by rfl


theorem act_20_x (st : State) : (act Sutra.S20 st).x = st.x + (20 : Rat) := by rfl


theorem act_20_y (st : State) : (act Sutra.S20 st).y = st.y - (20 : Rat) := by rfl


theorem mem_all_20 : Sutra.S20 ∈ Sutra.all := by native_decide


theorem sum_erase_20 : sumDelta (Sutra.all.erase Sutra.S20) = (415 : Rat) := by native_decide


theorem x_erase_20 (st : State) : (applyList (Sutra.all.erase Sutra.S20) st).x = st.x + (415 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S20) = (415 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_20 (st : State) : (applyList (Sutra.all.erase Sutra.S20) st).y = st.y - (415 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S20) = (415 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_20_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S20) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S20) st).x = st.x + (415 : Rat) := x_erase_20 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (415 : Rat) := by linarith
  norm_num at h


theorem delta_21 : delta Sutra.S21 = (21 : Rat) := by rfl


theorem act_21_x (st : State) : (act Sutra.S21 st).x = st.x + (21 : Rat) := by rfl


theorem act_21_y (st : State) : (act Sutra.S21 st).y = st.y - (21 : Rat) := by rfl


theorem mem_all_21 : Sutra.S21 ∈ Sutra.all := by native_decide


theorem sum_erase_21 : sumDelta (Sutra.all.erase Sutra.S21) = (414 : Rat) := by native_decide


theorem x_erase_21 (st : State) : (applyList (Sutra.all.erase Sutra.S21) st).x = st.x + (414 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S21) = (414 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_21 (st : State) : (applyList (Sutra.all.erase Sutra.S21) st).y = st.y - (414 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S21) = (414 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_21_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S21) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S21) st).x = st.x + (414 : Rat) := x_erase_21 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (414 : Rat) := by linarith
  norm_num at h


theorem delta_22 : delta Sutra.S22 = (22 : Rat) := by rfl


theorem act_22_x (st : State) : (act Sutra.S22 st).x = st.x + (22 : Rat) := by rfl


theorem act_22_y (st : State) : (act Sutra.S22 st).y = st.y - (22 : Rat) := by rfl


theorem mem_all_22 : Sutra.S22 ∈ Sutra.all := by native_decide


theorem sum_erase_22 : sumDelta (Sutra.all.erase Sutra.S22) = (413 : Rat) := by native_decide


theorem x_erase_22 (st : State) : (applyList (Sutra.all.erase Sutra.S22) st).x = st.x + (413 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S22) = (413 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_22 (st : State) : (applyList (Sutra.all.erase Sutra.S22) st).y = st.y - (413 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S22) = (413 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_22_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S22) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S22) st).x = st.x + (413 : Rat) := x_erase_22 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (413 : Rat) := by linarith
  norm_num at h


theorem delta_23 : delta Sutra.S23 = (23 : Rat) := by rfl


theorem act_23_x (st : State) : (act Sutra.S23 st).x = st.x + (23 : Rat) := by rfl


theorem act_23_y (st : State) : (act Sutra.S23 st).y = st.y - (23 : Rat) := by rfl


theorem mem_all_23 : Sutra.S23 ∈ Sutra.all := by native_decide


theorem sum_erase_23 : sumDelta (Sutra.all.erase Sutra.S23) = (412 : Rat) := by native_decide


theorem x_erase_23 (st : State) : (applyList (Sutra.all.erase Sutra.S23) st).x = st.x + (412 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S23) = (412 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_23 (st : State) : (applyList (Sutra.all.erase Sutra.S23) st).y = st.y - (412 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S23) = (412 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_23_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S23) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S23) st).x = st.x + (412 : Rat) := x_erase_23 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (412 : Rat) := by linarith
  norm_num at h


theorem delta_24 : delta Sutra.S24 = (24 : Rat) := by rfl


theorem act_24_x (st : State) : (act Sutra.S24 st).x = st.x + (24 : Rat) := by rfl


theorem act_24_y (st : State) : (act Sutra.S24 st).y = st.y - (24 : Rat) := by rfl


theorem mem_all_24 : Sutra.S24 ∈ Sutra.all := by native_decide


theorem sum_erase_24 : sumDelta (Sutra.all.erase Sutra.S24) = (411 : Rat) := by native_decide


theorem x_erase_24 (st : State) : (applyList (Sutra.all.erase Sutra.S24) st).x = st.x + (411 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S24) = (411 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_24 (st : State) : (applyList (Sutra.all.erase Sutra.S24) st).y = st.y - (411 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S24) = (411 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_24_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S24) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S24) st).x = st.x + (411 : Rat) := x_erase_24 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (411 : Rat) := by linarith
  norm_num at h


theorem delta_25 : delta Sutra.S25 = (25 : Rat) := by rfl


theorem act_25_x (st : State) : (act Sutra.S25 st).x = st.x + (25 : Rat) := by rfl


theorem act_25_y (st : State) : (act Sutra.S25 st).y = st.y - (25 : Rat) := by rfl


theorem mem_all_25 : Sutra.S25 ∈ Sutra.all := by native_decide


theorem sum_erase_25 : sumDelta (Sutra.all.erase Sutra.S25) = (410 : Rat) := by native_decide


theorem x_erase_25 (st : State) : (applyList (Sutra.all.erase Sutra.S25) st).x = st.x + (410 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S25) = (410 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_25 (st : State) : (applyList (Sutra.all.erase Sutra.S25) st).y = st.y - (410 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S25) = (410 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_25_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S25) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S25) st).x = st.x + (410 : Rat) := x_erase_25 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (410 : Rat) := by linarith
  norm_num at h


theorem delta_26 : delta Sutra.S26 = (26 : Rat) := by rfl


theorem act_26_x (st : State) : (act Sutra.S26 st).x = st.x + (26 : Rat) := by rfl


theorem act_26_y (st : State) : (act Sutra.S26 st).y = st.y - (26 : Rat) := by rfl


theorem mem_all_26 : Sutra.S26 ∈ Sutra.all := by native_decide


theorem sum_erase_26 : sumDelta (Sutra.all.erase Sutra.S26) = (409 : Rat) := by native_decide


theorem x_erase_26 (st : State) : (applyList (Sutra.all.erase Sutra.S26) st).x = st.x + (409 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S26) = (409 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_26 (st : State) : (applyList (Sutra.all.erase Sutra.S26) st).y = st.y - (409 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S26) = (409 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_26_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S26) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S26) st).x = st.x + (409 : Rat) := x_erase_26 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (409 : Rat) := by linarith
  norm_num at h


theorem delta_27 : delta Sutra.S27 = (27 : Rat) := by rfl


theorem act_27_x (st : State) : (act Sutra.S27 st).x = st.x + (27 : Rat) := by rfl


theorem act_27_y (st : State) : (act Sutra.S27 st).y = st.y - (27 : Rat) := by rfl


theorem mem_all_27 : Sutra.S27 ∈ Sutra.all := by native_decide


theorem sum_erase_27 : sumDelta (Sutra.all.erase Sutra.S27) = (408 : Rat) := by native_decide


theorem x_erase_27 (st : State) : (applyList (Sutra.all.erase Sutra.S27) st).x = st.x + (408 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S27) = (408 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_27 (st : State) : (applyList (Sutra.all.erase Sutra.S27) st).y = st.y - (408 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S27) = (408 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_27_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S27) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S27) st).x = st.x + (408 : Rat) := x_erase_27 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (408 : Rat) := by linarith
  norm_num at h


theorem delta_28 : delta Sutra.S28 = (28 : Rat) := by rfl


theorem act_28_x (st : State) : (act Sutra.S28 st).x = st.x + (28 : Rat) := by rfl


theorem act_28_y (st : State) : (act Sutra.S28 st).y = st.y - (28 : Rat) := by rfl


theorem mem_all_28 : Sutra.S28 ∈ Sutra.all := by native_decide


theorem sum_erase_28 : sumDelta (Sutra.all.erase Sutra.S28) = (407 : Rat) := by native_decide


theorem x_erase_28 (st : State) : (applyList (Sutra.all.erase Sutra.S28) st).x = st.x + (407 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S28) = (407 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_28 (st : State) : (applyList (Sutra.all.erase Sutra.S28) st).y = st.y - (407 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S28) = (407 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_28_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S28) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S28) st).x = st.x + (407 : Rat) := x_erase_28 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (407 : Rat) := by linarith
  norm_num at h


theorem delta_29 : delta Sutra.S29 = (29 : Rat) := by rfl


theorem act_29_x (st : State) : (act Sutra.S29 st).x = st.x + (29 : Rat) := by rfl


theorem act_29_y (st : State) : (act Sutra.S29 st).y = st.y - (29 : Rat) := by rfl


theorem mem_all_29 : Sutra.S29 ∈ Sutra.all := by native_decide


theorem sum_erase_29 : sumDelta (Sutra.all.erase Sutra.S29) = (406 : Rat) := by native_decide


theorem x_erase_29 (st : State) : (applyList (Sutra.all.erase Sutra.S29) st).x = st.x + (406 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S29) = (406 : Rat) := by native_decide
  simp [applyList_x, h]


theorem y_erase_29 (st : State) : (applyList (Sutra.all.erase Sutra.S29) st).y = st.y - (406 : Rat) := by
  have h : sumDelta (Sutra.all.erase Sutra.S29) = (406 : Rat) := by native_decide
  simp [applyList_y, h]


theorem all_vs_erase_29_x (st : State) : (applyAll st).x ≠ (applyList (Sutra.all.erase Sutra.S29) st).x := by
  have hx1 : (applyAll st).x = st.x + (435 : Rat) := by simp [applyAll_x]
  have hx2 : (applyList (Sutra.all.erase Sutra.S29) st).x = st.x + (406 : Rat) := x_erase_29 st
  rw [hx1, hx2]
  intro hEq
  have h : (435 : Rat) = (406 : Rat) := by linarith
  norm_num at h


theorem commute_x_1_1 (st : State) :
  (act Sutra.S1 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_1 (st : State) :
  (act Sutra.S1 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_2 (st : State) :
  (act Sutra.S1 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_2 (st : State) :
  (act Sutra.S1 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_3 (st : State) :
  (act Sutra.S1 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_3 (st : State) :
  (act Sutra.S1 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_4 (st : State) :
  (act Sutra.S1 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_4 (st : State) :
  (act Sutra.S1 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_5 (st : State) :
  (act Sutra.S1 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_5 (st : State) :
  (act Sutra.S1 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_6 (st : State) :
  (act Sutra.S1 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_6 (st : State) :
  (act Sutra.S1 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_7 (st : State) :
  (act Sutra.S1 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_7 (st : State) :
  (act Sutra.S1 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_8 (st : State) :
  (act Sutra.S1 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_8 (st : State) :
  (act Sutra.S1 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_9 (st : State) :
  (act Sutra.S1 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_9 (st : State) :
  (act Sutra.S1 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_10 (st : State) :
  (act Sutra.S1 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_10 (st : State) :
  (act Sutra.S1 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_11 (st : State) :
  (act Sutra.S1 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_11 (st : State) :
  (act Sutra.S1 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_12 (st : State) :
  (act Sutra.S1 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_12 (st : State) :
  (act Sutra.S1 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_13 (st : State) :
  (act Sutra.S1 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_13 (st : State) :
  (act Sutra.S1 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_14 (st : State) :
  (act Sutra.S1 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_14 (st : State) :
  (act Sutra.S1 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_15 (st : State) :
  (act Sutra.S1 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_15 (st : State) :
  (act Sutra.S1 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_16 (st : State) :
  (act Sutra.S1 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_16 (st : State) :
  (act Sutra.S1 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_17 (st : State) :
  (act Sutra.S1 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_17 (st : State) :
  (act Sutra.S1 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_18 (st : State) :
  (act Sutra.S1 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_18 (st : State) :
  (act Sutra.S1 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_19 (st : State) :
  (act Sutra.S1 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_19 (st : State) :
  (act Sutra.S1 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_20 (st : State) :
  (act Sutra.S1 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_20 (st : State) :
  (act Sutra.S1 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_21 (st : State) :
  (act Sutra.S1 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_21 (st : State) :
  (act Sutra.S1 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_22 (st : State) :
  (act Sutra.S1 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_22 (st : State) :
  (act Sutra.S1 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_23 (st : State) :
  (act Sutra.S1 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_23 (st : State) :
  (act Sutra.S1 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_24 (st : State) :
  (act Sutra.S1 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_24 (st : State) :
  (act Sutra.S1 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_25 (st : State) :
  (act Sutra.S1 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_25 (st : State) :
  (act Sutra.S1 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_26 (st : State) :
  (act Sutra.S1 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_26 (st : State) :
  (act Sutra.S1 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_27 (st : State) :
  (act Sutra.S1 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_27 (st : State) :
  (act Sutra.S1 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_28 (st : State) :
  (act Sutra.S1 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_28 (st : State) :
  (act Sutra.S1 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_1_29 (st : State) :
  (act Sutra.S1 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S1 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_1_29 (st : State) :
  (act Sutra.S1 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S1 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_1 (st : State) :
  (act Sutra.S2 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_1 (st : State) :
  (act Sutra.S2 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_2 (st : State) :
  (act Sutra.S2 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_2 (st : State) :
  (act Sutra.S2 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_3 (st : State) :
  (act Sutra.S2 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_3 (st : State) :
  (act Sutra.S2 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_4 (st : State) :
  (act Sutra.S2 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_4 (st : State) :
  (act Sutra.S2 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_5 (st : State) :
  (act Sutra.S2 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_5 (st : State) :
  (act Sutra.S2 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_6 (st : State) :
  (act Sutra.S2 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_6 (st : State) :
  (act Sutra.S2 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_7 (st : State) :
  (act Sutra.S2 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_7 (st : State) :
  (act Sutra.S2 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_8 (st : State) :
  (act Sutra.S2 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_8 (st : State) :
  (act Sutra.S2 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_9 (st : State) :
  (act Sutra.S2 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_9 (st : State) :
  (act Sutra.S2 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_10 (st : State) :
  (act Sutra.S2 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_10 (st : State) :
  (act Sutra.S2 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_11 (st : State) :
  (act Sutra.S2 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_11 (st : State) :
  (act Sutra.S2 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_12 (st : State) :
  (act Sutra.S2 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_12 (st : State) :
  (act Sutra.S2 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_13 (st : State) :
  (act Sutra.S2 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_13 (st : State) :
  (act Sutra.S2 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_14 (st : State) :
  (act Sutra.S2 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_14 (st : State) :
  (act Sutra.S2 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_15 (st : State) :
  (act Sutra.S2 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_15 (st : State) :
  (act Sutra.S2 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_16 (st : State) :
  (act Sutra.S2 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_16 (st : State) :
  (act Sutra.S2 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_17 (st : State) :
  (act Sutra.S2 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_17 (st : State) :
  (act Sutra.S2 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_18 (st : State) :
  (act Sutra.S2 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_18 (st : State) :
  (act Sutra.S2 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_19 (st : State) :
  (act Sutra.S2 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_19 (st : State) :
  (act Sutra.S2 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_20 (st : State) :
  (act Sutra.S2 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_20 (st : State) :
  (act Sutra.S2 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_21 (st : State) :
  (act Sutra.S2 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_21 (st : State) :
  (act Sutra.S2 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_22 (st : State) :
  (act Sutra.S2 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_22 (st : State) :
  (act Sutra.S2 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_23 (st : State) :
  (act Sutra.S2 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_23 (st : State) :
  (act Sutra.S2 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_24 (st : State) :
  (act Sutra.S2 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_24 (st : State) :
  (act Sutra.S2 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_25 (st : State) :
  (act Sutra.S2 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_25 (st : State) :
  (act Sutra.S2 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_26 (st : State) :
  (act Sutra.S2 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_26 (st : State) :
  (act Sutra.S2 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_27 (st : State) :
  (act Sutra.S2 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_27 (st : State) :
  (act Sutra.S2 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_28 (st : State) :
  (act Sutra.S2 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_28 (st : State) :
  (act Sutra.S2 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_2_29 (st : State) :
  (act Sutra.S2 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S2 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_2_29 (st : State) :
  (act Sutra.S2 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S2 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_1 (st : State) :
  (act Sutra.S3 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_1 (st : State) :
  (act Sutra.S3 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_2 (st : State) :
  (act Sutra.S3 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_2 (st : State) :
  (act Sutra.S3 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_3 (st : State) :
  (act Sutra.S3 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_3 (st : State) :
  (act Sutra.S3 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_4 (st : State) :
  (act Sutra.S3 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_4 (st : State) :
  (act Sutra.S3 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_5 (st : State) :
  (act Sutra.S3 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_5 (st : State) :
  (act Sutra.S3 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_6 (st : State) :
  (act Sutra.S3 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_6 (st : State) :
  (act Sutra.S3 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_7 (st : State) :
  (act Sutra.S3 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_7 (st : State) :
  (act Sutra.S3 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_8 (st : State) :
  (act Sutra.S3 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_8 (st : State) :
  (act Sutra.S3 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_9 (st : State) :
  (act Sutra.S3 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_9 (st : State) :
  (act Sutra.S3 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_10 (st : State) :
  (act Sutra.S3 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_10 (st : State) :
  (act Sutra.S3 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_11 (st : State) :
  (act Sutra.S3 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_11 (st : State) :
  (act Sutra.S3 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_12 (st : State) :
  (act Sutra.S3 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_12 (st : State) :
  (act Sutra.S3 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_13 (st : State) :
  (act Sutra.S3 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_13 (st : State) :
  (act Sutra.S3 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_14 (st : State) :
  (act Sutra.S3 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_14 (st : State) :
  (act Sutra.S3 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_15 (st : State) :
  (act Sutra.S3 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_15 (st : State) :
  (act Sutra.S3 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_16 (st : State) :
  (act Sutra.S3 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_16 (st : State) :
  (act Sutra.S3 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_17 (st : State) :
  (act Sutra.S3 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_17 (st : State) :
  (act Sutra.S3 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_18 (st : State) :
  (act Sutra.S3 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_18 (st : State) :
  (act Sutra.S3 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_19 (st : State) :
  (act Sutra.S3 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_19 (st : State) :
  (act Sutra.S3 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_20 (st : State) :
  (act Sutra.S3 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_20 (st : State) :
  (act Sutra.S3 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_21 (st : State) :
  (act Sutra.S3 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_21 (st : State) :
  (act Sutra.S3 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_22 (st : State) :
  (act Sutra.S3 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_22 (st : State) :
  (act Sutra.S3 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_23 (st : State) :
  (act Sutra.S3 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_23 (st : State) :
  (act Sutra.S3 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_24 (st : State) :
  (act Sutra.S3 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_24 (st : State) :
  (act Sutra.S3 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_25 (st : State) :
  (act Sutra.S3 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_25 (st : State) :
  (act Sutra.S3 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_26 (st : State) :
  (act Sutra.S3 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_26 (st : State) :
  (act Sutra.S3 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_27 (st : State) :
  (act Sutra.S3 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_27 (st : State) :
  (act Sutra.S3 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_28 (st : State) :
  (act Sutra.S3 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_28 (st : State) :
  (act Sutra.S3 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_3_29 (st : State) :
  (act Sutra.S3 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S3 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_3_29 (st : State) :
  (act Sutra.S3 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S3 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_1 (st : State) :
  (act Sutra.S4 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_1 (st : State) :
  (act Sutra.S4 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_2 (st : State) :
  (act Sutra.S4 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_2 (st : State) :
  (act Sutra.S4 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_3 (st : State) :
  (act Sutra.S4 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_3 (st : State) :
  (act Sutra.S4 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_4 (st : State) :
  (act Sutra.S4 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_4 (st : State) :
  (act Sutra.S4 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_5 (st : State) :
  (act Sutra.S4 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_5 (st : State) :
  (act Sutra.S4 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_6 (st : State) :
  (act Sutra.S4 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_6 (st : State) :
  (act Sutra.S4 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_7 (st : State) :
  (act Sutra.S4 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_7 (st : State) :
  (act Sutra.S4 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_8 (st : State) :
  (act Sutra.S4 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_8 (st : State) :
  (act Sutra.S4 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_9 (st : State) :
  (act Sutra.S4 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_9 (st : State) :
  (act Sutra.S4 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_10 (st : State) :
  (act Sutra.S4 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_10 (st : State) :
  (act Sutra.S4 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_11 (st : State) :
  (act Sutra.S4 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_11 (st : State) :
  (act Sutra.S4 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_12 (st : State) :
  (act Sutra.S4 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_12 (st : State) :
  (act Sutra.S4 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_13 (st : State) :
  (act Sutra.S4 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_13 (st : State) :
  (act Sutra.S4 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_14 (st : State) :
  (act Sutra.S4 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_14 (st : State) :
  (act Sutra.S4 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_15 (st : State) :
  (act Sutra.S4 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_15 (st : State) :
  (act Sutra.S4 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_16 (st : State) :
  (act Sutra.S4 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_16 (st : State) :
  (act Sutra.S4 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_17 (st : State) :
  (act Sutra.S4 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_17 (st : State) :
  (act Sutra.S4 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_18 (st : State) :
  (act Sutra.S4 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_18 (st : State) :
  (act Sutra.S4 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_19 (st : State) :
  (act Sutra.S4 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_19 (st : State) :
  (act Sutra.S4 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_20 (st : State) :
  (act Sutra.S4 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_20 (st : State) :
  (act Sutra.S4 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_21 (st : State) :
  (act Sutra.S4 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_21 (st : State) :
  (act Sutra.S4 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_22 (st : State) :
  (act Sutra.S4 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_22 (st : State) :
  (act Sutra.S4 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_23 (st : State) :
  (act Sutra.S4 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_23 (st : State) :
  (act Sutra.S4 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_24 (st : State) :
  (act Sutra.S4 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_24 (st : State) :
  (act Sutra.S4 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_25 (st : State) :
  (act Sutra.S4 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_25 (st : State) :
  (act Sutra.S4 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_26 (st : State) :
  (act Sutra.S4 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_26 (st : State) :
  (act Sutra.S4 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_27 (st : State) :
  (act Sutra.S4 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_27 (st : State) :
  (act Sutra.S4 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_28 (st : State) :
  (act Sutra.S4 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_28 (st : State) :
  (act Sutra.S4 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_4_29 (st : State) :
  (act Sutra.S4 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S4 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_4_29 (st : State) :
  (act Sutra.S4 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S4 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_1 (st : State) :
  (act Sutra.S5 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_1 (st : State) :
  (act Sutra.S5 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_2 (st : State) :
  (act Sutra.S5 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_2 (st : State) :
  (act Sutra.S5 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_3 (st : State) :
  (act Sutra.S5 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_3 (st : State) :
  (act Sutra.S5 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_4 (st : State) :
  (act Sutra.S5 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_4 (st : State) :
  (act Sutra.S5 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_5 (st : State) :
  (act Sutra.S5 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_5 (st : State) :
  (act Sutra.S5 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_6 (st : State) :
  (act Sutra.S5 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_6 (st : State) :
  (act Sutra.S5 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_7 (st : State) :
  (act Sutra.S5 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_7 (st : State) :
  (act Sutra.S5 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_8 (st : State) :
  (act Sutra.S5 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_8 (st : State) :
  (act Sutra.S5 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_9 (st : State) :
  (act Sutra.S5 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_9 (st : State) :
  (act Sutra.S5 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_10 (st : State) :
  (act Sutra.S5 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_10 (st : State) :
  (act Sutra.S5 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_11 (st : State) :
  (act Sutra.S5 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_11 (st : State) :
  (act Sutra.S5 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_12 (st : State) :
  (act Sutra.S5 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_12 (st : State) :
  (act Sutra.S5 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_13 (st : State) :
  (act Sutra.S5 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_13 (st : State) :
  (act Sutra.S5 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_14 (st : State) :
  (act Sutra.S5 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_14 (st : State) :
  (act Sutra.S5 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_15 (st : State) :
  (act Sutra.S5 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_15 (st : State) :
  (act Sutra.S5 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_16 (st : State) :
  (act Sutra.S5 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_16 (st : State) :
  (act Sutra.S5 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_17 (st : State) :
  (act Sutra.S5 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_17 (st : State) :
  (act Sutra.S5 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_18 (st : State) :
  (act Sutra.S5 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_18 (st : State) :
  (act Sutra.S5 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_19 (st : State) :
  (act Sutra.S5 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_19 (st : State) :
  (act Sutra.S5 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_20 (st : State) :
  (act Sutra.S5 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_20 (st : State) :
  (act Sutra.S5 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_21 (st : State) :
  (act Sutra.S5 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_21 (st : State) :
  (act Sutra.S5 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_22 (st : State) :
  (act Sutra.S5 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_22 (st : State) :
  (act Sutra.S5 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_23 (st : State) :
  (act Sutra.S5 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_23 (st : State) :
  (act Sutra.S5 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_24 (st : State) :
  (act Sutra.S5 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_24 (st : State) :
  (act Sutra.S5 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_25 (st : State) :
  (act Sutra.S5 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_25 (st : State) :
  (act Sutra.S5 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_26 (st : State) :
  (act Sutra.S5 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_26 (st : State) :
  (act Sutra.S5 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_27 (st : State) :
  (act Sutra.S5 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_27 (st : State) :
  (act Sutra.S5 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_28 (st : State) :
  (act Sutra.S5 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_28 (st : State) :
  (act Sutra.S5 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_5_29 (st : State) :
  (act Sutra.S5 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S5 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_5_29 (st : State) :
  (act Sutra.S5 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S5 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_1 (st : State) :
  (act Sutra.S6 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_1 (st : State) :
  (act Sutra.S6 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_2 (st : State) :
  (act Sutra.S6 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_2 (st : State) :
  (act Sutra.S6 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_3 (st : State) :
  (act Sutra.S6 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_3 (st : State) :
  (act Sutra.S6 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_4 (st : State) :
  (act Sutra.S6 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_4 (st : State) :
  (act Sutra.S6 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_5 (st : State) :
  (act Sutra.S6 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_5 (st : State) :
  (act Sutra.S6 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_6 (st : State) :
  (act Sutra.S6 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_6 (st : State) :
  (act Sutra.S6 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_7 (st : State) :
  (act Sutra.S6 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_7 (st : State) :
  (act Sutra.S6 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_8 (st : State) :
  (act Sutra.S6 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_8 (st : State) :
  (act Sutra.S6 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_9 (st : State) :
  (act Sutra.S6 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_9 (st : State) :
  (act Sutra.S6 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_10 (st : State) :
  (act Sutra.S6 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_10 (st : State) :
  (act Sutra.S6 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_11 (st : State) :
  (act Sutra.S6 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_11 (st : State) :
  (act Sutra.S6 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_12 (st : State) :
  (act Sutra.S6 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_12 (st : State) :
  (act Sutra.S6 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_13 (st : State) :
  (act Sutra.S6 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_13 (st : State) :
  (act Sutra.S6 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_14 (st : State) :
  (act Sutra.S6 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_14 (st : State) :
  (act Sutra.S6 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_15 (st : State) :
  (act Sutra.S6 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_15 (st : State) :
  (act Sutra.S6 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_16 (st : State) :
  (act Sutra.S6 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_16 (st : State) :
  (act Sutra.S6 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_17 (st : State) :
  (act Sutra.S6 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_17 (st : State) :
  (act Sutra.S6 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_18 (st : State) :
  (act Sutra.S6 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_18 (st : State) :
  (act Sutra.S6 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_19 (st : State) :
  (act Sutra.S6 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_19 (st : State) :
  (act Sutra.S6 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_20 (st : State) :
  (act Sutra.S6 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_20 (st : State) :
  (act Sutra.S6 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_21 (st : State) :
  (act Sutra.S6 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_21 (st : State) :
  (act Sutra.S6 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_22 (st : State) :
  (act Sutra.S6 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_22 (st : State) :
  (act Sutra.S6 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_23 (st : State) :
  (act Sutra.S6 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_23 (st : State) :
  (act Sutra.S6 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_24 (st : State) :
  (act Sutra.S6 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_24 (st : State) :
  (act Sutra.S6 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_25 (st : State) :
  (act Sutra.S6 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_25 (st : State) :
  (act Sutra.S6 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_26 (st : State) :
  (act Sutra.S6 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_26 (st : State) :
  (act Sutra.S6 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_27 (st : State) :
  (act Sutra.S6 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_27 (st : State) :
  (act Sutra.S6 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_28 (st : State) :
  (act Sutra.S6 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_28 (st : State) :
  (act Sutra.S6 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_6_29 (st : State) :
  (act Sutra.S6 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S6 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_6_29 (st : State) :
  (act Sutra.S6 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S6 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_1 (st : State) :
  (act Sutra.S7 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_1 (st : State) :
  (act Sutra.S7 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_2 (st : State) :
  (act Sutra.S7 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_2 (st : State) :
  (act Sutra.S7 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_3 (st : State) :
  (act Sutra.S7 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_3 (st : State) :
  (act Sutra.S7 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_4 (st : State) :
  (act Sutra.S7 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_4 (st : State) :
  (act Sutra.S7 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_5 (st : State) :
  (act Sutra.S7 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_5 (st : State) :
  (act Sutra.S7 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_6 (st : State) :
  (act Sutra.S7 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_6 (st : State) :
  (act Sutra.S7 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_7 (st : State) :
  (act Sutra.S7 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_7 (st : State) :
  (act Sutra.S7 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_8 (st : State) :
  (act Sutra.S7 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_8 (st : State) :
  (act Sutra.S7 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_9 (st : State) :
  (act Sutra.S7 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_9 (st : State) :
  (act Sutra.S7 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_10 (st : State) :
  (act Sutra.S7 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_10 (st : State) :
  (act Sutra.S7 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_11 (st : State) :
  (act Sutra.S7 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_11 (st : State) :
  (act Sutra.S7 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_12 (st : State) :
  (act Sutra.S7 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_12 (st : State) :
  (act Sutra.S7 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_13 (st : State) :
  (act Sutra.S7 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_13 (st : State) :
  (act Sutra.S7 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_14 (st : State) :
  (act Sutra.S7 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_14 (st : State) :
  (act Sutra.S7 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_15 (st : State) :
  (act Sutra.S7 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_15 (st : State) :
  (act Sutra.S7 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_16 (st : State) :
  (act Sutra.S7 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_16 (st : State) :
  (act Sutra.S7 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_17 (st : State) :
  (act Sutra.S7 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_17 (st : State) :
  (act Sutra.S7 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_18 (st : State) :
  (act Sutra.S7 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_18 (st : State) :
  (act Sutra.S7 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_19 (st : State) :
  (act Sutra.S7 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_19 (st : State) :
  (act Sutra.S7 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_20 (st : State) :
  (act Sutra.S7 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_20 (st : State) :
  (act Sutra.S7 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_21 (st : State) :
  (act Sutra.S7 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_21 (st : State) :
  (act Sutra.S7 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_22 (st : State) :
  (act Sutra.S7 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_22 (st : State) :
  (act Sutra.S7 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_23 (st : State) :
  (act Sutra.S7 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_23 (st : State) :
  (act Sutra.S7 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_24 (st : State) :
  (act Sutra.S7 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_24 (st : State) :
  (act Sutra.S7 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_25 (st : State) :
  (act Sutra.S7 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_25 (st : State) :
  (act Sutra.S7 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_26 (st : State) :
  (act Sutra.S7 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_26 (st : State) :
  (act Sutra.S7 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_27 (st : State) :
  (act Sutra.S7 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_27 (st : State) :
  (act Sutra.S7 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_28 (st : State) :
  (act Sutra.S7 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_28 (st : State) :
  (act Sutra.S7 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_7_29 (st : State) :
  (act Sutra.S7 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S7 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_7_29 (st : State) :
  (act Sutra.S7 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S7 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_1 (st : State) :
  (act Sutra.S8 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_1 (st : State) :
  (act Sutra.S8 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_2 (st : State) :
  (act Sutra.S8 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_2 (st : State) :
  (act Sutra.S8 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_3 (st : State) :
  (act Sutra.S8 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_3 (st : State) :
  (act Sutra.S8 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_4 (st : State) :
  (act Sutra.S8 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_4 (st : State) :
  (act Sutra.S8 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_5 (st : State) :
  (act Sutra.S8 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_5 (st : State) :
  (act Sutra.S8 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_6 (st : State) :
  (act Sutra.S8 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_6 (st : State) :
  (act Sutra.S8 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_7 (st : State) :
  (act Sutra.S8 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_7 (st : State) :
  (act Sutra.S8 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_8 (st : State) :
  (act Sutra.S8 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_8 (st : State) :
  (act Sutra.S8 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_9 (st : State) :
  (act Sutra.S8 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_9 (st : State) :
  (act Sutra.S8 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_10 (st : State) :
  (act Sutra.S8 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_10 (st : State) :
  (act Sutra.S8 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_11 (st : State) :
  (act Sutra.S8 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_11 (st : State) :
  (act Sutra.S8 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_12 (st : State) :
  (act Sutra.S8 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_12 (st : State) :
  (act Sutra.S8 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_13 (st : State) :
  (act Sutra.S8 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_13 (st : State) :
  (act Sutra.S8 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_14 (st : State) :
  (act Sutra.S8 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_14 (st : State) :
  (act Sutra.S8 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_15 (st : State) :
  (act Sutra.S8 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_15 (st : State) :
  (act Sutra.S8 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_16 (st : State) :
  (act Sutra.S8 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_16 (st : State) :
  (act Sutra.S8 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_17 (st : State) :
  (act Sutra.S8 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_17 (st : State) :
  (act Sutra.S8 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_18 (st : State) :
  (act Sutra.S8 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_18 (st : State) :
  (act Sutra.S8 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_19 (st : State) :
  (act Sutra.S8 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_19 (st : State) :
  (act Sutra.S8 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_20 (st : State) :
  (act Sutra.S8 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_20 (st : State) :
  (act Sutra.S8 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_21 (st : State) :
  (act Sutra.S8 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_21 (st : State) :
  (act Sutra.S8 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_22 (st : State) :
  (act Sutra.S8 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_22 (st : State) :
  (act Sutra.S8 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_23 (st : State) :
  (act Sutra.S8 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_23 (st : State) :
  (act Sutra.S8 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_24 (st : State) :
  (act Sutra.S8 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_24 (st : State) :
  (act Sutra.S8 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_25 (st : State) :
  (act Sutra.S8 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_25 (st : State) :
  (act Sutra.S8 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_26 (st : State) :
  (act Sutra.S8 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_26 (st : State) :
  (act Sutra.S8 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_27 (st : State) :
  (act Sutra.S8 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_27 (st : State) :
  (act Sutra.S8 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_28 (st : State) :
  (act Sutra.S8 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_28 (st : State) :
  (act Sutra.S8 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_8_29 (st : State) :
  (act Sutra.S8 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S8 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_8_29 (st : State) :
  (act Sutra.S8 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S8 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_1 (st : State) :
  (act Sutra.S9 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_1 (st : State) :
  (act Sutra.S9 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_2 (st : State) :
  (act Sutra.S9 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_2 (st : State) :
  (act Sutra.S9 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_3 (st : State) :
  (act Sutra.S9 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_3 (st : State) :
  (act Sutra.S9 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_4 (st : State) :
  (act Sutra.S9 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_4 (st : State) :
  (act Sutra.S9 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_5 (st : State) :
  (act Sutra.S9 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_5 (st : State) :
  (act Sutra.S9 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_6 (st : State) :
  (act Sutra.S9 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_6 (st : State) :
  (act Sutra.S9 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_7 (st : State) :
  (act Sutra.S9 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_7 (st : State) :
  (act Sutra.S9 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_8 (st : State) :
  (act Sutra.S9 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_8 (st : State) :
  (act Sutra.S9 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_9 (st : State) :
  (act Sutra.S9 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_9 (st : State) :
  (act Sutra.S9 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_10 (st : State) :
  (act Sutra.S9 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_10 (st : State) :
  (act Sutra.S9 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_11 (st : State) :
  (act Sutra.S9 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_11 (st : State) :
  (act Sutra.S9 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_12 (st : State) :
  (act Sutra.S9 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_12 (st : State) :
  (act Sutra.S9 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_13 (st : State) :
  (act Sutra.S9 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_13 (st : State) :
  (act Sutra.S9 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_14 (st : State) :
  (act Sutra.S9 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_14 (st : State) :
  (act Sutra.S9 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_15 (st : State) :
  (act Sutra.S9 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_15 (st : State) :
  (act Sutra.S9 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_16 (st : State) :
  (act Sutra.S9 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_16 (st : State) :
  (act Sutra.S9 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_17 (st : State) :
  (act Sutra.S9 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_17 (st : State) :
  (act Sutra.S9 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_18 (st : State) :
  (act Sutra.S9 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_18 (st : State) :
  (act Sutra.S9 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_19 (st : State) :
  (act Sutra.S9 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_19 (st : State) :
  (act Sutra.S9 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_20 (st : State) :
  (act Sutra.S9 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_20 (st : State) :
  (act Sutra.S9 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_21 (st : State) :
  (act Sutra.S9 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_21 (st : State) :
  (act Sutra.S9 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_22 (st : State) :
  (act Sutra.S9 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_22 (st : State) :
  (act Sutra.S9 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_23 (st : State) :
  (act Sutra.S9 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_23 (st : State) :
  (act Sutra.S9 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_24 (st : State) :
  (act Sutra.S9 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_24 (st : State) :
  (act Sutra.S9 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_25 (st : State) :
  (act Sutra.S9 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_25 (st : State) :
  (act Sutra.S9 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_26 (st : State) :
  (act Sutra.S9 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_26 (st : State) :
  (act Sutra.S9 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_27 (st : State) :
  (act Sutra.S9 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_27 (st : State) :
  (act Sutra.S9 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_28 (st : State) :
  (act Sutra.S9 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_28 (st : State) :
  (act Sutra.S9 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_9_29 (st : State) :
  (act Sutra.S9 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S9 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_9_29 (st : State) :
  (act Sutra.S9 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S9 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_1 (st : State) :
  (act Sutra.S10 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_1 (st : State) :
  (act Sutra.S10 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_2 (st : State) :
  (act Sutra.S10 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_2 (st : State) :
  (act Sutra.S10 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_3 (st : State) :
  (act Sutra.S10 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_3 (st : State) :
  (act Sutra.S10 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_4 (st : State) :
  (act Sutra.S10 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_4 (st : State) :
  (act Sutra.S10 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_5 (st : State) :
  (act Sutra.S10 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_5 (st : State) :
  (act Sutra.S10 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_6 (st : State) :
  (act Sutra.S10 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_6 (st : State) :
  (act Sutra.S10 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_7 (st : State) :
  (act Sutra.S10 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_7 (st : State) :
  (act Sutra.S10 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_8 (st : State) :
  (act Sutra.S10 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_8 (st : State) :
  (act Sutra.S10 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_9 (st : State) :
  (act Sutra.S10 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_9 (st : State) :
  (act Sutra.S10 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_10 (st : State) :
  (act Sutra.S10 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_10 (st : State) :
  (act Sutra.S10 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_11 (st : State) :
  (act Sutra.S10 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_11 (st : State) :
  (act Sutra.S10 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_12 (st : State) :
  (act Sutra.S10 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_12 (st : State) :
  (act Sutra.S10 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_13 (st : State) :
  (act Sutra.S10 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_13 (st : State) :
  (act Sutra.S10 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_14 (st : State) :
  (act Sutra.S10 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_14 (st : State) :
  (act Sutra.S10 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_15 (st : State) :
  (act Sutra.S10 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_15 (st : State) :
  (act Sutra.S10 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_16 (st : State) :
  (act Sutra.S10 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_16 (st : State) :
  (act Sutra.S10 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_17 (st : State) :
  (act Sutra.S10 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_17 (st : State) :
  (act Sutra.S10 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_18 (st : State) :
  (act Sutra.S10 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_18 (st : State) :
  (act Sutra.S10 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_19 (st : State) :
  (act Sutra.S10 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_19 (st : State) :
  (act Sutra.S10 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_20 (st : State) :
  (act Sutra.S10 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_20 (st : State) :
  (act Sutra.S10 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_21 (st : State) :
  (act Sutra.S10 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_21 (st : State) :
  (act Sutra.S10 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_22 (st : State) :
  (act Sutra.S10 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_22 (st : State) :
  (act Sutra.S10 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_23 (st : State) :
  (act Sutra.S10 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_23 (st : State) :
  (act Sutra.S10 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_24 (st : State) :
  (act Sutra.S10 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_24 (st : State) :
  (act Sutra.S10 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_25 (st : State) :
  (act Sutra.S10 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_25 (st : State) :
  (act Sutra.S10 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_26 (st : State) :
  (act Sutra.S10 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_26 (st : State) :
  (act Sutra.S10 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_27 (st : State) :
  (act Sutra.S10 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_27 (st : State) :
  (act Sutra.S10 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_28 (st : State) :
  (act Sutra.S10 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_28 (st : State) :
  (act Sutra.S10 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_10_29 (st : State) :
  (act Sutra.S10 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S10 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_10_29 (st : State) :
  (act Sutra.S10 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S10 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_1 (st : State) :
  (act Sutra.S11 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_1 (st : State) :
  (act Sutra.S11 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_2 (st : State) :
  (act Sutra.S11 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_2 (st : State) :
  (act Sutra.S11 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_3 (st : State) :
  (act Sutra.S11 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_3 (st : State) :
  (act Sutra.S11 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_4 (st : State) :
  (act Sutra.S11 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_4 (st : State) :
  (act Sutra.S11 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_5 (st : State) :
  (act Sutra.S11 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_5 (st : State) :
  (act Sutra.S11 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_6 (st : State) :
  (act Sutra.S11 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_6 (st : State) :
  (act Sutra.S11 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_7 (st : State) :
  (act Sutra.S11 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_7 (st : State) :
  (act Sutra.S11 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_8 (st : State) :
  (act Sutra.S11 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_8 (st : State) :
  (act Sutra.S11 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_9 (st : State) :
  (act Sutra.S11 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_9 (st : State) :
  (act Sutra.S11 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_10 (st : State) :
  (act Sutra.S11 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_10 (st : State) :
  (act Sutra.S11 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_11 (st : State) :
  (act Sutra.S11 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_11 (st : State) :
  (act Sutra.S11 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_12 (st : State) :
  (act Sutra.S11 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_12 (st : State) :
  (act Sutra.S11 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_13 (st : State) :
  (act Sutra.S11 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_13 (st : State) :
  (act Sutra.S11 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_14 (st : State) :
  (act Sutra.S11 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_14 (st : State) :
  (act Sutra.S11 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_15 (st : State) :
  (act Sutra.S11 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_15 (st : State) :
  (act Sutra.S11 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_16 (st : State) :
  (act Sutra.S11 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_16 (st : State) :
  (act Sutra.S11 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_17 (st : State) :
  (act Sutra.S11 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_17 (st : State) :
  (act Sutra.S11 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_18 (st : State) :
  (act Sutra.S11 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_18 (st : State) :
  (act Sutra.S11 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_19 (st : State) :
  (act Sutra.S11 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_19 (st : State) :
  (act Sutra.S11 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_20 (st : State) :
  (act Sutra.S11 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_20 (st : State) :
  (act Sutra.S11 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_21 (st : State) :
  (act Sutra.S11 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_21 (st : State) :
  (act Sutra.S11 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_22 (st : State) :
  (act Sutra.S11 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_22 (st : State) :
  (act Sutra.S11 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_23 (st : State) :
  (act Sutra.S11 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_23 (st : State) :
  (act Sutra.S11 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_24 (st : State) :
  (act Sutra.S11 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_24 (st : State) :
  (act Sutra.S11 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_25 (st : State) :
  (act Sutra.S11 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_25 (st : State) :
  (act Sutra.S11 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_26 (st : State) :
  (act Sutra.S11 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_26 (st : State) :
  (act Sutra.S11 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_27 (st : State) :
  (act Sutra.S11 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_27 (st : State) :
  (act Sutra.S11 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_28 (st : State) :
  (act Sutra.S11 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_28 (st : State) :
  (act Sutra.S11 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_11_29 (st : State) :
  (act Sutra.S11 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S11 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_11_29 (st : State) :
  (act Sutra.S11 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S11 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_1 (st : State) :
  (act Sutra.S12 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_1 (st : State) :
  (act Sutra.S12 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_2 (st : State) :
  (act Sutra.S12 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_2 (st : State) :
  (act Sutra.S12 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_3 (st : State) :
  (act Sutra.S12 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_3 (st : State) :
  (act Sutra.S12 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_4 (st : State) :
  (act Sutra.S12 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_4 (st : State) :
  (act Sutra.S12 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_5 (st : State) :
  (act Sutra.S12 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_5 (st : State) :
  (act Sutra.S12 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_6 (st : State) :
  (act Sutra.S12 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_6 (st : State) :
  (act Sutra.S12 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_7 (st : State) :
  (act Sutra.S12 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_7 (st : State) :
  (act Sutra.S12 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_8 (st : State) :
  (act Sutra.S12 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_8 (st : State) :
  (act Sutra.S12 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_9 (st : State) :
  (act Sutra.S12 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_9 (st : State) :
  (act Sutra.S12 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_10 (st : State) :
  (act Sutra.S12 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_10 (st : State) :
  (act Sutra.S12 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_11 (st : State) :
  (act Sutra.S12 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_11 (st : State) :
  (act Sutra.S12 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_12 (st : State) :
  (act Sutra.S12 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_12 (st : State) :
  (act Sutra.S12 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_13 (st : State) :
  (act Sutra.S12 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_13 (st : State) :
  (act Sutra.S12 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_14 (st : State) :
  (act Sutra.S12 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_14 (st : State) :
  (act Sutra.S12 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_15 (st : State) :
  (act Sutra.S12 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_15 (st : State) :
  (act Sutra.S12 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_16 (st : State) :
  (act Sutra.S12 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_16 (st : State) :
  (act Sutra.S12 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_17 (st : State) :
  (act Sutra.S12 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_17 (st : State) :
  (act Sutra.S12 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_18 (st : State) :
  (act Sutra.S12 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_18 (st : State) :
  (act Sutra.S12 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_19 (st : State) :
  (act Sutra.S12 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_19 (st : State) :
  (act Sutra.S12 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_20 (st : State) :
  (act Sutra.S12 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_20 (st : State) :
  (act Sutra.S12 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_21 (st : State) :
  (act Sutra.S12 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_21 (st : State) :
  (act Sutra.S12 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_22 (st : State) :
  (act Sutra.S12 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_22 (st : State) :
  (act Sutra.S12 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_23 (st : State) :
  (act Sutra.S12 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_23 (st : State) :
  (act Sutra.S12 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_24 (st : State) :
  (act Sutra.S12 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_24 (st : State) :
  (act Sutra.S12 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_25 (st : State) :
  (act Sutra.S12 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_25 (st : State) :
  (act Sutra.S12 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_26 (st : State) :
  (act Sutra.S12 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_26 (st : State) :
  (act Sutra.S12 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_27 (st : State) :
  (act Sutra.S12 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_27 (st : State) :
  (act Sutra.S12 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_28 (st : State) :
  (act Sutra.S12 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_28 (st : State) :
  (act Sutra.S12 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_12_29 (st : State) :
  (act Sutra.S12 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S12 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_12_29 (st : State) :
  (act Sutra.S12 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S12 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_1 (st : State) :
  (act Sutra.S13 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_1 (st : State) :
  (act Sutra.S13 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_2 (st : State) :
  (act Sutra.S13 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_2 (st : State) :
  (act Sutra.S13 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_3 (st : State) :
  (act Sutra.S13 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_3 (st : State) :
  (act Sutra.S13 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_4 (st : State) :
  (act Sutra.S13 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_4 (st : State) :
  (act Sutra.S13 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_5 (st : State) :
  (act Sutra.S13 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_5 (st : State) :
  (act Sutra.S13 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_6 (st : State) :
  (act Sutra.S13 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_6 (st : State) :
  (act Sutra.S13 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_7 (st : State) :
  (act Sutra.S13 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_7 (st : State) :
  (act Sutra.S13 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_8 (st : State) :
  (act Sutra.S13 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_8 (st : State) :
  (act Sutra.S13 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_9 (st : State) :
  (act Sutra.S13 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_9 (st : State) :
  (act Sutra.S13 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_10 (st : State) :
  (act Sutra.S13 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_10 (st : State) :
  (act Sutra.S13 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_11 (st : State) :
  (act Sutra.S13 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_11 (st : State) :
  (act Sutra.S13 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_12 (st : State) :
  (act Sutra.S13 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_12 (st : State) :
  (act Sutra.S13 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_13 (st : State) :
  (act Sutra.S13 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_13 (st : State) :
  (act Sutra.S13 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_14 (st : State) :
  (act Sutra.S13 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_14 (st : State) :
  (act Sutra.S13 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_15 (st : State) :
  (act Sutra.S13 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_15 (st : State) :
  (act Sutra.S13 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_16 (st : State) :
  (act Sutra.S13 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_16 (st : State) :
  (act Sutra.S13 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_17 (st : State) :
  (act Sutra.S13 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_17 (st : State) :
  (act Sutra.S13 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_18 (st : State) :
  (act Sutra.S13 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_18 (st : State) :
  (act Sutra.S13 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_19 (st : State) :
  (act Sutra.S13 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_19 (st : State) :
  (act Sutra.S13 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_20 (st : State) :
  (act Sutra.S13 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_20 (st : State) :
  (act Sutra.S13 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_21 (st : State) :
  (act Sutra.S13 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_21 (st : State) :
  (act Sutra.S13 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_22 (st : State) :
  (act Sutra.S13 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_22 (st : State) :
  (act Sutra.S13 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_23 (st : State) :
  (act Sutra.S13 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_23 (st : State) :
  (act Sutra.S13 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_24 (st : State) :
  (act Sutra.S13 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_24 (st : State) :
  (act Sutra.S13 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_25 (st : State) :
  (act Sutra.S13 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_25 (st : State) :
  (act Sutra.S13 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_26 (st : State) :
  (act Sutra.S13 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_26 (st : State) :
  (act Sutra.S13 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_27 (st : State) :
  (act Sutra.S13 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_27 (st : State) :
  (act Sutra.S13 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_28 (st : State) :
  (act Sutra.S13 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_28 (st : State) :
  (act Sutra.S13 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_13_29 (st : State) :
  (act Sutra.S13 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S13 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_13_29 (st : State) :
  (act Sutra.S13 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S13 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_1 (st : State) :
  (act Sutra.S14 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_1 (st : State) :
  (act Sutra.S14 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_2 (st : State) :
  (act Sutra.S14 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_2 (st : State) :
  (act Sutra.S14 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_3 (st : State) :
  (act Sutra.S14 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_3 (st : State) :
  (act Sutra.S14 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_4 (st : State) :
  (act Sutra.S14 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_4 (st : State) :
  (act Sutra.S14 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_5 (st : State) :
  (act Sutra.S14 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_5 (st : State) :
  (act Sutra.S14 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_6 (st : State) :
  (act Sutra.S14 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_6 (st : State) :
  (act Sutra.S14 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_7 (st : State) :
  (act Sutra.S14 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_7 (st : State) :
  (act Sutra.S14 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_8 (st : State) :
  (act Sutra.S14 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_8 (st : State) :
  (act Sutra.S14 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_9 (st : State) :
  (act Sutra.S14 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_9 (st : State) :
  (act Sutra.S14 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_10 (st : State) :
  (act Sutra.S14 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_10 (st : State) :
  (act Sutra.S14 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_11 (st : State) :
  (act Sutra.S14 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_11 (st : State) :
  (act Sutra.S14 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_12 (st : State) :
  (act Sutra.S14 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_12 (st : State) :
  (act Sutra.S14 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_13 (st : State) :
  (act Sutra.S14 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_13 (st : State) :
  (act Sutra.S14 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_14 (st : State) :
  (act Sutra.S14 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_14 (st : State) :
  (act Sutra.S14 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_15 (st : State) :
  (act Sutra.S14 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_15 (st : State) :
  (act Sutra.S14 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_16 (st : State) :
  (act Sutra.S14 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_16 (st : State) :
  (act Sutra.S14 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_17 (st : State) :
  (act Sutra.S14 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_17 (st : State) :
  (act Sutra.S14 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_18 (st : State) :
  (act Sutra.S14 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_18 (st : State) :
  (act Sutra.S14 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_19 (st : State) :
  (act Sutra.S14 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_19 (st : State) :
  (act Sutra.S14 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_20 (st : State) :
  (act Sutra.S14 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_20 (st : State) :
  (act Sutra.S14 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_21 (st : State) :
  (act Sutra.S14 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_21 (st : State) :
  (act Sutra.S14 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_22 (st : State) :
  (act Sutra.S14 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_22 (st : State) :
  (act Sutra.S14 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_23 (st : State) :
  (act Sutra.S14 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_23 (st : State) :
  (act Sutra.S14 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_24 (st : State) :
  (act Sutra.S14 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_24 (st : State) :
  (act Sutra.S14 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_25 (st : State) :
  (act Sutra.S14 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_25 (st : State) :
  (act Sutra.S14 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_26 (st : State) :
  (act Sutra.S14 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_26 (st : State) :
  (act Sutra.S14 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_27 (st : State) :
  (act Sutra.S14 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_27 (st : State) :
  (act Sutra.S14 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_28 (st : State) :
  (act Sutra.S14 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_28 (st : State) :
  (act Sutra.S14 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_14_29 (st : State) :
  (act Sutra.S14 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S14 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_14_29 (st : State) :
  (act Sutra.S14 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S14 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_1 (st : State) :
  (act Sutra.S15 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_1 (st : State) :
  (act Sutra.S15 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_2 (st : State) :
  (act Sutra.S15 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_2 (st : State) :
  (act Sutra.S15 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_3 (st : State) :
  (act Sutra.S15 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_3 (st : State) :
  (act Sutra.S15 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_4 (st : State) :
  (act Sutra.S15 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_4 (st : State) :
  (act Sutra.S15 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_5 (st : State) :
  (act Sutra.S15 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_5 (st : State) :
  (act Sutra.S15 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_6 (st : State) :
  (act Sutra.S15 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_6 (st : State) :
  (act Sutra.S15 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_7 (st : State) :
  (act Sutra.S15 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_7 (st : State) :
  (act Sutra.S15 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_8 (st : State) :
  (act Sutra.S15 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_8 (st : State) :
  (act Sutra.S15 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_9 (st : State) :
  (act Sutra.S15 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_9 (st : State) :
  (act Sutra.S15 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_10 (st : State) :
  (act Sutra.S15 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_10 (st : State) :
  (act Sutra.S15 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_11 (st : State) :
  (act Sutra.S15 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_11 (st : State) :
  (act Sutra.S15 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_12 (st : State) :
  (act Sutra.S15 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_12 (st : State) :
  (act Sutra.S15 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_13 (st : State) :
  (act Sutra.S15 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_13 (st : State) :
  (act Sutra.S15 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_14 (st : State) :
  (act Sutra.S15 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_14 (st : State) :
  (act Sutra.S15 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_15 (st : State) :
  (act Sutra.S15 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_15 (st : State) :
  (act Sutra.S15 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_16 (st : State) :
  (act Sutra.S15 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_16 (st : State) :
  (act Sutra.S15 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_17 (st : State) :
  (act Sutra.S15 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_17 (st : State) :
  (act Sutra.S15 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_18 (st : State) :
  (act Sutra.S15 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_18 (st : State) :
  (act Sutra.S15 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_19 (st : State) :
  (act Sutra.S15 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_19 (st : State) :
  (act Sutra.S15 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_20 (st : State) :
  (act Sutra.S15 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_20 (st : State) :
  (act Sutra.S15 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_21 (st : State) :
  (act Sutra.S15 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_21 (st : State) :
  (act Sutra.S15 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_22 (st : State) :
  (act Sutra.S15 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_22 (st : State) :
  (act Sutra.S15 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_23 (st : State) :
  (act Sutra.S15 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_23 (st : State) :
  (act Sutra.S15 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_24 (st : State) :
  (act Sutra.S15 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_24 (st : State) :
  (act Sutra.S15 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_25 (st : State) :
  (act Sutra.S15 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_25 (st : State) :
  (act Sutra.S15 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_26 (st : State) :
  (act Sutra.S15 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_26 (st : State) :
  (act Sutra.S15 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_27 (st : State) :
  (act Sutra.S15 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_27 (st : State) :
  (act Sutra.S15 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_28 (st : State) :
  (act Sutra.S15 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_28 (st : State) :
  (act Sutra.S15 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_15_29 (st : State) :
  (act Sutra.S15 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S15 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_15_29 (st : State) :
  (act Sutra.S15 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S15 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_1 (st : State) :
  (act Sutra.S16 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_1 (st : State) :
  (act Sutra.S16 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_2 (st : State) :
  (act Sutra.S16 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_2 (st : State) :
  (act Sutra.S16 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_3 (st : State) :
  (act Sutra.S16 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_3 (st : State) :
  (act Sutra.S16 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_4 (st : State) :
  (act Sutra.S16 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_4 (st : State) :
  (act Sutra.S16 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_5 (st : State) :
  (act Sutra.S16 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_5 (st : State) :
  (act Sutra.S16 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_6 (st : State) :
  (act Sutra.S16 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_6 (st : State) :
  (act Sutra.S16 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_7 (st : State) :
  (act Sutra.S16 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_7 (st : State) :
  (act Sutra.S16 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_8 (st : State) :
  (act Sutra.S16 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_8 (st : State) :
  (act Sutra.S16 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_9 (st : State) :
  (act Sutra.S16 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_9 (st : State) :
  (act Sutra.S16 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_10 (st : State) :
  (act Sutra.S16 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_10 (st : State) :
  (act Sutra.S16 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_11 (st : State) :
  (act Sutra.S16 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_11 (st : State) :
  (act Sutra.S16 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_12 (st : State) :
  (act Sutra.S16 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_12 (st : State) :
  (act Sutra.S16 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_13 (st : State) :
  (act Sutra.S16 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_13 (st : State) :
  (act Sutra.S16 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_14 (st : State) :
  (act Sutra.S16 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_14 (st : State) :
  (act Sutra.S16 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_15 (st : State) :
  (act Sutra.S16 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_15 (st : State) :
  (act Sutra.S16 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_16 (st : State) :
  (act Sutra.S16 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_16 (st : State) :
  (act Sutra.S16 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_17 (st : State) :
  (act Sutra.S16 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_17 (st : State) :
  (act Sutra.S16 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_18 (st : State) :
  (act Sutra.S16 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_18 (st : State) :
  (act Sutra.S16 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_19 (st : State) :
  (act Sutra.S16 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_19 (st : State) :
  (act Sutra.S16 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_20 (st : State) :
  (act Sutra.S16 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_20 (st : State) :
  (act Sutra.S16 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_21 (st : State) :
  (act Sutra.S16 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_21 (st : State) :
  (act Sutra.S16 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_22 (st : State) :
  (act Sutra.S16 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_22 (st : State) :
  (act Sutra.S16 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_23 (st : State) :
  (act Sutra.S16 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_23 (st : State) :
  (act Sutra.S16 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_24 (st : State) :
  (act Sutra.S16 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_24 (st : State) :
  (act Sutra.S16 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_25 (st : State) :
  (act Sutra.S16 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_25 (st : State) :
  (act Sutra.S16 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_26 (st : State) :
  (act Sutra.S16 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_26 (st : State) :
  (act Sutra.S16 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_27 (st : State) :
  (act Sutra.S16 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_27 (st : State) :
  (act Sutra.S16 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_28 (st : State) :
  (act Sutra.S16 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_28 (st : State) :
  (act Sutra.S16 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_16_29 (st : State) :
  (act Sutra.S16 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S16 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_16_29 (st : State) :
  (act Sutra.S16 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S16 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_1 (st : State) :
  (act Sutra.S17 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_1 (st : State) :
  (act Sutra.S17 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_2 (st : State) :
  (act Sutra.S17 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_2 (st : State) :
  (act Sutra.S17 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_3 (st : State) :
  (act Sutra.S17 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_3 (st : State) :
  (act Sutra.S17 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_4 (st : State) :
  (act Sutra.S17 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_4 (st : State) :
  (act Sutra.S17 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_5 (st : State) :
  (act Sutra.S17 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_5 (st : State) :
  (act Sutra.S17 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_6 (st : State) :
  (act Sutra.S17 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_6 (st : State) :
  (act Sutra.S17 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_7 (st : State) :
  (act Sutra.S17 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_7 (st : State) :
  (act Sutra.S17 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_8 (st : State) :
  (act Sutra.S17 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_8 (st : State) :
  (act Sutra.S17 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_9 (st : State) :
  (act Sutra.S17 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_9 (st : State) :
  (act Sutra.S17 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_10 (st : State) :
  (act Sutra.S17 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_10 (st : State) :
  (act Sutra.S17 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_11 (st : State) :
  (act Sutra.S17 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_11 (st : State) :
  (act Sutra.S17 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_12 (st : State) :
  (act Sutra.S17 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_12 (st : State) :
  (act Sutra.S17 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_13 (st : State) :
  (act Sutra.S17 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_13 (st : State) :
  (act Sutra.S17 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_14 (st : State) :
  (act Sutra.S17 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_14 (st : State) :
  (act Sutra.S17 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_15 (st : State) :
  (act Sutra.S17 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_15 (st : State) :
  (act Sutra.S17 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_16 (st : State) :
  (act Sutra.S17 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_16 (st : State) :
  (act Sutra.S17 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_17 (st : State) :
  (act Sutra.S17 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_17 (st : State) :
  (act Sutra.S17 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_18 (st : State) :
  (act Sutra.S17 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_18 (st : State) :
  (act Sutra.S17 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_19 (st : State) :
  (act Sutra.S17 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_19 (st : State) :
  (act Sutra.S17 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_20 (st : State) :
  (act Sutra.S17 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_20 (st : State) :
  (act Sutra.S17 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_21 (st : State) :
  (act Sutra.S17 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_21 (st : State) :
  (act Sutra.S17 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_22 (st : State) :
  (act Sutra.S17 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_22 (st : State) :
  (act Sutra.S17 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_23 (st : State) :
  (act Sutra.S17 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_23 (st : State) :
  (act Sutra.S17 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_24 (st : State) :
  (act Sutra.S17 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_24 (st : State) :
  (act Sutra.S17 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_25 (st : State) :
  (act Sutra.S17 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_25 (st : State) :
  (act Sutra.S17 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_26 (st : State) :
  (act Sutra.S17 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_26 (st : State) :
  (act Sutra.S17 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_27 (st : State) :
  (act Sutra.S17 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_27 (st : State) :
  (act Sutra.S17 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_28 (st : State) :
  (act Sutra.S17 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_28 (st : State) :
  (act Sutra.S17 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_17_29 (st : State) :
  (act Sutra.S17 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S17 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_17_29 (st : State) :
  (act Sutra.S17 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S17 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_1 (st : State) :
  (act Sutra.S18 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_1 (st : State) :
  (act Sutra.S18 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_2 (st : State) :
  (act Sutra.S18 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_2 (st : State) :
  (act Sutra.S18 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_3 (st : State) :
  (act Sutra.S18 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_3 (st : State) :
  (act Sutra.S18 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_4 (st : State) :
  (act Sutra.S18 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_4 (st : State) :
  (act Sutra.S18 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_5 (st : State) :
  (act Sutra.S18 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_5 (st : State) :
  (act Sutra.S18 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_6 (st : State) :
  (act Sutra.S18 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_6 (st : State) :
  (act Sutra.S18 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_7 (st : State) :
  (act Sutra.S18 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_7 (st : State) :
  (act Sutra.S18 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_8 (st : State) :
  (act Sutra.S18 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_8 (st : State) :
  (act Sutra.S18 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_9 (st : State) :
  (act Sutra.S18 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_9 (st : State) :
  (act Sutra.S18 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_10 (st : State) :
  (act Sutra.S18 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_10 (st : State) :
  (act Sutra.S18 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_11 (st : State) :
  (act Sutra.S18 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_11 (st : State) :
  (act Sutra.S18 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_12 (st : State) :
  (act Sutra.S18 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_12 (st : State) :
  (act Sutra.S18 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_13 (st : State) :
  (act Sutra.S18 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_13 (st : State) :
  (act Sutra.S18 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_14 (st : State) :
  (act Sutra.S18 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_14 (st : State) :
  (act Sutra.S18 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_15 (st : State) :
  (act Sutra.S18 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_15 (st : State) :
  (act Sutra.S18 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_16 (st : State) :
  (act Sutra.S18 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_16 (st : State) :
  (act Sutra.S18 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_17 (st : State) :
  (act Sutra.S18 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_17 (st : State) :
  (act Sutra.S18 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_18 (st : State) :
  (act Sutra.S18 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_18 (st : State) :
  (act Sutra.S18 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_19 (st : State) :
  (act Sutra.S18 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_19 (st : State) :
  (act Sutra.S18 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_20 (st : State) :
  (act Sutra.S18 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_20 (st : State) :
  (act Sutra.S18 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_21 (st : State) :
  (act Sutra.S18 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_21 (st : State) :
  (act Sutra.S18 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_22 (st : State) :
  (act Sutra.S18 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_22 (st : State) :
  (act Sutra.S18 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_23 (st : State) :
  (act Sutra.S18 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_23 (st : State) :
  (act Sutra.S18 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_24 (st : State) :
  (act Sutra.S18 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_24 (st : State) :
  (act Sutra.S18 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_25 (st : State) :
  (act Sutra.S18 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_25 (st : State) :
  (act Sutra.S18 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_26 (st : State) :
  (act Sutra.S18 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_26 (st : State) :
  (act Sutra.S18 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_27 (st : State) :
  (act Sutra.S18 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_27 (st : State) :
  (act Sutra.S18 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_28 (st : State) :
  (act Sutra.S18 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_28 (st : State) :
  (act Sutra.S18 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_18_29 (st : State) :
  (act Sutra.S18 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S18 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_18_29 (st : State) :
  (act Sutra.S18 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S18 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_1 (st : State) :
  (act Sutra.S19 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_1 (st : State) :
  (act Sutra.S19 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_2 (st : State) :
  (act Sutra.S19 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_2 (st : State) :
  (act Sutra.S19 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_3 (st : State) :
  (act Sutra.S19 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_3 (st : State) :
  (act Sutra.S19 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_4 (st : State) :
  (act Sutra.S19 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_4 (st : State) :
  (act Sutra.S19 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_5 (st : State) :
  (act Sutra.S19 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_5 (st : State) :
  (act Sutra.S19 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_6 (st : State) :
  (act Sutra.S19 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_6 (st : State) :
  (act Sutra.S19 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_7 (st : State) :
  (act Sutra.S19 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_7 (st : State) :
  (act Sutra.S19 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_8 (st : State) :
  (act Sutra.S19 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_8 (st : State) :
  (act Sutra.S19 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_9 (st : State) :
  (act Sutra.S19 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_9 (st : State) :
  (act Sutra.S19 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_10 (st : State) :
  (act Sutra.S19 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_10 (st : State) :
  (act Sutra.S19 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_11 (st : State) :
  (act Sutra.S19 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_11 (st : State) :
  (act Sutra.S19 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_12 (st : State) :
  (act Sutra.S19 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_12 (st : State) :
  (act Sutra.S19 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_13 (st : State) :
  (act Sutra.S19 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_13 (st : State) :
  (act Sutra.S19 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_14 (st : State) :
  (act Sutra.S19 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_14 (st : State) :
  (act Sutra.S19 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_15 (st : State) :
  (act Sutra.S19 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_15 (st : State) :
  (act Sutra.S19 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_16 (st : State) :
  (act Sutra.S19 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_16 (st : State) :
  (act Sutra.S19 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_17 (st : State) :
  (act Sutra.S19 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_17 (st : State) :
  (act Sutra.S19 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_18 (st : State) :
  (act Sutra.S19 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_18 (st : State) :
  (act Sutra.S19 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_19 (st : State) :
  (act Sutra.S19 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_19 (st : State) :
  (act Sutra.S19 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_20 (st : State) :
  (act Sutra.S19 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_20 (st : State) :
  (act Sutra.S19 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_21 (st : State) :
  (act Sutra.S19 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_21 (st : State) :
  (act Sutra.S19 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_22 (st : State) :
  (act Sutra.S19 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_22 (st : State) :
  (act Sutra.S19 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_23 (st : State) :
  (act Sutra.S19 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_23 (st : State) :
  (act Sutra.S19 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_24 (st : State) :
  (act Sutra.S19 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_24 (st : State) :
  (act Sutra.S19 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_25 (st : State) :
  (act Sutra.S19 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_25 (st : State) :
  (act Sutra.S19 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_26 (st : State) :
  (act Sutra.S19 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_26 (st : State) :
  (act Sutra.S19 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_27 (st : State) :
  (act Sutra.S19 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_27 (st : State) :
  (act Sutra.S19 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_28 (st : State) :
  (act Sutra.S19 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_28 (st : State) :
  (act Sutra.S19 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_19_29 (st : State) :
  (act Sutra.S19 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S19 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_19_29 (st : State) :
  (act Sutra.S19 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S19 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_1 (st : State) :
  (act Sutra.S20 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_1 (st : State) :
  (act Sutra.S20 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_2 (st : State) :
  (act Sutra.S20 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_2 (st : State) :
  (act Sutra.S20 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_3 (st : State) :
  (act Sutra.S20 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_3 (st : State) :
  (act Sutra.S20 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_4 (st : State) :
  (act Sutra.S20 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_4 (st : State) :
  (act Sutra.S20 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_5 (st : State) :
  (act Sutra.S20 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_5 (st : State) :
  (act Sutra.S20 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_6 (st : State) :
  (act Sutra.S20 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_6 (st : State) :
  (act Sutra.S20 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_7 (st : State) :
  (act Sutra.S20 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_7 (st : State) :
  (act Sutra.S20 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_8 (st : State) :
  (act Sutra.S20 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_8 (st : State) :
  (act Sutra.S20 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_9 (st : State) :
  (act Sutra.S20 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_9 (st : State) :
  (act Sutra.S20 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_10 (st : State) :
  (act Sutra.S20 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_10 (st : State) :
  (act Sutra.S20 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_11 (st : State) :
  (act Sutra.S20 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_11 (st : State) :
  (act Sutra.S20 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_12 (st : State) :
  (act Sutra.S20 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_12 (st : State) :
  (act Sutra.S20 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_13 (st : State) :
  (act Sutra.S20 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_13 (st : State) :
  (act Sutra.S20 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_14 (st : State) :
  (act Sutra.S20 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_14 (st : State) :
  (act Sutra.S20 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_15 (st : State) :
  (act Sutra.S20 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_15 (st : State) :
  (act Sutra.S20 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_16 (st : State) :
  (act Sutra.S20 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_16 (st : State) :
  (act Sutra.S20 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_17 (st : State) :
  (act Sutra.S20 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_17 (st : State) :
  (act Sutra.S20 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_18 (st : State) :
  (act Sutra.S20 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_18 (st : State) :
  (act Sutra.S20 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_19 (st : State) :
  (act Sutra.S20 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_19 (st : State) :
  (act Sutra.S20 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_20 (st : State) :
  (act Sutra.S20 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_20 (st : State) :
  (act Sutra.S20 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_21 (st : State) :
  (act Sutra.S20 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_21 (st : State) :
  (act Sutra.S20 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_22 (st : State) :
  (act Sutra.S20 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_22 (st : State) :
  (act Sutra.S20 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_23 (st : State) :
  (act Sutra.S20 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_23 (st : State) :
  (act Sutra.S20 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_24 (st : State) :
  (act Sutra.S20 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_24 (st : State) :
  (act Sutra.S20 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_25 (st : State) :
  (act Sutra.S20 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_25 (st : State) :
  (act Sutra.S20 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_26 (st : State) :
  (act Sutra.S20 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_26 (st : State) :
  (act Sutra.S20 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_27 (st : State) :
  (act Sutra.S20 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_27 (st : State) :
  (act Sutra.S20 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_28 (st : State) :
  (act Sutra.S20 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_28 (st : State) :
  (act Sutra.S20 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_20_29 (st : State) :
  (act Sutra.S20 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S20 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_20_29 (st : State) :
  (act Sutra.S20 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S20 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_1 (st : State) :
  (act Sutra.S21 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_1 (st : State) :
  (act Sutra.S21 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_2 (st : State) :
  (act Sutra.S21 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_2 (st : State) :
  (act Sutra.S21 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_3 (st : State) :
  (act Sutra.S21 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_3 (st : State) :
  (act Sutra.S21 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_4 (st : State) :
  (act Sutra.S21 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_4 (st : State) :
  (act Sutra.S21 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_5 (st : State) :
  (act Sutra.S21 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_5 (st : State) :
  (act Sutra.S21 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_6 (st : State) :
  (act Sutra.S21 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_6 (st : State) :
  (act Sutra.S21 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_7 (st : State) :
  (act Sutra.S21 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_7 (st : State) :
  (act Sutra.S21 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_8 (st : State) :
  (act Sutra.S21 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_8 (st : State) :
  (act Sutra.S21 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_9 (st : State) :
  (act Sutra.S21 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_9 (st : State) :
  (act Sutra.S21 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_10 (st : State) :
  (act Sutra.S21 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_10 (st : State) :
  (act Sutra.S21 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_11 (st : State) :
  (act Sutra.S21 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_11 (st : State) :
  (act Sutra.S21 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_12 (st : State) :
  (act Sutra.S21 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_12 (st : State) :
  (act Sutra.S21 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_13 (st : State) :
  (act Sutra.S21 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_13 (st : State) :
  (act Sutra.S21 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_14 (st : State) :
  (act Sutra.S21 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_14 (st : State) :
  (act Sutra.S21 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_15 (st : State) :
  (act Sutra.S21 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_15 (st : State) :
  (act Sutra.S21 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_16 (st : State) :
  (act Sutra.S21 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_16 (st : State) :
  (act Sutra.S21 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_17 (st : State) :
  (act Sutra.S21 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_17 (st : State) :
  (act Sutra.S21 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_18 (st : State) :
  (act Sutra.S21 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_18 (st : State) :
  (act Sutra.S21 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_19 (st : State) :
  (act Sutra.S21 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_19 (st : State) :
  (act Sutra.S21 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_20 (st : State) :
  (act Sutra.S21 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_20 (st : State) :
  (act Sutra.S21 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_21 (st : State) :
  (act Sutra.S21 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_21 (st : State) :
  (act Sutra.S21 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_22 (st : State) :
  (act Sutra.S21 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_22 (st : State) :
  (act Sutra.S21 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_23 (st : State) :
  (act Sutra.S21 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_23 (st : State) :
  (act Sutra.S21 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_24 (st : State) :
  (act Sutra.S21 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_24 (st : State) :
  (act Sutra.S21 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_25 (st : State) :
  (act Sutra.S21 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_25 (st : State) :
  (act Sutra.S21 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_26 (st : State) :
  (act Sutra.S21 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_26 (st : State) :
  (act Sutra.S21 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_27 (st : State) :
  (act Sutra.S21 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_27 (st : State) :
  (act Sutra.S21 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_28 (st : State) :
  (act Sutra.S21 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_28 (st : State) :
  (act Sutra.S21 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_21_29 (st : State) :
  (act Sutra.S21 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S21 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_21_29 (st : State) :
  (act Sutra.S21 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S21 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_1 (st : State) :
  (act Sutra.S22 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_1 (st : State) :
  (act Sutra.S22 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_2 (st : State) :
  (act Sutra.S22 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_2 (st : State) :
  (act Sutra.S22 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_3 (st : State) :
  (act Sutra.S22 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_3 (st : State) :
  (act Sutra.S22 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_4 (st : State) :
  (act Sutra.S22 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_4 (st : State) :
  (act Sutra.S22 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_5 (st : State) :
  (act Sutra.S22 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_5 (st : State) :
  (act Sutra.S22 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_6 (st : State) :
  (act Sutra.S22 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_6 (st : State) :
  (act Sutra.S22 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_7 (st : State) :
  (act Sutra.S22 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_7 (st : State) :
  (act Sutra.S22 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_8 (st : State) :
  (act Sutra.S22 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_8 (st : State) :
  (act Sutra.S22 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_9 (st : State) :
  (act Sutra.S22 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_9 (st : State) :
  (act Sutra.S22 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_10 (st : State) :
  (act Sutra.S22 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_10 (st : State) :
  (act Sutra.S22 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_11 (st : State) :
  (act Sutra.S22 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_11 (st : State) :
  (act Sutra.S22 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_12 (st : State) :
  (act Sutra.S22 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_12 (st : State) :
  (act Sutra.S22 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_13 (st : State) :
  (act Sutra.S22 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_13 (st : State) :
  (act Sutra.S22 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_14 (st : State) :
  (act Sutra.S22 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_14 (st : State) :
  (act Sutra.S22 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_15 (st : State) :
  (act Sutra.S22 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_15 (st : State) :
  (act Sutra.S22 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_16 (st : State) :
  (act Sutra.S22 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_16 (st : State) :
  (act Sutra.S22 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_17 (st : State) :
  (act Sutra.S22 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_17 (st : State) :
  (act Sutra.S22 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_18 (st : State) :
  (act Sutra.S22 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_18 (st : State) :
  (act Sutra.S22 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_19 (st : State) :
  (act Sutra.S22 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_19 (st : State) :
  (act Sutra.S22 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_20 (st : State) :
  (act Sutra.S22 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_20 (st : State) :
  (act Sutra.S22 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_21 (st : State) :
  (act Sutra.S22 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_21 (st : State) :
  (act Sutra.S22 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_22 (st : State) :
  (act Sutra.S22 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_22 (st : State) :
  (act Sutra.S22 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_23 (st : State) :
  (act Sutra.S22 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_23 (st : State) :
  (act Sutra.S22 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_24 (st : State) :
  (act Sutra.S22 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_24 (st : State) :
  (act Sutra.S22 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_25 (st : State) :
  (act Sutra.S22 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_25 (st : State) :
  (act Sutra.S22 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_26 (st : State) :
  (act Sutra.S22 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_26 (st : State) :
  (act Sutra.S22 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_27 (st : State) :
  (act Sutra.S22 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_27 (st : State) :
  (act Sutra.S22 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_28 (st : State) :
  (act Sutra.S22 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_28 (st : State) :
  (act Sutra.S22 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_22_29 (st : State) :
  (act Sutra.S22 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S22 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_22_29 (st : State) :
  (act Sutra.S22 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S22 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_1 (st : State) :
  (act Sutra.S23 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_1 (st : State) :
  (act Sutra.S23 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_2 (st : State) :
  (act Sutra.S23 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_2 (st : State) :
  (act Sutra.S23 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_3 (st : State) :
  (act Sutra.S23 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_3 (st : State) :
  (act Sutra.S23 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_4 (st : State) :
  (act Sutra.S23 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_4 (st : State) :
  (act Sutra.S23 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_5 (st : State) :
  (act Sutra.S23 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_5 (st : State) :
  (act Sutra.S23 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_6 (st : State) :
  (act Sutra.S23 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_6 (st : State) :
  (act Sutra.S23 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_7 (st : State) :
  (act Sutra.S23 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_7 (st : State) :
  (act Sutra.S23 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_8 (st : State) :
  (act Sutra.S23 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_8 (st : State) :
  (act Sutra.S23 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_9 (st : State) :
  (act Sutra.S23 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_9 (st : State) :
  (act Sutra.S23 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_10 (st : State) :
  (act Sutra.S23 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_10 (st : State) :
  (act Sutra.S23 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_11 (st : State) :
  (act Sutra.S23 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_11 (st : State) :
  (act Sutra.S23 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_12 (st : State) :
  (act Sutra.S23 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_12 (st : State) :
  (act Sutra.S23 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_13 (st : State) :
  (act Sutra.S23 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_13 (st : State) :
  (act Sutra.S23 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_14 (st : State) :
  (act Sutra.S23 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_14 (st : State) :
  (act Sutra.S23 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_15 (st : State) :
  (act Sutra.S23 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_15 (st : State) :
  (act Sutra.S23 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_16 (st : State) :
  (act Sutra.S23 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_16 (st : State) :
  (act Sutra.S23 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_17 (st : State) :
  (act Sutra.S23 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_17 (st : State) :
  (act Sutra.S23 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_18 (st : State) :
  (act Sutra.S23 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_18 (st : State) :
  (act Sutra.S23 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_19 (st : State) :
  (act Sutra.S23 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_19 (st : State) :
  (act Sutra.S23 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_20 (st : State) :
  (act Sutra.S23 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_20 (st : State) :
  (act Sutra.S23 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_21 (st : State) :
  (act Sutra.S23 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_21 (st : State) :
  (act Sutra.S23 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_22 (st : State) :
  (act Sutra.S23 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_22 (st : State) :
  (act Sutra.S23 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_23 (st : State) :
  (act Sutra.S23 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_23 (st : State) :
  (act Sutra.S23 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_24 (st : State) :
  (act Sutra.S23 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_24 (st : State) :
  (act Sutra.S23 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_25 (st : State) :
  (act Sutra.S23 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_25 (st : State) :
  (act Sutra.S23 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_26 (st : State) :
  (act Sutra.S23 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_26 (st : State) :
  (act Sutra.S23 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_27 (st : State) :
  (act Sutra.S23 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_27 (st : State) :
  (act Sutra.S23 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_28 (st : State) :
  (act Sutra.S23 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_28 (st : State) :
  (act Sutra.S23 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_23_29 (st : State) :
  (act Sutra.S23 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S23 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_23_29 (st : State) :
  (act Sutra.S23 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S23 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_1 (st : State) :
  (act Sutra.S24 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_1 (st : State) :
  (act Sutra.S24 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_2 (st : State) :
  (act Sutra.S24 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_2 (st : State) :
  (act Sutra.S24 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_3 (st : State) :
  (act Sutra.S24 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_3 (st : State) :
  (act Sutra.S24 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_4 (st : State) :
  (act Sutra.S24 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_4 (st : State) :
  (act Sutra.S24 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_5 (st : State) :
  (act Sutra.S24 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_5 (st : State) :
  (act Sutra.S24 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_6 (st : State) :
  (act Sutra.S24 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_6 (st : State) :
  (act Sutra.S24 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_7 (st : State) :
  (act Sutra.S24 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_7 (st : State) :
  (act Sutra.S24 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_8 (st : State) :
  (act Sutra.S24 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_8 (st : State) :
  (act Sutra.S24 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_9 (st : State) :
  (act Sutra.S24 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_9 (st : State) :
  (act Sutra.S24 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_10 (st : State) :
  (act Sutra.S24 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_10 (st : State) :
  (act Sutra.S24 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_11 (st : State) :
  (act Sutra.S24 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_11 (st : State) :
  (act Sutra.S24 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_12 (st : State) :
  (act Sutra.S24 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_12 (st : State) :
  (act Sutra.S24 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_13 (st : State) :
  (act Sutra.S24 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_13 (st : State) :
  (act Sutra.S24 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_14 (st : State) :
  (act Sutra.S24 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_14 (st : State) :
  (act Sutra.S24 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_15 (st : State) :
  (act Sutra.S24 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_15 (st : State) :
  (act Sutra.S24 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_16 (st : State) :
  (act Sutra.S24 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_16 (st : State) :
  (act Sutra.S24 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_17 (st : State) :
  (act Sutra.S24 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_17 (st : State) :
  (act Sutra.S24 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_18 (st : State) :
  (act Sutra.S24 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_18 (st : State) :
  (act Sutra.S24 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_19 (st : State) :
  (act Sutra.S24 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_19 (st : State) :
  (act Sutra.S24 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_20 (st : State) :
  (act Sutra.S24 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_20 (st : State) :
  (act Sutra.S24 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_21 (st : State) :
  (act Sutra.S24 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_21 (st : State) :
  (act Sutra.S24 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_22 (st : State) :
  (act Sutra.S24 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_22 (st : State) :
  (act Sutra.S24 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_23 (st : State) :
  (act Sutra.S24 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_23 (st : State) :
  (act Sutra.S24 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_24 (st : State) :
  (act Sutra.S24 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_24 (st : State) :
  (act Sutra.S24 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_25 (st : State) :
  (act Sutra.S24 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_25 (st : State) :
  (act Sutra.S24 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_26 (st : State) :
  (act Sutra.S24 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_26 (st : State) :
  (act Sutra.S24 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_27 (st : State) :
  (act Sutra.S24 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_27 (st : State) :
  (act Sutra.S24 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_28 (st : State) :
  (act Sutra.S24 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_28 (st : State) :
  (act Sutra.S24 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_24_29 (st : State) :
  (act Sutra.S24 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S24 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_24_29 (st : State) :
  (act Sutra.S24 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S24 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_1 (st : State) :
  (act Sutra.S25 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_1 (st : State) :
  (act Sutra.S25 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_2 (st : State) :
  (act Sutra.S25 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_2 (st : State) :
  (act Sutra.S25 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_3 (st : State) :
  (act Sutra.S25 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_3 (st : State) :
  (act Sutra.S25 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_4 (st : State) :
  (act Sutra.S25 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_4 (st : State) :
  (act Sutra.S25 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_5 (st : State) :
  (act Sutra.S25 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_5 (st : State) :
  (act Sutra.S25 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_6 (st : State) :
  (act Sutra.S25 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_6 (st : State) :
  (act Sutra.S25 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_7 (st : State) :
  (act Sutra.S25 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_7 (st : State) :
  (act Sutra.S25 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_8 (st : State) :
  (act Sutra.S25 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_8 (st : State) :
  (act Sutra.S25 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_9 (st : State) :
  (act Sutra.S25 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_9 (st : State) :
  (act Sutra.S25 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_10 (st : State) :
  (act Sutra.S25 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_10 (st : State) :
  (act Sutra.S25 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_11 (st : State) :
  (act Sutra.S25 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_11 (st : State) :
  (act Sutra.S25 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_12 (st : State) :
  (act Sutra.S25 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_12 (st : State) :
  (act Sutra.S25 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_13 (st : State) :
  (act Sutra.S25 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_13 (st : State) :
  (act Sutra.S25 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_14 (st : State) :
  (act Sutra.S25 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_14 (st : State) :
  (act Sutra.S25 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_15 (st : State) :
  (act Sutra.S25 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_15 (st : State) :
  (act Sutra.S25 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_16 (st : State) :
  (act Sutra.S25 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_16 (st : State) :
  (act Sutra.S25 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_17 (st : State) :
  (act Sutra.S25 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_17 (st : State) :
  (act Sutra.S25 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_18 (st : State) :
  (act Sutra.S25 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_18 (st : State) :
  (act Sutra.S25 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_19 (st : State) :
  (act Sutra.S25 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_19 (st : State) :
  (act Sutra.S25 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_20 (st : State) :
  (act Sutra.S25 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_20 (st : State) :
  (act Sutra.S25 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_21 (st : State) :
  (act Sutra.S25 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_21 (st : State) :
  (act Sutra.S25 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_22 (st : State) :
  (act Sutra.S25 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_22 (st : State) :
  (act Sutra.S25 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_23 (st : State) :
  (act Sutra.S25 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_23 (st : State) :
  (act Sutra.S25 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_24 (st : State) :
  (act Sutra.S25 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_24 (st : State) :
  (act Sutra.S25 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_25 (st : State) :
  (act Sutra.S25 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_25 (st : State) :
  (act Sutra.S25 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_26 (st : State) :
  (act Sutra.S25 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_26 (st : State) :
  (act Sutra.S25 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_27 (st : State) :
  (act Sutra.S25 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_27 (st : State) :
  (act Sutra.S25 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_28 (st : State) :
  (act Sutra.S25 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_28 (st : State) :
  (act Sutra.S25 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_25_29 (st : State) :
  (act Sutra.S25 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S25 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_25_29 (st : State) :
  (act Sutra.S25 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S25 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_1 (st : State) :
  (act Sutra.S26 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_1 (st : State) :
  (act Sutra.S26 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_2 (st : State) :
  (act Sutra.S26 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_2 (st : State) :
  (act Sutra.S26 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_3 (st : State) :
  (act Sutra.S26 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_3 (st : State) :
  (act Sutra.S26 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_4 (st : State) :
  (act Sutra.S26 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_4 (st : State) :
  (act Sutra.S26 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_5 (st : State) :
  (act Sutra.S26 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_5 (st : State) :
  (act Sutra.S26 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_6 (st : State) :
  (act Sutra.S26 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_6 (st : State) :
  (act Sutra.S26 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_7 (st : State) :
  (act Sutra.S26 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_7 (st : State) :
  (act Sutra.S26 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_8 (st : State) :
  (act Sutra.S26 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_8 (st : State) :
  (act Sutra.S26 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_9 (st : State) :
  (act Sutra.S26 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_9 (st : State) :
  (act Sutra.S26 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_10 (st : State) :
  (act Sutra.S26 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_10 (st : State) :
  (act Sutra.S26 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_11 (st : State) :
  (act Sutra.S26 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_11 (st : State) :
  (act Sutra.S26 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_12 (st : State) :
  (act Sutra.S26 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_12 (st : State) :
  (act Sutra.S26 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_13 (st : State) :
  (act Sutra.S26 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_13 (st : State) :
  (act Sutra.S26 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_14 (st : State) :
  (act Sutra.S26 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_14 (st : State) :
  (act Sutra.S26 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_15 (st : State) :
  (act Sutra.S26 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_15 (st : State) :
  (act Sutra.S26 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_16 (st : State) :
  (act Sutra.S26 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_16 (st : State) :
  (act Sutra.S26 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_17 (st : State) :
  (act Sutra.S26 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_17 (st : State) :
  (act Sutra.S26 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_18 (st : State) :
  (act Sutra.S26 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_18 (st : State) :
  (act Sutra.S26 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_19 (st : State) :
  (act Sutra.S26 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_19 (st : State) :
  (act Sutra.S26 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_20 (st : State) :
  (act Sutra.S26 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_20 (st : State) :
  (act Sutra.S26 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_21 (st : State) :
  (act Sutra.S26 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_21 (st : State) :
  (act Sutra.S26 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_22 (st : State) :
  (act Sutra.S26 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_22 (st : State) :
  (act Sutra.S26 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_23 (st : State) :
  (act Sutra.S26 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_23 (st : State) :
  (act Sutra.S26 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_24 (st : State) :
  (act Sutra.S26 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_24 (st : State) :
  (act Sutra.S26 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_25 (st : State) :
  (act Sutra.S26 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_25 (st : State) :
  (act Sutra.S26 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_26 (st : State) :
  (act Sutra.S26 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_26 (st : State) :
  (act Sutra.S26 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_27 (st : State) :
  (act Sutra.S26 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_27 (st : State) :
  (act Sutra.S26 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_28 (st : State) :
  (act Sutra.S26 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_28 (st : State) :
  (act Sutra.S26 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_26_29 (st : State) :
  (act Sutra.S26 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S26 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_26_29 (st : State) :
  (act Sutra.S26 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S26 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_1 (st : State) :
  (act Sutra.S27 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_1 (st : State) :
  (act Sutra.S27 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_2 (st : State) :
  (act Sutra.S27 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_2 (st : State) :
  (act Sutra.S27 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_3 (st : State) :
  (act Sutra.S27 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_3 (st : State) :
  (act Sutra.S27 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_4 (st : State) :
  (act Sutra.S27 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_4 (st : State) :
  (act Sutra.S27 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_5 (st : State) :
  (act Sutra.S27 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_5 (st : State) :
  (act Sutra.S27 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_6 (st : State) :
  (act Sutra.S27 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_6 (st : State) :
  (act Sutra.S27 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_7 (st : State) :
  (act Sutra.S27 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_7 (st : State) :
  (act Sutra.S27 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_8 (st : State) :
  (act Sutra.S27 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_8 (st : State) :
  (act Sutra.S27 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_9 (st : State) :
  (act Sutra.S27 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_9 (st : State) :
  (act Sutra.S27 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_10 (st : State) :
  (act Sutra.S27 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_10 (st : State) :
  (act Sutra.S27 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_11 (st : State) :
  (act Sutra.S27 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_11 (st : State) :
  (act Sutra.S27 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_12 (st : State) :
  (act Sutra.S27 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_12 (st : State) :
  (act Sutra.S27 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_13 (st : State) :
  (act Sutra.S27 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_13 (st : State) :
  (act Sutra.S27 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_14 (st : State) :
  (act Sutra.S27 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_14 (st : State) :
  (act Sutra.S27 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_15 (st : State) :
  (act Sutra.S27 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_15 (st : State) :
  (act Sutra.S27 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_16 (st : State) :
  (act Sutra.S27 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_16 (st : State) :
  (act Sutra.S27 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_17 (st : State) :
  (act Sutra.S27 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_17 (st : State) :
  (act Sutra.S27 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_18 (st : State) :
  (act Sutra.S27 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_18 (st : State) :
  (act Sutra.S27 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_19 (st : State) :
  (act Sutra.S27 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_19 (st : State) :
  (act Sutra.S27 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_20 (st : State) :
  (act Sutra.S27 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_20 (st : State) :
  (act Sutra.S27 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_21 (st : State) :
  (act Sutra.S27 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_21 (st : State) :
  (act Sutra.S27 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_22 (st : State) :
  (act Sutra.S27 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_22 (st : State) :
  (act Sutra.S27 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_23 (st : State) :
  (act Sutra.S27 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_23 (st : State) :
  (act Sutra.S27 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_24 (st : State) :
  (act Sutra.S27 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_24 (st : State) :
  (act Sutra.S27 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_25 (st : State) :
  (act Sutra.S27 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_25 (st : State) :
  (act Sutra.S27 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_26 (st : State) :
  (act Sutra.S27 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_26 (st : State) :
  (act Sutra.S27 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_27 (st : State) :
  (act Sutra.S27 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_27 (st : State) :
  (act Sutra.S27 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_28 (st : State) :
  (act Sutra.S27 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_28 (st : State) :
  (act Sutra.S27 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_27_29 (st : State) :
  (act Sutra.S27 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S27 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_27_29 (st : State) :
  (act Sutra.S27 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S27 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_1 (st : State) :
  (act Sutra.S28 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_1 (st : State) :
  (act Sutra.S28 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_2 (st : State) :
  (act Sutra.S28 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_2 (st : State) :
  (act Sutra.S28 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_3 (st : State) :
  (act Sutra.S28 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_3 (st : State) :
  (act Sutra.S28 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_4 (st : State) :
  (act Sutra.S28 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_4 (st : State) :
  (act Sutra.S28 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_5 (st : State) :
  (act Sutra.S28 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_5 (st : State) :
  (act Sutra.S28 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_6 (st : State) :
  (act Sutra.S28 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_6 (st : State) :
  (act Sutra.S28 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_7 (st : State) :
  (act Sutra.S28 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_7 (st : State) :
  (act Sutra.S28 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_8 (st : State) :
  (act Sutra.S28 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_8 (st : State) :
  (act Sutra.S28 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_9 (st : State) :
  (act Sutra.S28 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_9 (st : State) :
  (act Sutra.S28 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_10 (st : State) :
  (act Sutra.S28 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_10 (st : State) :
  (act Sutra.S28 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_11 (st : State) :
  (act Sutra.S28 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_11 (st : State) :
  (act Sutra.S28 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_12 (st : State) :
  (act Sutra.S28 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_12 (st : State) :
  (act Sutra.S28 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_13 (st : State) :
  (act Sutra.S28 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_13 (st : State) :
  (act Sutra.S28 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_14 (st : State) :
  (act Sutra.S28 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_14 (st : State) :
  (act Sutra.S28 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_15 (st : State) :
  (act Sutra.S28 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_15 (st : State) :
  (act Sutra.S28 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_16 (st : State) :
  (act Sutra.S28 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_16 (st : State) :
  (act Sutra.S28 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_17 (st : State) :
  (act Sutra.S28 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_17 (st : State) :
  (act Sutra.S28 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_18 (st : State) :
  (act Sutra.S28 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_18 (st : State) :
  (act Sutra.S28 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_19 (st : State) :
  (act Sutra.S28 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_19 (st : State) :
  (act Sutra.S28 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_20 (st : State) :
  (act Sutra.S28 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_20 (st : State) :
  (act Sutra.S28 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_21 (st : State) :
  (act Sutra.S28 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_21 (st : State) :
  (act Sutra.S28 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_22 (st : State) :
  (act Sutra.S28 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_22 (st : State) :
  (act Sutra.S28 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_23 (st : State) :
  (act Sutra.S28 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_23 (st : State) :
  (act Sutra.S28 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_24 (st : State) :
  (act Sutra.S28 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_24 (st : State) :
  (act Sutra.S28 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_25 (st : State) :
  (act Sutra.S28 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_25 (st : State) :
  (act Sutra.S28 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_26 (st : State) :
  (act Sutra.S28 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_26 (st : State) :
  (act Sutra.S28 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_27 (st : State) :
  (act Sutra.S28 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_27 (st : State) :
  (act Sutra.S28 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_28 (st : State) :
  (act Sutra.S28 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_28 (st : State) :
  (act Sutra.S28 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_28_29 (st : State) :
  (act Sutra.S28 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S28 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_28_29 (st : State) :
  (act Sutra.S28 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S28 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_1 (st : State) :
  (act Sutra.S29 (act Sutra.S1 st)).x = (act Sutra.S1 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_1 (st : State) :
  (act Sutra.S29 (act Sutra.S1 st)).y = (act Sutra.S1 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_2 (st : State) :
  (act Sutra.S29 (act Sutra.S2 st)).x = (act Sutra.S2 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_2 (st : State) :
  (act Sutra.S29 (act Sutra.S2 st)).y = (act Sutra.S2 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_3 (st : State) :
  (act Sutra.S29 (act Sutra.S3 st)).x = (act Sutra.S3 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_3 (st : State) :
  (act Sutra.S29 (act Sutra.S3 st)).y = (act Sutra.S3 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_4 (st : State) :
  (act Sutra.S29 (act Sutra.S4 st)).x = (act Sutra.S4 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_4 (st : State) :
  (act Sutra.S29 (act Sutra.S4 st)).y = (act Sutra.S4 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_5 (st : State) :
  (act Sutra.S29 (act Sutra.S5 st)).x = (act Sutra.S5 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_5 (st : State) :
  (act Sutra.S29 (act Sutra.S5 st)).y = (act Sutra.S5 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_6 (st : State) :
  (act Sutra.S29 (act Sutra.S6 st)).x = (act Sutra.S6 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_6 (st : State) :
  (act Sutra.S29 (act Sutra.S6 st)).y = (act Sutra.S6 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_7 (st : State) :
  (act Sutra.S29 (act Sutra.S7 st)).x = (act Sutra.S7 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_7 (st : State) :
  (act Sutra.S29 (act Sutra.S7 st)).y = (act Sutra.S7 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_8 (st : State) :
  (act Sutra.S29 (act Sutra.S8 st)).x = (act Sutra.S8 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_8 (st : State) :
  (act Sutra.S29 (act Sutra.S8 st)).y = (act Sutra.S8 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_9 (st : State) :
  (act Sutra.S29 (act Sutra.S9 st)).x = (act Sutra.S9 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_9 (st : State) :
  (act Sutra.S29 (act Sutra.S9 st)).y = (act Sutra.S9 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_10 (st : State) :
  (act Sutra.S29 (act Sutra.S10 st)).x = (act Sutra.S10 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_10 (st : State) :
  (act Sutra.S29 (act Sutra.S10 st)).y = (act Sutra.S10 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_11 (st : State) :
  (act Sutra.S29 (act Sutra.S11 st)).x = (act Sutra.S11 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_11 (st : State) :
  (act Sutra.S29 (act Sutra.S11 st)).y = (act Sutra.S11 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_12 (st : State) :
  (act Sutra.S29 (act Sutra.S12 st)).x = (act Sutra.S12 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_12 (st : State) :
  (act Sutra.S29 (act Sutra.S12 st)).y = (act Sutra.S12 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_13 (st : State) :
  (act Sutra.S29 (act Sutra.S13 st)).x = (act Sutra.S13 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_13 (st : State) :
  (act Sutra.S29 (act Sutra.S13 st)).y = (act Sutra.S13 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_14 (st : State) :
  (act Sutra.S29 (act Sutra.S14 st)).x = (act Sutra.S14 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_14 (st : State) :
  (act Sutra.S29 (act Sutra.S14 st)).y = (act Sutra.S14 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_15 (st : State) :
  (act Sutra.S29 (act Sutra.S15 st)).x = (act Sutra.S15 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_15 (st : State) :
  (act Sutra.S29 (act Sutra.S15 st)).y = (act Sutra.S15 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_16 (st : State) :
  (act Sutra.S29 (act Sutra.S16 st)).x = (act Sutra.S16 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_16 (st : State) :
  (act Sutra.S29 (act Sutra.S16 st)).y = (act Sutra.S16 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_17 (st : State) :
  (act Sutra.S29 (act Sutra.S17 st)).x = (act Sutra.S17 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_17 (st : State) :
  (act Sutra.S29 (act Sutra.S17 st)).y = (act Sutra.S17 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_18 (st : State) :
  (act Sutra.S29 (act Sutra.S18 st)).x = (act Sutra.S18 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_18 (st : State) :
  (act Sutra.S29 (act Sutra.S18 st)).y = (act Sutra.S18 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_19 (st : State) :
  (act Sutra.S29 (act Sutra.S19 st)).x = (act Sutra.S19 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_19 (st : State) :
  (act Sutra.S29 (act Sutra.S19 st)).y = (act Sutra.S19 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_20 (st : State) :
  (act Sutra.S29 (act Sutra.S20 st)).x = (act Sutra.S20 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_20 (st : State) :
  (act Sutra.S29 (act Sutra.S20 st)).y = (act Sutra.S20 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_21 (st : State) :
  (act Sutra.S29 (act Sutra.S21 st)).x = (act Sutra.S21 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_21 (st : State) :
  (act Sutra.S29 (act Sutra.S21 st)).y = (act Sutra.S21 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_22 (st : State) :
  (act Sutra.S29 (act Sutra.S22 st)).x = (act Sutra.S22 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_22 (st : State) :
  (act Sutra.S29 (act Sutra.S22 st)).y = (act Sutra.S22 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_23 (st : State) :
  (act Sutra.S29 (act Sutra.S23 st)).x = (act Sutra.S23 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_23 (st : State) :
  (act Sutra.S29 (act Sutra.S23 st)).y = (act Sutra.S23 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_24 (st : State) :
  (act Sutra.S29 (act Sutra.S24 st)).x = (act Sutra.S24 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_24 (st : State) :
  (act Sutra.S29 (act Sutra.S24 st)).y = (act Sutra.S24 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_25 (st : State) :
  (act Sutra.S29 (act Sutra.S25 st)).x = (act Sutra.S25 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_25 (st : State) :
  (act Sutra.S29 (act Sutra.S25 st)).y = (act Sutra.S25 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_26 (st : State) :
  (act Sutra.S29 (act Sutra.S26 st)).x = (act Sutra.S26 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_26 (st : State) :
  (act Sutra.S29 (act Sutra.S26 st)).y = (act Sutra.S26 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_27 (st : State) :
  (act Sutra.S29 (act Sutra.S27 st)).x = (act Sutra.S27 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_27 (st : State) :
  (act Sutra.S29 (act Sutra.S27 st)).y = (act Sutra.S27 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_28 (st : State) :
  (act Sutra.S29 (act Sutra.S28 st)).x = (act Sutra.S28 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_28 (st : State) :
  (act Sutra.S29 (act Sutra.S28 st)).y = (act Sutra.S28 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem commute_x_29_29 (st : State) :
  (act Sutra.S29 (act Sutra.S29 st)).x = (act Sutra.S29 (act Sutra.S29 st)).x := by
  simp [act, delta, add_assoc, add_left_comm, add_comm]


theorem commute_y_29_29 (st : State) :
  (act Sutra.S29 (act Sutra.S29 st)).y = (act Sutra.S29 (act Sutra.S29 st)).y := by
  simp [act, delta, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]


theorem prefix_x_0 (st : State) :
  (applyList (Sutra.all.take 0) st).x = st.x + (0 : Rat) := by
  have h : sumDelta (Sutra.all.take 0) = (0 : Rat) := by native_decide
  simpa [applyList_x, h]


theorem prefix_y_0 (st : State) :
  (applyList (Sutra.all.take 0) st).y = st.y - (0 : Rat) := by
  have h : sumDelta (Sutra.all.take 0) = (0 : Rat) := by native_decide
  simpa [applyList_y, h]


theorem prefix_x_1 (st : State) :
  (applyList (Sutra.all.take 1) st).x = st.x + (1 : Rat) := by
  have h : sumDelta (Sutra.all.take 1) = (1 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_1 (st : State) :
  (applyList (Sutra.all.take 1) st).y = st.y - (1 : Rat) := by
  have h : sumDelta (Sutra.all.take 1) = (1 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_2 (st : State) :
  (applyList (Sutra.all.take 2) st).x = st.x + (3 : Rat) := by
  have h : sumDelta (Sutra.all.take 2) = (3 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_2 (st : State) :
  (applyList (Sutra.all.take 2) st).y = st.y - (3 : Rat) := by
  have h : sumDelta (Sutra.all.take 2) = (3 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_3 (st : State) :
  (applyList (Sutra.all.take 3) st).x = st.x + (6 : Rat) := by
  have h : sumDelta (Sutra.all.take 3) = (6 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_3 (st : State) :
  (applyList (Sutra.all.take 3) st).y = st.y - (6 : Rat) := by
  have h : sumDelta (Sutra.all.take 3) = (6 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_4 (st : State) :
  (applyList (Sutra.all.take 4) st).x = st.x + (10 : Rat) := by
  have h : sumDelta (Sutra.all.take 4) = (10 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_4 (st : State) :
  (applyList (Sutra.all.take 4) st).y = st.y - (10 : Rat) := by
  have h : sumDelta (Sutra.all.take 4) = (10 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_5 (st : State) :
  (applyList (Sutra.all.take 5) st).x = st.x + (15 : Rat) := by
  have h : sumDelta (Sutra.all.take 5) = (15 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_5 (st : State) :
  (applyList (Sutra.all.take 5) st).y = st.y - (15 : Rat) := by
  have h : sumDelta (Sutra.all.take 5) = (15 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_6 (st : State) :
  (applyList (Sutra.all.take 6) st).x = st.x + (21 : Rat) := by
  have h : sumDelta (Sutra.all.take 6) = (21 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_6 (st : State) :
  (applyList (Sutra.all.take 6) st).y = st.y - (21 : Rat) := by
  have h : sumDelta (Sutra.all.take 6) = (21 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_7 (st : State) :
  (applyList (Sutra.all.take 7) st).x = st.x + (28 : Rat) := by
  have h : sumDelta (Sutra.all.take 7) = (28 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_7 (st : State) :
  (applyList (Sutra.all.take 7) st).y = st.y - (28 : Rat) := by
  have h : sumDelta (Sutra.all.take 7) = (28 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_8 (st : State) :
  (applyList (Sutra.all.take 8) st).x = st.x + (36 : Rat) := by
  have h : sumDelta (Sutra.all.take 8) = (36 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_8 (st : State) :
  (applyList (Sutra.all.take 8) st).y = st.y - (36 : Rat) := by
  have h : sumDelta (Sutra.all.take 8) = (36 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_9 (st : State) :
  (applyList (Sutra.all.take 9) st).x = st.x + (45 : Rat) := by
  have h : sumDelta (Sutra.all.take 9) = (45 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_9 (st : State) :
  (applyList (Sutra.all.take 9) st).y = st.y - (45 : Rat) := by
  have h : sumDelta (Sutra.all.take 9) = (45 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_10 (st : State) :
  (applyList (Sutra.all.take 10) st).x = st.x + (55 : Rat) := by
  have h : sumDelta (Sutra.all.take 10) = (55 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_10 (st : State) :
  (applyList (Sutra.all.take 10) st).y = st.y - (55 : Rat) := by
  have h : sumDelta (Sutra.all.take 10) = (55 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_11 (st : State) :
  (applyList (Sutra.all.take 11) st).x = st.x + (66 : Rat) := by
  have h : sumDelta (Sutra.all.take 11) = (66 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_11 (st : State) :
  (applyList (Sutra.all.take 11) st).y = st.y - (66 : Rat) := by
  have h : sumDelta (Sutra.all.take 11) = (66 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_12 (st : State) :
  (applyList (Sutra.all.take 12) st).x = st.x + (78 : Rat) := by
  have h : sumDelta (Sutra.all.take 12) = (78 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_12 (st : State) :
  (applyList (Sutra.all.take 12) st).y = st.y - (78 : Rat) := by
  have h : sumDelta (Sutra.all.take 12) = (78 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_13 (st : State) :
  (applyList (Sutra.all.take 13) st).x = st.x + (91 : Rat) := by
  have h : sumDelta (Sutra.all.take 13) = (91 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_13 (st : State) :
  (applyList (Sutra.all.take 13) st).y = st.y - (91 : Rat) := by
  have h : sumDelta (Sutra.all.take 13) = (91 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_14 (st : State) :
  (applyList (Sutra.all.take 14) st).x = st.x + (105 : Rat) := by
  have h : sumDelta (Sutra.all.take 14) = (105 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_14 (st : State) :
  (applyList (Sutra.all.take 14) st).y = st.y - (105 : Rat) := by
  have h : sumDelta (Sutra.all.take 14) = (105 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_15 (st : State) :
  (applyList (Sutra.all.take 15) st).x = st.x + (120 : Rat) := by
  have h : sumDelta (Sutra.all.take 15) = (120 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_15 (st : State) :
  (applyList (Sutra.all.take 15) st).y = st.y - (120 : Rat) := by
  have h : sumDelta (Sutra.all.take 15) = (120 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_16 (st : State) :
  (applyList (Sutra.all.take 16) st).x = st.x + (136 : Rat) := by
  have h : sumDelta (Sutra.all.take 16) = (136 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_16 (st : State) :
  (applyList (Sutra.all.take 16) st).y = st.y - (136 : Rat) := by
  have h : sumDelta (Sutra.all.take 16) = (136 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_17 (st : State) :
  (applyList (Sutra.all.take 17) st).x = st.x + (153 : Rat) := by
  have h : sumDelta (Sutra.all.take 17) = (153 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_17 (st : State) :
  (applyList (Sutra.all.take 17) st).y = st.y - (153 : Rat) := by
  have h : sumDelta (Sutra.all.take 17) = (153 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_18 (st : State) :
  (applyList (Sutra.all.take 18) st).x = st.x + (171 : Rat) := by
  have h : sumDelta (Sutra.all.take 18) = (171 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_18 (st : State) :
  (applyList (Sutra.all.take 18) st).y = st.y - (171 : Rat) := by
  have h : sumDelta (Sutra.all.take 18) = (171 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_19 (st : State) :
  (applyList (Sutra.all.take 19) st).x = st.x + (190 : Rat) := by
  have h : sumDelta (Sutra.all.take 19) = (190 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_19 (st : State) :
  (applyList (Sutra.all.take 19) st).y = st.y - (190 : Rat) := by
  have h : sumDelta (Sutra.all.take 19) = (190 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_20 (st : State) :
  (applyList (Sutra.all.take 20) st).x = st.x + (210 : Rat) := by
  have h : sumDelta (Sutra.all.take 20) = (210 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_20 (st : State) :
  (applyList (Sutra.all.take 20) st).y = st.y - (210 : Rat) := by
  have h : sumDelta (Sutra.all.take 20) = (210 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_21 (st : State) :
  (applyList (Sutra.all.take 21) st).x = st.x + (231 : Rat) := by
  have h : sumDelta (Sutra.all.take 21) = (231 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_21 (st : State) :
  (applyList (Sutra.all.take 21) st).y = st.y - (231 : Rat) := by
  have h : sumDelta (Sutra.all.take 21) = (231 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_22 (st : State) :
  (applyList (Sutra.all.take 22) st).x = st.x + (253 : Rat) := by
  have h : sumDelta (Sutra.all.take 22) = (253 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_22 (st : State) :
  (applyList (Sutra.all.take 22) st).y = st.y - (253 : Rat) := by
  have h : sumDelta (Sutra.all.take 22) = (253 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_23 (st : State) :
  (applyList (Sutra.all.take 23) st).x = st.x + (276 : Rat) := by
  have h : sumDelta (Sutra.all.take 23) = (276 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_23 (st : State) :
  (applyList (Sutra.all.take 23) st).y = st.y - (276 : Rat) := by
  have h : sumDelta (Sutra.all.take 23) = (276 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_24 (st : State) :
  (applyList (Sutra.all.take 24) st).x = st.x + (300 : Rat) := by
  have h : sumDelta (Sutra.all.take 24) = (300 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_24 (st : State) :
  (applyList (Sutra.all.take 24) st).y = st.y - (300 : Rat) := by
  have h : sumDelta (Sutra.all.take 24) = (300 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_25 (st : State) :
  (applyList (Sutra.all.take 25) st).x = st.x + (325 : Rat) := by
  have h : sumDelta (Sutra.all.take 25) = (325 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_25 (st : State) :
  (applyList (Sutra.all.take 25) st).y = st.y - (325 : Rat) := by
  have h : sumDelta (Sutra.all.take 25) = (325 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_26 (st : State) :
  (applyList (Sutra.all.take 26) st).x = st.x + (351 : Rat) := by
  have h : sumDelta (Sutra.all.take 26) = (351 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_26 (st : State) :
  (applyList (Sutra.all.take 26) st).y = st.y - (351 : Rat) := by
  have h : sumDelta (Sutra.all.take 26) = (351 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_27 (st : State) :
  (applyList (Sutra.all.take 27) st).x = st.x + (378 : Rat) := by
  have h : sumDelta (Sutra.all.take 27) = (378 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_27 (st : State) :
  (applyList (Sutra.all.take 27) st).y = st.y - (378 : Rat) := by
  have h : sumDelta (Sutra.all.take 27) = (378 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_28 (st : State) :
  (applyList (Sutra.all.take 28) st).x = st.x + (406 : Rat) := by
  have h : sumDelta (Sutra.all.take 28) = (406 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_28 (st : State) :
  (applyList (Sutra.all.take 28) st).y = st.y - (406 : Rat) := by
  have h : sumDelta (Sutra.all.take 28) = (406 : Rat) := by native_decide
  simp [applyList_y, h]


theorem prefix_x_29 (st : State) :
  (applyList (Sutra.all.take 29) st).x = st.x + (435 : Rat) := by
  have h : sumDelta (Sutra.all.take 29) = (435 : Rat) := by native_decide
  simp [applyList_x, h]


theorem prefix_y_29 (st : State) :
  (applyList (Sutra.all.take 29) st).y = st.y - (435 : Rat) := by
  have h : sumDelta (Sutra.all.take 29) = (435 : Rat) := by native_decide
  simp [applyList_y, h]


end SutraWS
