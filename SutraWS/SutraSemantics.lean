import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import SutraWS.Sutra
import SutraWS.State

namespace SutraWS

def delta : Sutra → Rat
| Sutra.S1 => 1
| Sutra.S2 => 2
| Sutra.S3 => 3
| Sutra.S4 => 4
| Sutra.S5 => 5
| Sutra.S6 => 6
| Sutra.S7 => 7
| Sutra.S8 => 8
| Sutra.S9 => 9
| Sutra.S10 => 10
| Sutra.S11 => 11
| Sutra.S12 => 12
| Sutra.S13 => 13
| Sutra.S14 => 14
| Sutra.S15 => 15
| Sutra.S16 => 16
| Sutra.S17 => 17
| Sutra.S18 => 18
| Sutra.S19 => 19
| Sutra.S20 => 20
| Sutra.S21 => 21
| Sutra.S22 => 22
| Sutra.S23 => 23
| Sutra.S24 => 24
| Sutra.S25 => 25
| Sutra.S26 => 26
| Sutra.S27 => 27
| Sutra.S28 => 28
| Sutra.S29 => 29

def act (u : Sutra) (s : State) : State :=
  let k := delta u
  { x := s.x + k
  , y := s.y - k
  , z := s.z
  , t := s.t
  , X := s.X + s.x*k
  , Y := s.Y + s.y*k
  , Z := s.Z + s.z*k }

def applyList (L : List Sutra) (s : State) : State :=
  List.foldl (fun st u => act u st) s L

def applyAll (s : State) : State :=
  applyList Sutra.all s

def sumDelta : List Sutra → Rat
| [] => 0
| u :: us => delta u + sumDelta us

theorem applyList_x (L : List Sutra) (s : State) :
  (applyList L s).x = s.x + sumDelta L := by
  induction L generalizing s with
  | nil =>
      simp [applyList, sumDelta]
  | cons u us ih =>
      simp [applyList, sumDelta, act, ih, Rat.add_assoc, Rat.add_left_comm, Rat.add_comm]

theorem applyList_y (L : List Sutra) (s : State) :
  (applyList L s).y = s.y - sumDelta L := by
  induction L generalizing s with
  | nil =>
      simp [applyList, sumDelta]
  | cons u us ih =>
      simp [applyList, sumDelta, act, ih, Rat.sub_eq_add_neg, Rat.add_assoc, Rat.add_left_comm, Rat.add_comm]

theorem sumDelta_all : sumDelta Sutra.all = (435 : Rat) := by native_decide
theorem applyAll_x (s : State) : (applyAll s).x = s.x + (435 : Rat) := by simpa [applyAll, applyList_x, sumDelta_all]
theorem applyAll_y (s : State) : (applyAll s).y = s.y - (435 : Rat) := by simpa [applyAll, applyList_y, sumDelta_all]

end SutraWS
