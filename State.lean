import Mathlib.Data.Rat.Basic

namespace SutraWS

structure State where
  x : Rat
  y : Rat
  z : Rat
  t : Rat
  X : Rat
  Y : Rat
  Z : Rat
deriving DecidableEq, Repr

end SutraWS
