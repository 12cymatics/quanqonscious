import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Ring.Rat

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
