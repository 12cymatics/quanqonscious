import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Tactic

namespace SutraWS

structure QI where
  lo : Rat
  hi : Rat
  hle : lo ≤ hi

namespace QI

def point (x : Rat) : QI := ⟨x, x, le_rfl⟩

def contains (I : QI) (x : Rat) : Prop := I.lo ≤ x ∧ x ≤ I.hi

def add (A B : QI) : QI := ⟨A.lo + B.lo, A.hi + B.hi, by exact add_le_add A.hle B.hle⟩

def neg (A : QI) : QI := ⟨-A.hi, -A.lo, by exact neg_le_neg A.hle⟩

def sub (A B : QI) : QI := add A (neg B)

def union (A B : QI) : QI := ⟨min A.lo B.lo, max A.hi B.hi, by
  have h1 : min A.lo B.lo ≤ A.lo := min_le_left _ _
  have h2 : A.hi ≤ max A.hi B.hi := le_max_left _ _
  exact le_trans (le_trans h1 A.hle) h2⟩

def mul (A B : QI) : QI :=
  let p1 := A.lo * B.lo
  let p2 := A.lo * B.hi
  let p3 := A.hi * B.lo
  let p4 := A.hi * B.hi
  let lo := min (min p1 p2) (min p3 p4)
  let hi := max (max p1 p2) (max p3 p4)
  -- min over the four corner products ≤ p1 ≤ max over the four corner products
  have hle : lo ≤ hi :=
    le_trans (le_trans (min_le_left _ _) (min_le_left p1 p2))
      (le_trans (le_max_left p1 p2) (le_max_left _ _))
  ⟨lo, hi, hle⟩

def abs (A : QI) : QI :=
  if _h0 : 0 ≤ A.lo then A else
  if h1 : A.hi ≤ 0 then neg A else
  ⟨0, max (-A.lo) A.hi, by
    -- h1 : ¬ A.hi ≤ 0, hence 0 < A.hi ≤ max (-A.lo) A.hi
    have hpos : (0:Rat) < A.hi := not_le.mp h1
    exact le_trans hpos.le (le_max_right _ _)⟩

theorem contains_point (x : Rat) : (point x).contains x := by constructor <;> rfl

theorem contains_add {A B : QI} {x y : Rat} :
  A.contains x → B.contains y → (add A B).contains (x + y) := by
  intro hx hy
  rcases hx with ⟨hAx, hxA⟩
  rcases hy with ⟨hBy, hyB⟩
  constructor
  · have := add_le_add hAx hBy
    simpa [add, contains] using this
  · have := add_le_add hxA hyB
    simpa [add, contains] using this

theorem contains_sub {A B : QI} {x y : Rat} :
  A.contains x → B.contains y → (sub A B).contains (x - y) := by
  intro hx hy
  have hy' : (neg B).contains (-y) := by
    rcases hy with ⟨h1, h2⟩
    constructor
    · have := neg_le_neg h2
      simpa [neg, contains] using this
    · have := neg_le_neg h1
      simpa [neg, contains] using this
  have := contains_add (A:=A) (B:=neg B) (x:=x) (y:=-y) hx hy'
  simpa [sub, Rat.sub_eq_add_neg] using this

end QI

end SutraWS
