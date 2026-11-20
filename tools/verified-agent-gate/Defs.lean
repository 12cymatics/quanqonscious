import Std
namespace ByteGate
structure Claims where
  source : List UInt8
  expected : List UInt8
  claimedLength : Nat
  signatureOK : Bool
  digestOK : Bool
  canonicalOK : Bool
  filesOK : Bool
  id : String
  parent : Option String
  expectedParent : Option String
  used : List String
  parents : List String
  deriving DecidableEq
def ParentKnown (c : Claims) : Prop :=
  match c.parent with
  | none => True
  | some p => p ∈ c.parents
instance (c : Claims) : Decidable (ParentKnown c) := by
  unfold ParentKnown
  split <;> infer_instance
def Accepted (c : Claims) : Prop :=
  c.source = c.expected ∧ c.claimedLength = c.source.length ∧
  c.signatureOK = true ∧ c.digestOK = true ∧ c.canonicalOK = true ∧
  c.filesOK = true ∧ c.id ∉ c.used ∧ c.parent = c.expectedParent ∧ ParentKnown c
instance (c : Claims) : Decidable (Accepted c) := by unfold Accepted; infer_instance
def check (c : Claims) : Bool := decide (Accepted c)
def consume (c : Claims) : Claims := { c with used := c.id :: c.used }
inductive Chain (source : List UInt8) : List Claims → Prop
  | empty : Chain source []
  | append {history : List Claims} {c : Claims} :
      Chain source history → c.expected = source → check c = true → Chain source (c :: history)
end ByteGate
