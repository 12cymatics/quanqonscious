import Std

set_option autoImplicit false
set_option warningAsError true

namespace VerifiedGate

abbrev Bytes := List UInt8

def magic : Bytes := [86, 71, 75, 51, 0]

structure Frame where
  instruction : Bytes
  policyDigest : Bytes
  instructionDigest : Bytes
  operation : Bytes
  parentDigest : Bytes
  handoffId : Bytes
  keyId : Bytes
  executionSpec : Bytes
  deriving DecidableEq, Repr

structure Expected where
  instruction : Bytes
  policyDigest : Bytes
  instructionDigest : Bytes
  parentDigest : Bytes
  deriving DecidableEq, Repr

def Frame.fields (f : Frame) : List Bytes :=
  [f.instruction, f.policyDigest, f.instructionDigest, f.operation,
   f.parentDigest, f.handoffId, f.keyId, f.executionSpec]

def be64 (n : Nat) : Bytes :=
  (List.range 8).map fun i => UInt8.ofNat ((n / 256 ^ (7 - i)) % 256)

def serialize (f : Frame) : Bytes :=
  magic ++ (f.fields.flatMap fun bs => be64 bs.length ++ bs)

def takeExact (n : Nat) (bs : Bytes) : Option (Bytes × Bytes) :=
  if n ≤ bs.length then some (bs.take n, bs.drop n) else none

def parseField (bs : Bytes) : Option (Bytes × Bytes) := do
  let (lengthBytes, rest) ← takeExact 8 bs
  let n := lengthBytes.foldl (fun acc b => acc * 256 + b.toNat) 0
  takeExact n rest

def parseFields : Nat → Bytes → Option (List Bytes × Bytes)
  | 0, bs => some ([], bs)
  | n + 1, bs => do
    let (field, rest) ← parseField bs
    let (fields, tail) ← parseFields n rest
    some (field :: fields, tail)

def ofFields : List Bytes → Option Frame
  | [a, b, c, d, e, f, g, h] => some ⟨a, b, c, d, e, f, g, h⟩
  | _ => none

def parse (bs : Bytes) : Option Frame := do
  let (header, rest) ← takeExact magic.length bs
  if header ≠ magic then none else do
    let (fields, tail) ← parseFields 8 rest
    if tail ≠ [] then none else ofFields fields

def decodeCanonical (bs : Bytes) : Option Frame :=
  match parse bs with
  | none => none
  | some f => if serialize f = bs then some f else none

def Shape (f : Frame) : Prop :=
  f.instruction ≠ [] ∧ f.policyDigest.length = 32 ∧
  f.instructionDigest.length = 32 ∧ f.operation ≠ [] ∧
  (f.parentDigest = [] ∨ f.parentDigest.length = 32) ∧
  f.handoffId.length = 32 ∧ f.keyId.length = 32 ∧
  f.executionSpec ≠ [] ∧
  (f.fields.all fun bs => decide (bs.length < 2 ^ 64)) = true

instance (f : Frame) : Decidable (Shape f) := inferInstanceAs (Decidable (
  f.instruction ≠ [] ∧ f.policyDigest.length = 32 ∧
  f.instructionDigest.length = 32 ∧ f.operation ≠ [] ∧
  (f.parentDigest = [] ∨ f.parentDigest.length = 32) ∧
  f.handoffId.length = 32 ∧ f.keyId.length = 32 ∧
  f.executionSpec ≠ [] ∧
  (f.fields.all fun bs => decide (bs.length < 2 ^ 64)) = true))

def encodeChecked (f : Frame) : Option Bytes :=
  if Shape f then
    if decodeCanonical (serialize f) = some f then some (serialize f) else none
  else none

theorem encodeChecked_roundtrip (f : Frame) (raw : Bytes)
    (h : encodeChecked f = some raw) : decodeCanonical raw = some f := by
  unfold encodeChecked at h
  split at h
  next =>
    split at h
    next hround =>
      have heq : serialize f = raw := Option.some.inj h
      exact heq ▸ hround
    next => contradiction
  next => contradiction

def Matches (f : Frame) (expected : Expected) : Prop :=
  f.instruction = expected.instruction ∧
  f.policyDigest = expected.policyDigest ∧
  f.instructionDigest = expected.instructionDigest ∧
  f.parentDigest = expected.parentDigest

instance (f : Frame) (e : Expected) : Decidable (Matches f e) :=
  inferInstanceAs (Decidable (
    f.instruction = e.instruction ∧ f.policyDigest = e.policyDigest ∧
    f.instructionDigest = e.instructionDigest ∧ f.parentDigest = e.parentDigest))

def verify (bs : Bytes) (expected : Expected) : Option Frame :=
  match decodeCanonical bs with
  | none => none
  | some f => if Shape f ∧ Matches f expected then some f else none

theorem decodeCanonical_sound (bs : Bytes) (f : Frame)
    (h : decodeCanonical bs = some f) : serialize f = bs := by
  unfold decodeCanonical at h
  cases hp : parse bs with
  | none => simp [hp] at h
  | some parsed =>
    simp only [hp] at h
    split at h
    next heq =>
      have hf : parsed = f := Option.some.inj h
      exact hf ▸ heq
    next => contradiction

theorem verify_sound (bs : Bytes) (expected : Expected) (f : Frame)
    (h : verify bs expected = some f) :
    serialize f = bs ∧ Shape f ∧ Matches f expected := by
  unfold verify at h
  cases hd : decodeCanonical bs with
  | none => simp [hd] at h
  | some parsed =>
    simp only [hd] at h
    split at h
    next hcheck =>
      have hf : parsed = f := Option.some.inj h
      subst parsed
      exact ⟨decodeCanonical_sound bs f hd, hcheck.1, hcheck.2⟩
    next => contradiction

theorem accepted_instruction_exact (bs : Bytes) (expected : Expected) (f : Frame)
    (h : verify bs expected = some f) : f.instruction = expected.instruction :=
  (verify_sound bs expected f h).2.2.1

theorem accepted_policy_exact (bs : Bytes) (expected : Expected) (f : Frame)
    (h : verify bs expected = some f) : f.policyDigest = expected.policyDigest :=
  (verify_sound bs expected f h).2.2.2.1

theorem accepted_source_digest_exact (bs : Bytes) (expected : Expected) (f : Frame)
    (h : verify bs expected = some f) : f.instructionDigest = expected.instructionDigest :=
  (verify_sound bs expected f h).2.2.2.2.1

theorem accepted_parent_exact (bs : Bytes) (expected : Expected) (f : Frame)
    (h : verify bs expected = some f) : f.parentDigest = expected.parentDigest :=
  (verify_sound bs expected f h).2.2.2.2.2

theorem accepted_frame_canonical (bs : Bytes) (expected : Expected) (f : Frame)
    (h : verify bs expected = some f) : serialize f = bs :=
  (verify_sound bs expected f h).1

theorem different_instructions_rejected (bs : Bytes) (expected : Expected) (f : Frame)
    (hne : f.instruction ≠ expected.instruction) : verify bs expected ≠ some f := by
  intro h
  exact hne (accepted_instruction_exact bs expected f h)

theorem child_inherits_exact (parent child : Frame) (raw : Bytes)
    (trustedPolicy trustedDigest parentEnvelopeDigest : Bytes)
    (h : verify raw ⟨parent.instruction, trustedPolicy, trustedDigest,
      parentEnvelopeDigest⟩ = some child) :
    child.instruction = parent.instruction ∧
    child.policyDigest = trustedPolicy ∧ child.instructionDigest = trustedDigest ∧
    child.parentDigest = parentEnvelopeDigest :=
  (verify_sound raw _ child h).2.2

def hexDigit (n : Nat) : Char :=
  Char.ofNat (if n < 10 then 48 + n else 87 + n)

def toHex (bs : Bytes) : String :=
  String.ofList (bs.flatMap fun b => [hexDigit (b.toNat / 16), hexDigit (b.toNat % 16)])

def hexValue (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - 48)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 87)
  else none

def parseHexChars : List Char → Option Bytes
  | [] => some []
  | a :: b :: rest => do
    let hi ← hexValue a
    let lo ← hexValue b
    let tail ← parseHexChars rest
    some (UInt8.ofNat (16 * hi + lo) :: tail)
  | _ => none

def fromHex (s : String) : Option Bytes := do
  let bs ← parseHexChars s.toList
  if toHex bs = s then some bs else none

def readFields (count : Nat) : IO (List Bytes) := do
  let stdin ← IO.getStdin
  let mut fields := []
  for _ in [:count] do
    let line ← stdin.getLine
    if line.isEmpty || !line.endsWith "\n" then
      throw (IO.userError "missing hex line")
    let value := String.ofList (line.toList.dropLast)
    match fromHex value with
    | none => throw (IO.userError "noncanonical hex")
    | some bs => fields := fields ++ [bs]
  if !(← stdin.readToEnd).isEmpty then
    throw (IO.userError "unexpected trailing input")
  return fields

def writeFields (fields : List Bytes) : IO Unit := do
  let stdout ← IO.getStdout
  for field in fields do
    stdout.putStrLn (toHex field)

def run (args : List String) : IO UInt32 := do
  match args with
  | ["encode"] =>
    let fields ← readFields 8
    match ofFields fields with
    | none => throw (IO.userError "expected eight fields")
    | some f =>
      match encodeChecked f with
      | none => throw (IO.userError "invalid fields or serialization roundtrip")
      | some raw =>
        writeFields [raw]
        return 0
  | ["decode"] =>
    let fields ← readFields 1
    match fields with
    | [raw] =>
      match decodeCanonical raw with
      | none => throw (IO.userError "invalid canonical frame")
      | some f =>
        if !(decide (Shape f)) then throw (IO.userError "invalid frame fields")
        writeFields f.fields
        return 0
    | _ => throw (IO.userError "expected payload")
  | ["verify"] =>
    let fields ← readFields 5
    match fields with
    | [raw, instruction, policyDigest, instructionDigest, parentDigest] =>
      match verify raw ⟨instruction, policyDigest, instructionDigest, parentDigest⟩ with
      | none => throw (IO.userError "frame verification rejected")
      | some f =>
        writeFields f.fields
        return 0
    | _ => throw (IO.userError "expected payload and four trusted values")
  | ["version"] =>
    (← IO.getStdout).putStrLn "verified-agent-gate-kernel-v3 Lean 4.36.0"
    return 0
  | _ => throw (IO.userError "usage: gate-kernel encode|decode|verify|version")

end VerifiedGate

def main (args : List String) : IO UInt32 := do
  try
    VerifiedGate.run args
  catch e =>
    (← IO.getStderr).putStrLn ("REJECTED: " ++ e.toString)
    return 1

#print axioms VerifiedGate.verify_sound
#print axioms VerifiedGate.accepted_instruction_exact
#print axioms VerifiedGate.accepted_frame_canonical
#print axioms VerifiedGate.child_inherits_exact

#print axioms VerifiedGate.encodeChecked_roundtrip
