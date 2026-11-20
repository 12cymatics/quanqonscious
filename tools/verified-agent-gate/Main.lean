import Gate
import Lean
open Lean ByteGate
def bytes (j : Json) : Except String (List UInt8) := do
  let a ← j.getArr?
  a.toList.mapM fun x => do
    let n ← x.getNat?
    if n < 256 then pure n.toUInt8 else throw "byte outside 0..255"
def optionalString (j : Json) : Except String (Option String) :=
  if j == Json.null then pure none else some <$> j.getStr?
def strings (j : Json) : Except String (List String) := do
  (← j.getArr?).toList.mapM Json.getStr?
def claimsFromJson (j : Json) : Except String Claims := do
  return {
    source := ← bytes (← j.getObjVal? "source")
    expected := ← bytes (← j.getObjVal? "expected")
    claimedLength := ← (← j.getObjVal? "claimedLength").getNat?
    signatureOK := ← (← j.getObjVal? "signatureOK").getBool?
    digestOK := ← (← j.getObjVal? "digestOK").getBool?
    canonicalOK := ← (← j.getObjVal? "canonicalOK").getBool?
    filesOK := ← (← j.getObjVal? "filesOK").getBool?
    id := ← (← j.getObjVal? "id").getStr?
    parent := ← optionalString (← j.getObjVal? "parent")
    expectedParent := ← optionalString (← j.getObjVal? "expectedParent")
    used := ← strings (← j.getObjVal? "used")
    parents := ← strings (← j.getObjVal? "parents")
  }
def main : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let input ← stdin.readToEnd
  match Json.parse input >>= claimsFromJson with
  | .error e =>
      stdout.putStrLn s!"REJECTED: {e}"
      return 2
  | .ok c =>
      if check c then
        stdout.putStrLn "ACCEPTED"
        return 0
      else
        stdout.putStrLn "REJECTED"
        return 1
