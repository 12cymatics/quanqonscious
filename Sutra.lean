import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Tactic

namespace SutraWS

inductive Sutra : Type
| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10
| S11 | S12 | S13 | S14 | S15 | S16 | S17 | S18 | S19 | S20
| S21 | S22 | S23 | S24 | S25 | S26 | S27 | S28 | S29
deriving DecidableEq, Repr

abbrev Sutra29 := Fin 29

def Sutra.ofFin : Sutra29 → Sutra
| ⟨0, _⟩ => Sutra.S1
| ⟨1, _⟩ => Sutra.S2
| ⟨2, _⟩ => Sutra.S3
| ⟨3, _⟩ => Sutra.S4
| ⟨4, _⟩ => Sutra.S5
| ⟨5, _⟩ => Sutra.S6
| ⟨6, _⟩ => Sutra.S7
| ⟨7, _⟩ => Sutra.S8
| ⟨8, _⟩ => Sutra.S9
| ⟨9, _⟩ => Sutra.S10
| ⟨10, _⟩ => Sutra.S11
| ⟨11, _⟩ => Sutra.S12
| ⟨12, _⟩ => Sutra.S13
| ⟨13, _⟩ => Sutra.S14
| ⟨14, _⟩ => Sutra.S15
| ⟨15, _⟩ => Sutra.S16
| ⟨16, _⟩ => Sutra.S17
| ⟨17, _⟩ => Sutra.S18
| ⟨18, _⟩ => Sutra.S19
| ⟨19, _⟩ => Sutra.S20
| ⟨20, _⟩ => Sutra.S21
| ⟨21, _⟩ => Sutra.S22
| ⟨22, _⟩ => Sutra.S23
| ⟨23, _⟩ => Sutra.S24
| ⟨24, _⟩ => Sutra.S25
| ⟨25, _⟩ => Sutra.S26
| ⟨26, _⟩ => Sutra.S27
| ⟨27, _⟩ => Sutra.S28
| ⟨28, _⟩ => Sutra.S29

def Sutra.all : List Sutra :=
  [Sutra.S1, Sutra.S2, Sutra.S3, Sutra.S4, Sutra.S5, Sutra.S6, Sutra.S7, Sutra.S8, Sutra.S9, Sutra.S10,
   Sutra.S11, Sutra.S12, Sutra.S13, Sutra.S14, Sutra.S15, Sutra.S16, Sutra.S17, Sutra.S18, Sutra.S19, Sutra.S20,
   Sutra.S21, Sutra.S22, Sutra.S23, Sutra.S24, Sutra.S25, Sutra.S26, Sutra.S27, Sutra.S28, Sutra.S29]

theorem Sutra.all_length : Sutra.all.length = 29 := by decide

end SutraWS
