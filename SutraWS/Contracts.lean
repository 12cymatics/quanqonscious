import SutraWS.Vertex
import SutraWS.Sutra
import SutraWS.SutraSemantics

/-!
# The α → 0 identity-collapse guarantee

The v18 kernel states this as a structural claim
(`vedic_v18.24_full_kernel.html:5520-5534`, §12Y "STRUCTURAL SINGULARITY SUPPRESSION"):

```
STRUCTURAL GUARANTEE: every operator type collapses to identity when α→0:
  MULTIPLICATIVE:  Ψ' = Ψ·(1 + α·(M-1))       → Ψ·1 = Ψ
  REFLECTIVE:      Ψ' = (1-α)·Ψ + α·R(Ψ)      → 1·Ψ + 0 = Ψ
  CONVOLUTIVE:     Ψ' = Ψ + α·(C⊛Ψ - Ψ)       → Ψ + 0 = Ψ
  DIFFUSIVE:       Ψ' = Ψ + α·ΔΨ               → Ψ + 0 = Ψ
  DIVISIVE:        Ψ' = Ψ + α·(Q(Ψ) - Ψ)      → Ψ + 0 = Ψ
  PERMUTATIVE:     Ψ' = (1-α)·Ψ + α·P(Ψ)      → Ψ
  MODULAR:         Ψ' = Ψ + α·(Ψ mod B - Ψ)   → Ψ
  CONSERVATION:    Ψ' = Ψ + α·(target - Ψ)     → Ψ
```

and `CONTRACTS.testIdentity` (`simulation v18:499-510`) checks it at runtime by zeroing an
operator's `strength` and asserting the field moves by less than `1e-10` in floats. Here it is
proved outright, for all 29 operators at once, in exact ℚ.

**STATUS: not yet machine-checked** (Mathlib cache blocked by egress policy; `lake build` has
never run). The `family` assignment was checked mechanically against the §12Z table: all 29
sutras covered exactly once, 4+5+3+5+4+3+5. If `identity_preserved`'s `cases u <;> simp [step,
family]` does not discharge every case, fall back to unfolding the combinators explicitly:
`cases u <;> funext i <;> simp [step, family, mulStep, affineStep, relaxStep, driftStep]`.

Those eight shapes reduce to four distinct combinators — `DIFFUSIVE` is the odd one out, being
`Ψ + α·G(Ψ)` rather than a relaxation `Ψ + α·(F(Ψ) - Ψ)`.

The per-operator maps (`M`, `R`, `C⊛·`, `Q`, `Δ`, `P`, `mod B`, `target`) are left abstract as
an `OpMap`: the guarantee is a property of the *coupling shape*, not of the individual sutra
kernels, and holds whatever those maps are.
-/

namespace SutraWS

/-- `Ψ'ᵢ = Ψᵢ · (1 + α·(M(Ψ)ᵢ - 1))` — MULTIPLICATIVE. -/
def mulStep (α : ℚ) (M : Psi → Psi) (P : Psi) : Psi :=
  fun i => P i * (1 + α * (M P i - 1))

/-- `Ψ'ᵢ = (1-α)·Ψᵢ + α·R(Ψ)ᵢ` — REFLECTIVE and PERMUTATIVE. -/
def affineStep (α : ℚ) (R : Psi → Psi) (P : Psi) : Psi :=
  fun i => (1 - α) * P i + α * R P i

/-- `Ψ'ᵢ = Ψᵢ + α·(F(Ψ)ᵢ - Ψᵢ)` — CONVOLUTIVE, DIVISIVE, MODULAR, CONSERVATION. -/
def relaxStep (α : ℚ) (F : Psi → Psi) (P : Psi) : Psi :=
  fun i => P i + α * (F P i - P i)

/-- `Ψ'ᵢ = Ψᵢ + α·G(Ψ)ᵢ` — DIFFUSIVE (a graph-Laplacian increment, not a relaxation). -/
def driftStep (α : ℚ) (G : Psi → Psi) (P : Psi) : Psi :=
  fun i => P i + α * G P i

@[simp] theorem mulStep_zero (M : Psi → Psi) (P : Psi) : mulStep 0 M P = P := by
  funext i; simp [mulStep]

@[simp] theorem affineStep_zero (R : Psi → Psi) (P : Psi) : affineStep 0 R P = P := by
  funext i; simp [affineStep]

@[simp] theorem relaxStep_zero (F : Psi → Psi) (P : Psi) : relaxStep 0 F P = P := by
  funext i; simp [relaxStep]

@[simp] theorem driftStep_zero (G : Psi → Psi) (P : Psi) : driftStep 0 G P = P := by
  funext i; simp [driftStep]

/-- The seven operator classes of §12Z (`vedic_v18.24_full_kernel.html:5540-5551`), plus the
`CONSERVATION` shape named in the §12Y guarantee banner. -/
inductive Family
  | multiplicative | reflective | convolutive | divisive
  | diffusive | permutative | modular | conservation
deriving DecidableEq, Repr

/-- The operator-class assignment, transcribed from the §12Z table. Every one of the 29 sutras
is assigned exactly once: 4 + 5 + 3 + 5 + 4 + 3 + 5 = 29. -/
def family : Sutra → Family
  -- MULTIPLICATIVE [S1, S10, S14, S15]
  | Sutra.S1 | Sutra.S10 | Sutra.S14 | Sutra.S15 => Family.multiplicative
  -- REFLECTIVE [S2, S5, S12, S22, S23]
  | Sutra.S2 | Sutra.S5 | Sutra.S12 | Sutra.S22 | Sutra.S23 => Family.reflective
  -- CONVOLUTIVE [S3, S11, S25]
  | Sutra.S3 | Sutra.S11 | Sutra.S25 => Family.convolutive
  -- DIVISIVE [S4, S8, S13, S16, S19]
  | Sutra.S4 | Sutra.S8 | Sutra.S13 | Sutra.S16 | Sutra.S19 => Family.divisive
  -- DIFFUSIVE [S9, S17, S27, S28]
  | Sutra.S9 | Sutra.S17 | Sutra.S27 | Sutra.S28 => Family.diffusive
  -- PERMUTATIVE [S6, S7, S26]
  | Sutra.S6 | Sutra.S7 | Sutra.S26 => Family.permutative
  -- MODULAR [S18, S20, S21, S24, S29]
  | Sutra.S18 | Sutra.S20 | Sutra.S21 | Sutra.S24 | Sutra.S29 => Family.modular

/-- The kernel's per-operator map, left abstract. -/
abbrev OpMap := Sutra → Psi → Psi

/-- One evolution step of operator `u` at coupling `α`, dispatched on its family. -/
def step (K : OpMap) (u : Sutra) (α : ℚ) (P : Psi) : Psi :=
  match family u with
  | Family.multiplicative => mulStep α (K u) P
  | Family.reflective     => affineStep α (K u) P
  | Family.permutative    => affineStep α (K u) P
  | Family.convolutive    => relaxStep α (K u) P
  | Family.divisive       => relaxStep α (K u) P
  | Family.modular        => relaxStep α (K u) P
  | Family.conservation   => relaxStep α (K u) P
  | Family.diffusive      => driftStep α (K u) P

/-- **The identity-preservation guarantee.** Every one of the 29 operators is the identity at
α = 0, for every field and every choice of the underlying kernel maps. This is the claim the
v18 banner makes ("29 discrete operators with an identity-preservation guarantee") and that
`CONTRACTS.testIdentity` only spot-checks numerically. -/
theorem identity_preserved (K : OpMap) (u : Sutra) (P : Psi) : step K u 0 P = P := by
  cases u <;> simp [step, family]

/-- `α(n) = strength · n/435` — §12Y's suppression coefficient, whose triangular denominator is
`T(29) = 435`, i.e. `sumDelta Sutra.all`. -/
def alpha (strength : ℚ) (u : Sutra) : ℚ := strength * (delta u / 435)

/-- The denominator in `alpha` is exactly the sum of the 29 operator weights. -/
theorem alpha_denominator : sumDelta Sutra.all = (435 : ℚ) := sumDelta_all

@[simp] theorem alpha_zero_strength (u : Sutra) : alpha 0 u = 0 := by simp [alpha]

/-- Zero strength ⇒ the whole 29-operator bank is the identity. -/
theorem identity_preserved_of_strength_zero (K : OpMap) (u : Sutra) (P : Psi) :
    step K u (alpha 0 u) P = P := by
  rw [alpha_zero_strength]
  exact identity_preserved K u P

end SutraWS
