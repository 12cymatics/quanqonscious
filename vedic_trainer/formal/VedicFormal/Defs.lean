import Mathlib

/-!
# Trusted definitions: vedic_trainer's exact kernel, transcribed over ℚ

Under `CODEX_OPENAI_MATH.md` (rule C1) this file is **trusted**: the
challenge module and the solution module both import it, so Comparator
guarantees each proof is about *these* definitions, and a reader must audit
these definitions against the Python they transcribe. Each docstring names
that Python. Nothing here is proved; nothing here contains `sorry`.

Conventions, matching `vedic_trainer/vedic/kernel/tesseract.py`:

* a vertex of Z₂⁴ is a `Fin 16`; bit `k` of `v` is `(v >> k) & 1`;
* a state Ψ is a function `Fin 16 → ℚ` (the Python `Q16` tuple);
* the complement of `v` is `v ⊕ 0b1111`. For a 4-bit vertex that is
  `15 - v`, which is `Fin.rev`; the challenge `complement_val` makes the
  xor form a checked theorem rather than a claim in this comment.
-/

namespace VedicFormal

open Finset

/-- A state Ψ ∈ ℚ¹⁶ on the sixteen vertices of Z₂⁴ (`vedic.kernel.q.Q16`). -/
abbrev Psi : Type := Fin 16 → ℚ

/-- Bit `k` of vertex `v`, i.e. `(v >> k) & 1`. -/
def bit (k : Fin 4) (v : Fin 16) : ℕ := v.val / 2 ^ k.val % 2

/-- Hamming weight (`tesseract.POPCOUNT`). -/
def popcount (v : Fin 16) : ℕ := ∑ k : Fin 4, bit k v

/-- Complement `v ⊕ 0b1111` (`tesseract.COMPLEMENT`). -/
def complement (v : Fin 16) : Fin 16 := Fin.rev v

/-- `z2_primitives._mean`: `sum(psi) / 16`. -/
def mean (ψ : Psi) : ℚ := (∑ v, ψ v) / 16

/-- `z2_primitives.s5_shunyam_samya`: `Ψ_v − mean(Ψ)`. -/
def s5 (ψ : Psi) : Psi := fun v => ψ v - mean ψ

/-- The shell mean used by `z2_primitives.s11_vyasti_samasti`: the mean of Ψ
over the vertices whose popcount equals that of `v`. (Python's empty-shell
branch is unreachable here: the shell of `v` contains `v`.) -/
def shellMean (ψ : Psi) (v : Fin 16) : ℚ :=
  (∑ u ∈ univ.filter (fun u => popcount u = popcount v), ψ u) /
    ((univ.filter (fun u => popcount u = popcount v)).card : ℚ)

/-- `z2_primitives.s11_vyasti_samasti` at its canonical weight
`S11_WEIGHT = 1/4`: `Ψ_v − (1/4)·shell_mean(v)`. -/
def s11 (ψ : Psi) : Psi := fun v => ψ v - (1 / 4 : ℚ) * shellMean ψ v

/-- `z2_primitives.s29_mean_drive` at its canonical weight
`S29_WEIGHT = 1/2`: `(1 − w)·Ψ_v + w·mean(Ψ)`. -/
def s29 (ψ : Psi) : Psi := fun v => (1 - (1 / 2 : ℚ)) * ψ v + (1 / 2 : ℚ) * mean ψ

/-- Symmetric part of `z2_primitives.s7_sankalana_vyavakalana` at the full
mask: `(Ψ_v + Ψ_{v̄}) · 1/2`. -/
def sym (ψ : Psi) : Psi := fun v => (ψ v + ψ (complement v)) * (1 / 2)

/-- Antisymmetric part of `z2_primitives.s7_sankalana_vyavakalana` at the
full mask: `(Ψ_v − Ψ_{v̄}) · 1/2`. -/
def anti (ψ : Psi) : Psi := fun v => (ψ v - ψ (complement v)) * (1 / 2)

/-- `conservation_exact.r1_trace_closure` for an integer trace:
`numerator % T29` with `T29 = 29·30/2 = 435`. Python's `%` with a positive
modulus and Lean's `Int` `%` (Euclidean) both return a value in `[0, 435)`. -/
def r1 (t : ℤ) : ℤ := t % 435

/-- `conservation_exact.r2_complement_pair_sum`:
`Σ_{v < v̄} (Ψ_v + Ψ_{v̄}) − Σ_v Ψ_v`. -/
def r2 (ψ : Psi) : ℚ :=
  (∑ v ∈ univ.filter (fun v => v < complement v), (ψ v + ψ (complement v))) - ∑ v, ψ v

/-- `conservation_exact.r3_s29_mean_preservation`: `mean(S29 Ψ) − mean(Ψ)`. -/
def r3 (ψ : Psi) : ℚ := mean (s29 ψ) - mean ψ

/-- `conservation_exact.r4_beltrami_orthogonality`: `⟨S(Ψ), A(Ψ)⟩`. -/
def r4 (ψ : Psi) : ℚ := ∑ v, sym ψ v * anti ψ v

/-- `audit_filter.audit_closed`: all four residuals are zero. -/
def auditClosed (ψ : Psi) (t : ℤ) : Prop :=
  r1 t = 0 ∧ r2 ψ = 0 ∧ r3 ψ = 0 ∧ r4 ψ = 0

/-- The mass term of `training/losses.py` `L_cons`, before squaring:
`Σ_v (S29 Ψ)_v − Σ_v Ψ_v`. -/
def dMass (ψ : Psi) : ℚ := (∑ v, s29 ψ v) - ∑ v, ψ v

/-- Row `k` of `wht.WHT_AXIS_Q`: `(−1)^{bit k v}`. -/
def axisRow (k : Fin 4) (v : Fin 16) : ℚ := (-1) ^ bit k v

/-- `training/losses.py` `L_dual` for one example, before the batch mean:
with `f = S11(S5 Ψ)`, reconstruct `f` from its mean and its four axis
coefficients `⟨f, h_k⟩/16`, and return the squared residual. -/
def lDual (ψ : Psi) : ℚ :=
  let f := s11 (s5 ψ)
  let star := mean f
  let coeffs : Fin 4 → ℚ := fun k => (∑ v, f v * axisRow k v) / 16
  let recon : Psi := fun v => star + ∑ k, coeffs k * axisRow k v
  ∑ v, (f v - recon v) ^ 2

/-- The Walsh character of Z₂⁴: `χ_S(v) = (−1)^{|S ∧ v|}`. -/
def chi (S v : Fin 16) : ℚ := (-1) ^ (∑ k : Fin 4, bit k S * bit k v)

/-- The Walsh–Fourier coefficient `f̂(S) = (1/16) Σ_v f(v) χ_S(v)`. -/
def fourier (f : Psi) (S : Fin 16) : ℚ := (∑ v, f v * chi S v) / 16

end VedicFormal
