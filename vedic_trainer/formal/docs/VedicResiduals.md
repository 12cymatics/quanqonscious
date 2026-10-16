# Scope note: the residuals and auxiliary losses behind vedic_trainer's experiment

Written to rule C3 of `CODEX_OPENAI_MATH.md` at the repository root. This
note says exactly what the formalization proves and what it does not. Read
it, not the theorem names, to know what has been checked.

**Checked by.** Comparator, with two configurations, each pairing a
challenge module (statements only) with a solution module, permitted axioms
`propext`, `Quot.sound`, `Classical.choice`. Lean v4.34.1, Mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`, comparator `d03acab`,
lean4export `076e8e5`, landrun `811cfff`.

- `ComparatorChallenges/VedicResiduals.json`:
  `ComparatorChallenges.VedicResiduals` against `VedicFormal.Residuals`
  (items 1–6 below). Result: *"Lean default kernel accepts the solution"*.
  The same gate went RED on a disposable copy when the challenge claimed
  `15·W≥2` (statement mismatch) and when the complement was broken to the
  identity (kernel rejection).
- `ComparatorChallenges/VedicLCons.json`:
  `ComparatorChallenges.VedicLCons` against `VedicFormal.LCons` (items 7–9).
  Result, run from an empty `.lake/build` over both configurations as the
  `formal-comparator` job does: *"Lean default kernel accepts the solution"*
  for each. On a disposable worktree of the commit that added it (`9ec32f8`),
  the unmodified copy went GREEN as a control, then RED twice, one defect at a
  time: with `9/16` changed to `1/2` in the challenge's closed form only
  (*"Challenge and solution theorem statement do not match:
  'VedicFormal.lConsRow_closed_form'"*), and with `+ d_anti * d_anti` dropped
  from the trusted `lConsRow` (the solution no longer builds). That second
  defect makes the closed form false, not merely its proof stale: at
  Ψ = 𝟙_{0} − 𝟙_{15} the broken `lConsRow` evaluates to 0 and the right-hand
  side to 9/4 (evaluated in Lean, not proved).

## Scope

The trusted definitions in `VedicFormal/Defs.lean` transcribe, over ℚ and
for a state Ψ : Fin 16 → ℚ, the following Python. Each definition's
docstring names its source.

- `vedic/kernel/z2_primitives.py`: the mean; S5; S11 at its canonical
  weight 1/4 with its shell mean; S29 at its canonical weight 1/2; the
  symmetric and antisymmetric parts of S7 at the full mask.
- `vedic/kernel/conservation_exact.py`: R1 (for an integer trace), R2, R3,
  R4.
- `vedic/kernel/audit_filter.py`: `audit_closed`.
- `vedic/training/losses.py`: `L_dual` for one example; the mass term
  that `L_cons` used to include, before it was squared; and `L_cons` for one
  example as it now stands (`lConsRow`), transcribed line for line and left
  unsimplified — four energies `Σ_v x_v·x_v` over all sixteen vertices, the
  drifts `d_sym`, `d_anti`, and `d_sym·d_sym + d_anti·d_anti`.
- `vedic/kernel/wht.py`: the four axis rows. The Walsh characters and
  Walsh–Fourier coefficients of Z₂⁴ are defined directly.

On those definitions the formalization proves, for **every** Ψ ∈ ℚ¹⁶ (not a
sample):

1. The complement used is `v ⊕ 0b1111`.
2. R2 = 0, R3 = 0 and R4 = 0.
3. `audit_closed(Ψ, t)` holds exactly when 435 divides `t`, and therefore
   two states — two models — can never differ on it at the same trace. This
   is the formal form of the README's verdict on falsification criterion 2.
4. The mass term of `L_cons` is identically zero: S29 preserves the sum.
5. `L_dual(Ψ) = 16 · Σ_{|S| ≥ 2} f̂(S)²` with `f = S11(S5 Ψ)`: sixteen times
   the Walsh–Fourier weight of `f` above degree one. The identity is proved
   for an arbitrary `f`, so it is Parseval for the projection onto the
   constant and the four single-axis characters.
6. `L_dual` is not identically zero (witness: the indicator of vertex 3).
   So the image of S11∘S5 does **not** lie in the span of the constant and
   the four axis characters, contrary to the docstring of `L_dual`.
7. Per example, `L_cons(Ψ) = (9/16)·((Σ_v (S_v − mean Ψ)²)² + (Σ_v A_v²)²)`,
   with S, A the symmetric and antisymmetric parts of S7
   (`lConsRow_closed_form`). This is the closed form the `L_cons` docstring
   states and `vedic/training/tests/test_losses.py` pins on samples; it is
   now proved for every Ψ ∈ ℚ¹⁶.
8. Adding the same constant to every vertex leaves it unchanged
   (`lConsRow_shift`).
9. It is zero exactly when Ψ is constant (`lConsRow_eq_zero_iff`): the
   kernel of the penalty is span{𝟙}, the fixed points of S29.

## Outside this scope

- **The transcription is trusted, not proved.** Nothing machine-checks that
  `Defs.lean` says what the Python does; a reader must compare them. Item 1
  turns one of those choices (complement as `Fin.rev`) into a checked
  theorem; the others are not. Items 7–9 are about `lConsRow`, so they hold
  for the Python `L_cons` only as far as `lConsRow` transcribes it. One
  difference a reader will see is resolved over ℚ, not by proof: the torch
  `S7` divides by `2.0` where `sym` and `anti` multiply by `1/2`. As
  evidence, not proof, the Python `L_cons` in float64 (exact on dyadic
  inputs) equalled Lean's evaluation of `lConsRow` exactly on two states,
  giving 60455961/4096 and 2953244025/16777216.
- **The training path is not covered.** The losses that actually train run
  in float32 torch (`vedic/kernel/sutras_torch.py`,
  `vedic/training/losses.py`). They are tied to the exact kernel only by
  the package's structural buffer tests and fixtures, and float rounding is
  outside every statement here.
- `L_chi`, `L_curv`, the batch mean of every term (each result here is per
  example), and the weighted sum in `total_loss` are not formalized. In
  particular the overlap the `L_cons` docstring states — its antisymmetric
  half `(9/16)·‖A‖⁴` is the square of the per-example `L_chi` — is not a
  theorem here, because `L_chi` has no Lean definition.
- S11 and S29 are covered only at their canonical weights.
- The 29 canonical sutras in `vedic/kernel/sutras_canonical.py` are not in
  scope.
- Nothing here says whether the auxiliary losses help a model. That is an
  empirical question for the experiment, which is unformalizable and is
  labelled as such (rule C6).
