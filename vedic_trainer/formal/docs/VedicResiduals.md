# Scope note: the residuals and auxiliary losses behind vedic_trainer's experiment

Written to rule C3 of `CODEX_OPENAI_MATH.md` at the repository root. This
note says exactly what the formalization proves and what it does not. Read
it, not the theorem names, to know what has been checked.

**Checked by.** Comparator, with `ComparatorChallenges/VedicResiduals.json`:
the challenge module `ComparatorChallenges.VedicResiduals` (statements only)
against the solution module `VedicFormal.Residuals`, permitted axioms
`propext`, `Quot.sound`, `Classical.choice`. Lean v4.34.1, Mathlib
`d13f23b723b8a846827a245b89c10fc7d3f11612`, comparator `d03acab`,
lean4export `076e8e5`, landrun `811cfff`. Result: *"Lean default kernel
accepts the solution"*. The same gate went RED on a disposable copy when the
challenge claimed `15·W≥2` (statement mismatch) and when the complement was
broken to the identity (kernel rejection).

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
- `vedic/training/losses.py`: `L_dual` for one example, and the mass term
  of `L_cons` before it is squared.
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

## Outside this scope

- **The transcription is trusted, not proved.** Nothing machine-checks that
  `Defs.lean` says what the Python does; a reader must compare them. Item 1
  turns one of those choices (complement as `Fin.rev`) into a checked
  theorem; the others are not.
- **The training path is not covered.** The losses that actually train run
  in float32 torch (`vedic/kernel/sutras_torch.py`,
  `vedic/training/losses.py`). They are tied to the exact kernel only by
  the package's structural buffer tests and fixtures, and float rounding is
  outside every statement here.
- `L_chi`, `L_curv`, the two energy terms of `L_cons`, the batch mean, and
  the weighted sum in `total_loss` are not formalized.
- S11 and S29 are covered only at their canonical weights.
- The 29 canonical sutras in `vedic/kernel/sutras_canonical.py` are not in
  scope.
- Nothing here says whether the auxiliary losses help a model. That is an
  empirical question for the experiment, which is unformalizable and is
  labelled as such (rule C6).
