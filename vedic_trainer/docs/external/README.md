# `vedic.external` — sidecar adapters

Optional adapters that surface code from sibling branches of the parent
repository. None of them are imported by the kernel, memory, training,
data, or eval layers — they're shipped so users can:

- Compute the **classical-arithmetic** interpretation of the 29 sutras
  (NumPy float64) alongside the **structural Z₂⁴-algebra** interpretation
  used by the LLM trainer.
- Run the kernel pipeline across many inputs in **serial / threaded /
  process** modes.
- Cross-check canonical algebraic identities through the **Lean 4
  theorem prover** when the `lean` binary is installed.

## Files

| Module | Source branch | What it gives you |
| ------ | ------------- | ----------------- |
| `vedic_engine.py` | `codex/replace-blocks-with-fixed-implementations` | `VedicSutraEngine`: 29 sutras as NumPy float operations. |
| `hypercube.py` | `codex/replace-blocks-with-fixed-implementations` | `Hypercube`: weighted-hypercube, Λ, Ω, Υ operators. |
| `proof_validation.py` | `codex/replace-blocks-with-fixed-implementations` | `ProofTester`: invokes every operator of the two adapters above and records shapes, emptiness and finiteness. It compares no value against a reference, so it does not establish correctness — see the module header, which opens "Not a smoke test." |
| `executor.py` | `codex/locate-runnable-simulations-in-repos` | `SutraExecutor(mode=...)` running the full ℚ-exact pipeline across many inputs. |
| `lean4_mirror.py` | `codex/fix-package-exports-in-__all__-definition` | `Lean4Mirror`: drives the Lean 4 compiler over Bool-valued sutra statements. |
| `lean_props.py` | new | Renders our 30 algebraic identities as Lean 4 `Bool` props using `Rat` literals built from Python `Fraction`s. |

## Two interpretations, one algebra

The 29 sutras admit at least two formalisations relevant to this
repository:

1. **Z₂⁴ structural algebra (`vedic.kernel.sutras_exact`)** — every
   sutra is a function on length-16 tuples of `fractions.Fraction`,
   acting on the 16 vertices of the Boolean cube. This is the
   ground-truth implementation for the LLM-training kernel and is the
   one verified bit-exactly against the simulator fixtures.

2. **Classical arithmetic (`vedic.external.vedic_engine`)** — every
   sutra is a recipe on NumPy arrays of real numbers. This is the form
   used by older mainline GRVQ / hybrid-quantum simulations in this
   repo and remains useful when you want to compute numerical sutra
   transforms outside the LLM-training pipeline.

The two interpretations agree on the **mathematical principle** behind
each sutra (e.g. *Ekadhikena Purvena* = "by one more than the previous"
both XOR-toggles the bit-0 index in (1) and adds 1.0 to an array in
(2)). They differ in their carrier type. Neither is wrong; they answer
different questions.

## Running the executor

```python
from fractions import Fraction
from vedic.external import ExecutionMode, SutraExecutor

inputs = [tuple(Fraction(v - 8, 16) for v in range(16)) for _ in range(100)]
results = SutraExecutor(mode=ExecutionMode.PROCESSES, max_workers=4).execute(inputs)
```

`results[i]` is the length-16 `Fraction` tuple obtained by applying
every unary operator in the canonical pipeline order (see
`executor._PIPELINE`).

## Running the Lean 4 mirror

```bash
brew install lean   # or: see https://lean-lang.org/lean4/doc/quickstart.html
```

```python
from vedic.external import Lean4Mirror, Lean4SessionConfig, build_lean_props
from vedic.external.lean_props import _enumerate_canonical_psi

_, psi = next(iter(_enumerate_canonical_psi()))
props = build_lean_props(psi)

with Lean4Mirror(Lean4SessionConfig()) as mirror:
    results = mirror.run_parallel(props)

for r in results:
    print(f"{r.sutra}: {'✓' if r.success else '✗'} ({r.duration:.2f}s)")
```

Each identity is rendered as `decide (lhs = rhs) && decide (lhs = rhs) …`
over 16 components, using `Rat` literals built from the Python
`Fraction`'s numerator and denominator — so the Lean compiler must
agree with the Python rationals on every component for the prop to
return `true`.

If the Lean compiler is not installed locally, the mirror raises
`FileNotFoundError`; `build_lean_props` itself still works (it only
emits the Bool string and does not invoke Lean).

## Further reference material

- `CONSOLIDATED_ALGORITHMS.md` (400 lines) — full documentation pulled
  from `claude/run-palindrome-simulations-…`. Catalogues the palindrome
  sub-sutras 10–13 with mathematical foundations and quantum / classical
  / hybrid execution recipes.
- `reference/extended_subsutras_palindrome.py` (519 lines) — the
  palindrome sub-sutras 10–13 implementation from the same branch. This
  file depends on `primarysutra.VedicSutras` (the older mainline engine)
  and on `cirq` / `cudaq` / `torch`, so it is not wired into the kernel.
  It is preserved here as reference for future work on bringing
  sub-sutra 10–13 palindrome semantics into the Z₂⁴ kernel.

  **It does not meet this package's standards and must not be imported.**
  `sopaantyadvayamantyam` computes `x * 2 ** steps` under the comment
  "dominant eigenvalue approximation", with three identical branches; the
  kernel forbids approximations and dead branches alike. The file is kept
  byte-identical because it is a record of what that branch contains, and
  editing a record is not the same as correcting code — the same reason
  `runs/*.json` keeps the withdrawn subset evaluations. What is forbidden
  is depending on it, which
  `vedic/external/tests/test_reference_archive_is_not_live_code.py`
  enforces.
