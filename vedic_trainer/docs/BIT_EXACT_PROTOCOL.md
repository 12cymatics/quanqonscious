# Bit-exact protocol

Two separate comparisons live under this heading. Keep them apart: one is a
regression reference, the other establishes correctness.

## 1. The 29 canonical operators against the upstream kernel — **verified**

`vedic_v18.24_full_kernel.html` is **tracked at the repository root**. Several
documents used to say it "lives on the user's machine, not in this
repository", and the path gate exempted it as external on that basis — an
exemption that then stopped the premise being checked, since a declared
exemption short-circuits resolution.

It is here, so the comparison runs. `scripts/extract_upstream_kernel.js` lifts
`STRICT_SUTRA_KERNEL` and its dependencies (`Q`, `VTX`, `ALPHA`, `SUTRA_KIND`)
out of the HTML **by source slicing** — nothing is reimplemented, because a
reimplementation would compare this package against itself — and evaluates
them under node.

**Result: 6,380 of 6,380 (Ψ, strength, sutra) triples agree exactly.** All 29
operators, the full enumerated Ψ corpus, strengths {0, 1, 50, 100, 250},
exact rational equality with no tolerance on either side.
`vedic/kernel/tests/test_upstream_agreement.py` runs it every time.

**Which upstream path is the definition.** The HTML carries two. The
`SUTRAS[].evolve()` bodies are the *display* path: they round-trip Ψ through
IEEE-754 via `Q.fl`, use `Math.log10`/`floor`/`pow`, carry an epsilon
(`+1 // prevent division by zero`) and re-quantise to 1e-4 via
`Bi(x*10000)/10000n`. `STRICT_SUTRA_KERNEL` (line 6527) is float-free and is
the definition, which `sutras_canonical.py` has said all along. Comparing
against `evolve()` compares against the wrong half of the file.

**The one difference.** `ALPHA.computeQ` begins `BigInt(Math.round(strength))`,
so upstream cannot represent a fractional strength: 7/3 becomes 2. This kernel
carries it exactly, which is what α(n) = (n/435)·(strength/100) says. The port
is the more faithful of the two, and that is why the comparison above is over
integer strengths — a restriction with a reason, pinned by its own test.

`vedic_v18.51.1_exact_phi.html` is also tracked and is a later revision with a
different architecture: 29 individually implemented operators over the C4/K2(√5)
extension, with per-sutra named operand specs. Nothing here ports it. That is
an open question, not a finding.

## 2. The Z₂⁴ primitives against `vedic_v18.16_strict_kernel.html` — unrun

The Fraction kernel in `vedic/kernel/z2_primitives.py` and the JavaScript
kernel inside the v18.16 simulator are two independent implementations of the
**same** exact-rational algorithm. (This paragraph named `sutras_exact.py` for
as long as that file had been renamed.) **v18.16 is genuinely not in this
repository** — unlike v18.24 — so this half remains unrun. Bit-exact
means: given the same seed and the same `n` random Ψ inputs, both
implementations produce the same JSON file byte-for-byte.

## Fixture format (exact ℚ)

`fixtures/psi_inputs.json`:

```json
{
  "seed": 3235889118,
  "n": 32,
  "denom_max": 1000,
  "inputs": [
    [{"num": -3, "den": 7}, {"num": 1, "den": 2}, ...]
  ]
}
```

`fixtures/sutra_outputs.json` records each input alongside every
operator's output. That is **all 29**, in 30 keys — S7 splits into
`S7_sym`/`S7_anti` — and it does include the three binary operators
S3/S17/S23, evaluated against the second field Φ. This paragraph used to say
"S1..S29 except S3/S17/S23 which are binary", and while it said so nothing
compared those three: the records were there and unread. Scalars are stored
as `{num, den}` Fractions.

`fixtures/conservation_residuals.json` stores three trace samples per
input (`trace = 0`, `435`, `7·435` = 3045) so R1's modular closure is
exercised on both sides of the boundary. 32 inputs × 3 samples = 96 records.

`fixtures/canonical_inputs.json` and `fixtures/canonical_sutra_outputs.json`
are a second, separate reference covering the α-weighted operators in
`vedic/kernel/sutras_canonical.py`: 8 inputs × 29 sutras × 3 strengths = 696
records. They are written by `scripts/build_canonical_fixtures.py` and are
checked by the same gate. Unlike the v18.16 fixtures they are produced by the
kernel they are compared against, so they detect **drift, not error** — see
the README's falsification criterion 3 for what that distinction buys.

## Building the fixtures from Python (authoritative source)

```bash
python scripts/build_fixtures.py --seed 0xC0DEC0DE --n 32 --out fixtures
```

## Hooking the v18.16 simulator

The simulator HTML must expose a single JavaScript function on the
window object:

```js
window.exportFixtures = function(seed, n) {
    // Run the same deterministic-random Ψ generation as
    // scripts/build_fixtures.py:_random_q16:
    //   for v in 0..15:
    //     den = rng.nextInt(1, denom_max=1000);
    //     num = rng.nextInt(-denom_max, +denom_max);
    //     psi[v] = Fraction(num, den);
    // Then compute every sutra output and every conservation residual
    // (using the same trace_sum samples 0, 435, 7·435 as in
    // scripts/build_fixtures.py).
    //
    // Return { psi_inputs, sutra_outputs, conservation_residuals }
    // with the same JSON schema as fixtures/*.json.
};
```

The seeded RNG must be deterministic and identical to Python's
`random.Random(seed).randint(...)` *or* an equivalent Mersenne Twister
implementation seeded the same way. If the simulator uses a different
RNG, drive it from a pre-computed list of (num, den) pairs that match
Python's output (the simulator's recorder has an "input table" mode
documented in its UI).

## Verification

Once `fixtures/*.json` is populated either by `build_fixtures.py` or by
`export_simulator_fixtures.py`, run:

```bash
python scripts/verify_bit_exact.py
```

It exits 0 if the ℚ kernel reproduces every fixture record exactly, and
prints what it checked: 32 inputs, 32 sutra records, 96 conservation records,
then 8 inputs and 696 canonical-29 records. An unchecked fixture key is
itself a failure — the gate refuses to run at all against a missing reference
rather than rebuilding one from the code under test. If
the simulator export disagrees with `build_fixtures.py`, the simulator
has a bug; if the Python kernel disagrees with both, the kernel has a
bug. Either way: stop, fix, restart.

## Why exact ℚ matters for the LLM trainer

Float arithmetic introduces approximations that compound silently
through 29 sutra applications. The kernel is the only ground truth we
have for "this auxiliary-loss landscape is genuinely the one the
algebra describes". Without exact ℚ verification, we cannot
distinguish "the experiment failed because the algebra is wrong" from
"the experiment failed because float round-off broke the algebra". The
ℚ layer is the falsifiability anchor.
