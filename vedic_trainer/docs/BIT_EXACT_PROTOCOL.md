# Bit-exact protocol with `vedic_v18.16_strict_kernel.html`

The Fraction kernel in `vedic/kernel/z2_primitives.py` and the
JavaScript kernel inside the v18.16 simulator are two independent
implementations of the **same** exact-rational algorithm. (This paragraph
named `sutras_exact.py` for as long as that file had been renamed.) Bit-exact
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
