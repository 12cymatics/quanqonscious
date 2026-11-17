# Bit-exact protocol

The fixtures below are compared against the kernel that wrote them.

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
checked by the same gate. Like the primitive fixtures they are produced by the
kernel they are compared against, so they detect **drift, not error** — see
the README's falsification criterion 3 for what that distinction buys.

## Building the fixtures from Python (authoritative source)

```bash
python scripts/build_fixtures.py --seed 0xC0DEC0DE --n 32 --out fixtures
```

## Verification

Once `fixtures/*.json` is populated by `build_fixtures.py`, run:

```bash
python scripts/verify_bit_exact.py
```

It exits 0 if the ℚ kernel reproduces every fixture record exactly, and
prints what it checked: 32 inputs, 32 sutra records, 96 conservation records,
then 8 inputs and 696 canonical-29 records. An unchecked fixture key is
itself a failure — the gate refuses to run at all against a missing reference
rather than rebuilding one from the code under test. If the Python kernel
disagrees with a committed fixture, it has drifted: stop, find which changed
and why, and only then regenerate.

## Why exact ℚ matters for the LLM trainer

Float arithmetic introduces approximations that compound silently
through 29 sutra applications. The kernel is the only ground truth we
have for "this auxiliary-loss landscape is genuinely the one the
algebra describes". Without exact ℚ verification, we cannot
distinguish "the experiment failed because the algebra is wrong" from
"the experiment failed because float round-off broke the algebra". The
ℚ layer is the falsifiability anchor.
