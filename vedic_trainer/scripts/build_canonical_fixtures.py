"""Build a regression reference for the CANONICAL 29 sutras.

What this is, and is not
------------------------
`verify_bit_exact.py` gates `vedic/kernel/z2_primitives.py`. That module says
of itself, in its first paragraph: *"This module is not an implementation of
the Vedic sutras. The authority for those is sutras_canonical"* -- and the
two use conflicting numbering (here S19 is Lopana-Sthapanabhyam and S24 is
Kevalaih Saptakam; the engine has them the other way round).

So the module the README calls "the single authority" for the 29 sutras had
no fixture-backed gate at all, while the README's falsification criterion 3
("any bit-exactness mismatch between the ℚ kernel and the committed
fixtures") read as though it covered them.

**These fixtures cannot prove correctness.** They are written by the same
kernel the gate compares against, so they detect *drift*, not error -- the
distinction that made the old regenerate-on-missing fallback worthless. What
they give is a regression reference: if an operator's output changes, the
gate says so instead of the change passing silently.

The correctness cross-check is external and stays external: the user's
`vedic_v18.24_full_kernel.html` is the upstream definition
`sutras_canonical.py` was ported from, and agreement with it is checked by
exporting from that file, not by anything in this repository.

Two strengths are recorded per input. Strength 0 exercises the §12Y
guarantee (α → 0 ⇒ identity) and would be satisfied by an operator that does
nothing at all, so it is never recorded alone.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

from vedic.kernel import sutras_canonical as K
from vedic.kernel.q import Q16

REPO = Path(__file__).resolve().parents[1]
STRENGTHS = (0, 50, 100)


def _obj(x: Fraction) -> dict[str, int]:
    return {"num": x.numerator, "den": x.denominator}


def _q(psi: Q16) -> list[dict[str, int]]:
    return [_obj(v) for v in psi]


def _random_q16(rng: random.Random, denom_max: int = 1000) -> Q16:
    return tuple(Fraction(rng.randint(-denom_max, denom_max),
                          rng.randint(1, denom_max)) for _ in range(16))


def build(seed: int, n_inputs: int, out_dir: Path) -> int:
    rng = random.Random(seed)
    inputs = [_random_q16(rng) for _ in range(n_inputs)]

    records = []
    for i, psi in enumerate(inputs):
        for sid in K.ALL:
            for strength in STRENGTHS:
                records.append({
                    "input": i,
                    "sutra": sid,
                    "name": K.NAMES[sid],
                    "strength": strength,
                    "alpha": _obj(K.alpha(sid, Fraction(strength))),
                    "output": _q(K.apply_sutra(sid, psi, Fraction(strength))),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "canonical_inputs.json").write_text(json.dumps(
        {"seed": seed, "n": n_inputs, "denom_max": 1000,
         "inputs": [_q(p) for p in inputs]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (out_dir / "canonical_sutra_outputs.json").write_text(json.dumps(
        {"strengths": list(STRENGTHS), "records": records},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "fixtures")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--inputs", type=int, default=8)
    args = ap.parse_args()
    n = build(args.seed, args.inputs, args.out)
    print(f"wrote {n} canonical records "
          f"({args.inputs} inputs x 29 sutras x {len(STRENGTHS)} strengths) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
