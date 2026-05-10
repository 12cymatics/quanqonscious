"""Generate the synthetic LoRA fine-tuning corpus.

For each base sentence in the seed text file (one per line):

    * One contradiction pair (P, ¬P) with auxiliary axis = polarity.
    * Four axis-emphasis paraphrase pairs (one per axis 0..3).

Each example is written as one JSONL line containing:

    text, label, source_text, axis (or null), psi (16-tuple of {num, den}),
    audit_closed, residuals (4-tuple of {num, den}).

The audit closure is computed against ``trace_sum = idx`` so the R1
residual closes deterministically every 435 examples.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from vedic.data.audit_filter import audit_psi
from vedic.data.synthetic_contradiction import generate_contradiction_pair
from vedic.data.synthetic_paraphrase import generate_paraphrase_pair
from vedic.kernel.q import Q16


def _frac_to_obj(x: Fraction) -> dict[str, int]:
    return {"num": x.numerator, "den": x.denominator}


def _q16_to_obj(psi: Q16) -> list[dict[str, int]]:
    return [_frac_to_obj(x) for x in psi]


def _write_record(
    f, *, idx: int, text: str, label: str, source: str, axis: int | None, psi: Q16
) -> None:
    audit = audit_psi(psi, Fraction(idx))
    f.write(
        json.dumps(
            {
                "idx": idx,
                "text": text,
                "label": label,
                "source": source,
                "axis": axis,
                "psi": _q16_to_obj(psi),
                "audit_closed": audit.closed,
                "residuals": [_frac_to_obj(r) for r in audit.residuals],
            },
            sort_keys=True,
        )
    )
    f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vedic synthetic corpus.")
    parser.add_argument("--input", type=Path, required=True,
                        help="One sentence per line; UTF-8.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as out:
        for line in src:
            text = line.strip()
            if not text:
                continue
            pair = generate_contradiction_pair(text)
            _write_record(out, idx=idx, text=pair.base_text, label="base", source=text, axis=None, psi=pair.base_psi); idx += 1
            _write_record(out, idx=idx, text=pair.contradiction_text, label="contradictory", source=text, axis=0, psi=pair.contradiction_psi); idx += 1
            for axis in range(4):
                pp = generate_paraphrase_pair(text, axis)
                _write_record(out, idx=idx, text=pp.text_a, label="paraphrase_pos", source=text, axis=axis, psi=pp.psi_a); idx += 1
                _write_record(out, idx=idx, text=pp.text_b, label="paraphrase_neg", source=text, axis=axis, psi=pp.psi_b); idx += 1
    print(f"wrote {idx} records to {args.output}")


if __name__ == "__main__":
    main()
