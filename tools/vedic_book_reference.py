#!/usr/bin/env python3
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vedic_book import evaluate  # noqa: E402
from vedic_book_cases import EXAMPLE_PARAMETERS  # noqa: E402
from vedic_field import FieldBinding  # noqa: E402


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    return Fraction(value)


def encode(value):
    value = Fraction(int(value.p), int(value.q)) if hasattr(value, "p") else Fraction(value)
    return f"{value.numerator}/{value.denominator}"


NEIGHBORS = tuple(tuple(i ^ (1 << bit) for bit in range(4)) for i in range(16))


def run(case):
    sid = case["id"]
    parameters = EXAMPLE_PARAMETERS[sid] if case.get("operands") is None else {
        name: decode(value) for name, value in case["operands"].items()}
    try:
        result = evaluate(sid, **parameters)
    except (ValueError, ArithmeticError, ZeroDivisionError) as error:
        return {"refused": str(error)}
    out = {"terms": [encode(c) for c in result.spectral_coefficients]}
    if "psi" in case:
        binding = FieldBinding.from_parameters(sid, None if case.get("operands") is None else parameters)
        field = binding.rotate_matched(tuple(decode(case["psi"])), NEIGHBORS, Fraction(case["strength"]))
        out["psi"] = [encode(v) for v in field]
    return out


if __name__ == "__main__":
    print(json.dumps({"results": [run(case) for case in json.load(sys.stdin)["cases"]]}))
