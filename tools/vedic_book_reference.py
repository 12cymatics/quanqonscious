#!/usr/bin/env python3
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.set_int_max_str_digits(0)

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


def operands_of(sid, psi):
    w = [value.numerator // value.denominator for value in psi]
    m = [abs(value) for value in w]

    def base(n):
        return 10 ** len(str(max(n, 1)))

    def pairs(count):
        return [[w[v], w[15 - v]] for v in range(count)]

    top = max(m[0], m[15])

    def nz(v):
        return w[v] + 1 if w[v] >= 0 else w[v] - 1

    def odd(v):
        return 2 * w[v] + 1

    def even(v):
        return 2 * w[v]

    def period(d):
        if d > 999:
            raise ValueError("the repetend is refused above three digits")
        return d

    if sid == 1:
        return {"denominator": period(10 * m[0] + 9)}
    if sid == 2:
        return {"a": w[0], "b": w[15], "base": base(top)}
    if sid == 3:
        return {"a": w[:8], "b": w[8:]}
    if sid == 4:
        return {"dividend": w[:8], "divisor": [w[15], 1]}
    if sid == 5:
        return {"a": w[0], "b": w[1] * w[15], "c": w[1], "d": w[0] * w[15]}
    if sid == 6:
        return {"a": odd(0), "b": even(1), "c": w[15] * even(1), "d": even(2), "e": odd(3), "f": w[15] * odd(3)}
    if sid == 7:
        return {"a": odd(0), "b": even(1), "c": w[2], "d": w[3]}
    if sid == 8:
        return {"a": nz(0), "b": w[1], "c": w[2]}
    if sid == 9:
        return {"p": w}
    if sid == 10:
        return {"n": w[0], "base": base(m[0])}
    if sid == 11:
        return {"a": w[0], "b": w[15], "total": w[1]}
    if sid == 12:
        return {"denominator": period(10 * m[0] + 7)}
    if sid == 13:
        return {"first": [w[0], odd(1)], "difference": [w[2], even(3)]}
    if sid == 14:
        return {"n": w[0], "places": len(str(max(m[0], 1)))}
    if sid == 15:
        return {"factors": pairs(4)}
    if sid == 16:
        return {"factors": pairs(3)}
    if sid == 17:
        return {"a": w[0], "b": w[15], "reference_base": base(top),
                "ratio": Fraction(int(str(max(top, 1))[0]) + 1, 10)}
    if sid == 18:
        return {"dividend": w[0], "divisors": [m[15] + 1]}
    if sid == 19:
        return {"quadratic": [nz(0) * w[2], nz(0) * nz(3) + nz(1) * w[2], nz(1) * nz(3)], "factor": [nz(0), nz(1)]}
    if sid == 20:
        return {"numerator": 1 + m[0] % 6}
    if sid == 21:
        return {"n": m[0], "divisor": 10 * m[15] + 7}
    if sid == 22:
        return {"n": w[0], "base": base(m[0])}
    if sid == 23:
        return {"n": w[15], "base": base(m[15])}
    if sid == 24:
        last = 1 + m[15] % 9
        return {"a": 10 * m[0] + last, "b": 10 * m[0] + 10 - last}
    if sid == 25:
        return {"a": w[0:2], "b": [w[2], nz(3)], "common": [w[4], nz(5)], "numerator_last": w[6], "denominator_last": w[7]}
    if sid == 26:
        return {"a": w[0:4], "b": w[4:8]}
    if sid == 27:
        return {"a": nz(0), "b": w[1] * w[3], "c": w[2] * w[4], "xy": w[3] + nz(0) * w[1],
                "yz": w[1] * w[4] + w[2] * w[3], "zx": w[4] + nz(0) * w[2]}
    if sid == 28:
        return {"total": nz(0) + Fraction(1, nz(0)), "split": nz(0)}
    return {"factors": pairs(8)}


def run(case):
    sid = case["id"]
    if case.get("from_field"):
        parameters = operands_of(sid, decode(case["psi"]))
    elif case.get("operands") is None:
        parameters = EXAMPLE_PARAMETERS[sid]
    else:
        parameters = {name: decode(value) for name, value in case["operands"].items()}
    try:
        result = evaluate(sid, **parameters)
    except (ValueError, ArithmeticError, ZeroDivisionError) as error:
        return {"refused": str(error)}
    out = {"terms": [encode(c) for c in result.spectral_coefficients]}
    if "psi" in case:
        binding = FieldBinding.from_parameters(
            sid, None if case.get("operands") is None and not case.get("from_field") else parameters)
        field = binding.rotate_matched(tuple(decode(case["psi"])), NEIGHBORS, Fraction(case["strength"]))
        out["psi"] = [encode(v) for v in field]
    return out


if __name__ == "__main__":
    print(json.dumps({"results": [run(case) for case in json.load(sys.stdin)["cases"]]}))
