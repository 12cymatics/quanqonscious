"""Independent exact comparisons of the repository's 16+13 book applications.

The arithmetic inputs vary over each method's stated domain.  The generated
operands are experiment data, never quantum couplings.  SymPy expansion,
linear solves, root sets and integer long division provide direct references.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from fractions import Fraction
from pathlib import Path

import sympy as sp

import vedic_book as book


X, Y, Z = sp.symbols("x y z")


def _poly(values):
    return sum(sp.Rational(v) * X**i for i, v in enumerate(values))


def _coefficients(expression, length=None):
    polynomial = sp.Poly(sp.expand(expression), X)
    values = tuple(reversed(polynomial.all_coeffs()))
    if length is not None:
        if len(values) > length and any(values[length:]):
            raise ArithmeticError("Reference polynomial exceeds the requested degree")
        values = values[:length] + (sp.S.Zero,) * max(0, length - len(values))
    return values


def _long_division(numerator, denominator):
    remainder, digits = numerator, []
    while True:
        digit, remainder = divmod(10 * remainder, denominator)
        digits.append(digit)
        if remainder == numerator:
            return tuple(digits)


def _poly_data(rng, degree):
    values = [Fraction(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(degree)]
    values.append(Fraction(rng.choice((-1, 1)) * rng.randint(1, 9), rng.randint(1, 7)))
    return values


def _parameters(sid, index, rng):
    n = index + 1
    if sid == 1:
        return {"denominator": 10 * (index % 18 + 1) - 1}
    if sid == 2:
        base = 10 ** (index % 5 + 1)
        return {"a": base + Fraction(rng.randint(-9, 9), 7),
                "b": base + Fraction(rng.randint(-9, 9), 11), "base": base}
    if sid == 3:
        return {"a": _poly_data(rng, index % 6 + 1),
                "b": _poly_data(rng, index % 4 + 1)}
    if sid == 4:
        return {"dividend": _poly_data(rng, index % 7 + 2),
                "divisor": _poly_data(rng, index % 3 + 1)}
    if sid == 5:
        if index % 4 == 0:
            return {"a": n, "b": n + 2, "c": n + 2, "d": n}
        return {"a": n, "b": 6 * n, "c": 2 * n, "d": 3 * n}
    if sid == 6:
        a, b, d, e = n + 1, n + 2, n + 2, n + 4
        ratio = Fraction((-1) ** index * (n + 2), n + 3)
        c, f = (ratio * b, ratio * e) if index % 2 else (ratio * a, ratio * d)
        return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}
    if sid == 7:
        return {"a": n + 2, "b": (-1) ** index,
                "c": Fraction(n, 7), "d": Fraction(2 - n, 11)}
    if sid == 8:
        return {"a": Fraction(n, n + 1), "b": (-1) ** index * (n + 2),
                "c": 0 if index % 3 == 0 else n + 3}
    if sid == 9:
        return {"p": _poly_data(rng, index % 9)}
    if sid in (10, 23):
        base = 10 ** (index % 5 + 1)
        return {"n": base + Fraction((-1) ** index * n, 13), "base": base}
    if sid == 11:
        a, b, root = n + 1, n - 1, Fraction(index % 4 - 2, 3)
        return {"a": a, "b": b, "total": (root + a)**4 + (root + b)**4}
    if sid == 12:
        denominators = (3, 7, 11, 13, 17, 19, 21, 27, 31, 37, 41, 49, 59, 73, 81, 101)
        return {"denominator": denominators[index % len(denominators)]}
    if sid == 13:
        if index % 5 == 0:
            return {"first": [n, 1], "difference": [0, 0]}
        return {"first": [n + 1, index % 3 + 1],
                "difference": [1, index % 2]}
    if sid == 14:
        return {"n": rng.randint(-10**8, 10**8), "places": index % 7 + 1}
    if sid in (15, 16, 29):
        return {"factors": [_poly_data(rng, (index + j) % 4)
                            for j in range(index % 4 + 1)]}
    if sid == 17:
        return {"a": Fraction(n + 9, 3), "b": Fraction(2 * n - 3, 7),
                "reference_base": 10 ** (index % 3 + 1),
                "ratio": Fraction(index % 5 + 1, index % 7 + 1)}
    if sid == 18:
        divisors = [2**k for k in range(1, index % 6 + 3)]
        return {"dividend": divisors[-1] * (n + 1) + 1, "divisors": divisors}
    if sid == 19:
        first, other = [n + 1, (-1) ** index], [Fraction(2, 7), n + 2]
        quadratic = _coefficients(_poly(first) * _poly(other), 3)
        return {"quadratic": quadratic, "factor": first}
    if sid == 20:
        return {"numerator": index % 6 + 1}
    if sid == 21:
        divisors = (7, 11, 13, 17, 19, 29, 31, 37)
        divisor = divisors[index % len(divisors)]
        return {"n": rng.randint(0, 10**10) * divisor + index % 2, "divisor": divisor}
    if sid == 22:
        base = 10 ** (index % 5 + 1)
        return {"n": base + Fraction((-1) ** index * n, 3), "base": base}
    if sid == 24:
        base = 10 ** (index % 4 + 1)
        tail = rng.randint(1, base - 1)
        return {"a": n * base + tail, "b": n * base + base - tail}
    if sid == 25:
        return {"a": [n, 1], "b": [n + 2, index % 2 + 1],
                "common": [index % 3, 1], "numerator_last": index % 4,
                "denominator_last": index % 5 + 1}
    if sid == 26:
        return {"a": _poly_data(rng, index % 7), "b": _poly_data(rng, index % 5)}
    if sid == 27:
        l = (Fraction(n + 1, 3), Fraction((-1) ** index, 2), Fraction(n, 7))
        r = (Fraction(n + 2, 5), Fraction(n + 3, 7), Fraction(2 - n, 3))
        return {"a": l[0] * r[0], "b": l[1] * r[1], "c": l[2] * r[2],
                "xy": l[0] * r[1] + l[1] * r[0],
                "yz": l[1] * r[2] + l[2] * r[1],
                "zx": l[2] * r[0] + l[0] * r[2]}
    if sid == 28:
        split, sign = Fraction(n + 1, n + 2), (-1) ** index
        return {"total": split + sign / split, "split": split,
                "numerator": [n, 1], "denominator": [n + 2, 1], "sign": sign}
    raise ValueError(f"Unknown sutra {sid}")


def _root_set(numerator, denominators):
    roots = sp.solveset(sp.expand(numerator), X, sp.S.Complexes)
    excluded = sp.Union(*(sp.solveset(p, X, sp.S.Complexes) for p in denominators))
    return sp.Complement(roots, excluded)


def _reference(sid, p, value):
    if sid in (1, 12):
        return _long_division(1, int(p["denominator"])), value
    if sid in (2, 17, 24):
        return sp.Rational(p["a"]) * sp.Rational(p["b"]), value
    if sid == 3:
        length = len(p["a"]) + len(p["b"]) - 1
        return _coefficients(_poly(p["a"]) * _poly(p["b"]), length), value
    if sid == 4:
        a, b = _poly(p["dividend"]), _poly(p["divisor"])
        q, r = sp.div(a, b, X, domain=sp.QQ)
        return (_coefficients(q, len(value[0])), _coefficients(r, len(value[1]))), value
    if sid == 5:
        a, b, c, d = (sp.Rational(p[k]) for k in ("a", "b", "c", "d"))
        return sp.solveset((X+a)*(X+b)-(X+c)*(X+d), X, sp.S.Complexes), value
    if sid == 6:
        matrix = sp.Matrix([[p["a"], p["b"]], [p["d"], p["e"]]])
        return tuple(matrix.inv() * sp.Matrix([p["c"], p["f"]])), value
    if sid == 7:
        matrix = sp.Matrix([[p["a"], p["b"]], [p["b"], p["a"]]])
        return tuple(matrix.inv() * sp.Matrix([p["c"], p["d"]])), value
    if sid == 8:
        a, b, c = (sp.Rational(p[k]) for k in ("a", "b", "c"))
        _, h, k, roots = value
        residual = sp.expand(a * (X-h)**2+k-(a*X**2+b*X+c))
        direct_roots = sp.solveset(a*X**2+b*X+c, X, sp.S.Complexes)
        return (sp.S.Zero, direct_roots), (residual, sp.FiniteSet(*roots))
    if sid == 9:
        return _coefficients(sp.diff(_poly(p["p"]), X), len(value)), value
    if sid in (10, 23):
        return sp.Rational(p["n"])**2, value
    if sid == 11:
        a, b, total = (sp.Rational(p[k]) for k in ("a", "b", "total"))
        original = (X+a)**4 + (X+b)**4 - total
        reconstructed = 2 * sp.prod(X-r for r in value)
        return sp.S.Zero, sp.simplify(sp.expand(reconstructed-original))
    if sid == 13:
        a, d = _poly(p["first"]), _poly(p["difference"])
        A, B, C, D = (a+k*d for k in range(4))
        return _root_set(C*D+B*D-B*C-A*D, (A, B, C, D)), value
    if sid == 14:
        return p["n"] * (10**p["places"]-1), value
    if sid in (15, 26, 29):
        factors = p["factors"] if sid in (15, 29) else (p["a"], p["b"])
        product = sp.expand(sp.prod(_poly(f) for f in factors))
        total = product.subs(X, 1)
        return ((total, total) if sid == 29 else total), value
    if sid == 16:
        product = sp.prod(_poly(f) for f in p["factors"])
        return _coefficients(sp.diff(product, X), len(value)), value
    if sid == 18:
        return tuple((p["dividend"] // d, p["dividend"] % d) for d in p["divisors"]), value
    if sid == 19:
        quotient, remainder = sp.div(_poly(p["quadratic"]), _poly(p["factor"]), X, domain=sp.QQ)
        if remainder != 0:
            raise ArithmeticError("Generator produced an invalid factor")
        return _coefficients(quotient, 2), value
    if sid == 20:
        return _long_division(p["numerator"], 7), value
    if sid == 21:
        direct = p["n"] % p["divisor"] == 0
        preserved = all((v % p["divisor"] == 0) == direct for chain in value[1:] for v in chain)
        return (direct, True), (value[0], preserved)
    if sid == 22:
        return sp.Rational(p["n"])**3, value
    if sid == 25:
        A, B, C = _poly(p["a"]), _poly(p["b"]), _poly(p["common"])
        d, e = sp.Rational(p["numerator_last"]), sp.Rational(p["denominator_last"])
        return _root_set(A*(B*C+e)-B*(A*C+d), (B, B*C+e)), value
    if sid == 27:
        a, b, c, xy, yz, zx = (sp.Rational(p[k]) for k in ("a", "b", "c", "xy", "yz", "zx"))
        original = a*X**2+b*Y**2+c*Z**2+xy*X*Y+yz*Y*Z+zx*Z*X
        reconstructed = sp.prod(sum(c*v for c, v in zip(f, (X,Y,Z))) for f in value)
        return sp.S.Zero, sp.expand(reconstructed-original)
    if sid == 28:
        A, B = _poly(p["numerator"]), _poly(p["denominator"])
        total, sign = sp.Rational(p["total"]), p["sign"]
        return _root_set(A**2+sign*B**2-total*A*B, (A,B)), value
    raise ValueError(f"Unknown reference {sid}")


def _equivalent(expected, actual):
    if isinstance(expected, (tuple, list)):
        return isinstance(actual, (tuple, list)) and len(expected) == len(actual) and all(
            _equivalent(left, right) for left, right in zip(expected, actual))
    if isinstance(expected, sp.Set):
        return expected == actual
    if isinstance(expected, bool):
        return bool(actual) == expected
    return sp.simplify(expected-actual) == 0


def _encode(value):
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def run_arithmetic_comparison(samples_per_method=32, seed=20261122):
    if type(samples_per_method) is not int or samples_per_method < 1:
        raise ValueError("samples_per_method must be a positive integer")
    rng = random.Random(seed)
    rows, traces = [], []
    start = time.perf_counter_ns()
    for sid in range(1, 30):
        source = book.SUTRAS[sid-1]
        arithmetic_ns = reference_ns = 0
        digest = hashlib.sha256()
        unique_operands = set()
        for index in range(samples_per_method):
            parameters = _parameters(sid, index, rng)
            t0 = time.perf_counter_ns()
            result = book.evaluate(sid, **parameters)
            arithmetic_ns += time.perf_counter_ns()-t0
            t0 = time.perf_counter_ns()
            expected, observed = _reference(sid, parameters, result.value)
            verified = _equivalent(expected, observed)
            reference_ns += time.perf_counter_ns()-t0
            if not verified:
                raise ArithmeticError(f"Sutra {sid}, case {index}: expected {expected}, observed {observed}")
            record = {"sutra_id": sid, "case": index, "operands": _encode(parameters),
                      "value": _encode(result.value), "reference": _encode(expected),
                      "comparison_value": _encode(observed), "equal": True}
            unique_operands.add(json.dumps(record["operands"], sort_keys=True))
            digest.update(json.dumps(record, sort_keys=True).encode())
            traces.append(record)
        rows.append({"id": sid, "name": source.name, "application": source.application,
                     "book_pages": list(source.pages), "cases": samples_per_method,
                     "unique_operand_sets": len(unique_operands),
                     "equal": samples_per_method, "arithmetic_ns": arithmetic_ns,
                     "reference_and_comparison_ns": reference_ns,
                     "case_digest_sha256": digest.hexdigest()})
    source_path = Path(book.__file__).resolve()
    return {"schema": "grvq-29-arithmetic-comparison-v1", "seed": seed,
            "samples_per_method": samples_per_method, "total_cases": len(traces),
            "methods": rows, "cases": traces,
            "elapsed_ns": time.perf_counter_ns()-start,
            "source": {"path": "vedic_book.py", "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
            "arithmetic": "exact rational and symbolic algebra",
            "reference": "independent symbolic expansion, differentiation, matrix inverse, root sets and integer long division",
            "timing_note": "Separate operation timings and reference-plus-comparison timings; these are not equivalent-work speed benchmarks."}


def run_arithmetic_suite(seed=20261122, cases_per_sutra=64):
    return run_arithmetic_comparison(samples_per_method=cases_per_sutra, seed=seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-method", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = run_arithmetic_comparison(args.samples_per_method, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"total_cases": output["total_cases"], "methods": len(output["methods"]),
                      "output": str(args.output)}))
