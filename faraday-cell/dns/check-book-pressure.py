#!/usr/bin/env python3
"""Exact arithmetic checks for pressure in a surface-following coordinate.

Run with the unchanged, pinned upstream arithmetic module:
  python dns/check-book-pressure.py
An explicit --book /path/to/vedic_book.py can select the same pinned source.

The named book procedures are used as implemented in that module: coefficient
convolution (Urdhva), polynomial differentiation (Calana), and the complete
factor product derivative (Gunakasamuccaya). Coordinate bookkeeping and the
chain rule retain their ordinary mathematical names. No fluid solver is
replaced, approximated, or relabelled by this verification program.

The manufactured surface is H=h+A*(1-(x*x+y*y)/R**2)**2, with h,A,R positive.
It has pinned elevation at the circular rim. Its additional zero rim slope
belongs to this test field and is not imposed as an extra solver condition.
For z=sigma*H and q=q0+a*x+b*y+c*sigma*H, the complete coordinate chain rule
must give the physical gradient (a,b,c), including every metric-slope term.

This verifies exact arithmetic and an analytic identity. It does not execute
the production discretization or establish convergence, fluid validation,
or experimental agreement. Those remain separate, required solver checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import sympy as sp


UPSTREAM_COMMIT = "e951e23ef93b683b4194b2dde3154952ff7dea73"
UPSTREAM_BLOB = "98d7d240670ade6701aaaaae4811ec8510203f90"
UPSTREAM_URL = (
    "https://github.com/12cymatics/quanqonscious/blob/"
    + UPSTREAM_COMMIT + "/vedic_book.py"
)
BOOK_URL = "https://archive.org/details/vedic-mathematics-bharati-krishna-tirth-ji-maharaj"
CORRECTION_URL = "https://github.com/12cymatics/quanqonscious/pull/172"
CORRECTION_COMMIT = "aee61f40a967ea08bf5e8879edd1da9c9e47d624"
ZERO_EXPONENT = (0, 0, 0)


class CheckFailure(RuntimeError):
    pass


def require(condition, message):
    # Ordinary assert would disappear under python -O; these checks must not.
    if not condition:
        raise CheckFailure(message)


def load_book(path):
    source = path.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(source)).encode("ascii") + b"\0" + source).hexdigest()
    require(actual == UPSTREAM_BLOB,
            f"Book source differs from pinned Git blob: {actual}; expected {UPSTREAM_BLOB}")
    spec = importlib.util.spec_from_file_location("pressure_verified_book", path)
    require(spec is not None and spec.loader is not None, "Cannot load pinned book module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        # Verification must not add cache files to the reproducible bundle.
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


class CoordinatePolynomials:
    """Sparse Q[x,y,sigma] bookkeeping around univariate book operations.

    Products use an explicit Kronecker encoding. Each exponent radix is larger
    than the sum of the input degrees, so no exponent carry or collision can
    occur. All encoded coefficients passed to the book are rationals; no
    symbolic operand is smuggled into its rational-only interface.
    """

    def __init__(self, book):
        self.book = book
        self.convolutions = 0
        self.derivative_fibres = 0

    def clean(self, terms):
        result = {}
        for exponents, coefficient in terms.items():
            require(len(exponents) == 3 and all(type(e) is int and e >= 0 for e in exponents),
                    "Invalid polynomial exponent")
            value = self.book.rational(coefficient)
            if value != 0:
                result[exponents] = value
        return result

    def constant(self, value):
        return self.clean({ZERO_EXPONENT: value})

    def variable(self, axis):
        e = [0, 0, 0]
        e[axis] = 1
        return {tuple(e): sp.S.One}

    def add(self, *polynomials):
        out = {}
        for polynomial in polynomials:
            for exponent, value in polynomial.items():
                out[exponent] = out.get(exponent, sp.S.Zero) + value
        return self.clean(out)

    def scale(self, polynomial, value):
        value = self.book.rational(value)
        return self.clean({e: value * c for e, c in polynomial.items()})

    def multiply(self, left, right):
        if not left or not right:
            return {}
        limits = [max(e[i] for e in left) + max(e[i] for e in right) for i in range(3)]
        radix_x, radix_y = limits[0] + 1, limits[1] + 1

        def index(e):
            return e[0] + radix_x * (e[1] + radix_y * e[2])

        def encode(p):
            encoded = [sp.S.Zero] * (max(index(e) for e in p) + 1)
            for e, value in p.items():
                encoded[index(e)] = value
            return encoded

        encoded = self.book.urdhva(encode(left), encode(right)).value
        self.convolutions += 1
        out = {}
        for k, coefficient in enumerate(encoded):
            if coefficient == 0:
                continue
            quotient, x = divmod(k, radix_x)
            sigma, y = divmod(quotient, radix_y)
            e = (x, y, sigma)
            require(all(e[i] <= limits[i] for i in range(3)), "Convolution escaped exponent bounds")
            out[e] = coefficient
        return self.clean(out)

    def derivative(self, polynomial, axis):
        fibres = {}
        for exponents, value in polynomial.items():
            key = list(exponents)
            degree = key[axis]
            key[axis] = 0
            fibres.setdefault(tuple(key), {})[degree] = value
        out = {}
        for key, fibre in fibres.items():
            sequence = [fibre.get(i, sp.S.Zero) for i in range(max(fibre) + 1)]
            derived = self.book.calana(sequence).value
            self.derivative_fibres += 1
            for degree, value in enumerate(derived):
                exponents = list(key)
                exponents[axis] = degree
                out[tuple(exponents)] = value
        return self.clean(out)

    @staticmethod
    def expression(polynomial, variables):
        return sum((coefficient * sp.prod(v ** e for v, e in zip(variables, exponents))
                    for exponents, coefficient in polynomial.items()), sp.S.Zero)


def derive_pressure(book):
    p = CoordinatePolynomials(book)
    h, amplitude, radius = sp.Rational(1, 250), sp.Rational(1, 1000), sp.Rational(17, 1000)
    q0, a, b, c = sp.Rational(7, 3), sp.Rational(11, 7), sp.Rational(-13, 5), sp.Rational(137, 9)
    require(h > 0 and amplitude > 0 and radius > 0, "Manufactured surface requires positive dimensions")
    x, y, sigma = (p.variable(i) for i in range(3))
    radial_square = p.add(p.multiply(x, x), p.multiply(y, y))
    rim_factor = p.add(p.constant(1), p.scale(radial_square, -1 / radius ** 2))
    height = p.add(p.constant(h), p.scale(p.multiply(rim_factor, rim_factor), amplitude))
    pressure = p.add(p.constant(q0), p.scale(x, a), p.scale(y, b),
                     p.scale(p.multiply(sigma, height), c))
    hx, hy = p.derivative(height, 0), p.derivative(height, 1)
    qx, qy, qs = (p.derivative(pressure, i) for i in range(3))

    # Independently express the chosen fields, without using the encoded
    # convolution or the book derivative results as the expected values.
    X, Y, S = sp.symbols("x y sigma")
    variables = (X, Y, S)
    expected_height = h + amplitude * (1 - (X ** 2 + Y ** 2) / radius ** 2) ** 2
    expected_pressure = q0 + a * X + b * Y + c * S * expected_height
    independent = {
        "height": (height, expected_height),
        "pressure": (pressure, expected_pressure),
        "height_x": (hx, sp.diff(expected_height, X)),
        "height_y": (hy, sp.diff(expected_height, Y)),
        "pressure_x": (qx, sp.diff(expected_pressure, X)),
        "pressure_y": (qy, sp.diff(expected_pressure, Y)),
        "pressure_sigma": (qs, sp.diff(expected_pressure, S)),
    }
    for name, (polynomial, expected) in independent.items():
        require(sp.expand(p.expression(polynomial, variables) - expected) == 0,
                f"Independent exact derivation failed: {name}")

    # H is strictly positive: h>0 plus a positive multiple of a square.
    # Multiplying the physical-gradient identities by H is therefore
    # equivalent, and permits coefficient-wise polynomial equality checks.
    residuals = {
        "horizontal_x": p.add(p.multiply(height, qx),
                              p.scale(p.multiply(p.multiply(sigma, hx), qs), -1),
                              p.scale(height, -a)),
        "horizontal_y": p.add(p.multiply(height, qy),
                              p.scale(p.multiply(p.multiply(sigma, hy), qs), -1),
                              p.scale(height, -b)),
        "vertical": p.add(qs, p.scale(height, -c)),
    }
    for name, residual in residuals.items():
        require(not residual, f"Nonzero exact pressure metric residual: {name}: {residual}")
    physical = (sp.diff(expected_pressure, X) - S * sp.diff(expected_height, X) * sp.diff(expected_pressure, S) / expected_height,
                sp.diff(expected_pressure, Y) - S * sp.diff(expected_height, Y) * sp.diff(expected_pressure, S) / expected_height,
                sp.diff(expected_pressure, S) / expected_height)
    require(all(sp.cancel(value - target) == 0 for value, target in zip(physical, (a, b, c))),
            "Independent physical chain rule failed")

    # Exact product-rule checks use rational slices because the book accepts
    # rational coefficients, not multivariate symbolic coefficients.
    product_rule_checks = 0
    for axis, variable in enumerate((X, Y)):
        other = Y if axis == 0 else X
        for fixed in (sp.S.Zero, radius / 3, -3 * radius / 5, radius):
            factor = (1 - fixed ** 2 / radius ** 2, sp.S.Zero, -1 / radius ** 2)
            coefficients = book.gunakasamuccaya((factor, factor)).value
            derived_slice = amplitude * book.polynomial(coefficients, variable)
            expected_slice = sp.diff(expected_height, variable).subs(other, fixed)
            require(sp.expand(derived_slice - expected_slice) == 0, "Full book product derivative failed")
            product_rule_checks += 1

    rim_checks = 0
    for cosine, sine in ((1, 0), (0, 1), (sp.Rational(3, 5), sp.Rational(4, 5)),
                         (sp.Rational(-5, 13), sp.Rational(12, 13))):
        require(cosine ** 2 + sine ** 2 == 1, "Rim point is not on the circle")
        require(expected_height.subs({X: radius * cosine, Y: radius * sine}) == h,
                "Manufactured rim elevation is not pinned")
        rim_checks += 1
    rho2 = sp.Symbol("rho2")
    require((amplitude * (1 - rho2 / radius ** 2) ** 2).subs(rho2, radius ** 2) == 0,
            "Symbolic pinned-rim identity failed")

    # A missing metric term must be observable on this non-flat test field.
    require(bool(p.add(p.multiply(height, qx), p.scale(height, -a))),
            "Fixture cannot detect an omitted x metric-slope term")
    require(bool(p.add(p.multiply(height, qy), p.scale(height, -b))),
            "Fixture cannot detect an omitted y metric-slope term")
    return {
        "surface": "H=h+A*(1-(x^2+y^2)/R^2)^2; z=sigma*H",
        "pressure": "q=q0+a*x+b*y+c*sigma*H",
        "parameters": {k: str(v) for k, v in {"h": h, "A": amplitude, "R": radius,
                                             "q0": q0, "a": a, "b": b, "c": c}.items()},
        "physical_gradient": [str(v) for v in (a, b, c)],
        "identities": ["H*q_x-sigma*H_x*q_sigma-a*H = 0",
                       "H*q_y-sigma*H_y*q_sigma-b*H = 0", "q_sigma-c*H = 0"],
        "residual_nonzero_coefficients": {name: len(value) for name, value in residuals.items()},
        "independent_symbolic_checks": len(independent) + 1,
        "book_convolutions": p.convolutions,
        "book_derivative_fibres": p.derivative_fibres,
        "book_full_product_derivative_checks": product_rule_checks,
        "pinned_rim_point_checks": rim_checks,
        "symbolic_pinned_rim_check": "PASS",
        "omitted_metric_term_sensitivity": "PASS, both horizontal directions",
    }


def corrected_method_checks(book):
    # Corrections documented by Claude in PR172. They validate the unchanged
    # arithmetic dependency; these are not fluid-physics or solver tests.
    X = sp.Symbol("x")
    examples = (
        ((-12, 9, -7, -4), (-4, 2), (sp.Rational(-21, 2), sp.Rational(-15, 2), -2), (-54,)),
        ((-5, -1, 3), (-7, 3), (2, 1), (9,)),
    )
    for dividend, divisor, expected_q, expected_r in examples:
        quotient, remainder = book.shishyate_divisor(dividend, divisor).value
        require(quotient == expected_q and remainder == expected_r, "Printed page57 worked example failed")
        rebuilt = (book.polynomial(divisor, X) * book.polynomial(quotient, X)
                   + book.polynomial(remainder, X) - book.polynomial(dividend, X))
        require(sp.expand(rebuilt) == 0, "Divisor*quotient+remainder did not reconstruct dividend")
    require(book.shishyate(65, (2, 4, 8, 16, 32, 64)).value ==
            ((32, 1), (16, 1), (8, 1), (4, 1), (2, 1), (1, 1)), "Printed page19 example failed")
    require(book.tavadunam_cube(9989, 10000).value == 996703628669,
            "Printed cube-of9989 example failed")
    cubes = 0
    for n in range(2, 401):
        for base in (10 ** (len(str(n)) - 1), 10 ** len(str(n))):
            require(book.tavadunam_cube(n, base).value == n ** 3, "Deficiency cube disagrees with integer cube")
            cubes += 1
    invalid = (
        ("different remainders", ValueError, lambda: book.shishyate(65, (2, 3))),
        ("zero leading divisor", ValueError, lambda: book.shishyate_divisor((1, 2, 3), (4, 0))),
        ("float operand", TypeError, lambda: book.rational(0.1)),
        ("boolean operand", TypeError, lambda: book.rational(True)),
    )
    for name, expected_exception, operation in invalid:
        try:
            operation()
        except expected_exception:
            pass
        else:
            raise CheckFailure(f"Invalid input accepted: {name}")
    return {"printed_page57_divisions": len(examples), "printed_page19_constant_remainder": "PASS",
            "printed_cube9989": "PASS", "independent_cube_checks": cubes,
            "invalid_operand_rejections": len(invalid)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, default=Path(__file__).resolve().parent.parent / "vedic_book.py",
                        help="Unchanged vedic_book.py from the pinned upstream commit (default: repository root)")
    args = parser.parse_args()
    book = load_book(args.book)
    report = {
        "status": "PASS",
        "notice": "Exact arithmetic and analytic coordinate identity only; production solver and physical validation are separate.",
        "book": {"title": "Vedic Mathematics", "author": "Bharati Krishna Tirthaji",
                 "edition": "1992 revised edition, 2010/2012 reprint", "url": BOOK_URL,
                 "method_pages": {"coefficient_convolution": [33, 34, 35, 36],
                                  "polynomial_derivative": [145, 175, 176, 177, 178],
                                  "full_factor_product_derivative": [175, 176, 177, 178]}},
        "unchanged_source": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "git_blob": UPSTREAM_BLOB},
        "historical_correction": {"url": CORRECTION_URL, "commit": CORRECTION_COMMIT,
                                  "attribution": "Claude-authored correction; independently checked here",
                                  "printed_pages": [19, 57, 288, 289], "introductory_cube_page": "xliii"},
        "pressure_derivation": derive_pressure(book),
        "corrected_arithmetic_regressions": corrected_method_checks(book),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (CheckFailure, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
