#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import sympy as sp

HERE = Path(__file__).resolve().parent
BITS = 320
ONE = 1 << BITS
RESOLUTIONS = ((16, 24), (32, 48), (64, 96), (128, 192))
CONTROL_RESOLUTIONS = RESOLUTIONS[1:3]
CONTACTS = ("free", "pinned")
QUANTITIES = (("kappa", "curvature"), ("slope2", "slope2"), ("sr", "radial_slope"), ("st", "azimuthal_slope"))
NAMES = ("eta", "slope2", "numerator", "laplacian", "r_eta_r", "eta_theta")
SECOND_ORDER = (7, 2)
FIRST_ORDER = (2, 1)
DECREASING = (1, 1)


def load_pressure_check():
    path = HERE / "check-book-pressure.py"
    spec = importlib.util.spec_from_file_location("book_pressure_check", path)
    if spec is None or spec.loader is None:
        raise OSError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


pressure = load_pressure_check()
require = pressure.require
CheckFailure = pressure.CheckFailure


class SurfacePolynomials(pressure.CoordinatePolynomials):
    def __init__(self, book):
        super().__init__(book)
        self.product_tests = 0

    def multiply(self, left, right):
        product = super().multiply(left, right)
        total = lambda polynomial: sum(polynomial.values(), sp.S.Zero)
        require(total(product) == total(left) * total(right),
                "Gunita-samuccaya: the sum of a product's coefficients is not the product of the sums")
        self.product_tests += 1
        return product


def surfaces(p, radius):
    x, y = p.variable(0), p.variable(1)
    xx, yy = p.multiply(x, x), p.multiply(y, y)
    rim = p.add(p.constant(1), p.scale(p.add(xx, yy), -1 / radius ** 2))
    cap = p.scale(p.multiply(rim, rim), radius / 2)
    lobes = p.add(p.constant(1), p.scale(p.add(xx, p.scale(yy, -1)), 2 / radius ** 2))
    return {"cap": cap, "lobed": p.multiply(cap, lobes)}


def independent_surfaces(radius):
    X, Y = sp.symbols("x y")
    cap = radius / 2 * (1 - (X ** 2 + Y ** 2) / radius ** 2) ** 2
    return (X, Y), {"cap": cap, "lobed": cap * (1 + 2 * (X ** 2 - Y ** 2) / radius ** 2)}


def derive_geometry(p, eta, scale=sp.S.One):
    eta = p.scale(eta, scale)
    x, y = p.variable(0), p.variable(1)
    ex, ey = p.derivative(eta, 0), p.derivative(eta, 1)
    exx, exy, eyy = p.derivative(ex, 0), p.derivative(ex, 1), p.derivative(ey, 1)
    require(p.derivative(ey, 0) == exy, "The mixed derivatives differ")
    ex2, ey2 = p.multiply(ex, ex), p.multiply(ey, ey)
    q = p.add(ex2, ey2)
    one = p.constant(1)
    numerator = p.add(p.multiply(p.add(one, ey2), exx),
                      p.scale(p.multiply(p.multiply(ex, ey), exy), -2),
                      p.multiply(p.add(one, ex2), eyy))
    laplacian = p.add(exx, eyy)
    by_divergence = p.add(p.multiply(p.add(one, q), laplacian),
                          p.scale(p.add(p.multiply(ex, p.derivative(q, 0)),
                                        p.multiply(ey, p.derivative(q, 1))), sp.Rational(-1, 2)))
    require(by_divergence == numerator,
            "The curvature numerator from the divergence differs from the expanded one")
    radial = p.add(p.multiply(x, ex), p.multiply(y, ey))
    azimuthal = p.add(p.multiply(x, ey), p.scale(p.multiply(y, ex), -1))
    rho2 = p.add(p.multiply(x, x), p.multiply(y, y))
    require(p.add(p.multiply(radial, radial), p.multiply(azimuthal, azimuthal)) == p.multiply(rho2, q),
            "(r eta_r)^2 + eta_theta^2 is not r^2 times the squared slope")
    return {"eta": eta, "eta_x": ex, "eta_y": ey, "slope2": q, "numerator": numerator,
            "laplacian": laplacian, "r_eta_r": radial, "eta_theta": azimuthal}


def times_root_cubed(expression, base):
    w = sp.Symbol("w")
    rewritten = expression.replace(
        lambda e: e.is_Pow and e.base == base and (2 * e.exp).is_integer and not e.exp.is_integer,
        lambda e: w ** (2 * e.exp))
    require(all(power.exp.is_integer for power in rewritten.atoms(sp.Pow)),
            "A radical other than a power of the root remained")
    product = sp.Poly(sp.expand(rewritten * w ** 3), w)
    require(all(degree % 2 == 0 for (degree,) in product.monoms()),
            "The root did not cancel from the curvature times its cube")
    return sum((coefficient * base ** (degree // 2)
                for (degree,), coefficient in zip(product.monoms(), product.coeffs())), sp.S.Zero)


def check_independently(p, derived, expression, variables, radius):
    X, Y = variables
    S = sp.Symbol("sigma")
    as_expression = lambda name: p.expression(derived[name], (X, Y, S))
    ex, ey = sp.diff(expression, X), sp.diff(expression, Y)
    s2 = 1 + ex ** 2 + ey ** 2
    s = sp.sqrt(s2)
    cartesian = sp.diff(ex / s, X) + sp.diff(ey / s, Y)
    expected = {
        "eta": expression, "eta_x": ex, "eta_y": ey, "slope2": s2 - 1,
        "numerator": times_root_cubed(cartesian, s2),
        "laplacian": sp.diff(expression, X, 2) + sp.diff(expression, Y, 2),
        "r_eta_r": X * ex + Y * ey, "eta_theta": X * ey - Y * ex,
    }
    for name, value in expected.items():
        require(sp.expand(as_expression(name) - value) == 0, f"Independent derivation differs: {name}")
    r, theta, c, sn = sp.symbols("r theta c sn")
    polar = expression.subs({X: r * sp.cos(theta), Y: r * sp.sin(theta)})
    er, et = sp.diff(polar, r), sp.diff(polar, theta)
    base = 1 + er ** 2 + (et / r) ** 2
    root = sp.sqrt(base)
    model = sp.diff(r * er / root, r) / r + sp.diff(et / root, theta) / r ** 2
    target = as_expression("numerator").subs({X: r * sp.cos(theta), Y: r * sp.sin(theta)})
    difference = sp.expand((times_root_cubed(model, base) - target) * r ** 4)
    difference = difference.subs({sp.cos(theta): c, sp.sin(theta): sn})
    require(not difference.has(theta) and difference.is_polynomial(r, c, sn),
            "MODEL.md's curvature times s^3 did not reduce to a polynomial")
    require(sp.rem(sp.Poly(difference, sn), sp.Poly(sn ** 2 + c ** 2 - 1, sn)).is_zero,
            "MODEL.md's cylindrical curvature is not the derived numerator over s^3")
    rim_points = ((1, 0), (0, 1), (sp.Rational(3, 5), sp.Rational(4, 5)), (sp.Rational(-5, 13), sp.Rational(12, 13)))
    for cosine, sine in rim_points:
        require(cosine ** 2 + sine ** 2 == 1, "Rim point is not on the circle")
        at = {X: radius * cosine, Y: radius * sine}
        require(as_expression("eta").subs(at) == 0, "The surface is not pinned at the rim")
        require(as_expression("r_eta_r").subs(at) == 0, "The surface's radial slope is not zero at the rim")
    return len(expected) + 1 + 2 * len(rim_points)


def iv(value):
    value = Fraction(value)
    scaled = value.numerator << BITS
    return (scaled // value.denominator, -((-scaled) // value.denominator))


def iv_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def iv_sub(a, b):
    return (a[0] - b[1], a[1] - b[0])


def iv_mul(a, b):
    p1, p2, p3, p4 = a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]
    return (min(p1, p2, p3, p4) >> BITS, -((-max(p1, p2, p3, p4)) >> BITS))


def iv_square(a):
    lo, hi = abs(a[0]), abs(a[1])
    low = 0 if a[0] <= 0 <= a[1] else min(lo, hi) ** 2
    return (low >> BITS, -((-max(lo, hi) ** 2) >> BITS))


def iv_over_integer(a, n):
    return (a[0] // n, -((-a[1]) // n))


def iv_divide(a, b):
    require(b[0] > 0, "Interval division needs a positive divisor")
    low = (a[0] << BITS) // (b[1] if a[0] >= 0 else b[0])
    high = -((-a[1] << BITS) // (b[0] if a[1] >= 0 else b[1]))
    return (low, high)


def iv_sqrt(a):
    require(a[0] > 0, "Interval square root needs a positive argument")
    return (math.isqrt(a[0] << BITS), math.isqrt(a[1] << BITS) + 1)


def iv_greatest_magnitude(a):
    return max(abs(a[0]), abs(a[1]))


def machin_pi():
    def arctangent_of_reciprocal(n):
        total, lower, upper, j = Fraction(0), None, None, 0
        while lower is None or upper is None or upper - lower >= Fraction(1, 1 << (BITS + 16)):
            term = Fraction(1, (2 * j + 1) * n ** (2 * j + 1))
            if j % 2 == 0:
                total += term
                upper = total
            else:
                total -= term
                lower = total
            j += 1
        return lower, upper
    low5, high5 = arctangent_of_reciprocal(5)
    low239, high239 = arctangent_of_reciprocal(239)
    return (iv(16 * low5 - 4 * high239)[0], iv(16 * high5 - 4 * low239)[1])


def iv_cos_sin(t):
    require(iv_greatest_magnitude(t) < 8 * ONE, "Angle outside the range the series is run on")
    t2 = iv_square(t)
    cosine, sine = (ONE, ONE), t
    term_c, term_s = (ONE, ONE), t
    j = 0
    while True:
        j += 1
        term_c = iv_over_integer(iv_mul(term_c, t2), (2 * j - 1) * (2 * j))
        term_s = iv_over_integer(iv_mul(term_s, t2), (2 * j) * (2 * j + 1))
        cosine = iv_sub(cosine, term_c) if j % 2 else iv_add(cosine, term_c)
        sine = iv_sub(sine, term_s) if j % 2 else iv_add(sine, term_s)
        last = max(iv_greatest_magnitude(term_c), iv_greatest_magnitude(term_s))
        if last <= 2 and (2 * j + 1) * (2 * j + 2) * ONE > t2[1]:
            return (cosine[0] - last - 1, cosine[1] + last + 1), (sine[0] - last - 1, sine[1] + last + 1)


def compile_polynomial(polynomial):
    terms = []
    for (a, b, sigma), coefficient in polynomial.items():
        require(sigma == 0, "A surface polynomial has no sigma")
        terms.append((a, b, iv(Fraction(int(coefficient.p), int(coefficient.q)))))
    return terms, max((t[0] for t in terms), default=0), max((t[1] for t in terms), default=0)


def powers(value, degree):
    out = [(ONE, ONE)]
    for _ in range(degree):
        out.append(iv_mul(out[-1], value))
    return out


def evaluate(compiled, xp, yp):
    low = high = 0
    for a, b, coefficient in compiled[0]:
        value = iv_mul(coefficient, iv_mul(xp[a], yp[b]))
        low += value[0]
        high += value[1]
    return (low, high)


def to_double(interval, what):
    value = float(Fraction(interval[0], ONE))
    require(value == float(Fraction(interval[1], ONE)), f"{what}: the enclosure does not decide the nearest double")
    return value


def midpoint(interval):
    return float(Fraction(interval[0] + interval[1], 2 * ONE))


def probe(mode, cases):
    done = subprocess.run(["node", str(HERE / "book-surface-probe.mjs")],
                          input=json.dumps({"mode": mode, "cases": cases}),
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise CheckFailure(f"The solver probe failed ({mode}): {done.stderr.strip()}")
    return json.loads(done.stdout)


def reference_on_grid(compiled, grid, pi):
    nr, nth = grid["nr"], grid["nth"]
    degree_x = max(c[1] for c in compiled.values())
    degree_y = max(c[2] for c in compiled.values())
    trig = [iv_cos_sin(iv_over_integer((pi[0] * (2 * k + 1), pi[1] * (2 * k + 1)), nth)) for k in range(nth)]
    cells = [None] * grid["cells"]
    for i in range(nr):
        radius = iv(Fraction(grid["rc"][i]))
        for k in range(nth):
            cosine, sine = trig[k]
            xp = powers(iv_mul(radius, cosine), degree_x)
            yp = powers(iv_mul(radius, sine), degree_y)
            value = {name: evaluate(c, xp, yp) for name, c in compiled.items()}
            root = iv_sqrt(iv_add((ONE, ONE), value["slope2"]))
            cells[grid["order"][i * nth + k]] = {
                "eta": value["eta"],
                "curvature": iv_divide(value["numerator"], iv_mul(root, iv_square(root))),
                "laplacian": value["laplacian"],
                "slope2": value["slope2"],
                "radial_slope": iv_divide(value["r_eta_r"], radius),
                "azimuthal_slope": iv_divide(value["eta_theta"], radius),
            }
    return cells


def error_sums(grid, measured, reference, radius, quantity, exact_name, region):
    nr, nth = grid["nr"], grid["nth"]
    error2, exact2, worst, area = (0, 0), (0, 0), 0, Fraction(0)
    for i in range(nr):
        centre = Fraction(grid["rc"][i])
        if (i != nr - 1) if region == "last row" else (region == "rim") == (centre * 10 <= 9 * radius):
            continue
        weight = iv(centre)
        area += centre * nth
        for k in range(nth):
            cell = grid["order"][i * nth + k]
            exact = reference[cell][exact_name]
            error = iv_sub(iv(Fraction(measured[quantity][cell])), exact)
            error2 = iv_add(error2, iv_mul(weight, iv_square(error)))
            exact2 = iv_add(exact2, iv_mul(weight, iv_square(exact)))
            worst = max(worst, iv_greatest_magnitude(error))
    return {"error2": error2, "exact2": exact2, "worst": worst, "area": area}


def falls_by(coarse, fine, ratio):
    numerator, denominator = ratio
    return (coarse["error2"][0] * fine["exact2"][0] * denominator ** 2
            >= numerator ** 2 * fine["error2"][1] * coarse["exact2"][1])


def strictly_falls(coarse, fine):
    return coarse["error2"][0] * fine["exact2"][0] > fine["error2"][1] * coarse["exact2"][1]


def relative_rms(sums):
    return math.sqrt(midpoint(sums["error2"]) / midpoint(sums["exact2"]))


def required_rate(quantity, region, contact):
    if region == "inner":
        return SECOND_ORDER
    if quantity == "kappa":
        return FIRST_ORDER if contact == "free" else DECREASING
    if quantity == "sr":
        return SECOND_ORDER if contact == "free" else FIRST_ORDER
    return SECOND_ORDER


def main():
    parser = argparse.ArgumentParser(
        description="The free surface's slope, normal and curvature by the book's procedures, and the solver held to them.")
    parser.add_argument("--book", type=Path, default=HERE.parent / "vedic_book.py",
                        help="Unchanged vedic_book.py from the pinned upstream commit (default: repository root)")
    args = parser.parse_args()
    book = pressure.load_book(args.book)
    radius, depth = sp.Rational(1, 64), sp.Rational(1, 32)
    R = Fraction(int(radius.p), int(radius.q))
    base = {"R": float(radius), "h": float(depth)}
    require(Fraction(base["R"]) == R and Fraction(base["h"]) == 2 * R, "R and h are not exact doubles")
    p = SurfacePolynomials(book)
    built = surfaces(p, radius)
    variables, expressions = independent_surfaces(radius)
    derived, independent_checks = {}, 0
    for name, eta in built.items():
        derived[name] = derive_geometry(p, eta)
        independent_checks += check_independently(p, derived[name], expressions[name], variables, radius)
    taller = {name: derive_geometry(p, eta, sp.Rational(101, 100)) for name, eta in built.items()}
    pi = machin_pi()
    grids = {}
    for contact in CONTACTS:
        answers = probe("grid", [dict(base, nr=nr, nth=nth, contact=contact) for nr, nth in RESOLUTIONS])
        for (nr, nth), grid in zip(RESOLUTIONS, answers):
            require(grid["nr"] == nr and grid["nth"] == nth and grid["cells"] == nr * nth
                    and sorted(grid["order"]) == list(range(nr * nth)), "The probe's grid is not the one asked for")
            grids[(contact, nr)] = grid
    for nr, _ in RESOLUTIONS:
        require(grids[("free", nr)]["rc"] == grids[("pinned", nr)]["rc"]
                and grids[("free", nr)]["order"] == grids[("pinned", nr)]["order"],
                "The two contact conditions were given different grids")
    report = {}
    claims = refusals = 0
    for name in built:
        compiled = {key: compile_polynomial(derived[name][key]) for key in NAMES}
        compiled_taller = {key: compile_polynomial(taller[name][key]) for key in NAMES}
        references = {nr: reference_on_grid(compiled, grids[("free", nr)], pi) for nr, _ in RESOLUTIONS}
        references_taller = {nr: reference_on_grid(compiled_taller, grids[("free", nr)], pi) for nr, _ in CONTROL_RESOLUTIONS}
        axisymmetric = not derived[name]["eta_theta"]
        report[name] = {"steepest_slope": max(math.sqrt(midpoint(cell["slope2"])) for cell in references[RESOLUTIONS[-1][0]])}
        for contact in CONTACTS:
            cases = [dict(base, nr=nr, nth=nth, contact=contact,
                          eta=[to_double(cell["eta"], f"{name} {nr}x{nth} eta") for cell in references[nr]])
                     for nr, nth in RESOLUTIONS]
            measured = dict(zip((nr for nr, _ in RESOLUTIONS), probe("measure", cases)))
            entry = {}
            for quantity, exact_name in QUANTITIES:
                for region in ("inner", "rim"):
                    sums = [error_sums(grids[(contact, nr)], measured[nr], references[nr], R, quantity, exact_name, region)
                            for nr, _ in RESOLUTIONS]
                    label = f"{name}, {contact} contact, {exact_name}, {region}"
                    if quantity == "st" and axisymmetric:
                        require(all(s["worst"] == 0 and s["exact2"] == (0, 0) for s in sums),
                                f"{label}: an axisymmetric surface has an azimuthal slope")
                        entry[f"{exact_name} {region}"] = "exactly zero"
                        claims += 1
                        continue
                    rate = required_rate(quantity, region, contact)
                    for coarse, fine, (nr, _) in zip(sums, sums[1:], RESOLUTIONS[1:]):
                        held = strictly_falls(coarse, fine) if rate == DECREASING else falls_by(coarse, fine, rate)
                        require(held, f"{label}: halving the cell to nr = {nr} took the relative error from "
                                      f"{relative_rms(coarse):.3e} to {relative_rms(fine):.3e}, which is not "
                                      + ("a fall" if rate == DECREASING else f"a fall by {rate[0]}/{rate[1]}"))
                        claims += 1
                    entry[f"{exact_name} {region}"] = {
                        "required_fall_per_halving": "any" if rate == DECREASING else f"{rate[0]}/{rate[1]}",
                        "relative_error": [relative_rms(s) for s in sums],
                        "worst_cell_over_rms_exact": [
                            float(Fraction(s["worst"], ONE)) / math.sqrt(midpoint(s["exact2"]) / float(s["area"]))
                            for s in sums]}
            entry["curvature last row, relative_error"] = [
                relative_rms(error_sums(grids[(contact, nr)], measured[nr], references[nr], R, "kappa", "curvature", "last row"))
                for nr, _ in RESOLUTIONS]
            controls = {
                "linearised curvature eta_xx + eta_yy": (references, "laplacian"),
                "curvature of the surface one per cent taller": (references_taller, "curvature"),
            }
            for control, (reference, exact_name) in controls.items():
                coarse, fine = (error_sums(grids[(contact, nr)], measured[nr], reference[nr], R, "kappa", exact_name, "inner")
                                for nr, _ in CONTROL_RESOLUTIONS)
                require(not falls_by(coarse, fine, SECOND_ORDER),
                        f"{name}, {contact} contact: the solver was accepted as converging on the {control}")
                entry[f"refused: {control}"] = [relative_rms(coarse), relative_rms(fine)]
                refusals += 1
            report[name][contact] = entry
    print(json.dumps({
        "status": "PASS",
        "notice": "Exact free-surface geometry and the solver's convergence on it only; the fluid equations, pressure solve, time stepping and experiment are separate.",
        "book": {"method_pages": {"polynomial_derivative": [145, 175, 176, 177, 178],
                                  "coefficient_convolution": [33, 34, 35, 36],
                                  "coefficient_sum_test_of_a_product": [76]},
                 "git_blob": pressure.UPSTREAM_BLOB},
        "cell": {"R": str(radius), "h": str(depth), "grids": [f"{nr}x{nth}" for nr, nth in RESOLUTIONS]},
        "surfaces": {"cap": "eta = (R/2)(1 - rho^2/R^2)^2", "lobed": "eta = (R/2)(1 - rho^2/R^2)^2 (1 + 2(x^2 - y^2)/R^2)"},
        "interval_bits": BITS,
        "norm": "area-weighted root mean square over the region, relative to the exact quantity's; inner is r <= 9R/10, rim is beyond it",
        "book_convolutions": p.convolutions,
        "book_product_tests": p.product_tests,
        "book_derivative_fibres": p.derivative_fibres,
        "independent_symbolic_checks": independent_checks,
        "convergence_claims_decided": claims,
        "mismatched_references_refused": refusals,
        "measured": report,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (CheckFailure, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
