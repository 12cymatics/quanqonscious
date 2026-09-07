#!/usr/bin/env python3
"""
Gates on the exact polar geometry in vedic_cymatic_pde_engine.

What this replaces mattered: the previous code computed a lattice point's
"radius" as

    r = safe_inverse_distance(r_sq)          # 1 / (r_sq + 1e-12)
    r = 1 / r                                # r_sq + 1e-12

and then fed that into J_n(k*r) and cos(theta) = dj/r. So the variable named r
held r SQUARED, plus an epsilon added on every evaluation whether or not the
point was anywhere near the origin. Both the wrong power and the epsilon are
gated below.

Runs standalone or under pytest.
"""
import math
import sys
import os
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vedic_cymatic_pde_engine import (          # noqa: E402
    ExactPolarGeometry, VedicPDESolver,
    IrrationalRadius, UndefinedAtOrigin, TruncationBoundInvalid,
)

G = ExactPolarGeometry
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}   {detail}")
        FAILURES.append(name)


def raises(name, exc, fn, *a, **kw):
    try:
        got = fn(*a, **kw)
    except exc as e:
        print(f"  ok   {name}  -> {type(e).__name__}: {str(e).split('.')[0][:70]}")
        return
    except Exception as e:                       # noqa: BLE001
        check(name, False, f"raised {type(e).__name__}, wanted {exc.__name__}: {e}")
        return
    check(name, False, f"returned {got}, wanted {exc.__name__}")


# ---------------------------------------------------------------------------
# An independent J_n. The integral representation
#     J_n(x) = (1/pi) * integral_0^pi cos(n*tau - x*sin tau) d tau
# shares no algebra with the power series under test.
# ---------------------------------------------------------------------------
def bessel_reference(n, x, panels=200000):
    h = math.pi / panels
    total = 0.0
    for i in range(panels + 1):
        tau = i * h
        w = 1 if i in (0, panels) else (2 if i % 2 == 0 else 4)
        total += w * math.cos(n * tau - x * math.sin(tau))
    return (total * h / 3) / math.pi


print("1. solid harmonic against r^m cos(m theta) computed in floating point")
for (m, di, dj) in [(0, 2, 3), (1, 2, 3), (2, 2, 3), (3, 2, 3),
                    (4, -5, 1), (5, 7, -2), (6, 0, 4), (3, -3, -3)]:
    exact = G.solid_harmonic_re(m, F(di), F(dj))
    r = math.hypot(di, dj)
    theta = math.atan2(di, dj)
    ref = (r ** m) * math.cos(m * theta)
    check(f"H_{m}({di},{dj}) = {exact}",
          abs(float(exact) - ref) <= 1e-9 * max(1.0, abs(ref)),
          f"exact {float(exact)!r} vs reference {ref!r}")

check("H_3(2,3) is exactly -9, by (3+2i)^3 = -9+46i",
      G.solid_harmonic_re(3, F(2), F(3)) == F(-9))
check("the solid harmonic is a Fraction, not a float",
      isinstance(G.solid_harmonic_re(4, F(2), F(3)), F))
check("the solid harmonic is finite at the origin",
      G.solid_harmonic_re(5, F(0), F(0)) == F(0))
check("H_0 at the origin is 1",
      G.solid_harmonic_re(0, F(0), F(0)) == F(1))

print("\n2. exact rational square root: found where it exists, refused where it does not")
check("sqrt(25/4) = 5/2", G.exact_rational_sqrt(F(25, 4)) == F(5, 2))
check("sqrt(0) = 0", G.exact_rational_sqrt(F(0)) == F(0))
check("sqrt(2) is None, not 1.41...", G.exact_rational_sqrt(F(2)) is None)
check("sqrt(13) is None", G.exact_rational_sqrt(F(13)) is None)
check("sqrt(1/3) is None (denominator not square)", G.exact_rational_sqrt(F(1, 3)) is None)
check("a huge perfect square is still found exactly",
      G.exact_rational_sqrt(F(10 ** 40)) == F(10 ** 20))
raises("a negative argument is rejected", ValueError, G.exact_rational_sqrt, F(-1))

print("\n3. radius_power: exact for even exponents, refused for irrational odd ones")
check("r^0 = 1 even at the origin", G.radius_power(F(0), 0) == F(1))
check("r^2 from r_sq = 13 is 13", G.radius_power(F(13), 2) == F(13))
check("r^4 from r_sq = 13 is 169", G.radius_power(F(13), 4) == F(169))
check("r^-2 from r_sq = 13 is 1/13", G.radius_power(F(13), -2) == F(1, 13))
check("r^1 from r_sq = 25 is exactly 5", G.radius_power(F(25), 1) == F(5))
check("r^3 from r_sq = 25 is exactly 125", G.radius_power(F(25), 3) == F(125))
raises("r^1 from r_sq = 13 refuses rather than rounding",
       IrrationalRadius, G.radius_power, F(13), 1)
raises("r^-1 from r_sq = 2 refuses", IrrationalRadius, G.radius_power, F(2), -1)
raises("a negative power of r = 0 is reported as the pole it is",
       ZeroDivisionError, G.radius_power, F(0), -2)
check("r^3 at the origin is 0, not an epsilon", G.radius_power(F(0), 3) == F(0))

print("\n4. the Bessel partial sum sits within its own stated tail bound")
for (n, k, r_sq, terms) in [(0, F(1, 2), F(4), 12), (1, F(1, 2), F(9), 12),
                            (2, F(1), F(4), 14), (0, F(1), F(1), 10),
                            (3, F(1, 4), F(16), 10)]:
    value, bound = G.bessel_radial_polynomial(n, k, r_sq, terms)
    r = math.sqrt(float(r_sq))
    # value is J_n(k r) / r^n; scale back up to compare with the reference
    got = float(value) * (r ** n)
    ref = bessel_reference(n, float(k) * r)
    err = abs(got - ref)
    allowed = float(bound) * (r ** n) + 1e-9
    check(f"J_{n}({float(k)*r:.3f}) = {got:.12f} within its bound",
          err <= allowed, f"error {err:.3e} > allowed {allowed:.3e} (ref {ref:.12f})")

print("\n5. the tail bound is refused where the series is not yet decreasing")
raises("a large argument with too few terms refuses to quote a bound",
       TruncationBoundInvalid, G.bessel_radial_polynomial, 0, F(20), F(400), 3)
value, bound = G.bessel_radial_polynomial(0, F(20), F(400), 200)
check("the same point succeeds once enough terms are taken",
      isinstance(value, F) and bound >= 0)
raises("terms must be at least 1", ValueError,
       G.bessel_radial_polynomial, 0, F(1), F(1), 0)
raises("a negative order is rejected", ValueError,
       G.bessel_radial_polynomial, -1, F(1), F(1), 5)

print("\n6. the whole mode value equals J_n(k r) cos(m theta)")
solver = VedicPDESolver()
for (n, m, di, dj, k) in [(1, 1, 3, 4, F(1, 2)), (2, 2, 1, 2, F(1, 2)),
                          (3, 1, 2, 3, F(1, 4)), (2, 0, 5, 1, F(1, 3)),
                          (0, 0, 2, 2, F(1, 2)), (4, 2, -3, 1, F(1, 4))]:
    r_sq = F(di * di + dj * dj)
    got = solver._helmholtz_mode_value(k, r_sq, F(di), F(dj), n, m, terms=16)
    r = math.sqrt(float(r_sq))
    ref = bessel_reference(n, float(k) * r) * math.cos(m * math.atan2(di, dj))
    check(f"J_{n}(k r) cos({m} theta) at ({di},{dj}) = {float(got):.10f}",
          abs(float(got) - ref) <= 1e-8 * max(1.0, abs(ref)),
          f"got {float(got)!r} vs reference {ref!r}")
    check(f"  ... and it is a Fraction at ({di},{dj})", isinstance(got, F))

print("\n7. no epsilon reaches the result")
# n = m = 1 makes r^(n-m) = 1, so the value is a product of two exact rationals
# with no root anywhere. Recompute it independently, term by term.
k, di, dj, terms = F(1, 2), F(3), F(4), 6
r_sq = di * di + dj * dj                      # 25
half = k / 2
expected_radial = sum(
    (F(-1) ** t) * (half ** (1 + 2 * t)) * (r_sq ** t)
    / (math.factorial(t) * math.factorial(1 + t))
    for t in range(terms))
expected = expected_radial * G.solid_harmonic_re(1, di, dj)
got = solver._helmholtz_mode_value(k, r_sq, di, dj, 1, 1, terms=terms)
check("the value is bit-for-bit the independently summed rational",
      got == expected, f"got {got} expected {expected}")
check("their difference is exactly zero, not below a tolerance",
      (got - expected) == 0)
# 1e-12 was the old epsilon. If any survived, r_sq would not be exactly 25.
check("adding the old epsilon would have changed the answer",
      solver._helmholtz_mode_value(k, r_sq + F(1, 10 ** 12), di, dj, 1, 1,
                                   terms=terms) != got)

print("\n8. the origin is exact, and undefined only where it truly is")
check("radial order 1 at the origin is exactly 0",
      solver._helmholtz_mode_value(F(1, 2), F(0), F(0), F(0), 1, 0) == F(0))
check("radial order 3 at the origin is exactly 0",
      solver._helmholtz_mode_value(F(1, 2), F(0), F(0), F(0), 3, 2) == F(0))
check("J_0(0) cos(0) at the origin is exactly 1",
      solver._helmholtz_mode_value(F(1, 2), F(0), F(0), F(0), 0, 0) == F(1))
raises("J_0(0) cos(2 theta) at the origin is refused, not invented",
       UndefinedAtOrigin, solver._helmholtz_mode_value,
       F(1, 2), F(0), F(0), F(0), 0, 2)

print("\n9. an odd radial-angular gap on an irrational radius is refused")
raises("orders 2 and 1 at (2,3), where r = sqrt(13), refuse",
       IrrationalRadius, solver._helmholtz_mode_value,
       F(1, 2), F(13), F(2), F(3), 2, 1)
check("the same orders succeed at (3,4), where r = 5 exactly",
      isinstance(solver._helmholtz_mode_value(F(1, 2), F(25), F(3), F(4), 2, 1), F))
check("equal orders never need a root, so they always succeed",
      isinstance(solver._helmholtz_mode_value(F(1, 2), F(13), F(2), F(3), 3, 3), F))

print("\n10. the suppression machinery is gone, not merely unused")
import vedic_cymatic_pde_engine as engine      # noqa: E402
source = open(engine.__file__).read()
for banned, why in [
        ("safe_inverse_distance", "added 1e-12 to r^2 on every evaluation"),
        ("safe_divide", "swapped in a different algorithm near zero"),
        ("regularize_field", "overwrote outliers with neighbour means"),
        ("SingularitySuppression", "the class holding all three")]:
    check(f"{banned} is absent ({why})",
          f"def {banned}" not in source and f"class {banned}" not in source)
check("no 1e-12 threshold survives in code",
      "Fraction(1, 10**12)" not in source and "Fraction(1, 1000000)" not in source)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")


def test_exact_polar_geometry():
    """pytest entry point: the module-level checks above must all have passed."""
    assert not FAILURES, FAILURES
