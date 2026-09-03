"""Sulba Sutra geometric construction vs direct computation.

The Sulba Sutras are construction rules -- rope-and-peg procedures for altar
geometry -- so the comparison that matters is not only speed but ACCURACY and
EXACTNESS: each rule yields an exact rational, where the modern route yields a
binary float. Both are approximations to an irrational; they are wrong in
different ways, and this measures how.

Nothing here is read from the repository's Sulba modules: `sulbasutraws.py`
does not import (NameError on `Union` -- it is a class-body fragment), and
`sulba.py` is a waveform shaper, not the constructions. The rules below are
therefore written directly from the sutras they name, and each is checked
against the quantity it is supposed to produce.
"""
from fractions import Fraction
from decimal import Decimal, getcontext
import math
import time

getcontext().prec = 60


def sulba_sqrt2() -> Fraction:
    """Baudhayana 1.61-2 (dvi-karani): 1 + 1/3 + 1/(3*4) - 1/(3*4*34)."""
    return 1 + Fraction(1, 3) + Fraction(1, 3 * 4) - Fraction(1, 3 * 4 * 34)


def sulba_square_to_circle(side: Fraction, root2: Fraction) -> Fraction:
    """Baudhayana 2.9: radius = side * (2 + sqrt2) / 6."""
    return side * (2 + root2) / 6


def sulba_circle_to_square(radius: Fraction) -> Fraction:
    """Baudhayana 2.10: side = 2r * (1 - 1/8 + 1/(8*29) - 1/(8*29*6) + 1/(8*29*6*8))."""
    series = (1 - Fraction(1, 8) + Fraction(1, 8 * 29)
              - Fraction(1, 8 * 29 * 6) + Fraction(1, 8 * 29 * 6 * 8))
    return 2 * radius * series


def sulba_triples():
    """The triples named in the Sulba texts, used to set out right angles."""
    return [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (12, 35, 37)]


def err(approx, true_value) -> Decimal:
    return abs(Decimal(approx.numerator) / Decimal(approx.denominator) - true_value)


def timed(fn, iters=20000):
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    return (time.perf_counter_ns() - t0) / iters


TRUE_SQRT2 = Decimal(2).sqrt()
TRUE_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")

print("\n=== GEOMETRY: accuracy ===")
print(f"  {'construction':34} {'sulba (exact rational)':>26} {'error':>14} {'digits':>7}")

r2 = sulba_sqrt2()
e = err(r2, TRUE_SQRT2)
print(f"  {'sqrt(2) [dvi-karani]':34} {str(r2):>26} {float(e):>14.2e} "
      f"{int(-e.log10()):>7}")
assert r2 == Fraction(577, 408), r2

# Implied pi from each circle rule: the rule equates a square and a circle,
# so the pi it assumes is recoverable and can be scored against the real one.
side = Fraction(1)
rad = sulba_square_to_circle(side, r2)
pi_from_squaring = side * side / (rad * rad)
e1 = err(pi_from_squaring, TRUE_PI)
print(f"  {'square->circle implied pi':34} {str(pi_from_squaring)[:26]:>26} "
      f"{float(e1):>14.2e} {int(-e1.log10()):>7}")

radius = Fraction(1)
sq = sulba_circle_to_square(radius)
pi_from_circling = sq * sq / (radius * radius)
e2 = err(pi_from_circling, TRUE_PI)
print(f"  {'circle->square implied pi':34} {str(pi_from_circling)[:26]:>26} "
      f"{float(e2):>14.2e} {int(-e2.log10()):>7}")

bad = [(a, b, c) for a, b, c in sulba_triples() if a * a + b * b != c * c]
print(f"  {'pythagorean triples':34} {'5 named in the texts':>26} "
      f"{'exact' if not bad else 'FAILED':>14} {'inf':>7}")

print("\n=== GEOMETRY: speed, sutra construction vs direct float ===")
print(f"  {'construction':34} {'sulba ns':>10} {'direct ns':>10} {'ratio':>9}")
for name, sut, dirf in [
    ("sqrt(2)", sulba_sqrt2, lambda: math.sqrt(2)),
    ("square->circle radius", lambda: sulba_square_to_circle(side, r2),
     lambda: 1.0 * (2 + math.sqrt(2)) / 6),
    ("circle->square side", lambda: sulba_circle_to_square(radius),
     lambda: 1.0 * math.sqrt(math.pi)),
]:
    s, d = timed(sut), timed(dirf)
    print(f"  {name:34} {s:>10.1f} {d:>10.1f} {s / d:>8.2f}x")

print(f"\n  float64 sqrt(2) error for comparison: "
      f"{abs(Decimal(repr(math.sqrt(2))) - TRUE_SQRT2):.2e}")
print(f"  float64 pi error for comparison:      "
      f"{abs(Decimal(repr(math.pi)) - TRUE_PI):.2e}")
