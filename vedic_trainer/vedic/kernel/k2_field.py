"""K2 = ℚ(√2, i) — the certified base field, and C4 = K2(√5) for φ.

Source: *Exact Kernel Evolution Blueprint*, 31 July 2026, §Coefficient-field
architecture.

Certified base
--------------
    z = a + b√2 + i(c + d√2),    a, b, c, d ∈ ℚ

Two independent involutions, which must commute exactly:

    σ₂(√2) = −√2,  σ₂(i) =  i
    σᵢ(i)  = −i,   σᵢ(√2) = √2
    σ₂(σᵢ(z)) = σᵢ(σ₂(z))

Two distinct norms — the blueprint is explicit that these are different
objects and that *an energy definition must name the selected norm*:

    H(z) = z†z = (a+b√2)² + (c+d√2)²           ∈ ℚ(√2)   [Hermitian]
    N(z) = z·σ₂(z)·σᵢ(z)·σ₂σᵢ(z)                ∈ ℚ       [total field norm]

A rational coefficient, a trace, or a renderer projection is **not** silently
interchangeable with either one.

Golden-ratio extension
----------------------
φ and √5 are not in K2. Representing them as rationals would be false, so the
canonical runtime extension is

    C4 = K2(√5),   x = u + v√5,   u, v ∈ K2

and φ = (1 + √5)/2 lives there exactly — not as a Fibonacci convergent.

Every arithmetic coordinate is a ``Fraction``. Exact constructors reject
floating-point inputs; the explicitly named ``to_float`` renderers are the
only conversion out of the exact field.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from fractions import Fraction
from numbers import Rational
from typing import Tuple

__all__ = [
    "Q2", "K2", "C4", "phi", "phi_cubed",
    "SQRT2", "I", "SQRT5",
]


# ----------------------------------------------------------------------
# ℚ(√2) — the Hermitian-norm codomain
# ----------------------------------------------------------------------


# The only two floats in this module. They exist solely for to_float().
_SQRT2_F = math.sqrt(2.0)
_SQRT5_F = math.sqrt(5.0)


def _rational_coordinate(value: object, name: str) -> Fraction:
    """Accept rational-domain values without importing a floating-point state."""
    if isinstance(value, bool) or not isinstance(value, Rational):
        raise TypeError(f"{name} must be an exact rational (integer or Fraction)")
    return Fraction(int(value.numerator), int(value.denominator))


@dataclass(frozen=True)
class Q2:
    """a + b√2 with a, b ∈ ℚ."""

    a: Fraction = Fraction(0)
    b: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _rational_coordinate(self.a, "Q2.a"))
        object.__setattr__(self, "b", _rational_coordinate(self.b, "Q2.b"))

    def to_float(self) -> float:
        """Render to IEEE-754. THE boundary: nothing upstream may call this.

        Every operation in this module is exact; a float appears only when a
        value is being displayed or plotted. Keeping that conversion in one
        named method is what makes "no float contamination in the arithmetic
        path" a checkable statement rather than an intention.
        """
        return float(self.a) + float(self.b) * _SQRT2_F

    def __add__(self, o: "Q2") -> "Q2":
        return Q2(self.a + o.a, self.b + o.b)

    def __sub__(self, o: "Q2") -> "Q2":
        return Q2(self.a - o.a, self.b - o.b)

    def __neg__(self) -> "Q2":
        return Q2(-self.a, -self.b)

    def __mul__(self, o: "Q2") -> "Q2":
        # (a+b√2)(c+d√2) = (ac + 2bd) + (ad + bc)√2
        return Q2(self.a * o.a + 2 * self.b * o.b,
                  self.a * o.b + self.b * o.a)

    def inverse(self) -> "Q2":
        """Return (a − b√2)/(a² − 2b²); zero has no field inverse."""
        if self.is_zero():
            raise ZeroDivisionError("cannot invert zero in Q2")
        denominator = self.norm()
        return Q2(self.a / denominator, -self.b / denominator)

    def __truediv__(self, o: "Q2") -> "Q2":
        """Divide two Q2 elements in the original coefficient field."""
        if not isinstance(o, Q2):
            return NotImplemented
        return self * o.inverse()

    def conj2(self) -> "Q2":
        """σ₂: √2 ↦ −√2."""
        return Q2(self.a, -self.b)

    def norm(self) -> Fraction:
        """z·σ₂(z) = a² − 2b² ∈ ℚ."""
        return self.a * self.a - 2 * self.b * self.b

    def is_rational(self) -> bool:
        return self.b == 0

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0

    def __repr__(self) -> str:
        return f"Q2({self.a}, {self.b})"


# ----------------------------------------------------------------------
# K2 = ℚ(√2, i)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class K2:
    """z = a + b√2 + i(c + d√2), stored as (re, im) over ℚ(√2)."""

    re: Q2 = Q2()
    im: Q2 = Q2()

    def __post_init__(self) -> None:
        if not isinstance(self.re, Q2) or not isinstance(self.im, Q2):
            raise TypeError("K2.re and K2.im must be Q2 elements")

    @staticmethod
    def from_coords(a: int | Fraction, b: int | Fraction, c: int | Fraction, d: int | Fraction) -> "K2":
        """The blueprint's four reduced rational coordinates."""
        return K2(Q2(a, b), Q2(c, d))

    @staticmethod
    def from_rational(q: int | Fraction) -> "K2":
        """Embed an exact rational, rejecting floating-point inputs."""
        return K2(Q2(q, Fraction(0)), Q2())

    def coords(self) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
        return (self.re.a, self.re.b, self.im.a, self.im.b)

    def __add__(self, o: "K2") -> "K2":
        return K2(self.re + o.re, self.im + o.im)

    def __sub__(self, o: "K2") -> "K2":
        return K2(self.re - o.re, self.im - o.im)

    def __neg__(self) -> "K2":
        return K2(-self.re, -self.im)

    def __mul__(self, o: "K2") -> "K2":
        # (x + iy)(u + iv) = (xu − yv) + i(xv + yu)
        return K2(self.re * o.re - self.im * o.im,
                  self.re * o.im + self.im * o.re)

    def inverse(self) -> "K2":
        """Return z†/H(z), retaining the Q2-valued Hermitian denominator.

        The rational total norm is a separate quantity and is not used in
        place of H(z).
        """
        if self.is_zero():
            raise ZeroDivisionError("cannot invert zero in K2")
        inverse_h = self.hermitian_norm().inverse()
        return K2(self.re * inverse_h, -self.im * inverse_h)

    def __truediv__(self, o: "K2") -> "K2":
        """Divide two K2 elements without a projection to rational scalars."""
        if not isinstance(o, K2):
            return NotImplemented
        return self * o.inverse()

    # -- the two independent involutions --------------------------------

    def sigma2(self) -> "K2":
        """σ₂: √2 ↦ −√2, i ↦ i. Acts on both components."""
        return K2(self.re.conj2(), self.im.conj2())

    def sigma_i(self) -> "K2":
        """σᵢ: i ↦ −i, √2 ↦ √2. Complex conjugation."""
        return K2(self.re, -self.im)

    def dagger(self) -> "K2":
        """z† — complex conjugate, i.e. σᵢ."""
        return self.sigma_i()

    # -- the two distinct norms -----------------------------------------

    def hermitian_norm(self) -> Q2:
        """H(z) = z†z = (a+b√2)² + (c+d√2)²  ∈ ℚ(√2).

        NOT interchangeable with the total norm. Name which one you mean.
        """
        return self.re * self.re + self.im * self.im

    def total_norm(self) -> Fraction:
        """N(z) = z·σ₂(z)·σᵢ(z)·σ₂σᵢ(z)  ∈ ℚ — the full Galois product."""
        prod = self * self.sigma2() * self.sigma_i() * self.sigma2().sigma_i()
        if not (prod.im.a == 0 and prod.im.b == 0 and prod.re.b == 0):
            raise ValueError(
                f"total_norm must land in ℚ; got {prod!r} — the Galois "
                "product is not closing, which means the involutions are wrong"
            )
        return prod.re.a

    def is_zero(self) -> bool:
        return self.coords() == (Fraction(0),) * 4

    def __repr__(self) -> str:
        a, b, c, d = self.coords()
        return f"K2({a}, {b}, {c}, {d})"


SQRT2: K2 = K2.from_coords(0, 1, 0, 0)
I: K2 = K2.from_coords(0, 0, 1, 0)
ONE: K2 = K2.from_coords(1, 0, 0, 0)


# ----------------------------------------------------------------------
# C4 = K2(√5) — where φ actually lives
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class C4:
    """x = u + v√5 with u, v ∈ K2. The canonical runtime extension."""

    u: K2 = K2()
    v: K2 = K2()

    def __post_init__(self) -> None:
        if not isinstance(self.u, K2) or not isinstance(self.v, K2):
            raise TypeError("C4.u and C4.v must be K2 elements")

    def to_float(self) -> complex:
        """Render to a complex float. See Q2.to_float -- this is the boundary."""
        return (complex(self.u.re.to_float(), self.u.im.to_float())
                + complex(self.v.re.to_float(), self.v.im.to_float()) * _SQRT5_F)

    @staticmethod
    def from_k2(z: K2) -> "C4":
        return C4(z, K2())

    @staticmethod
    def from_rational(q: int | Fraction) -> "C4":
        """Embed an exact rational through K2's strict domain boundary."""
        return C4(K2.from_rational(q), K2())

    def __add__(self, o: "C4") -> "C4":
        return C4(self.u + o.u, self.v + o.v)

    def __sub__(self, o: "C4") -> "C4":
        return C4(self.u - o.u, self.v - o.v)

    def __neg__(self) -> "C4":
        return C4(-self.u, -self.v)

    def __mul__(self, o: "C4") -> "C4":
        # (u + v√5)(p + q√5) = (up + 5vq) + (uq + vp)√5
        five = K2.from_rational(5)
        return C4(self.u * o.u + five * self.v * o.v,
                  self.u * o.v + self.v * o.u)

    def inverse(self) -> "C4":
        """Return (u − v√5)/(u² − 5v²), with denominator inverted in K2."""
        if self.is_zero():
            raise ZeroDivisionError("cannot invert zero in C4")
        denominator = self.u * self.u - K2.from_rational(5) * self.v * self.v
        inverse_denominator = denominator.inverse()
        return C4(self.u * inverse_denominator, -self.v * inverse_denominator)

    def __truediv__(self, o: "C4") -> "C4":
        """Divide two C4 elements with every field coordinate retained."""
        if not isinstance(o, C4):
            return NotImplemented
        return self * o.inverse()

    def conj5(self) -> "C4":
        """√5 ↦ −√5."""
        return C4(self.u, -self.v)

    def is_in_k2(self) -> bool:
        """True when the √5 part vanishes, i.e. the value really lives in K2."""
        return self.v.is_zero()

    def is_zero(self) -> bool:
        return self.u.is_zero() and self.v.is_zero()

    def __repr__(self) -> str:
        return f"C4({self.u!r} + {self.v!r}·√5)"


SQRT5: C4 = C4(K2(), K2.from_rational(1))


def phi() -> C4:
    """φ = (1 + √5)/2 — exact, in C4. Not a Fibonacci convergent."""
    half = Fraction(1, 2)
    return C4(K2.from_rational(half), K2.from_rational(half))


def phi_cubed() -> C4:
    """φ³ = 2 + √5, exactly.

    φ² = φ + 1 and φ³ = 2φ + 1 = 2·(1+√5)/2 + 1 = 2 + √5.
    This is the Φ³ dielectric–curvature scaling factor; it is an algebraic
    integer of C4, never the float 4.236…
    """
    p = phi()
    return p * p * p
