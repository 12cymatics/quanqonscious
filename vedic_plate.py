from dataclasses import dataclass
from functools import lru_cache
from math import isqrt

import sympy as sp

from vedic_book import Evaluation, coefficients, integer


@dataclass(frozen=True)
class Interval:
    lower: int
    upper: int
    scale: int

    def __add__(self, other):
        if self.scale != other.scale:
            raise ValueError("Interval scales differ")
        return Interval(self.lower + other.lower, self.upper + other.upper, self.scale)

    def __neg__(self):
        return Interval(-self.upper, -self.lower, self.scale)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if self.scale != other.scale:
            raise ValueError("Interval scales differ")
        products = (self.lower * other.lower, self.lower * other.upper,
                    self.upper * other.lower, self.upper * other.upper)
        return Interval(min(products) // self.scale, -(-max(products) // self.scale), self.scale)

    def times(self, rational):
        p, q = int(rational.p), int(rational.q)
        a, b = self.lower * p, self.upper * p
        return Interval(min(a, b) // q, -(-max(a, b) // q), self.scale)

    def sqrt(self):
        if self.lower < 0:
            raise ValueError("A nonnegative interval is required")
        lo = isqrt(self.lower * self.scale)
        hi = isqrt(self.upper * self.scale)
        if hi * hi != self.upper * self.scale:
            hi += 1
        return Interval(lo, hi, self.scale)

    def absolute(self):
        if self.lower >= 0:
            return self
        if self.upper <= 0:
            return -self
        return Interval(0, max(-self.lower, self.upper), self.scale)


@lru_cache(maxsize=None)
def sine_bounds(intervals, bits):
    if intervals < 2 or intervals & (intervals - 1):
        raise ValueError("The interval count must be a power of two >= 2")
    scale = 1 << bits
    zero, one = Interval(0, 0, scale), Interval(scale, scale, scale)
    cosine, sine, order = zero, one, 2
    while order < intervals:
        sine = (one - cosine).times(sp.Rational(1, 2)).sqrt()
        cosine = (one + cosine).times(sp.Rational(1, 2)).sqrt()
        order *= 2
    s, c = zero, one
    quarter = [zero]
    for k in range(1, intervals // 2):
        s, c = s * cosine + c * sine, c * cosine - s * sine
        quarter.append(s)
    quarter.append(one)
    half = quarter + quarter[-2:0:-1]
    return tuple(half + [-v for v in half])


@dataclass(frozen=True)
class PlateField:
    coefficients: tuple

    def __post_init__(self):
        object.__setattr__(self, "coefficients", coefficients(self.coefficients))

    @classmethod
    def from_evaluation(cls, result):
        if not isinstance(result, Evaluation):
            raise TypeError("A book-sutra evaluation is required")
        return cls(result.spectral_coefficients)

    @property
    def maximum_mode(self):
        return len(self.coefficients)

    @property
    def amplitude_bound(self):
        return 2 * sum(abs(c) for c in self.coefficients)

    @property
    def default_intervals(self):
        return 1 << max(5, (2 * self.maximum_mode - 1).bit_length())

    def expression(self, x, y, tau=sp.S.Zero):
        terms = []
        for m, coefficient in enumerate(self.coefficients, 1):
            shape = sp.Add(sp.sin(m * sp.pi * x) * sp.sin(sp.pi * y),
                           sp.sin(sp.pi * x) * sp.sin(m * sp.pi * y), evaluate=False)
            terms.append(sp.Mul(coefficient, shape,
                                sp.cos(sp.pi ** 2 * (m * m + 1) * tau), evaluate=False))
        return sp.Add(*terms, evaluate=False)

    def validate_grid(self, intervals):
        intervals = integer(intervals)
        if intervals < 2 or intervals & (intervals - 1):
            raise ValueError("The interval count must be a power of two >= 2")
        if intervals <= self.maximum_mode:
            raise ValueError(f"Aliasing: {intervals} intervals cannot resolve mode {self.maximum_mode}; "
                             f"use at least {self.default_intervals}")
        return intervals

    def sample_bound(self, x, y, nx, ny, bits):
        sx, sy = sine_bounds(nx, bits), sine_bounds(ny, bits)
        total = Interval(0, 0, 1 << bits)
        for m, coefficient in enumerate(self.coefficients, 1):
            mode = sx[(m * x) % (2 * nx)] * sy[y % (2 * ny)]
            mode += sx[x % (2 * nx)] * sy[(m * y) % (2 * ny)]
            total += mode.times(coefficient)
        return total

    def level(self, x, y, nx, ny, levels):
        bound = self.amplitude_bound
        if bound == 0:
            return 0
        bits = 32
        while True:
            value = self.sample_bound(x, y, nx, ny, bits).absolute().times(sp.Rational(levels - 1) / bound)
            lower = value.lower // value.scale
            upper = value.upper // value.scale
            if lower == upper:
                if not 0 <= lower < levels:
                    raise ArithmeticError("The amplitude exceeded its proven bound")
                return lower
            if bits == 64:
                sample = self.expression(sp.Rational(x, nx), sp.Rational(y, ny)).doit()
                exact = sp.floor((levels - 1) * sp.Abs(sample) / bound)
                if exact.is_Integer is True:
                    result = int(exact)
                    if not 0 <= result < levels:
                        raise ArithmeticError("The exact amplitude exceeded its proven bound")
                    return result
            bits *= 2

    def render(self, intervals=None, chars="@#Xx=+;:. "):
        intervals = self.validate_grid(self.default_intervals if intervals is None else intervals)
        if len(chars) < 2:
            raise ValueError("At least two amplitude levels are required")
        nx, ny = 2 * intervals, intervals
        rows = []
        for y in range(ny + 1):
            rows.append("".join(chars[self.level(x, y, nx, ny, len(chars))]
                                for x in range(nx + 1)))
        return "\n".join(rows)

    def metadata(self):
        x, y, tau = sp.symbols("x y tau", real=True)
        return {
            "model": "simply_supported_square_plate_modal_superposition",
            "mapping": "authored_ordered_sutra_coefficients_to_symmetric_modes_m_1",
            "domain": {"x": [0, 1], "y": [0, 1]},
            "time": "tau = t*sqrt(D/(rho*h))/L**2",
            "pde": "u_tau_tau + u_xxxx + 2*u_xxyy + u_yyyy = 0",
            "boundary": "u=0; u_nn+nu*u_ss=0",
            "coefficients": [str(c) for c in self.coefficients],
            "expression": str(self.expression(x, y, tau)),
            "raster": "certified_exact_magnitude_bins_at_tau_zero",
            "maximum_mode": self.maximum_mode,
            "default_intervals": self.default_intervals,
        }
