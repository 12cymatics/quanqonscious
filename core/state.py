"""
Field State Module (CODEX 2.1)

Implements the field representation Ψ: Ω → ℂ (or ℚ[i] in exact mode).

Features:
- Exact rational complex arithmetic by default (Fraction-based)
- Optional float mode with explicit error bounds
- Derived fields (stress/tension, phase, energy density, cymatic mask)
- Immutable state snapshots for trace/replay
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from typing import Dict, Optional, Callable, Tuple, Union, List, Any
from enum import Enum
import copy
import hashlib
import math

from .lattice import ToroidalHypercube, LatticePoint

# CRITICAL: math module FORBIDDEN - violates exact arithmetic
# All operations must use Fraction and Vedic sutra functions ONLY


class ArithmeticMode(Enum):
    """Arithmetic precision mode."""
    EXACT = "exact"           # ℚ[i] - exact rational complex
    FLOAT = "float"           # IEEE-754 float64
    MIXED = "mixed"           # Exact internal, float output


@dataclass(frozen=True)
class RationalComplex:
    """
    Exact complex number over rationals: ℚ[i].

    Represents a + bi where a, b ∈ ℚ (Fraction).
    All arithmetic is exact with no rounding errors.
    """
    real: Fraction
    imag: Fraction

    def __post_init__(self):
        # Ensure real and imag are Fractions
        if not isinstance(self.real, Fraction):
            object.__setattr__(self, 'real', Fraction(self.real))
        if not isinstance(self.imag, Fraction):
            object.__setattr__(self, 'imag', Fraction(self.imag))

    @classmethod
    def from_int(cls, n: int) -> 'RationalComplex':
        return cls(Fraction(n), Fraction(0))

    @classmethod
    def from_real(cls, r: Union[int, Fraction, float]) -> 'RationalComplex':
        if isinstance(r, float):
            return cls(Fraction(r).limit_denominator(10**12), Fraction(0))
        return cls(Fraction(r), Fraction(0))

    @classmethod
    def from_complex(cls, c: complex, precision: int = 10**12) -> 'RationalComplex':
        return cls(
            Fraction(c.real).limit_denominator(precision),
            Fraction(c.imag).limit_denominator(precision)
        )

    @classmethod
    def zero(cls) -> 'RationalComplex':
        return cls(Fraction(0), Fraction(0))

    @classmethod
    def one(cls) -> 'RationalComplex':
        return cls(Fraction(1), Fraction(0))

    @classmethod
    def i(cls) -> 'RationalComplex':
        return cls(Fraction(0), Fraction(1))

    def __add__(self, other: 'RationalComplex') -> 'RationalComplex':
        if isinstance(other, (int, Fraction)):
            other = RationalComplex.from_real(other)
        return RationalComplex(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other: Union[int, Fraction]) -> 'RationalComplex':
        return self + RationalComplex.from_real(other)

    def __sub__(self, other: 'RationalComplex') -> 'RationalComplex':
        if isinstance(other, (int, Fraction)):
            other = RationalComplex.from_real(other)
        return RationalComplex(self.real - other.real, self.imag - other.imag)

    def __rsub__(self, other: Union[int, Fraction]) -> 'RationalComplex':
        return RationalComplex.from_real(other) - self

    def __mul__(self, other: 'RationalComplex') -> 'RationalComplex':
        if isinstance(other, (int, Fraction)):
            other = RationalComplex.from_real(other)
        # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return RationalComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def __rmul__(self, other: Union[int, Fraction]) -> 'RationalComplex':
        return self * RationalComplex.from_real(other)

    def __truediv__(self, other: 'RationalComplex') -> 'RationalComplex':
        if isinstance(other, (int, Fraction)):
            other = RationalComplex.from_real(other)
        # (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
        denom = other.real * other.real + other.imag * other.imag
        if denom == 0:
            raise ZeroDivisionError("Division by zero in RationalComplex")
        return RationalComplex(
            (self.real * other.real + self.imag * other.imag) / denom,
            (self.imag * other.real - self.real * other.imag) / denom
        )

    def __neg__(self) -> 'RationalComplex':
        return RationalComplex(-self.real, -self.imag)

    def conjugate(self) -> 'RationalComplex':
        return RationalComplex(self.real, -self.imag)

    def norm_squared(self) -> Fraction:
        """Compute |z|² = a² + b² (exact)."""
        return self.real * self.real + self.imag * self.imag

    def norm(self) -> float:
        """
        FORBIDDEN: Uses math.sqrt() which violates exact arithmetic.
        Use norm_squared() instead for exact Fraction comparisons.
        """
        raise NotImplementedError(
            "norm() forbidden - uses float approximation. "
            "Use norm_squared() for exact rational arithmetic."
        )

    def phase(self) -> float:
        """
        FORBIDDEN: Uses math.atan2() which violates exact arithmetic.
        Phase operations must use sutra-based methods only.
        """
        raise NotImplementedError(
            "phase() forbidden - uses float approximation. "
            "Use Vedic sutra functions for phase operations."
        )

    def to_complex(self) -> complex:
        """Convert to Python complex."""
        return complex(float(self.real), float(self.imag))

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0

    def is_real(self) -> bool:
        return self.imag == 0

    def __repr__(self) -> str:
        if self.imag == 0:
            return f"RationalComplex({self.real})"
        sign = "+" if self.imag >= 0 else ""
        return f"RationalComplex({self.real}{sign}{self.imag}i)"


# Type alias for field values
FieldValue = Union[RationalComplex, complex, float]


@dataclass
class DerivedFields:
    """
    Derived field quantities computed from Ψ (CODEX 7.1).

    These are computed lazily and cached for efficiency.
    """
    # Core derived fields
    amplitude: Optional[Dict[Tuple[int, ...], float]] = None        # |Ψ|
    phase: Optional[Dict[Tuple[int, ...], float]] = None            # arg(Ψ)
    intensity: Optional[Dict[Tuple[int, ...], float]] = None        # |Ψ|²
    cymatic_mask: Optional[Dict[Tuple[int, ...], bool]] = None      # nodal pattern
    stress_field: Optional[Dict[Tuple[int, ...], Fraction]] = None  # MSTVQ S(x,y,z,t)
    tension_field: Optional[Dict[Tuple[int, ...], Fraction]] = None # MSTVQ tension
    r4_energy: Optional[Dict[Tuple[int, ...], Fraction]] = None     # R4 coupling energy

    def invalidate(self):
        """Clear all cached derived fields."""
        self.amplitude = None
        self.phase = None
        self.intensity = None
        self.cymatic_mask = None
        self.stress_field = None
        self.tension_field = None
        self.r4_energy = None


@dataclass
class FieldState:
    """
    Field state Ψ: Ω → ℂ (CODEX 2.1).

    The primary state object for the cymatic field simulation.
    Stores field values at each lattice point with support for:
    - Exact rational complex arithmetic (default)
    - Float mode with error tracking
    - Derived field computation
    - Immutable snapshots for trace/replay
    """

    lattice: ToroidalHypercube
    mode: ArithmeticMode = ArithmeticMode.EXACT
    timestep: int = 0

    # Field storage: maps lattice coordinates to field values
    _psi: Dict[Tuple[int, ...], RationalComplex] = dataclass_field(default_factory=dict)

    # Derived fields (computed on demand)
    _derived: DerivedFields = dataclass_field(default_factory=DerivedFields)

    # Error bounds for float mode
    _error_bounds: Optional[Dict[Tuple[int, ...], float]] = None

    # Metadata for trace/replay
    _metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Initialize field to zero if empty."""
        if not self._psi:
            for point in self.lattice.iterate_all():
                self._psi[point.coords] = RationalComplex.zero()

    def get(self, point: LatticePoint) -> RationalComplex:
        """Get field value at a lattice point."""
        coords = self.lattice.wrap_index(point.coords)
        return self._psi.get(coords, RationalComplex.zero())

    def get_by_coords(self, *coords: int) -> RationalComplex:
        """Get field value by raw coordinates (with auto-wrap)."""
        wrapped = self.lattice.wrap_index(tuple(coords))
        return self._psi.get(wrapped, RationalComplex.zero())

    def set(self, point: LatticePoint, value: FieldValue) -> None:
        """Set field value at a lattice point."""
        coords = self.lattice.wrap_index(point.coords)
        if isinstance(value, RationalComplex):
            self._psi[coords] = value
        elif isinstance(value, complex):
            self._psi[coords] = RationalComplex.from_complex(value)
        elif isinstance(value, (int, float, Fraction)):
            self._psi[coords] = RationalComplex.from_real(value)
        else:
            raise TypeError(f"Unsupported field value type: {type(value)}")
        self._derived.invalidate()

    def set_by_coords(self, coords: Tuple[int, ...], value: FieldValue) -> None:
        """Set field value by raw coordinates (with auto-wrap)."""
        point = LatticePoint(coords, self.lattice.shape)
        self.set(point, value)

    def apply_func(self, func: Callable[[Tuple[int, ...], RationalComplex], RationalComplex]) -> None:
        """Apply a function to all field values."""
        for coords in list(self._psi.keys()):
            self._psi[coords] = func(coords, self._psi[coords])
        self._derived.invalidate()

    def copy(self) -> 'FieldState':
        """Create a deep copy of the state."""
        new_state = FieldState(
            lattice=self.lattice,
            mode=self.mode,
            timestep=self.timestep
        )
        new_state._psi = copy.deepcopy(self._psi)
        new_state._metadata = copy.deepcopy(self._metadata)
        return new_state

    def snapshot(self) -> 'FieldStateSnapshot':
        """Create an immutable snapshot for trace/replay."""
        return FieldStateSnapshot(
            lattice_shape=self.lattice.shape,
            mode=self.mode,
            timestep=self.timestep,
            psi=tuple((coords, (val.real, val.imag))
                      for coords, val in sorted(self._psi.items())),
            metadata=tuple(sorted(self._metadata.items()))
        )

    # Derived field accessors

    def amplitude(self, point: LatticePoint) -> Fraction:
        """
        DEPRECATED: Returns intensity |Ψ(x)|² instead of amplitude.
        Amplitude requires sqrt which violates exact arithmetic.
        """
        return self.get(point).norm_squared()

    def phase(self, point: LatticePoint) -> Fraction:
        """
        FORBIDDEN: Phase requires atan2 which violates exact arithmetic.
        Use Vedic sutra functions for phase operations.
        """
        raise NotImplementedError("Phase forbidden - use sutra functions")

    def intensity(self, point: LatticePoint) -> Fraction:
        """Compute |Ψ(x)|² (exact)."""
        return self.get(point).norm_squared()

    def compute_amplitude_field(self) -> Dict[Tuple[int, ...], Fraction]:
        """
        DEPRECATED: Returns intensity field |Ψ|² instead of amplitude.
        Amplitude requires sqrt which violates exact arithmetic.
        """
        if self._derived.amplitude is None:
            self._derived.amplitude = {
                coords: val.norm_squared() for coords, val in self._psi.items()
            }
        return self._derived.amplitude

    def compute_phase_field(self) -> Dict[Tuple[int, ...], float]:
        """
        FORBIDDEN: Phase computation requires atan2 which violates exact arithmetic.
        Use Vedic sutra functions for phase operations.
        """
        raise NotImplementedError("Phase field forbidden - use sutra functions")

    def compute_intensity_field(self) -> Dict[Tuple[int, ...], Fraction]:
        """Compute intensity at all points (exact)."""
        if self._derived.intensity is None:
            self._derived.intensity = {
                coords: val.norm_squared() for coords, val in self._psi.items()
            }
        return self._derived.intensity

    def compute_cymatic_mask(self, threshold: Fraction = Fraction(1, 10)) -> Dict[Tuple[int, ...], bool]:
        """
        Compute cymatic nodal pattern mask.
        Points below threshold intensity are marked as nodes.
        """
        if self._derived.cymatic_mask is None:
            self._derived.cymatic_mask = {
                coords: val.norm_squared() < threshold
                for coords, val in self._psi.items()
            }
        return self._derived.cymatic_mask

    # Aggregate observables

    def total_norm_squared(self) -> Fraction:
        """Compute total ∑|Ψ|² (exact)."""
        return sum(val.norm_squared() for val in self._psi.values())

    def max_amplitude(self) -> Fraction:
        """
        DEPRECATED: Returns max intensity |Ψ|² instead of amplitude.
        Amplitude requires sqrt which violates exact arithmetic.
        """
        return max(val.norm_squared() for val in self._psi.values())

    def mean_amplitude(self) -> Fraction:
        """
        DEPRECATED: Returns mean intensity |Ψ|² instead of amplitude.
        Amplitude requires sqrt which violates exact arithmetic.
        """
        total = sum(val.norm_squared() for val in self._psi.values())
        return Fraction(total, len(self._psi))

    # Metadata management

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    # Validation

    def validate_bounded(self, max_norm: Fraction) -> bool:
        """
        Invariant check (CODEX 7.2): Verify field is bounded.
        Returns True if all |Ψ(x)|² <= max_norm².
        """
        max_norm_sq = max_norm * max_norm
        for val in self._psi.values():
            if val.norm_squared() > max_norm_sq:
                return False
        return True

    def validate_all_points_set(self) -> bool:
        """Verify all lattice points have values."""
        return len(self._psi) == self.lattice.total_sites


@dataclass(frozen=True)
class FieldStateSnapshot:
    """
    Immutable snapshot of field state for trace/replay.

    All data is stored as tuples for hashability and immutability.
    """
    lattice_shape: Tuple[int, ...]
    mode: ArithmeticMode
    timestep: int
    psi: Tuple[Tuple[Tuple[int, ...], Tuple[Fraction, Fraction]], ...]
    metadata: Tuple[Tuple[str, Any], ...]

    def to_state(self, lattice: ToroidalHypercube) -> FieldState:
        """Reconstruct FieldState from snapshot."""
        state = FieldState(lattice=lattice, mode=self.mode, timestep=self.timestep)
        for coords, (real, imag) in self.psi:
            state._psi[coords] = RationalComplex(real, imag)
        state._metadata = dict(self.metadata)
        return state

    def __hash__(self) -> int:
        # Custom hash for large states
        return hash((self.lattice_shape, self.mode, self.timestep, len(self.psi)))


# Factory functions

def create_zero_field(lattice: ToroidalHypercube,
                      mode: ArithmeticMode = ArithmeticMode.EXACT) -> FieldState:
    """Create a field initialized to zero everywhere."""
    return FieldState(lattice=lattice, mode=mode)


def create_uniform_field(lattice: ToroidalHypercube,
                         value: FieldValue,
                         mode: ArithmeticMode = ArithmeticMode.EXACT) -> FieldState:
    """Create a field with uniform value everywhere."""
    state = FieldState(lattice=lattice, mode=mode)
    if isinstance(value, RationalComplex):
        val = value
    elif isinstance(value, complex):
        val = RationalComplex.from_complex(value)
    else:
        val = RationalComplex.from_real(value)

    for point in lattice.iterate_all():
        state._psi[point.coords] = val
    return state


def create_gaussian_field(lattice: ToroidalHypercube,
                          center: Tuple[int, ...],
                          sigma: float,
                          amplitude: float = 1.0,
                          mode: ArithmeticMode = ArithmeticMode.EXACT) -> FieldState:
    """Create a Gaussian-peaked field centered at given point.

    The seed values are float-derived and this is not fixable: exp(-r²/2σ²)
    is transcendental, so a true Gaussian has no exact representation in ℚ.
    `math.exp` produces a double and `RationalComplex.from_real` stores it as
    the dyadic rational that double denotes -- exactly, but as an exact copy
    of an approximation, which is why fields from this constructor carry
    twelve-digit denominators from the start.

    `mode` is therefore not a promise about these values. It sets the
    arithmetic of everything done to the field afterwards, which is exact;
    the seed is as exact as a Gaussian can be.

    Anything asserting exactness end to end wants a field built from stated
    rationals instead -- see the fixtures in `tests/test_invariants.py`, which
    set a handful of sites by hand for that reason.
    """
    state = FieldState(lattice=lattice, mode=mode)
    center_point = LatticePoint(center, lattice.shape)

    for point in lattice.iterate_all():
        r2 = lattice.distance_squared(center_point, point)
        # Gaussian: A * exp(-r²/(2σ²))
        val = amplitude * math.exp(-float(r2) / (2 * sigma * sigma))
        state.set(point, RationalComplex.from_real(val))

    return state


# Self-test
def _self_test():
    """Run basic invariant tests."""
    from .lattice import create_3d_lattice

    lattice = create_3d_lattice(4, 4, 4)

    # Test RationalComplex arithmetic
    z1 = RationalComplex(Fraction(1, 2), Fraction(1, 3))
    z2 = RationalComplex(Fraction(1, 4), Fraction(-1, 6))
    z_sum = z1 + z2
    assert z_sum.real == Fraction(3, 4), f"Addition failed: {z_sum}"
    z_prod = z1 * z2
    expected_real = Fraction(1, 8) + Fraction(1, 18)  # (1/2)(1/4) - (1/3)(-1/6) = 1/8 + 1/18
    assert z_prod.real == expected_real, f"Multiplication failed: {z_prod}"

    # Test field state
    state = create_zero_field(lattice)
    assert state.total_norm_squared() == 0

    p = lattice.point(1, 2, 3)
    state.set(p, RationalComplex(Fraction(3), Fraction(4)))
    assert state.get(p).norm_squared() == Fraction(25)  # 3² + 4² = 25

    # Test snapshot roundtrip
    snap = state.snapshot()
    state2 = snap.to_state(lattice)
    assert state2.get(p).real == state.get(p).real

    # Test bounds validation
    assert state.validate_bounded(Fraction(10))
    assert not state.validate_bounded(Fraction(3))


# ---------------------------------------------------------------------------
# Canonical state digest
# ---------------------------------------------------------------------------
#
# One hash function, used by every consumer that needs to say "this is the
# same state". There used to be two: StateCheckpoint._compute_hash walked
# every site and returned 64 hex characters, while OperatorTrace._state_hash
# hashed a 1-in-(N/100) SAMPLE of sites and truncated to 16. They therefore
# disagreed on every state, which is why TraceReplayer.replay -- which
# compares one against the other -- could not pass on any input at all.
#
# The sampling version was also the weaker check on its own terms: on a 64
# lattice it looked at 100 sites and would not have noticed an operator that
# changed any of the rest.


def _safe_fraction_repr(value: Fraction) -> str:
    """Stringify a rational, degrading only for values too large to print.

    The 4096-bit guard bounds the cost of `str()` on a rational whose
    denominator has grown without limit. Above it the value is represented by
    its float magnitude, so two states differing only beyond that point hash
    alike -- an accepted, bounded loss, applied identically everywhere because
    every caller reaches this one function.
    """
    if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
        return f"{float(value):.12e}"
    return str(value)


def state_digest(state: "FieldState") -> str:
    """Deterministic SHA-256 over the whole field.

    Every occupied site in sorted-coordinate order, then the total norm. No
    sampling and no truncation: two states hash alike exactly when they agree
    at every site.
    """
    data = []
    for coords in sorted(state._psi.keys()):
        val = state._psi[coords]
        data.append((coords, _safe_fraction_repr(val.real), _safe_fraction_repr(val.imag)))
    data.append(('_norm', _safe_fraction_repr(state.total_norm_squared())))
    return hashlib.sha256(str(data).encode()).hexdigest()


if __name__ == "__main__":
    _self_test()
