"""
Toroidal Hypercube Lattice Module (CODEX 2.1 / 5.1 / 5.2)

Implements the discrete domain Ω = (ℤ_{N0} × ... × ℤ_{N(d-1)}) with:
- Toroidal (wraparound) boundary conditions per axis
- R4 adjacency kernel for cross-lattice coupling
- Exact rational coupling weights

All indexing uses modular arithmetic: x_k := x_k mod N_k
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple, Iterator, Dict, Optional, FrozenSet
import itertools


@dataclass(frozen=True)
class LatticePoint:
    """Immutable lattice coordinate with automatic modular normalization."""
    coords: Tuple[int, ...]
    dimensions: Tuple[int, ...]

    def __post_init__(self):
        # Enforce modular wrap on construction
        normalized = tuple(c % d for c, d in zip(self.coords, self.dimensions))
        object.__setattr__(self, 'coords', normalized)

    def __add__(self, other: Tuple[int, ...]) -> 'LatticePoint':
        """Add offset with modular wrap."""
        if len(other) != len(self.coords):
            raise ValueError(f"Offset dimension mismatch: {len(other)} vs {len(self.coords)}")
        new_coords = tuple((c + o) % d for c, o, d in zip(self.coords, other, self.dimensions))
        return LatticePoint(new_coords, self.dimensions)

    def __sub__(self, other: 'LatticePoint') -> Tuple[int, ...]:
        """Compute signed distance (minimum image convention)."""
        diffs = []
        for c1, c2, d in zip(self.coords, other.coords, self.dimensions):
            diff = (c1 - c2) % d
            if diff > d // 2:
                diff -= d
            diffs.append(diff)
        return tuple(diffs)

    def __hash__(self) -> int:
        return hash((self.coords, self.dimensions))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticePoint):
            return False
        return self.coords == other.coords and self.dimensions == other.dimensions

    def linear_index(self) -> int:
        """Convert to linear (flat) index using row-major ordering."""
        idx = 0
        stride = 1
        for c, d in zip(reversed(self.coords), reversed(self.dimensions)):
            idx += c * stride
            stride *= d
        return idx

    @classmethod
    def from_linear_index(cls, idx: int, dimensions: Tuple[int, ...]) -> 'LatticePoint':
        """Construct from linear index."""
        coords = []
        for d in reversed(dimensions):
            coords.append(idx % d)
            idx //= d
        return cls(tuple(reversed(coords)), dimensions)


@dataclass
class R4AdjacencyKernel:
    """
    R4 coupling adjacency kernel (CODEX 5.2).

    Maintains explicit adjacency function N_R4(x) ⊆ Ω with:
    - Deterministic neighborhood ordering
    - Exact rational coupling weights

    The R4 topology creates cross-lattice couplings beyond nearest neighbors,
    enabling entanglement-like correlations in the classical evolution.
    """

    # Required fields first
    dimensions: int
    r4_radius: int = 2
    coupling_weights: Dict[int, Fraction] = field(default_factory=dict)

    # Computed shell offsets (set in __post_init__)
    SHELL_1: Tuple[Tuple[int, ...], ...] = field(default=(), init=False)
    SHELL_2: Tuple[Tuple[int, ...], ...] = field(default=(), init=False)
    SHELL_3: Tuple[Tuple[int, ...], ...] = field(default=(), init=False)

    def __post_init__(self):
        """Build offset shells for the given dimensionality."""
        # Shell 1: nearest neighbors (±1 along each axis)
        shell_1 = []
        for d in range(self.dimensions):
            for sign in [-1, 1]:
                offset = [0] * self.dimensions
                offset[d] = sign
                shell_1.append(tuple(offset))
        object.__setattr__(self, 'SHELL_1', tuple(shell_1))

        # Shell 2: face diagonals (±1 along two axes)
        shell_2 = []
        for d1, d2 in itertools.combinations(range(self.dimensions), 2):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    offset = [0] * self.dimensions
                    offset[d1] = s1
                    offset[d2] = s2
                    shell_2.append(tuple(offset))
        object.__setattr__(self, 'SHELL_2', tuple(shell_2))

        # Shell 3: body diagonals (±1 along all axes) + R4 extensions
        shell_3 = []
        if self.dimensions >= 3:
            for signs in itertools.product([-1, 1], repeat=self.dimensions):
                shell_3.append(tuple(signs))
        # R4 extension: ±2 along single axis (extended coupling)
        for d in range(self.dimensions):
            for sign in [-2, 2]:
                offset = [0] * self.dimensions
                offset[d] = sign
                shell_3.append(tuple(offset))
        object.__setattr__(self, 'SHELL_3', tuple(shell_3))

        # Default coupling weights (exact rationals)
        if not self.coupling_weights:
            self.coupling_weights = {
                1: Fraction(1, 1),      # Shell 1: full coupling
                2: Fraction(1, 2),      # Shell 2: sqrt(2) normalized
                3: Fraction(1, 4),      # Shell 3: weaker coupling
            }

    def get_neighbors(self, point: LatticePoint) -> List[Tuple[LatticePoint, Fraction]]:
        """
        Get all R4 neighbors of a point with their coupling weights.

        Returns: List of (neighbor_point, coupling_weight) tuples in deterministic order.
        """
        neighbors = []

        # Shell 1 neighbors
        for offset in self.SHELL_1:
            neighbor = point + offset
            neighbors.append((neighbor, self.coupling_weights.get(1, Fraction(1))))

        # Shell 2 neighbors (if within radius)
        if self.r4_radius >= 2:
            for offset in self.SHELL_2:
                neighbor = point + offset
                neighbors.append((neighbor, self.coupling_weights.get(2, Fraction(1, 2))))

        # Shell 3 / R4 extension neighbors
        if self.r4_radius >= 3:
            for offset in self.SHELL_3:
                neighbor = point + offset
                neighbors.append((neighbor, self.coupling_weights.get(3, Fraction(1, 4))))

        return neighbors

    def adjacency_set(self, point: LatticePoint) -> FrozenSet[LatticePoint]:
        """Return the set N_R4(x) of all adjacent points."""
        return frozenset(n for n, _ in self.get_neighbors(point))


@dataclass
class ToroidalHypercube:
    """
    Toroidal hypercube lattice domain Ω = (ℤ_{N0} × ... × ℤ_{N(d-1)}).

    Implements CODEX sections 2.1, 5.1, 5.2:
    - All lattice indexing uses modular wrap per axis
    - Explicit R4 adjacency kernel for cross-lattice coupling
    - Deterministic neighborhood ordering with exact rational weights

    Attributes:
        shape: Tuple of axis sizes (N0, N1, ..., N(d-1))
        r4_kernel: The R4 adjacency kernel
        total_sites: Total number of lattice sites
    """

    shape: Tuple[int, ...]
    r4_radius: int = 2
    r4_kernel: R4AdjacencyKernel = field(init=False)

    def __post_init__(self):
        if any(n <= 0 for n in self.shape):
            raise ValueError(f"All dimensions must be positive: {self.shape}")
        self.r4_kernel = R4AdjacencyKernel(
            dimensions=len(self.shape),
            r4_radius=self.r4_radius
        )

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.shape)

    @property
    def total_sites(self) -> int:
        """Total number of lattice sites."""
        result = 1
        for n in self.shape:
            result *= n
        return result

    def point(self, *coords: int) -> LatticePoint:
        """Create a lattice point with automatic modular wrap."""
        if len(coords) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} coordinates, got {len(coords)}")
        return LatticePoint(tuple(coords), self.shape)

    def wrap_index(self, coords: Tuple[int, ...]) -> Tuple[int, ...]:
        """Apply modular wrap to arbitrary coordinates."""
        if len(coords) != len(self.shape):
            raise ValueError(f"Dimension mismatch: {len(coords)} vs {len(self.shape)}")
        return tuple(c % n for c, n in zip(coords, self.shape))

    def iterate_all(self) -> Iterator[LatticePoint]:
        """Iterate over all lattice points in deterministic order."""
        for coords in itertools.product(*(range(n) for n in self.shape)):
            yield LatticePoint(coords, self.shape)

    def iterate_with_indices(self) -> Iterator[Tuple[int, LatticePoint]]:
        """Iterate with linear indices in deterministic order."""
        for idx, point in enumerate(self.iterate_all()):
            yield idx, point

    def neighbors(self, point: LatticePoint) -> List[Tuple[LatticePoint, Fraction]]:
        """Get R4 neighbors of a point with coupling weights."""
        return self.r4_kernel.get_neighbors(point)

    def nearest_neighbors(self, point: LatticePoint) -> List[LatticePoint]:
        """Get only nearest (von Neumann) neighbors."""
        return [point + offset for offset in self.r4_kernel.SHELL_1]

    def distance_squared(self, p1: LatticePoint, p2: LatticePoint) -> int:
        """Compute squared Euclidean distance with minimum image convention."""
        diff = p1 - p2
        return sum(d * d for d in diff)

    def radial_shells(self, center: LatticePoint, max_radius: int) -> Dict[int, List[LatticePoint]]:
        """
        Organize all points within radius into shells by squared distance.

        Returns: Dict mapping squared_distance -> list of points
        """
        shells: Dict[int, List[LatticePoint]] = {}
        for point in self.iterate_all():
            r2 = self.distance_squared(center, point)
            if r2 <= max_radius * max_radius:
                if r2 not in shells:
                    shells[r2] = []
                shells[r2].append(point)
        return shells

    def create_sublattice(self, stride: Tuple[int, ...]) -> 'ToroidalHypercube':
        """Create a coarser sublattice with given stride per axis."""
        if len(stride) != len(self.shape):
            raise ValueError(f"Stride dimension mismatch")
        new_shape = tuple(n // s for n, s in zip(self.shape, stride))
        if any(ns <= 0 for ns in new_shape):
            raise ValueError(f"Stride too large for lattice shape")
        return ToroidalHypercube(new_shape, self.r4_radius)

    def is_palindromic(self) -> bool:
        """Check if lattice has palindromic symmetry (all dimensions equal)."""
        return len(set(self.shape)) == 1

    def validate_closure(self, point: LatticePoint) -> bool:
        """
        Invariant check (CODEX 7.2): Verify toroidal index closure.
        All coordinates must be within [0, N_k) after wrap.
        """
        for c, n in zip(point.coords, self.shape):
            if c < 0 or c >= n:
                return False
        return True


# Convenience factory functions

def create_3d_lattice(nx: int, ny: int, nz: int, r4_radius: int = 2) -> ToroidalHypercube:
    """Create a 3D toroidal lattice."""
    return ToroidalHypercube((nx, ny, nz), r4_radius)


def create_4d_hypercube(n: int, r4_radius: int = 2) -> ToroidalHypercube:
    """Create a symmetric 4D hypercube (palindromic lattice)."""
    return ToroidalHypercube((n, n, n, n), r4_radius)


def create_cubic_lattice(n: int, dims: int = 3, r4_radius: int = 2) -> ToroidalHypercube:
    """Create a symmetric cubic lattice in any dimension."""
    return ToroidalHypercube(tuple([n] * dims), r4_radius)


# Self-test for invariants
def _self_test():
    """Run basic invariant tests on module load."""
    # Test 3D lattice
    lattice = create_3d_lattice(8, 8, 8)
    assert lattice.total_sites == 512
    assert lattice.is_palindromic()

    # Test modular wrap
    p = lattice.point(10, -1, 8)
    assert p.coords == (2, 7, 0), f"Wrap failed: {p.coords}"
    assert lattice.validate_closure(p)

    # Test neighbors
    center = lattice.point(0, 0, 0)
    neighbors = lattice.neighbors(center)
    assert len(neighbors) > 0

    # Verify all neighbors are valid
    for n, weight in neighbors:
        assert lattice.validate_closure(n), f"Invalid neighbor: {n}"
        assert isinstance(weight, Fraction)

    # Test linear index roundtrip
    for idx in [0, 100, 511]:
        p = LatticePoint.from_linear_index(idx, lattice.shape)
        assert p.linear_index() == idx, f"Index roundtrip failed for {idx}"


# Run self-test on import
_self_test()
