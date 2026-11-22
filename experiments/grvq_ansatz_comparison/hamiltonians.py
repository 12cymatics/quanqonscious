"""Exact rational objective Hamiltonians for the GRVQ comparison experiment.

These are explicitly defined finite experiment models. They do not identify a
graph model with the GRVQ field PDE, a chemical Hamiltonian, or a new physical
law. Q_n bit-flip adjacency follows ``vedic_trainer.vedic.kernel.q4_complex``;
the Cartesian modular topology and row-major indexing of ``torus_graph``
follow ``core/lattice.py:ToroidalHypercube``. Spin Hamiltonians are constructed
directly from the Pauli actions written in their constructor docstrings.

Every stored coefficient, state-vector contraction, Rayleigh quotient, and
variance is a ``fractions.Fraction``. Constructor scalar arguments may also
be Python integers, which have an exact documented conversion. Floating-point
and boolean scalar inputs are rejected. ``dense_float`` is the only numerical
projection and exists solely for independent numerical eigensystem diagnostics.

Basis convention: integer state index has qubit q in bit q, so qubit zero is
the least significant bit. Graph site indices are amplitude basis indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import prod
from typing import Any, Mapping, Sequence


ZERO = Fraction(0)
ONE = Fraction(1)
SparseRow = tuple[tuple[int, Fraction], ...]
ExactState = tuple[Fraction, ...]


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _rational(value: Any, name: str) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    if type(value) is Fraction:
        return value
    raise TypeError(f"{name} must be an exact Fraction or Python integer")


def _potential(values: Sequence[Fraction | int], dimension: int) -> ExactState:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("potential must be an explicit ordered exact sequence")
    if len(values) != dimension:
        raise ValueError(f"potential requires exactly {dimension} entries")
    return tuple(_rational(value, f"potential[{i}]") for i, value in enumerate(values))


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class ExactHamiltonian:
    """A real symmetric sparse matrix with exact rational coefficients.

    ``rows[i]`` is a sorted tuple of ``(j, Fraction)`` entries. Duplicate
    columns and stored zeros are forbidden. Every column is in ``[0,2**n)``.
    Exact symmetry is checked at construction; it is not inferred from a
    floating-point tolerance. The metadata records the experiment definition.
    """

    name: str
    nqubits: int
    rows: tuple[SparseRow, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a nonempty string")
        _integer(self.nqubits, "nqubits", minimum=1)
        if type(self.rows) is not tuple or len(self.rows) != self.dimension:
            raise ValueError(f"rows must be a tuple of exactly {self.dimension} rows")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        row_maps = []
        for i, row in enumerate(self.rows):
            if type(row) is not tuple:
                raise TypeError(f"row {i} must be a tuple")
            previous = -1
            mapping = {}
            for entry in row:
                if type(entry) is not tuple or len(entry) != 2:
                    raise TypeError(f"row {i} entries must be (column,Fraction) tuples")
                j, value = entry
                _integer(j, f"row {i} column", minimum=0)
                if j >= self.dimension or j <= previous:
                    raise ValueError(f"row {i} columns must be strictly increasing and in range")
                if type(value) is not Fraction:
                    raise TypeError(f"matrix coefficient ({i},{j}) must be a Fraction")
                if not value:
                    raise ValueError(f"zero coefficient ({i},{j}) must not be stored")
                previous = j
                mapping[j] = value
            row_maps.append(mapping)
        for i, row in enumerate(row_maps):
            for j, value in row.items():
                if row_maps[j].get(i, ZERO) != value:
                    raise ValueError(f"matrix is not exactly symmetric at ({i},{j})")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dimension(self) -> int:
        return 1 << self.nqubits

    @property
    def nnz(self) -> int:
        return sum(len(row) for row in self.rows)

    def _state(self, state: ExactState) -> ExactState:
        if type(state) is not tuple or len(state) != self.dimension:
            raise ValueError(f"state must be a tuple with {self.dimension} amplitudes")
        if any(type(value) is not Fraction for value in state):
            raise TypeError("every state amplitude must be a Fraction")
        return state

    def matvec(self, state: ExactState) -> ExactState:
        state = self._state(state)
        return tuple(sum((value * state[j] for j, value in row), ZERO) for row in self.rows)

    def norm_squared(self, state: ExactState) -> Fraction:
        return sum((value * value for value in self._state(state)), ZERO)

    def energy(self, state: ExactState) -> Fraction:
        """Return <psi|H|psi>/<psi|psi>, without normalizing or altering psi."""
        state = self._state(state)
        norm = self.norm_squared(state)
        if not norm:
            raise ValueError("the zero vector has no Rayleigh quotient")
        image = self.matvec(state)
        return sum((a * b for a, b in zip(state, image)), ZERO) / norm

    def variance(self, state: ExactState) -> Fraction:
        """Return the exact normalized squared eigen-residual.

        For this real symmetric matrix this is <H**2>-<H>**2 and is zero
        exactly for a nonzero eigenvector, including a degenerate eigenvector.
        """
        state = self._state(state)
        norm = self.norm_squared(state)
        if not norm:
            raise ValueError("the zero vector has no energy variance")
        image = self.matvec(state)
        energy = sum((a * b for a, b in zip(state, image)), ZERO) / norm
        return sum(((h - energy * x) ** 2 for h, x in zip(image, state)), ZERO) / norm

    def dense_float(self):
        """Project to NumPy float64 for diagnostics; never feed it into exact state updates."""
        import numpy as np

        matrix = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for i, row in enumerate(self.rows):
            for j, value in row:
                matrix[i, j] = float(value)
        if not np.isfinite(matrix).all():
            raise OverflowError("Hamiltonian cannot be represented by a finite float64 diagnostic matrix")
        return matrix


def _matrix(
    name: str, nqubits: int, rows: list[dict[int, Fraction]], metadata: Mapping[str, Any]
) -> ExactHamiltonian:
    sparse = tuple(tuple((j, value) for j, value in sorted(row.items()) if value) for row in rows)
    return ExactHamiltonian(name, nqubits, sparse, metadata)


def _add(row: dict[int, Fraction], column: int, value: Fraction) -> None:
    row[column] = row.get(column, ZERO) + value


def q4_laplacian(nqubits: int, potential: Sequence[Fraction | int]) -> ExactHamiltonian:
    """Q_n graph Laplacian plus an explicitly supplied rational potential.

    H_ij = n*delta_ij - sum_q delta_(j,i XOR 2**q) + V_i*delta_ij.
    For n=4 this is the repository's Q4 adjacency; arbitrary n retains its
    bit-flip graph definition rather than inventing further geometric physics.
    """
    nqubits = _integer(nqubits, "nqubits", minimum=1)
    dimension = 1 << nqubits
    potential = _potential(potential, dimension)
    rows = []
    for i in range(dimension):
        row = {i: Fraction(nqubits) + potential[i]}
        for q in range(nqubits):
            _add(row, i ^ (1 << q), -ONE)
        rows.append(row)
    return _matrix(f"Q{nqubits}_laplacian", nqubits, rows, {
        "family": "hypercube_schrodinger",
        "formula": "H=L(Q_n)+diag(V)",
        "topology_source": "vedic_trainer/vedic/kernel/q4_complex.py:neighbours",
        "potential": tuple(str(x) for x in potential),
        "potential_status": "explicit experiment input",
        "units": "dimensionless",
    })


def _chain_bonds(nqubits: int, periodic: bool) -> tuple[tuple[int, int], ...]:
    bonds = {(q, q + 1) for q in range(nqubits - 1)}
    if periodic and nqubits > 1:
        bonds.add((0, nqubits - 1))
    return tuple(sorted(bonds))


def tfim(
    nqubits: int, J: Fraction | int = ONE, h: Fraction | int = ONE, periodic: bool = True
) -> ExactHamiltonian:
    """Transverse-field Ising H=-J sum_(a,b) Za Zb - h sum_q Xq.

    Each undirected chain bond appears once. A two-site periodic chain has
    one bond, and a one-site chain has none. Z|b>=(-1)**b|b>; X flips b.
    These conventions fully specify the finite-size experiment.
    """
    nqubits = _integer(nqubits, "nqubits", minimum=1)
    periodic = _bool(periodic, "periodic")
    J, h = _rational(J, "J"), _rational(h, "h")
    bonds = _chain_bonds(nqubits, periodic)
    rows = []
    for i in range(1 << nqubits):
        diagonal = sum((-J * (1 if ((i >> a) ^ (i >> b)) & 1 == 0 else -1)
                        for a, b in bonds), ZERO)
        row = {i: diagonal}
        for q in range(nqubits):
            _add(row, i ^ (1 << q), -h)
        rows.append(row)
    return _matrix(f"TFIM_{nqubits}", nqubits, rows, {
        "family": "tfim", "formula": "H=-J sum_bonds Za Zb-h sum_q Xq",
        "J": str(J), "h": str(h), "periodic": periodic, "bonds": bonds,
        "bond_convention": "unique undirected nearest-neighbor bonds",
        "pauli_convention": "Z|0>=|0>; Z|1>=-|1>; X flips one bit",
        "units": "dimensionless experiment coupling units",
    })


def xxz(
    nqubits: int, Jxy: Fraction | int = ONE, Jz: Fraction | int = ONE,
    field: Fraction | int = ZERO, periodic: bool = True,
) -> ExactHamiltonian:
    """Pauli XXZ H=sum_bonds[Jxy(XX+YY)+Jz ZZ]+field sum_q Zq.

    These are Pauli matrices, not spin operators sigma/2. Thus XX+YY has
    exchange coefficient 2 between opposite-bit configurations and zero
    between equal-bit configurations. The longitudinal field has a plus sign.
    Bonds use the same unique-edge convention as ``tfim``.
    """
    nqubits = _integer(nqubits, "nqubits", minimum=1)
    periodic = _bool(periodic, "periodic")
    Jxy, Jz = _rational(Jxy, "Jxy"), _rational(Jz, "Jz")
    field = _rational(field, "field")
    bonds = _chain_bonds(nqubits, periodic)
    rows = []
    for i in range(1 << nqubits):
        diagonal = field * (nqubits - 2 * i.bit_count())
        row = {}
        for a, b in bonds:
            opposite = ((i >> a) ^ (i >> b)) & 1
            diagonal += -Jz if opposite else Jz
            if opposite:
                _add(row, i ^ (1 << a) ^ (1 << b), 2 * Jxy)
        _add(row, i, diagonal)
        rows.append(row)
    return _matrix(f"XXZ_{nqubits}", nqubits, rows, {
        "family": "xxz", "formula": "H=sum_bonds[Jxy(XX+YY)+Jz ZZ]+field sum_q Zq",
        "Jxy": str(Jxy), "Jz": str(Jz), "field": str(field),
        "periodic": periodic, "bonds": bonds,
        "bond_convention": "unique undirected nearest-neighbor bonds",
        "pauli_convention": "Pauli matrices; no implicit spin-half factor",
        "units": "dimensionless experiment coupling units",
    })


def torus_graph(
    shape: tuple[int, ...], potential: Sequence[Fraction | int]
) -> ExactHamiltonian:
    """Periodic Cartesian finite-difference graph Laplacian plus diag(V).

    For every axis, (L psi)_x includes 2 psi_x-psi_(x+e)-psi_(x-e).
    A size-two axis therefore has the same neighbor twice with total weight
    -2. This is the periodic stencil convention, deliberately distinct from
    a simple-edge hypercube. Every axis has at least two sites, and the product
    must be a power of two to match the amplitude register exactly.
    """
    if type(shape) is not tuple or not shape:
        raise TypeError("shape must be a nonempty tuple of Python integers")
    for axis, size in enumerate(shape):
        _integer(size, f"shape[{axis}]", minimum=2)
    dimension = prod(shape)
    if dimension & (dimension - 1):
        raise ValueError("torus site count must be a power of two; padding is not permitted")
    nqubits = dimension.bit_length() - 1
    potential = _potential(potential, dimension)
    strides = tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))
    rows = []
    for i, coords in enumerate(product(*(range(size) for size in shape))):
        row = {i: Fraction(2 * len(shape)) + potential[i]}
        for axis, size in enumerate(shape):
            for direction in (-1, 1):
                wrapped = (coords[axis] + direction) % size
                j = i + (wrapped - coords[axis]) * strides[axis]
                _add(row, j, -ONE)
        rows.append(row)
    return _matrix("torus_" + "x".join(map(str, shape)), nqubits, rows, {
        "family": "toroidal_schrodinger", "formula": "H=sum_axes(2I-T_axis-T_axis^-1)+diag(V)",
        "shape": shape, "boundary": "periodic", "indexing": "row major, last axis fastest",
        "neighbor_multiplicity": "both signed neighbors retained, including size-two axes",
        "topology_source": "core/lattice.py:ToroidalHypercube.wrap_index and iterate_all",
        "potential": tuple(str(x) for x in potential), "units": "dimensionless",
    })


def schrodinger_1d(
    nqubits: int, spacing: Fraction | int, potential: Sequence[Fraction | int],
    kinetic: Fraction | int = ONE, boundary: str = "dirichlet",
) -> ExactHamiltonian:
    """Finite-grid -kinetic*d²/dx²+V with an explicit rational spacing.

    Dirichlet uses zero exterior amplitudes, so the first and last diagonal
    entries remain 2*kinetic/spacing². Periodic uses both signed wrapped
    neighbors, including multiplicity for a two-site grid. This is a finite
    second-order discretization; it is not an exact continuum PDE solution.
    """
    nqubits = _integer(nqubits, "nqubits", minimum=1)
    spacing, kinetic = _rational(spacing, "spacing"), _rational(kinetic, "kinetic")
    if spacing <= 0 or kinetic <= 0:
        raise ValueError("spacing and kinetic must be positive exact rationals")
    if boundary not in ("dirichlet", "periodic"):
        raise ValueError("boundary must be 'dirichlet' or 'periodic'")
    dimension = 1 << nqubits
    potential = _potential(potential, dimension)
    scale = kinetic / (spacing * spacing)
    rows = []
    for i in range(dimension):
        row = {i: 2 * scale + potential[i]}
        for j in (i - 1, i + 1):
            if boundary == "periodic":
                _add(row, j % dimension, -scale)
            elif 0 <= j < dimension:
                _add(row, j, -scale)
        rows.append(row)
    return _matrix(f"schrodinger_1d_{dimension}_{boundary}", nqubits, rows, {
        "family": "finite_grid_schrodinger", "formula": "H=kinetic/spacing²*(2I-T-T^-1)+diag(V)",
        "spacing": str(spacing), "kinetic": str(kinetic), "boundary": boundary,
        "potential": tuple(str(x) for x in potential), "discretization_order": 2,
        "status": "declared finite discretization, not an exact continuum solution",
    })


def self_check() -> dict[str, Any]:
    """Execute exact analytical eigenpair and convention checks.

    These establish constructor behavior against independently known simple
    eigenvectors, rather than counting Hermiticity or finite outputs as an
    objective-level test. No numerical tolerance is involved.
    """
    checks = []

    def eigenpair(label: str, H: ExactHamiltonian, state: ExactState, expected: Fraction) -> None:
        if H.matvec(state) != tuple(expected * x for x in state):
            raise AssertionError(f"{label}: exact eigenvector identity failed")
        if H.energy(state) != expected or H.variance(state) != ZERO:
            raise AssertionError(f"{label}: energy or variance failed")
        checks.append({"name": label, "energy": str(expected), "variance": "0"})

    for n in (4, 6):
        d = 1 << n
        uniform = (ONE,) * d
        constant = Fraction(3, 7)
        H = q4_laplacian(n, (constant,) * d)
        eigenpair(f"Q{n} constant ground mode", H, uniform, constant)
        checker = tuple(Fraction(-1 if i.bit_count() & 1 else 1) for i in range(d))
        eigenpair(f"Q{n} highest Walsh mode", H, checker, constant + 2 * n)
        ferro = (ONE,) + (ZERO,) * (d - 1)
        eigenpair(f"TFIM{n} zero-field ferromagnet", tfim(n, J=Fraction(5, 3), h=ZERO), ferro, Fraction(-5 * n, 3))
        eigenpair(f"TFIM{n} decoupled transverse field", tfim(n, J=ZERO, h=Fraction(2, 3)), uniform, Fraction(-2 * n, 3))
    singlet = (ZERO, ONE, -ONE, ZERO)
    eigenpair("XXZ two-spin singlet", xxz(2, Jxy=Fraction(2, 3), Jz=Fraction(5, 7), field=Fraction(4, 9)), singlet, -Fraction(4, 3) - Fraction(5, 7))
    eigenpair("XXZ two-spin symmetric exchange", xxz(2, Jxy=Fraction(2, 3), Jz=Fraction(5, 7)), (ZERO, ONE, ONE, ZERO), Fraction(4, 3) - Fraction(5, 7))
    for shape in ((4, 4), (2, 2, 2, 2)):
        d = prod(shape)
        H = torus_graph(shape, (ZERO,) * d)
        eigenpair(f"torus {shape} constant ground mode", H, (ONE,) * d, ZERO)
        checker = tuple(Fraction(-1 if sum(coords) & 1 else 1) for coords in product(*(range(s) for s in shape)))
        eigenpair(f"torus {shape} alternating mode", H, checker, Fraction(4 * len(shape)))
    eigenpair("two-site Dirichlet lowest mode", schrodinger_1d(1, Fraction(1, 2), (ZERO, ZERO), kinetic=Fraction(3, 2)), (ONE, ONE), Fraction(6))
    eigenpair("two-site Dirichlet highest mode", schrodinger_1d(1, Fraction(1, 2), (ZERO, ZERO), kinetic=Fraction(3, 2)), (ONE, -ONE), Fraction(18))
    return {"status": "passed", "arithmetic": "exact Fraction", "checks": checks}


__all__ = ["ExactHamiltonian", "q4_laplacian", "tfim", "xxz", "torus_graph", "schrodinger_1d", "self_check"]
