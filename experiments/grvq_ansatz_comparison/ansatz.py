"""Exact projective GRVQ states and explicit rational quantum rotations.

The product is the source's prod(1 + alpha_j S_j) R f construction.
Shape values, carrier, radial coordinates and shell widths are supplied by the
experiment; no polynomial is substituted for an unspecified shape function.
The state retains unnormalized rational amplitudes. Observables divide by its
squared norm exactly. Only OpenQASM angle serialization is numerical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from hashlib import sha256
from math import gcd, lcm
from typing import Iterable, Sequence


def rational(value) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("Exact arithmetic requires int or Fraction operands")
    return Fraction(value)


def polynomial_value(coefficients: Sequence[Fraction], x: Fraction) -> Fraction:
    """Evaluate every coefficient of an ascending-power polynomial."""
    x = rational(x)
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * x + rational(coefficient)
    return result


def radial_factor(radius, widths: Sequence[Fraction], power: int) -> Fraction:
    radius = rational(radius)
    if radius < 0:
        raise ValueError("A radial coordinate must be nonnegative")
    if type(power) is not int or power not in (2, 4):
        raise ValueError("The comparison defines radial powers two and four")
    if not widths:
        raise ValueError("Supply the radial shell widths explicitly")
    result = Fraction(1)
    for width in widths:
        width = rational(width)
        if width <= 0:
            raise ValueError("Radial shell widths must be positive")
        raised = width ** power
        result *= raised / (radius ** power + raised)
    return result


def cayley_parts(parameter) -> tuple[int, int, int]:
    """Return denominator, cosine numerator, sine numerator for a Cayley map."""
    parameter = rational(parameter)
    p, q = parameter.numerator, parameter.denominator
    denominator, cosine, sine = q*q + p*p, q*q - p*p, 2*p*q
    common = gcd(gcd(denominator, cosine), sine)
    return denominator // common, cosine // common, sine // common


@dataclass(frozen=True)
class ExactSparseOperator:
    rows: tuple[tuple[tuple[int, int], ...], ...]
    denominator: int

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[tuple[int, Fraction]]]):
        rows = tuple(tuple((index, rational(value)) for index, value in row) for row in rows)
        if not rows:
            raise ValueError("An operator must have nonempty rows")
        size = len(rows)
        if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < size
               for row in rows for index, _ in row):
            raise ValueError("An operator column is outside the state dimension")
        denominator = lcm(*(value.denominator for row in rows for _, value in row))
        integers = tuple(tuple((index, value.numerator * (denominator // value.denominator))
                               for index, value in row) for row in rows)
        return cls(integers, denominator)

    @property
    def size(self):
        return len(self.rows)

    def apply_numerators(self, numerators: Sequence[int]) -> tuple[int, ...]:
        if len(numerators) != self.size:
            raise ValueError("Operator and state dimensions differ")
        return tuple(sum(coefficient * numerators[index] for index, coefficient in row)
                     for row in self.rows)

    def is_symmetric(self) -> bool:
        entries = {}
        for i, row in enumerate(self.rows):
            for j, value in row:
                entries[i, j] = entries.get((i, j), 0) + value
        return all(value == entries.get((j, i), 0) for (i, j), value in entries.items())


@dataclass(frozen=True)
class ExactState:
    numerators: tuple[int, ...]
    denominator: int = 1

    def __post_init__(self):
        numerators = tuple(self.numerators)
        if not numerators or any(isinstance(value, bool) or not isinstance(value, int)
                                 for value in numerators):
            raise TypeError("A state requires an ordered nonempty integer numerator vector")
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int) or self.denominator <= 0:
            raise ValueError("The common denominator must be a positive integer")
        if not any(numerators):
            raise ValueError("The zero vector does not define a quantum state")
        common = reduce(gcd, numerators, self.denominator)
        object.__setattr__(self, "numerators", tuple(value // common for value in numerators))
        object.__setattr__(self, "denominator", self.denominator // common)

    @classmethod
    def from_fractions(cls, values: Sequence[Fraction]):
        values = tuple(rational(value) for value in values)
        if not values:
            raise ValueError("A state requires amplitudes")
        denominator = lcm(*(value.denominator for value in values))
        return cls(tuple(value.numerator * (denominator // value.denominator) for value in values), denominator)

    @property
    def size(self) -> int:
        return len(self.numerators)

    @property
    def n_qubits(self) -> int:
        if self.size < 2 or self.size & (self.size - 1):
            raise ValueError("Quantum export requires a power-of-two dimension of at least two")
        return self.size.bit_length() - 1

    @property
    def integer_norm_squared(self) -> int:
        return sum(value*value for value in self.numerators)

    @property
    def norm_squared(self) -> Fraction:
        return Fraction(self.integer_norm_squared, self.denominator*self.denominator)

    def to_fractions(self) -> tuple[Fraction, ...]:
        return tuple(Fraction(value, self.denominator) for value in self.numerators)

    def probabilities(self) -> tuple[Fraction, ...]:
        norm = self.integer_norm_squared
        return tuple(Fraction(value*value, norm) for value in self.numerators)

    def measurement_probabilities(self, basis: str) -> tuple[Fraction, ...]:
        """Exact X/Y/Z tensor-basis probabilities, with letters in qubit order."""
        n = self.n_qubits
        if len(basis) != n or any(letter not in "XYZ" for letter in basis):
            raise ValueError("Supply one X, Y, or Z measurement basis letter per qubit")
        real, imag = list(self.numerators), [0]*self.size
        for q, letter in enumerate(basis):
            if letter == "Z":
                continue
            bit = 1 << q
            if letter == "Y":
                for index in range(self.size):
                    if index & bit:
                        real[index], imag[index] = imag[index], -real[index]
            for left in range(self.size):
                if not left & bit:
                    right = left | bit
                    a, b = real[left], real[right]
                    c, d = imag[left], imag[right]
                    real[left], real[right] = a+b, a-b
                    imag[left], imag[right] = c+d, c-d
        squares = tuple(a*a+b*b for a, b in zip(real, imag))
        total = sum(squares)
        return tuple(Fraction(value, total) for value in squares)

    def expectation(self, operator) -> Fraction:
        if not isinstance(operator, ExactSparseOperator):
            operator = ExactSparseOperator.from_rows(operator)
        applied = operator.apply_numerators(self.numerators)
        numerator = sum(left*right for left, right in zip(self.numerators, applied))
        return Fraction(numerator, operator.denominator*self.integer_norm_squared)

    def variance(self, operator) -> Fraction:
        if not isinstance(operator, ExactSparseOperator):
            operator = ExactSparseOperator.from_rows(operator)
        if not operator.is_symmetric():
            raise ValueError("Quantum energy variance requires a symmetric real operator")
        applied = operator.apply_numerators(self.numerators)
        second = Fraction(sum(value*value for value in applied),
                          operator.denominator**2*self.integer_norm_squared)
        return second - self.expectation(operator)**2

    def fidelity(self, other: "ExactState") -> Fraction:
        if self.size != other.size:
            raise ValueError("State dimensions differ")
        overlap = sum(left*right for left, right in zip(self.numerators, other.numerators))
        return Fraction(overlap*overlap, self.integer_norm_squared*other.integer_norm_squared)

    def apply_cayley(self, edge: tuple[int, int], parameter) -> "ExactState":
        return self.apply_cayley_matching((edge,), (parameter,))

    def apply_cayley_matching(self, edges: Sequence[tuple[int, int]],
                             parameters: Sequence[Fraction]) -> "ExactState":
        edges = tuple(tuple(edge) for edge in edges)
        parameters = tuple(parameters)
        if len(edges) != len(parameters):
            raise ValueError("Every edge requires its explicit Cayley parameter")
        used = set()
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("Every rotation requires a pair of vertices")
            left, right = edge
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.size for i in edge):
                raise ValueError("A rotation vertex is outside the state")
            if left == right or left in used or right in used:
                raise ValueError("A matching requires distinct nonoverlapping vertex pairs")
            used.update(edge)
        parts = tuple(cayley_parts(parameter) for parameter in parameters)
        denominator = lcm(*(part[0] for part in parts))
        out = [value*denominator for value in self.numerators]
        for (left, right), (base, cosine, sine) in zip(edges, parts):
            multiplier = denominator // base
            u, v = self.numerators[left], self.numerators[right]
            out[left] = (cosine*u - sine*v)*multiplier
            out[right] = (sine*u + cosine*v)*multiplier
        return ExactState(tuple(out), self.denominator*denominator)

    def apply_cayley_schedule(self, gates: Iterable[tuple[int, int, Fraction]]) -> "ExactState":
        current = self
        for left, right, parameter in gates:
            current = current.apply_cayley((left, right), parameter)
        return current

    def apply_cz(self, first: int, second: int) -> "ExactState":
        n = self.n_qubits
        if first == second or any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < n
                                 for q in (first, second)):
            raise ValueError("CZ requires two distinct qubit indices")
        mask = (1 << first) | (1 << second)
        return ExactState(tuple(-value if index & mask == mask else value
                                for index, value in enumerate(self.numerators)), self.denominator)

    def to_dict(self) -> dict:
        return {"numerators_hex": [hex(value) for value in self.numerators],
                "denominator_hex": hex(self.denominator),
                "normalization": "amplitude_i / sqrt(sum_j amplitude_j**2)",
                "basis_index": "sum_q bit(q)*2**q"}

    def digest(self) -> str:
        value = "|".join([hex(self.denominator)] + [hex(x) for x in self.numerators])
        return sha256(value.encode()).hexdigest()

    def to_qasm(self, *, decimal_digits: int = 40, basis: str | None = None,
                measure: bool = True) -> str:
        """Prepare this complete real state with uniformly controlled Ry gates.

        The exact integer amplitude ratios determine every angle. Decimal gate
        angles are evaluated only here using extra arbitrary precision. The
        circuit uses n-1 controlled rotation layers and includes all amplitudes.
        Bit q is bit 2**q in the integer basis index. Basis letters use q order.
        """
        import mpmath as mp

        n = self.n_qubits
        if not isinstance(decimal_digits, int) or decimal_digits < 17:
            raise ValueError("Use at least seventeen decimal digits for exported angles")
        if basis is None:
            basis = "Z"*n
        if len(basis) != n or any(letter not in "XYZ" for letter in basis):
            raise ValueError("Supply one X, Y, or Z measurement basis letter per qubit")
        lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];']
        if measure:
            lines.append(f'creg c[{n}];')
        with mp.workdps(decimal_digits + 30):
            for target in reversed(range(n)):
                control_count = n-target-1
                angles = []
                block_size = 1 << (target+1)
                half = block_size // 2
                for prefix in range(1 << control_count):
                    start = prefix*block_size
                    if target == 0:
                        left, right = self.numerators[start:start+2]
                        angle = mp.mpf(0) if left == right == 0 else 2*mp.atan2(mp.mpf(right), mp.mpf(left))
                    else:
                        left = sum(value*value for value in self.numerators[start:start+half])
                        right = sum(value*value for value in self.numerators[start+half:start+block_size])
                        angle = mp.mpf(0) if left == right == 0 else 2*mp.atan2(mp.sqrt(right), mp.sqrt(left))
                    angles.append(angle)
                count = len(angles)
                for j in range(count):
                    gray = j ^ (j >> 1)
                    angle = sum((-1 if (i & gray).bit_count() % 2 else 1)*value
                                for i, value in enumerate(angles)) / count
                    encoded = mp.nstr(angle, n=decimal_digits, min_fixed=-8, max_fixed=8)
                    if "e" in encoded:
                        mantissa, exponent = encoded.split("e")
                        encoded = mantissa + "e" + str(int(exponent))
                    lines.append(f'ry({encoded}) q[{target}];')
                    if control_count:
                        next_j = (j+1) % count
                        next_gray = next_j ^ (next_j >> 1)
                        changed = (gray ^ next_gray).bit_length()-1
                        control = target+1+changed
                        lines.append(f'cx q[{control}],q[{target}];')
        for q, letter in enumerate(basis):
            if letter == "X":
                lines.append(f'ry(-pi/2) q[{q}];')
            elif letter == "Y":
                lines.extend((f'rz(-pi/2) q[{q}];', f'ry(-pi/2) q[{q}];'))
        if measure:
            lines.append('measure q -> c;')
        return "\n".join(lines)+"\n"


@dataclass(frozen=True)
class GRVQProductAnsatz:
    shape_table: tuple[tuple[Fraction, ...], ...]
    radii: tuple[Fraction, ...]
    carrier: tuple[Fraction, ...]
    widths: tuple[Fraction, ...]
    _radial_carriers: tuple[tuple[Fraction, ...], tuple[Fraction, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        shapes = tuple(tuple(rational(value) for value in row) for row in self.shape_table)
        radii = tuple(rational(value) for value in self.radii)
        carrier = tuple(rational(value) for value in self.carrier)
        widths = tuple(rational(value) for value in self.widths)
        if not shapes or not shapes[0] or len(shapes) != len(radii) or len(shapes) != len(carrier):
            raise ValueError("Supply one complete shape row, radius and carrier value per state component")
        if any(len(row) != len(shapes[0]) for row in shapes):
            raise ValueError("All shape rows must have the same number of modes")
        if any(radius < 0 for radius in radii) or not widths or any(width <= 0 for width in widths):
            raise ValueError("Radii must be nonnegative and all explicit shell widths positive")
        object.__setattr__(self, "shape_table", shapes)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "_radial_carriers", tuple(
            tuple(value*radial_factor(radius, widths, power)
                  for radius, value in zip(radii, carrier))
            for power in (2,4)))

    @property
    def parameter_count(self) -> int:
        return len(self.shape_table[0])

    def evaluate(self, alphas: Sequence[Fraction], *, power: int = 4) -> ExactState:
        if type(power) is not int or power not in (2,4):
            raise ValueError("The comparison defines radial powers two and four")
        alphas = tuple(rational(value) for value in alphas)
        if len(alphas) != self.parameter_count:
            raise ValueError("Supply exactly one alpha for every shape function")
        values = []
        radial_carrier = self._radial_carriers[0 if power == 2 else 1]
        for row, initial in zip(self.shape_table, radial_carrier):
            value = initial
            for alpha, shape in zip(alphas, row):
                value *= 1+alpha*shape
            values.append(value)
        return ExactState.from_fractions(values)


def hypercube_matchings(n_qubits: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("Supply a positive integer qubit count")
    return tuple(tuple((i, i | (1 << bit)) for i in range(1 << n_qubits) if not i & (1 << bit))
                 for bit in range(n_qubits))


def mix_hypercube(state: ExactState, parameters: Sequence[Fraction], *,
                  ordering: Sequence[int] | None = None,
                  conditional_signs: bool = False,
                  control_mode: str = "parity") -> ExactState:
    """Apply one matching per cube dimension, optionally parity-conditioned.

    Parity conditioning uses t*(-1)**popcount(other vertex bits). Neighbor
    conditioning uses t*(-1)**bit((d+1) mod n). The parity generators commute;
    the neighbor generators generally do not for three or more qubits. These
    are explicit circuit constructions, not renamed arithmetic sutras.
    """
    n = state.n_qubits
    parameters = tuple(rational(value) for value in parameters)
    if control_mode not in ("parity", "neighbor"):
        raise ValueError("Control mode must be parity or neighbor")
    if len(parameters) != n:
        raise ValueError("Supply one Cayley parameter for every cube dimension")
    if ordering is None:
        ordering = tuple(range(n))
    ordering = tuple(ordering)
    if sorted(ordering) != list(range(n)):
        raise ValueError("Ordering must contain every cube dimension exactly once")
    matchings = hypercube_matchings(n)
    current = state
    for bit in ordering:
        edges = matchings[bit]
        signs = tuple(left.bit_count() % 2 if control_mode == "parity"
                      else (left >> ((bit+1) % n)) & 1 for left,_ in edges)
        angles = tuple(-parameters[bit] if conditional_signs and sign else parameters[bit] for sign in signs)
        current = current.apply_cayley_matching(edges, angles)
    return current


def source_ry_cz_state(n_qubits: int, parameters: Sequence[Fraction], *,
                       layers: int = 1, ring: bool = False) -> ExactState:
    """The source H/Ry/CZ topology, using exact rational Ry parametrization.

    A parameter t denotes physical Ry angle 4*atan(t). The initial all-ones
    vector is exactly the projective uniform-superposition state.
    """
    parameters = tuple(rational(value) for value in parameters)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("Supply a positive integer layer count")
    if len(parameters) != n_qubits*layers:
        raise ValueError("Supply one rotation parameter per qubit and layer")
    matchings = hypercube_matchings(n_qubits)
    current = ExactState((1,)*(1 << n_qubits))
    for layer in range(layers):
        for bit, edges in enumerate(matchings):
            parameter = parameters[layer*n_qubits+bit]
            current = current.apply_cayley_matching(edges, (parameter,)*len(edges))
        for bit in range(n_qubits-1):
            current = current.apply_cz(bit, bit+1)
        if ring and n_qubits > 2:
            current = current.apply_cz(n_qubits-1, 0)
    return current


def _uniform_ry_qasm(angles, target, controls_lsb, decimal_digits, mp):
    count = len(angles)
    if count != 1 << len(controls_lsb):
        raise ValueError("A uniformly controlled rotation requires every control angle")
    lines = []
    for j in range(count):
        gray = j ^ (j >> 1)
        angle = sum((-1 if (i & gray).bit_count() % 2 else 1)*value
                    for i, value in enumerate(angles)) / count
        encoded = mp.nstr(angle, n=decimal_digits, min_fixed=-8, max_fixed=8)
        if "e" in encoded:
            mantissa, exponent = encoded.split("e")
            encoded = mantissa + "e" + str(int(exponent))
        lines.append(f'ry({encoded}) q[{target}];')
        if controls_lsb:
            next_j = (j+1) % count
            next_gray = next_j ^ (next_j >> 1)
            changed = (gray ^ next_gray).bit_length()-1
            lines.append(f'cx q[{controls_lsb[changed]}],q[{target}];')
    return lines


def _measurement_qasm(lines, n_qubits, basis, measure):
    if basis is None:
        basis = "Z"*n_qubits
    if len(basis) != n_qubits or any(letter not in "XYZ" for letter in basis):
        raise ValueError("Supply one X, Y, or Z measurement basis letter per qubit")
    for q, letter in enumerate(basis):
        if letter == "X":
            lines.append(f'ry(-pi/2) q[{q}];')
        elif letter == "Y":
            lines.extend((f'rz(-pi/2) q[{q}];', f'ry(-pi/2) q[{q}];'))
    if measure:
        lines.append('measure q -> c;')
    return "\n".join(lines)+"\n"


def qasm_product_mixer(product_state: ExactState, parameters: Sequence[Fraction], *,
                       ordering: Sequence[int] | None = None,
                       conditional_signs: bool = False,
                       control_mode: str = "parity",
                       basis: str | None = None, decimal_digits: int = 40,
                       measure: bool = True) -> str:
    """Export product preparation followed by the actual learned cube mixers.

    This retains the algorithm's two stages instead of compiling its final
    state into a new preparation circuit. Exact parameters remain rational;
    only physical gate angles 4*atan(t) are serialized as decimals.
    """
    import mpmath as mp

    n = product_state.n_qubits
    parameters = tuple(rational(value) for value in parameters)
    if control_mode not in ("parity", "neighbor"):
        raise ValueError("Control mode must be parity or neighbor")
    if len(parameters) != n:
        raise ValueError("Supply one Cayley parameter for every cube dimension")
    if ordering is None:
        ordering = tuple(range(n))
    ordering = tuple(ordering)
    if sorted(ordering) != list(range(n)):
        raise ValueError("Ordering must contain every cube dimension exactly once")
    lines = product_state.to_qasm(decimal_digits=decimal_digits, measure=False).splitlines()
    if measure:
        lines.insert(3, f'creg c[{n}];')
    with mp.workdps(decimal_digits+30):
        for bit in ordering:
            parameter = parameters[bit]
            angle = 4*mp.atan(mp.mpf(parameter.numerator)/parameter.denominator)
            controls_lsb = tuple(q for q in range(n) if q != bit)
            neighbor = (bit+1) % n
            signs = tuple(prefix.bit_count() % 2 if control_mode == "parity"
                          else (prefix >> controls_lsb.index(neighbor)) & 1 if neighbor != bit else 0
                          for prefix in range(1 << (n-1)))
            angles = tuple(-angle if conditional_signs and sign else angle for sign in signs)
            lines.extend(_uniform_ry_qasm(angles, bit, controls_lsb, decimal_digits, mp))
    return _measurement_qasm(lines, n, basis, measure)


def qasm_source_ry_cz(n_qubits: int, parameters: Sequence[Fraction], *,
                      layers: int = 1, ring: bool = False,
                      basis: str | None = None, decimal_digits: int = 40,
                      measure: bool = True) -> str:
    """Export the source H/Ry/CZ topology without dense state compilation."""
    import mpmath as mp

    parameters = tuple(rational(value) for value in parameters)
    hypercube_matchings(n_qubits)
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ValueError("Supply a positive integer layer count")
    if len(parameters) != n_qubits*layers:
        raise ValueError("Supply one rotation parameter per qubit and layer")
    if not isinstance(decimal_digits, int) or decimal_digits < 17:
        raise ValueError("Use at least seventeen decimal digits for exported angles")
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n_qubits}];']
    if measure:
        lines.append(f'creg c[{n_qubits}];')
    for bit in range(n_qubits):
        lines.append(f'ry(pi/2) q[{bit}];')
    with mp.workdps(decimal_digits+30):
        for layer in range(layers):
            for bit in range(n_qubits):
                parameter = parameters[layer*n_qubits+bit]
                angle = 4*mp.atan(mp.mpf(parameter.numerator)/parameter.denominator)
                encoded = mp.nstr(angle, n=decimal_digits, min_fixed=-8, max_fixed=8)
                if "e" in encoded:
                    mantissa, exponent = encoded.split("e")
                    encoded = mantissa + "e" + str(int(exponent))
                lines.append(f'ry({encoded}) q[{bit}];')
            edges = tuple((bit, bit+1) for bit in range(n_qubits-1))
            if ring and n_qubits > 2:
                edges += ((n_qubits-1, 0),)
            for control, target in edges:
                lines.extend((f'ry(pi/2) q[{target}];',
                              f'cx q[{control}],q[{target}];',
                              f'ry(-pi/2) q[{target}];'))
    return _measurement_qasm(lines, n_qubits, basis, measure)
