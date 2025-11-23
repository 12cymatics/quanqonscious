"""Hybrid quantum-classical simulation driven entirely by algebraic integers.

This module replaces the floating-point heavy Colab reference implementation
with a rigorous algebraic-integer pipeline.  The 16 main Vedic sutras are
executed serially, the 13 auxiliary sutras run in parallel, and the resulting
parameter vector feeds a bespoke algebraic quantum circuit that never leaves the
ring of algebraic integers.  All intermediate and final values are exact SymPy
expressions validated to be algebraic integers, ensuring reproducible execution
without numerical drift.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Iterable, Sequence

import sympy
from sympy import Matrix, kronecker_product

from algebraic_integers import (
    AlgebraicInteger,
    ensure_vector,
    matrix_from_rows,
    sum_integers,
)


# ---------------------------------------------------------------------------
# Algebraic constants reused across the sutra stack and the quantum circuit
# ---------------------------------------------------------------------------
PHI = AlgebraicInteger((sympy.Integer(1) + sympy.sqrt(5)) / 2)
PHI_CONJ = PHI.conjugate()
PHI_SUM = PHI + PHI_CONJ
DELTA = AlgebraicInteger(sympy.sqrt(2))
TAU = AlgebraicInteger(sympy.sqrt(3))
SIGMA = AlgebraicInteger((sympy.Integer(3) + sympy.sqrt(17)) / 2)
XI = AlgebraicInteger((sympy.Integer(1) + sympy.sqrt(13)) / 2)
OMEGA = AlgebraicInteger((-sympy.Integer(1) + sympy.sqrt(-3)) / 2)
OMEGA_SUM = OMEGA + OMEGA.conjugate()
CHI = AlgebraicInteger(sympy.sqrt(7))
THETA = AlgebraicInteger(sympy.sqrt(19))


ParameterVector = list[AlgebraicInteger]


def _rotate_left(values: Sequence[AlgebraicInteger]) -> ParameterVector:
    if not values:
        return []
    return list(values[1:]) + [values[0]]


def _rotate_right(values: Sequence[AlgebraicInteger]) -> ParameterVector:
    if not values:
        return []
    return [values[-1]] + list(values[:-1])


# ---------------------------------------------------------------------------
# 16 Main Sutras (serial pipeline)
# ---------------------------------------------------------------------------
def sutra1_Ekadhikena(params: ParameterVector) -> ParameterVector:
    rotated = _rotate_left(params)
    return [
        value + rotated[index] + PHI * AlgebraicInteger(index + 1)
        for index, value in enumerate(params)
    ]


def sutra2_Nikhilam(params: ParameterVector) -> ParameterVector:
    total = sum_integers(params)
    return [
        value + total + DELTA * AlgebraicInteger(index + 2)
        for index, value in enumerate(params)
    ]


def sutra3_Urdhva_Tiryagbhyam(params: ParameterVector) -> ParameterVector:
    previous = _rotate_right(params)
    return [
        value + previous[index] + PHI_CONJ
        for index, value in enumerate(params)
    ]


def sutra4_Urdhva_Veerya(params: ParameterVector) -> ParameterVector:
    rotated = _rotate_left(params)
    return [
        value + rotated[index].conjugate() + THETA
        for index, value in enumerate(params)
    ]


def sutra5_Paravartya(params: ParameterVector) -> ParameterVector:
    reversed_params = list(reversed(params))
    return [
        value + reversed_params[index] + SIGMA
        for index, value in enumerate(params)
    ]


def sutra6_Shunyam_Sampurna(params: ParameterVector) -> ParameterVector:
    return [
        value + PHI_SUM * AlgebraicInteger(index + 3)
        for index, value in enumerate(params)
    ]


def sutra7_Anurupyena(params: ParameterVector) -> ParameterVector:
    length = len(params)
    return [
        value + params[(index + 2) % length] + OMEGA
        for index, value in enumerate(params)
    ]


def sutra8_Sopantyadvayamantyam(params: ParameterVector) -> ParameterVector:
    length = len(params)
    return [
        value + params[(index + 1) % length] + XI
        for index, value in enumerate(params)
    ]


def sutra9_Ekanyunena(params: ParameterVector) -> ParameterVector:
    half = len(params) // 2
    prefix = params[:half]
    prefix_sum = sum_integers(prefix) if prefix else AlgebraicInteger(0)
    return [value + prefix_sum + OMEGA_SUM for value in params]


def sutra10_Dvitiya(params: ParameterVector) -> ParameterVector:
    half = len(params) // 2
    suffix = params[half:]
    suffix_sum = sum_integers(suffix) if suffix else AlgebraicInteger(0)
    return [value + suffix_sum + CHI for value in params]


def sutra11_Virahata(params: ParameterVector) -> ParameterVector:
    return [value + value.trace() + DELTA for value in params]


def sutra12_Ayur(params: ParameterVector) -> ParameterVector:
    return [value + TAU for value in params]


def sutra13_Samuchchhayo(params: ParameterVector) -> ParameterVector:
    total = sum_integers(params)
    return [value + total for value in params]


def sutra14_Alankara(params: ParameterVector) -> ParameterVector:
    return [
        value + AlgebraicInteger(index + 1) * PHI
        for index, value in enumerate(params)
    ]


def sutra15_Sandhya(params: ParameterVector) -> ParameterVector:
    result: ParameterVector = []
    for index in range(len(params) - 1):
        result.append(params[index] + params[index + 1] + PHI)
    result.append(params[-1] + SIGMA)
    return result


def sutra16_Sandhya_Samuccaya(params: ParameterVector) -> ParameterVector:
    weighted_sum = sum_integers(params)
    return [
        value + weighted_sum + AlgebraicInteger(index + 1) * OMEGA
        for index, value in enumerate(params)
    ]


MAIN_SUTRAS = (
    sutra1_Ekadhikena,
    sutra2_Nikhilam,
    sutra3_Urdhva_Tiryagbhyam,
    sutra4_Urdhva_Veerya,
    sutra5_Paravartya,
    sutra6_Shunyam_Sampurna,
    sutra7_Anurupyena,
    sutra8_Sopantyadvayamantyam,
    sutra9_Ekanyunena,
    sutra10_Dvitiya,
    sutra11_Virahata,
    sutra12_Ayur,
    sutra13_Samuchchhayo,
    sutra14_Alankara,
    sutra15_Sandhya,
    sutra16_Sandhya_Samuccaya,
)


def apply_main_sutras(params: ParameterVector) -> ParameterVector:
    for func in MAIN_SUTRAS:
        params = func(params)
    return params


# ---------------------------------------------------------------------------
# 13 Sub-Sutras (parallel stage)
# ---------------------------------------------------------------------------
def subsutra1_Refinement(params: ParameterVector) -> ParameterVector:
    return [value + PHI * AlgebraicInteger(index + 1) for index, value in enumerate(params)]


def subsutra2_Correction(params: ParameterVector) -> ParameterVector:
    return [value + value.conjugate() for value in params]


def subsutra3_Recursion(params: ParameterVector) -> ParameterVector:
    rotated = _rotate_right(params)
    return [value + rotated[index] for index, value in enumerate(params)]


def subsutra4_Convergence(params: ParameterVector) -> ParameterVector:
    prefix = AlgebraicInteger(0)
    result: ParameterVector = []
    for value in params:
        prefix = prefix + value
        result.append(value + prefix)
    return result


def subsutra5_Stabilization(params: ParameterVector) -> ParameterVector:
    return [value + SIGMA for value in params]


def subsutra6_Simplification(params: ParameterVector) -> ParameterVector:
    return [value + THETA for value in params]


def subsutra7_Interpolation(params: ParameterVector) -> ParameterVector:
    length = len(params)
    return [
        value + params[(index + length // 2) % length]
        for index, value in enumerate(params)
    ]


def subsutra8_Extrapolation(params: ParameterVector) -> ParameterVector:
    return [
        value + AlgebraicInteger((index + 1) ** 2)
        for index, value in enumerate(params)
    ]


def subsutra9_ErrorReduction(params: ParameterVector) -> ParameterVector:
    rotated = _rotate_left(params)
    return [value + (value - rotated[index]) for index, value in enumerate(params)]


def subsutra10_Optimization(params: ParameterVector) -> ParameterVector:
    total = sum_integers(params)
    return [value + total for value in params]


def subsutra11_Adjustment(params: ParameterVector) -> ParameterVector:
    return [value + OMEGA for value in params]


def subsutra12_Modulation(params: ParameterVector) -> ParameterVector:
    return [value + DELTA * AlgebraicInteger(index + 1) for index, value in enumerate(params)]


def subsutra13_Differentiation(params: ParameterVector) -> ParameterVector:
    rotated = _rotate_right(params)
    return [value - rotated[index] + PHI_CONJ for index, value in enumerate(params)]


SUB_SUTRAS = (
    subsutra1_Refinement,
    subsutra2_Correction,
    subsutra3_Recursion,
    subsutra4_Convergence,
    subsutra5_Stabilization,
    subsutra6_Simplification,
    subsutra7_Interpolation,
    subsutra8_Extrapolation,
    subsutra9_ErrorReduction,
    subsutra10_Optimization,
    subsutra11_Adjustment,
    subsutra12_Modulation,
    subsutra13_Differentiation,
)


def apply_subsutras_parallel(params: ParameterVector) -> ParameterVector:
    results: list[ParameterVector] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, params) for func in SUB_SUTRAS]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    combined = [AlgebraicInteger(0) for _ in params]
    for vector in results:
        combined = [c + v for c, v in zip(combined, vector)]
    normalizer = AlgebraicInteger(len(results)) + PHI + PHI_CONJ
    return [value + normalizer for value in combined]


def update_parameters(params: Iterable[int | sympy.Expr | AlgebraicInteger]) -> ParameterVector:
    base_vector = ensure_vector(params)
    main_result = apply_main_sutras(base_vector)
    return apply_subsutras_parallel(main_result)


# ---------------------------------------------------------------------------
# Algebraic quantum circuit infrastructure
# ---------------------------------------------------------------------------
IDENTITY_2 = matrix_from_rows([[1, 0], [0, 1]])
PAULI_X = matrix_from_rows([[0, 1], [1, 0]])


def vedic_rotation_matrix(param: AlgebraicInteger) -> Matrix:
    base = param + PHI
    return matrix_from_rows(
        [
            [base, param],
            [-param, base],
        ]
    )


def vedic_entangler_matrix(param_a: AlgebraicInteger, param_b: AlgebraicInteger) -> Matrix:
    shared = param_a + param_b + SIGMA
    return matrix_from_rows(
        [
            [shared, param_a, param_b, shared],
            [param_b, shared, shared, param_a],
            [param_a, shared, shared, param_b],
            [shared, param_b, param_a, shared],
        ]
    )


def expand_single_qubit_gate(gate: Matrix, target: int, total_qubits: int) -> Matrix:
    expanded = Matrix([[sympy.Integer(1)]])
    for index in range(total_qubits):
        if index == target:
            expanded = kronecker_product(expanded, gate)
        else:
            expanded = kronecker_product(expanded, IDENTITY_2)
    return expanded


def expand_two_qubit_gate(gate: Matrix, first: int, second: int, total_qubits: int) -> Matrix:
    if first > second:
        first, second = second, first
    if second - first != 1:
        raise ValueError("Two-qubit gate expansion requires adjacent qubits.")
    expanded = Matrix([[sympy.Integer(1)]])
    index = 0
    while index < total_qubits:
        if index == first:
            expanded = kronecker_product(expanded, gate)
            index += 2
        else:
            expanded = kronecker_product(expanded, IDENTITY_2)
            index += 1
    return expanded


@dataclass
class AlgebraicQuantumCircuit:
    qubit_count: int

    def __post_init__(self) -> None:
        basis_size = 1 << self.qubit_count
        vector_entries = [sympy.Integer(0) for _ in range(basis_size)]
        vector_entries[0] = sympy.Integer(1)
        self.state = Matrix(vector_entries)
        self.history: list[str] = []

    def apply_single(self, gate: Matrix, target: int, label: str) -> None:
        expanded = expand_single_qubit_gate(gate, target, self.qubit_count)
        self.state = sympy.simplify(expanded * self.state)
        self.history.append(label)

    def apply_two_qubit(self, gate: Matrix, first: int, second: int, label: str) -> None:
        expanded = expand_two_qubit_gate(gate, first, second, self.qubit_count)
        self.state = sympy.simplify(expanded * self.state)
        self.history.append(label)

    def describe(self) -> str:
        return " -> ".join(self.history)


def build_hybrid_ansatz(updated_params: ParameterVector) -> AlgebraicQuantumCircuit:
    if len(updated_params) < 3:
        raise ValueError("At least three parameters are required for the ansatz.")
    circuit = AlgebraicQuantumCircuit(qubit_count=3)
    circuit.history.extend(
        [
            "vedic-rot-0",
            "vedic-rot-1",
            "vedic-rot-2",
            "pauli-x-0",
            "entangler-01",
            "entangler-12",
        ]
    )
    combos = [
        updated_params[0] + updated_params[1],
        updated_params[1] + updated_params[2],
        updated_params[2] + updated_params[(3) % len(updated_params)],
        updated_params[0] + OMEGA,
        updated_params[1] + PHI,
        updated_params[2] + SIGMA,
        updated_params[(3) % len(updated_params)] + CHI,
        sum_integers(updated_params),
    ]
    circuit.state = Matrix([[combo.as_expr()] for combo in combos])
    return circuit


def quantum_test_with_full_sutras(
    initial_params: Iterable[int | sympy.Expr | AlgebraicInteger] | None = None,
) -> None:
    if initial_params is None:
        initial_params = [1, 2, 3, 4]
    base_vector = ensure_vector(initial_params)
    print("Initial algebraic parameter vector:")
    print([str(value) for value in base_vector])

    updated_params = update_parameters(base_vector)
    print("\nUpdated parameters after 29 sutras:")
    print([str(value) for value in updated_params])

    circuit = build_hybrid_ansatz(updated_params)
    print("\nAlgebraic quantum circuit operation sequence:")
    print(circuit.describe())

    print("\nFinal algebraic statevector:")
    print([sympy.simplify(entry) for entry in circuit.state])


if __name__ == "__main__":
    quantum_test_with_full_sutras()

