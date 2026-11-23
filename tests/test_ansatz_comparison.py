"""Independent mathematics and exported-circuit checks for the real experiment."""
from fractions import Fraction as F
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
import sympy as sp

from experiments.grvq_ansatz_comparison.ansatz import (
    ExactSparseOperator, ExactState, GRVQProductAnsatz, mix_hypercube,
    qasm_product_mixer, qasm_source_ry_cz, source_ry_cz_state,
)
from experiments.grvq_ansatz_comparison.spectral import load_and_verify_polynomials


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def polynomial_data():
    completed = subprocess.run(
        ["node", str(ROOT / "experiments/grvq_ansatz_comparison/compile_polynomials.cjs")],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def test_all_source_polynomial_outputs_have_independent_exact_references(polynomial_data):
    data, report = load_and_verify_polynomials(polynomial_data)
    assert report["all_equal"]
    assert all(F(value) > 0 for value in report["mass_leading_minors"])
    assert {method["id"] for method in data["methods"]} == set(range(1,30))
    assert all(method["calls"] > 0 for method in data["methods"])


@pytest.mark.parametrize("power", [2,4])
def test_full_product_at_sixteen_distinct_source_points(polynomial_data, power):
    indices = tuple(range(1,65,4))
    points = tuple(F(polynomial_data["shape_points"][i]) for i in indices)
    samples = tuple(tuple(F(x) for x in polynomial_data["shape_samples"][i]) for i in indices)
    nodes = tuple(F(x) for x in polynomial_data["nodes"])
    assert len(points) == len(set(points)) == 16
    assert all(all(value != 0 for value in row) for row in samples)
    reference_shapes = tuple(tuple(math.prod((x-nodes[k])/(nodes[j]-nodes[k])
                                              for k in range(4) if k != j)
                                   for j in range(4)) for x in points)
    assert samples == reference_shapes
    assert all(sum(row) == 1 for row in samples)
    alphas = (F(1,7),F(2,9),F(3,11),F(4,13))
    widths = (F(1,2),F(1),F(3,2),F(2))
    carrier = tuple(1+x*x for x in points)
    model = GRVQProductAnsatz(samples,points,carrier,widths)
    state = model.evaluate(alphas,power=power)
    reference = tuple(c * math.prod(1+a*s for a,s in zip(alphas,row)) /
                       math.prod(1+(r/w)**power for w in widths)
                       for row,r,c in zip(reference_shapes,points,carrier))
    assert state.to_fractions() == reference
    for mode in range(4):
        changed = tuple(F(0) if j==mode else a for j,a in enumerate(alphas))
        assert model.evaluate(changed,power=power).fidelity(state) < 1
    assert model.evaluate(alphas,power=power).to_fractions() == reference


def test_exact_rayleigh_and_variance_match_independent_matrix_algebra():
    values = (F(2,3),F(-3,7),F(5,11),F(7,13))
    matrix = sp.Matrix([[2,sp.Rational(1,3),0,-1],
                        [sp.Rational(1,3),-1,sp.Rational(2,5),0],
                        [0,sp.Rational(2,5),3,sp.Rational(1,7)],
                        [-1,0,sp.Rational(1,7),4]])
    rows = tuple(tuple((j,F(matrix[i,j])) for j in range(4) if matrix[i,j]) for i in range(4))
    operator = ExactSparseOperator.from_rows(rows)
    state = ExactState.from_fractions(values)
    column = sp.Matrix(values)
    norm = (column.T*column)[0]
    energy = (column.T*matrix*column)[0]/norm
    variance = (column.T*matrix*matrix*column)[0]/norm-energy**2
    assert state.expectation(operator) == F(energy)
    assert state.variance(operator) == F(variance)
    scaled = ExactState.from_fractions(tuple(F(-11,17)*x for x in values))
    assert scaled.expectation(operator) == state.expectation(operator)
    assert scaled.variance(operator) == state.variance(operator)
    assert scaled.fidelity(state) == 1


@pytest.mark.parametrize("n_qubits", [1,2,4,6,8])
def test_cayley_gate_matches_independent_rational_matrix(n_qubits):
    values = tuple(F(3*i-5,11) for i in range(1 << n_qubits))
    state = ExactState.from_fractions(values)
    t = F(2,7)
    left, right = 0, state.size-1
    denominator = 1+t*t
    c, s = (1-t*t)/denominator, 2*t/denominator
    reference = list(values)
    reference[left] = c*values[left]-s*values[right]
    reference[right] = s*values[left]+c*values[right]
    result = state.apply_cayley((left,right),t)
    assert result.to_fractions() == tuple(reference)
    assert result.norm_squared == state.norm_squared
    assert result.apply_cayley((left,right),-t).to_fractions() == values


def _execute_qasm(qasm, n_qubits):
    state = np.zeros(1 << n_qubits,dtype=complex)
    state[0] = 1
    for line in qasm.splitlines():
        rotation = re.fullmatch(r"(ry|rz)\(([^)]+)\) q\[(\d+)\];",line)
        if rotation:
            gate, encoded, qubit = rotation.groups()
            angle = math.pi/2 if encoded == "pi/2" else -math.pi/2 if encoded == "-pi/2" else float(encoded)
            bit = 1 << int(qubit)
            if gate == "ry":
                c,s = math.cos(angle/2),math.sin(angle/2)
                for left in range(len(state)):
                    if not left & bit:
                        right = left | bit
                        a,b = state[left],state[right]
                        state[left],state[right] = c*a-s*b,s*a+c*b
            else:
                for index in range(len(state)):
                    state[index] *= np.exp((1j if index & bit else -1j)*angle/2)
            continue
        controlled = re.fullmatch(r"cx q\[(\d+)\],q\[(\d+)\];",line)
        if controlled:
            control,target = (1 << int(x) for x in controlled.groups())
            for left in range(len(state)):
                if left & control and not left & target:
                    right = left | target
                    state[left],state[right] = state[right],state[left]
            continue
        assert line.startswith(("OPENQASM ","include ","qreg ","creg ","measure ")), line
    return state


@pytest.mark.parametrize("n_qubits", [1,2,4,6,8])
@pytest.mark.parametrize("control_mode", ["parity","neighbor"])
def test_product_mixer_export_is_the_same_signed_state(n_qubits,control_mode):
    state = ExactState(tuple(3*i%17-8 for i in range(1 << n_qubits)))
    parameters = tuple(F(i+1,13) for i in range(n_qubits))
    ordering = tuple(reversed(range(n_qubits)))
    result = mix_hypercube(state,parameters,ordering=ordering,conditional_signs=True,control_mode=control_mode)
    assert result.norm_squared == state.norm_squared
    qasm = qasm_product_mixer(state,parameters,ordering=ordering,conditional_signs=True,control_mode=control_mode,measure=False)
    actual = _execute_qasm(qasm,n_qubits)
    reference = np.array([float(x) for x in result.to_fractions()])
    reference /= np.linalg.norm(reference)
    assert np.max(np.abs(actual-reference)) < 1e-13
    for basis in ("X"*n_qubits,"Y"*n_qubits,("XYZ"*n_qubits)[:n_qubits]):
        qasm = qasm_product_mixer(state,parameters,ordering=ordering,conditional_signs=True,control_mode=control_mode,basis=basis,measure=False)
        observed = np.abs(_execute_qasm(qasm,n_qubits))**2
        expected = np.array([float(x) for x in result.measurement_probabilities(basis)])
        assert np.max(np.abs(observed-expected)) < 1e-13


@pytest.mark.parametrize("n_qubits", [2,4,6,8])
def test_source_baseline_export_retains_the_actual_layer_topology(n_qubits):
    parameters = tuple(F(2*i-5,17) for i in range(2*n_qubits))
    state = source_ry_cz_state(n_qubits,parameters,layers=2,ring=True)
    qasm = qasm_source_ry_cz(n_qubits,parameters,layers=2,ring=True,measure=False)
    actual = _execute_qasm(qasm,n_qubits)
    reference = np.array([float(x) for x in state.to_fractions()])
    reference /= np.linalg.norm(reference)
    assert np.max(np.abs(actual-reference)) < 1e-13


def test_exact_measurement_distributions_include_phase_sensitive_interference():
    bell = ExactState((1,0,0,1))
    assert bell.measurement_probabilities("XX") == (F(1,2),F(0),F(0),F(1,2))
    assert bell.measurement_probabilities("YY") == (F(0),F(1,2),F(1,2),F(0))
    assert sum(bell.measurement_probabilities("XY")) == 1


def test_neighbor_controls_produce_a_noncommuting_refinement_without_changing_parity():
    state = ExactState(tuple(range(1,9)))
    parameters = (F(1,3),F(1,5),F(1,7))
    forward = mix_hypercube(state,parameters,conditional_signs=True,control_mode="neighbor")
    reverse = mix_hypercube(state,parameters,conditional_signs=True,control_mode="neighbor",ordering=(2,1,0))
    assert forward.fidelity(reverse) < 1
    parity = mix_hypercube(state,parameters,conditional_signs=True)
    parity_reverse = mix_hypercube(state,parameters,conditional_signs=True,ordering=(2,1,0))
    assert parity.to_fractions() == parity_reverse.to_fractions()
    assert forward.norm_squared == reverse.norm_squared == state.norm_squared
