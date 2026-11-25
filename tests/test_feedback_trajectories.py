"""Independent circuit and source feedback checks for the recovered mathematics.

This test file was reconstructed after loss of the original execution host.
It executes the recovered quantum mathematics; it does not claim byte identity
with the earlier test file or use historical test results as current evidence.
"""
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def node(script, value=None):
    completed = subprocess.run(["node", "-e", script], cwd=ROOT,
                               input="" if value is None else json.dumps(value),
                               text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)


def quantum_records(parameters):
    return node(r"""
const H=require('./experiments/grvq_ansatz_comparison/hybrid_quantum.cjs');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(input.map(p=>{
 const m=H.moments(H.circuitState(p)),T=H.rational.text;
 return {amplitudes:m.state.map(T),norm:T(m.norm),z:m.z.map(T),zz:m.zz.map(T)};
})));
""", parameters)


def test_exact_norm_and_final_entangler_chain():
    zero = ["0"] * 12
    anchor = zero.copy()
    anchor[0] = "1"
    records = quantum_records([zero, anchor])
    assert records[0]["amplitudes"] == ["1"] + ["0"] * 15
    # Three complete CNOT chains send the first Ry flip to basis index 3.
    assert records[1]["amplitudes"] == ["0"] * 3 + ["1"] + ["0"] * 12
    assert all(Fraction(record["norm"]) == 1 for record in records)


def independent_dense_state(parameters):
    state = np.eye(16, dtype=float)[:, 0]
    for layer in range(3):
        for qubit in range(4):
            t = Fraction(parameters[4 * layer + qubit])
            c = float((1 - t*t) / (1 + t*t))
            s = float(2*t / (1 + t*t))
            rotation = np.array([[c, -s], [s, c]], dtype=float)
            operator = np.array([[1.0]])
            # q0 is the least significant bit, so the tensor order is q3..q0.
            for tensor_qubit in range(3, -1, -1):
                operator = np.kron(operator, rotation if tensor_qubit == qubit else np.eye(2))
            state = operator @ state
        for control in range(3):
            gate = np.zeros((16, 16), dtype=float)
            for before in range(16):
                after = before ^ (1 << (control + 1)) if before & (1 << control) else before
                gate[after, before] = 1
            state = gate @ state
    return state


def test_circuit_against_independent_dense_gates():
    rng = random.Random(101)
    parameters = [[str(Fraction(rng.randint(-7, 7), 8)) for _ in range(12)] for _ in range(8)]
    records = quantum_records(parameters)
    for p, record in zip(parameters, records):
        exact = tuple(map(Fraction, record["amplitudes"]))
        assert sum(a*a for a in exact) == 1
        np.testing.assert_allclose(np.array(list(map(float, exact))), independent_dense_state(p), atol=3e-15, rtol=0)
        probabilities = tuple(a*a for a in exact)
        for q in range(4):
            expected = sum(probability * (-1 if i & (1 << q) else 1)
                           for i, probability in enumerate(probabilities))
            assert Fraction(record["z"][q]) == expected
        for q in range(3):
            expected = sum(probability * (-1 if ((i >> q) ^ (i >> (q + 1))) & 1 else 1)
                           for i, probability in enumerate(probabilities))
            assert Fraction(record["zz"][q]) == expected


def test_actual_source_feedback_projection_matches_fraction_identity():
    field = [Fraction(i*i - 3*i + 7, 13) for i in range(16)]
    deltas = [Fraction(-3, 7000), Fraction(2, 9000), Fraction(-1, 11000), Fraction(4, 13000)]
    result = node(r"""
const E=require('./experiments/grvq_ansatz_comparison/latest_hybrid_executor.cjs');
const K=E.loadKernel({arithmetic:'exact_dag'}),Q=K.Q;
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const before=input.field.map(Q.parse),deltas=input.deltas.map(Q.parse);
let after=before;
for(let q=0;q<4;q++)after=K.injectLambda(after,q,deltas[q]);
process.stdout.write(JSON.stringify({before_lambda:K.inverseL(before).map(Q.toString),
 after_lambda:K.inverseL(after).map(Q.toString),field:after.map(Q.toString)}));
""", {"field": list(map(str, field)), "deltas": list(map(str, deltas))})
    expected = [value + sum((1 if i & (1 << q) else -1) * deltas[q] for q in range(4))
                for i, value in enumerate(field)]
    assert tuple(map(Fraction, result["field"])) == tuple(expected)
    before_lambda = tuple(sum((1 if i & (1 << q) else -1) * value for i, value in enumerate(field)) / 16
                          for q in range(4))
    assert tuple(map(Fraction, result["before_lambda"])) == before_lambda
    assert tuple(map(Fraction, result["after_lambda"])) == tuple(a+b for a, b in zip(before_lambda, deltas))
