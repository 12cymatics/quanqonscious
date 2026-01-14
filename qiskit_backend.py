"""Qiskit integration utilities for hybrid Vedic sutra simulations.

This module provides a gateway to IBM's Qiskit framework beyond trivial
examples, allowing the QuanQonscious platform to orchestrate multi-qubit
experiments that mirror the concurrent execution of the 29 Vedic sutras.

The primary entry point is :func:`execute_ghz`, which constructs a GHZ
(Greenberger–Horne–Zeilinger) state across an arbitrary number of qubits.
In the default configuration we entangle 29 qubits, mapping one-to-one with
our sutra set to emphasize simultaneous coherence.  The circuit is fully
transpiled and executed using the Qiskit Aer simulator, returning
measurement counts suitable for downstream classical-quantum fusion.
"""

from __future__ import annotations

from typing import Dict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.result import Counts


def execute_ghz(num_qubits: int = 29, shots: int = 1024) -> Counts:
    """Create and measure a GHZ state on ``num_qubits`` qubits.

    The procedure follows established Qiskit documentation for GHZ-state
    preparation:

    1. Apply a Hadamard gate to the first qubit to create superposition.
    2. Cascade ``CX`` gates from the first qubit to all others to generate a
       maximally entangled GHZ state.
    3. Measure each qubit in the computational basis.

    Parameters
    ----------
    num_qubits:
        Number of qubits to entangle. Defaults to 29 to align with the
        quantity of Vedic sutras.
    shots:
        Number of repetitions for circuit execution on the Aer simulator.

    Returns
    -------
    Counts
        Mapping of bitstrings to observed frequencies after measurement.
    """

    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")

    circuit = QuantumCircuit(num_qubits, num_qubits, name="ghz")
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    circuit.measure(range(num_qubits), range(num_qubits))

    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts: Counts = result.get_counts(circuit)
    return counts


if __name__ == "__main__":  # pragma: no cover - manual execution
    print(execute_ghz())
