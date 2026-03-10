"""Qiskit integration utilities for hybrid Vedic sutra simulations.

This module provides a gateway to IBM's Qiskit framework beyond trivial
examples, allowing the QuanQonscious platform to orchestrate multi-qubit
experiments that mirror the concurrent execution of the 29 Vedic sutras.

When Qiskit is unavailable in the execution environment, the module falls
back to a deterministic GHZ-like count distribution so the rest of the
hybrid pipeline remains runnable.
"""

from __future__ import annotations

from typing import Dict

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.result import Counts
    QISKIT_AVAILABLE = True
except ModuleNotFoundError:
    QuantumCircuit = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    AerSimulator = None  # type: ignore[assignment]
    Counts = Dict[str, int]  # type: ignore[misc,assignment]
    QISKIT_AVAILABLE = False


def _fallback_ghz_counts(num_qubits: int, shots: int) -> Dict[str, int]:
    """Generate deterministic GHZ-style counts without Qiskit.

    The ideal GHZ state over ``num_qubits`` measures as all-zeros or all-ones.
    We split shots as evenly as possible to preserve that structure.
    """

    zeros = "0" * num_qubits
    ones = "1" * num_qubits
    high = shots // 2
    low = shots - high
    return {zeros: high, ones: low}


def execute_ghz(num_qubits: int = 29, shots: int = 1024) -> Counts:
    """Create and measure a GHZ state on ``num_qubits`` qubits.

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
    if shots < 1:
        raise ValueError("shots must be positive")

    if not QISKIT_AVAILABLE:
        return _fallback_ghz_counts(num_qubits, shots)

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
