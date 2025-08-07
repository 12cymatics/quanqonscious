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

from qiskit import IBMQ, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.result import Counts


# User-provided IBM Quantum API token enabling remote execution on IBM's
# cloud backends.  The key is embedded directly per user request to allow the
# framework to authenticate without external configuration files.
IBMQ_API_KEY = "ApiKey-4781ceaa-523c-4404-bc6d-a991cc1d847d"


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


def execute_ghz_ibmq(
    num_qubits: int = 29,
    shots: int = 1024,
    backend_name: str = "ibmq_qasm_simulator",
) -> Counts:
    """Execute a GHZ state on an IBM Quantum backend authenticated by API key.

    This routine mirrors :func:`execute_ghz` but targets IBM's cloud
    infrastructure.  It programmatically enables the user's account using the
    embedded API key, fetches the specified backend, and runs the fully
    transpiled circuit.

    Parameters
    ----------
    num_qubits:
        Number of qubits to entangle. Defaults to 29, aligning with the Vedic
        sutras.
    shots:
        Repetition count for backend execution.
    backend_name:
        Name of the IBM Quantum backend to use. ``"ibmq_qasm_simulator"`` is
        selected by default for broad availability.

    Returns
    -------
    Counts
        Mapping of bitstrings to observed frequencies after measurement.
    """

    if num_qubits < 1:
        raise ValueError("num_qubits must be positive")

    if not IBMQ.active_account():
        IBMQ.enable_account(IBMQ_API_KEY)

    provider = IBMQ.get_provider(hub="ibm-q")
    backend = provider.get_backend(backend_name)

    circuit = QuantumCircuit(num_qubits, num_qubits, name="ghz")
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    circuit.measure(range(num_qubits), range(num_qubits))

    compiled = transpile(circuit, backend)
    job = backend.run(compiled, shots=shots)
    result = job.result()
    counts: Counts = result.get_counts(circuit)
    return counts


if __name__ == "__main__":  # pragma: no cover - manual execution
    print("Aer simulator:", execute_ghz())
    print("IBM Quantum:", execute_ghz_ibmq())
