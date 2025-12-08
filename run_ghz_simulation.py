#!/usr/bin/env python
"""Run a GHZ quantum entanglement simulation using Qiskit Aer"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def execute_ghz(num_qubits=29, shots=1024):
    """Create and measure a GHZ (Greenberger-Horne-Zeilinger) state.

    This creates a maximally entangled state across multiple qubits:
    1. Apply Hadamard gate to first qubit (creates superposition)
    2. Apply cascading CX gates to entangle all qubits
    3. Measure all qubits

    Args:
        num_qubits: Number of qubits to entangle (default 29, matching Vedic sutras)
        shots: Number of measurement repetitions

    Returns:
        Measurement counts dictionary
    """
    print(f"Creating GHZ circuit with {num_qubits} qubits...")

    # Build quantum circuit
    circuit = QuantumCircuit(num_qubits, num_qubits, name="ghz")
    circuit.h(0)  # Hadamard on first qubit

    # Cascade CX gates to entangle all qubits
    for i in range(1, num_qubits):
        circuit.cx(0, i)

    # Measure all qubits
    circuit.measure(range(num_qubits), range(num_qubits))

    print(f"Circuit depth: {circuit.depth()}")
    print(f"Circuit operations: {circuit.count_ops()}")

    # Execute on Aer simulator
    print(f"\nRunning simulation with {shots} shots...")
    simulator = AerSimulator()
    compiled = transpile(circuit, simulator)
    job = simulator.run(compiled, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)

    return counts


def analyze_ghz_results(counts, num_qubits):
    """Analyze GHZ measurement results"""
    print(f"\n{'='*60}")
    print(f"GHZ State Results ({num_qubits} qubits)")
    print(f"{'='*60}")

    total_shots = sum(counts.values())

    # Expected outcomes for ideal GHZ state: all 0s or all 1s
    all_zeros = '0' * num_qubits
    all_ones = '1' * num_qubits

    zeros_count = counts.get(all_zeros, 0)
    ones_count = counts.get(all_ones, 0)
    other_count = total_shots - zeros_count - ones_count

    print(f"\nTotal measurements: {total_shots}")
    print(f"\nIdeal GHZ outcomes:")
    print(f"  |{'0'*num_qubits}⟩: {zeros_count:4d} ({zeros_count/total_shots*100:6.2f}%)")
    print(f"  |{'1'*num_qubits}⟩: {ones_count:4d} ({ones_count/total_shots*100:6.2f}%)")
    print(f"  Other states:  {other_count:4d} ({other_count/total_shots*100:6.2f}%)")

    fidelity = (zeros_count + ones_count) / total_shots
    print(f"\nGHZ state fidelity: {fidelity*100:.2f}%")

    # Show top 5 measurement outcomes
    print(f"\nTop 5 measured states:")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for state, count in sorted_counts[:5]:
        print(f"  |{state}⟩: {count:4d} ({count/total_shots*100:6.2f}%)")

    return fidelity


if __name__ == "__main__":
    # Run with different qubit counts to demonstrate scalability

    print("="*60)
    print("GHZ Quantum Entanglement Simulation")
    print("="*60)

    for num_qubits in [5, 10, 29]:
        print(f"\n\n{'#'*60}")
        print(f"# Experiment: {num_qubits}-qubit GHZ state")
        print(f"{'#'*60}\n")

        counts = execute_ghz(num_qubits=num_qubits, shots=2048)
        fidelity = analyze_ghz_results(counts, num_qubits)

    print(f"\n\n{'='*60}")
    print("✓ All GHZ simulations completed successfully!")
    print(f"{'='*60}")
