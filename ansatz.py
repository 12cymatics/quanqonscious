# QuanQonscious/ansatz.py

import math
import numpy as np
import cirq  # Cirq for quantum circuits (will be used if CUDA-Q is unavailable)
try:
    import cudaq  # NVIDIA CUDA Quantum (for GPU acceleration)
except ImportError:
    cudaq = None

# Use relative imports so the module works regardless of the package name on
# disk.
from . import core_engine, maya_cipher

class GRVQAnsatz:
    """
    Construct and manage the GRVQ variational quantum ansatz circuit.
    
    This ansatz integrates R^4 singularity suppression and Vedic sutra-based 
    transformations into a quantum circuit. It supports multiple backends:
    - Cirq (CPU simulation or quantum hardware)
    - NVIDIA CUDA-Q (GPU-accelerated simulation), if available.
    
    Attributes:
        n_qubits (int): Number of qubits in the ansatz circuit.
        use_cuda (bool): Whether to use CUDA-Q backend if available.
    """
    def __init__(self, n_qubits: int, use_cuda: bool = None):
        """
        Initialize the GRVQAnsatz.
        
        Args:
            n_qubits: The number of qubits for the quantum circuit.
            use_cuda: If True, forces use of CUDA-Q backend if available. If False, use Cirq.
                      If None, auto-select (use CUDA-Q if available, else Cirq).
        """
        self.n_qubits = n_qubits
        # Determine backend
        if use_cuda is None:
            self.backend = "cudaq" if (cudaq is not None) else "cirq"
        else:
            if use_cuda and cudaq is None:
                # Warn if requested CUDA but not available
                print("[GRVQAnsatz] Warning: CUDA-Q requested but not available, falling back to Cirq.")
            self.backend = "cudaq" if (use_cuda and cudaq is not None) else "cirq"
        # Prepare Cirq qubits (if Cirq will be used)
        self._cirq_qubits = [cirq.LineQubit(i) for i in range(n_qubits)] if self.backend == "cirq" else None

    def _suppress_singularity(self, params: np.ndarray) -> np.ndarray:
        """
        Apply R^4 singularity suppression to parameter values.
        
        This function modifies ansatz parameters to avoid singularities by damping extreme values.
        The suppression uses a fourth-order decay: param_new = param / (1 + (param/k)^4) for large param.
        Args:
            params: Array of ansatz parameters.
        Returns:
            np.ndarray of adjusted parameters.
        """
        # Define a damping scale k (threshold where values are considered "large")
        k = 1.0
        suppressed = np.array([p / (1.0 + (p/k)**4) for p in params], dtype=float)
        return suppressed

    def build_circuit(self, params: np.ndarray, maya_phase_key: int = None) -> object:
        """
        Build the quantum circuit for the GRVQ ansatz using the given parameters.
        
        Optionally, a Maya Sutra phase modulation can be embedded by providing a key.
        This will apply an encrypted phase rotation to one qubit, adding robustness and 
        cryptographic masking to the ansatz phase.
        
        Args:
            params: Array of variational parameters for the ansatz gates.
            maya_phase_key: Optional integer key for Maya encryption to modulate a phase.
        Returns:
            A quantum circuit object. 
            - If backend is Cirq, returns a cirq.Circuit.
            - If backend is CUDA-Q, returns a cudaq.Kernel (quantum program).
        """
        params = np.array(params, dtype=float)
        # 1. Suppress singularities in parameters
        params = self._suppress_singularity(params)
        # 2. (Optional) Refine parameters using Vedic sutras for better initial guess
        try:
            params = core_engine.apply_main_sutras(params)
        except Exception:
            # If core_engine is not available or fails, proceed without additional refinement
            pass

        if self.backend == "cirq":
            # Build circuit with Cirq
            circuit = cirq.Circuit()
            # Example ansatz: apply Hadamard and parameterized rotations and entanglers
            for q in self._cirq_qubits:
                circuit.append(cirq.H.on(q))
            # Layer of parameterized rotations (e.g., RY rotations using params)
            for q, theta in zip(self._cirq_qubits, params):
                circuit.append(cirq.ry(theta).on(q))
            # Entangle neighboring qubits (e.g., using CNOT or CZ gates in a chain)
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CZ.on(self._cirq_qubits[i], self._cirq_qubits[i+1]))
            # (Optional) Maya phase encryption: apply an extra phase rotation based on cipher text
            if maya_phase_key is not None:
                cipher = maya_cipher.MayaCipher(key=maya_phase_key)
                # Encrypt a fixed plaintext (e.g., 0) to generate a pseudo-random phase
                cipher_text = cipher.encrypt_block(0)  # encrypt the number 0
                phase_angle = (cipher_text % 256) * (2 * math.pi / 256.0)  # map to [0, 2π)
                # Apply a phase rotation (Z rotation) on the first qubit using the encrypted angle
                circuit.append(cirq.rz(phase_angle).on(self._cirq_qubits[0]))
            return circuit

        else:  # self.backend == "cudaq"
            # Build quantum circuit using NVIDIA CUDA Quantum
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(self.n_qubits)
            # Apply initial Hadamard gates on all qubits
            for q in range(self.n_qubits):
                kernel.h(qubits[q])
            # Apply parameterized rotations (use RY rotations as example)
            for q, theta in enumerate(params):
                kernel.ry(theta, qubits[q])
            # Apply chain of CZ entangling gates
            for q in range(self.n_qubits - 1):
                kernel.cz(qubits[q], qubits[q+1])
            # Optional Maya phase modulation as an extra Z rotation
            if maya_phase_key is not None:
                cipher = maya_cipher.MayaCipher(key=maya_phase_key)
                cipher_text = cipher.encrypt_block(0)
                phase_angle = (cipher_text % 256) * (2 * math.pi / 256.0)
                kernel.rz(phase_angle, qubits[0])
            return kernel

    def run(self, params: np.ndarray, shots: int = 0) -> object:
        """
        Execute the ansatz circuit with given parameters and return results.
        
        If using Cirq, this will simulate the circuit statevector (if shots=0) or sample measurements.
        If using CUDA-Q, it will sample the circuit using the GPU simulator.
        
        Args:
            params: Ansatz parameters.
            shots: Number of measurement shots. If 0, return the statevector for Cirq backend.
        Returns:
            Simulation result (state vector or measurement result object).
        """
        circuit = self.build_circuit(params)
        if self.backend == "cirq":
            simulator = cirq.Simulator()
            if shots and shots > 0:
                result = simulator.run(circuit, repetitions=shots)
            else:
                result = simulator.simulate(circuit).final_state_vector
            return result
        else:
            # Using CUDA-Q backend
            if shots and shots > 0:
                # sample (measure) the kernel a given number of times
                return cudaq.sample(circuit, shots)
            else:
                # For statevector, CUDA-Q might have a state retrieval (if supported)
                try:
                    return cudaq.simulate(circuit)
                except Exception:
                    # If statevector simulation not directly available, default to sampling 1 shot for simplicity
                    return cudaq.sample(circuit, shots=1)
