import numpy as np
import cirq
import cudaq
import torch
import matplotlib.pyplot as plt
import scipy.linalg as la
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
import logging
import time
import sympy as sp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VedicSutras")

class SutraMode(Enum):
    """Enumeration of operation modes for Vedic sutras"""
    CLASSICAL = 0
    QUANTUM = 1
    HYBRID = 2
    MAYA_ILLUSION = 3
    SULBA = 4

@dataclass
class SutraContext:
    """Context for sutra execution with configuration parameters"""
    mode: SutraMode = SutraMode.CLASSICAL
    quantum_backend: Optional[Any] = None
    precision: int = 32  # Bit precision
    base: float = 10.0   # Default base for complement calculations
    epsilon: float = 1e-10  # Numerical stability factor
    max_iterations: int = 100  # For recursive applications
    use_gpu: bool = False  # GPU acceleration flag
    device: Any = None    # GPU device if applicable
    record_performance: bool = True  # Track execution metrics
    visualization: bool = False  # Generate visual representations
    parallel: bool = True  # Use parallel processing when available

class VedicSutras:
    """
    Comprehensive implementation of all 29 Vedic sutras (16 primary + 13 sub-sutras)
    with full mathematical logic, quantum integration, and inter-sutra interactions.
    """
    
    def __init__(self, context: Optional[SutraContext] = None):
        """
        Initialize the Vedic Sutras system with the specified context.
        
        Args:
            context: Configuration context for sutra execution
        """
        self.context = context if context else SutraContext()
        
        # Initialize GPU if requested
        if self.context.use_gpu and torch.cuda.is_available():
            self.context.device = torch.device("cuda")
            logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
        else:
            self.context.use_gpu = False
            self.context.device = torch.device("cpu")
            logger.info("Using CPU for computations")
            
        # Initialize quantum backend if in quantum or hybrid mode
        if self.context.mode in [SutraMode.QUANTUM, SutraMode.HYBRID]:
            if self.context.quantum_backend is None:
                # Default to CUDAQ simulator
                self.quantum_platform = cudaq.get_platform()
                logger.info(f"Using CUDAQ platform: {self.quantum_platform.name()}")
            else:
                self.quantum_platform = self.context.quantum_backend
        
        # Performance tracking
        self.performance_history = []
        self.sutra_interactions = {}
        
        logger.info(f"Initialized Vedic Sutras system in {self.context.mode.name} mode")
    
    def _record_performance(self, sutra_name: str, start_time: float, 
                           end_time: float, success: bool, data_size: int,
                           error: Optional[str] = None) -> None:
        """Record performance metrics for a sutra execution"""
        if not self.context.record_performance:
            return
            
        self.performance_history.append({
            'sutra': sutra_name,
            'execution_time': end_time - start_time,
            'success': success,
            'data_size': data_size,
            'error': error,
            'timestamp': time.time(),
            'mode': self.context.mode.name
        })
    
    def _to_device(self, x):
        """Convert input to appropriate device (GPU tensor or CPU array)"""
        if self.context.use_gpu:
            if isinstance(x, torch.Tensor):
                return x.to(self.context.device)
            elif isinstance(x, np.ndarray):
                return torch.tensor(x, device=self.context.device, dtype=torch.float32)
            elif isinstance(x, (int, float, complex)):
                return torch.tensor([x], device=self.context.device, dtype=torch.float32)[0]
            else:
                return x  # Return as is if can't be converted
        return x
    
    def _from_device(self, x, original_type):
        """Convert result back to original type from device"""
        if self.context.use_gpu and isinstance(x, torch.Tensor):
            if isinstance(original_type, np.ndarray):
                return x.cpu().numpy()
            elif isinstance(original_type, (int, float, complex)):
                return x.item()
        return x

    # ========== PRIMARY SUTRAS (1-8) ==========
    
    def ekadhikena_purvena(self, x: Union[float, np.ndarray, torch.Tensor], 
                           iterations: int = 1, 
                           ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 1: Ekadhikena Purvena - "By one more than the previous one"
        
        Mathematical logic: Implements incremental expansion through recursion.
        For a number x, calculate x + 1 iteratively 'iterations' times.
        
        Classical applications:
        - Series expansions in transcendental functions
        - Progressive incrementation in numerical methods
        - Parameter stepping in optimization algorithms
        
        Quantum applications:
        - Quantum counter implementation
        - Controlled rotation angle incrementation
        - Phase kickback operations
        
        Args:
            x: Input value or array
            iterations: Number of recursive applications
            ctx: Optional execution context override
            
        Returns:
            Incrementally expanded value or array
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._ekadhikena_purvena_quantum(x, iterations, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._ekadhikena_purvena_hybrid(x, iterations, context)
            
            # Classical implementation (default)
            result = x_device
            for _ in range(iterations):
                if isinstance(result, torch.Tensor):
                    result = result + 1
                elif isinstance(result, np.ndarray):
                    result = result + 1
                else:
                    result = result + 1
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("ekadhikena_purvena", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in ekadhikena_purvena: {error_msg}")
            self._record_performance("ekadhikena_purvena", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _ekadhikena_purvena_quantum(self, x, iterations, context):
        """Quantum implementation of ekadhikena_purvena using Cirq"""
        # Determine bit width needed for the operation
        if isinstance(x, (np.ndarray, list)):
            max_val = max(np.max(x) + iterations, 0)
        else:
            max_val = max(x + iterations, 0)
            
        num_qubits = max(1, int(np.ceil(np.log2(max_val + 1))))
        
        # Create quantum circuit
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        
        # Encode initial value
        if isinstance(x, (int, float)):
            binary = bin(int(x))[2:].zfill(num_qubits)
            for i, bit in enumerate(reversed(binary)):
                if bit == '1':
                    circuit.append(cirq.X(qubits[i]))
        
        # Perform incrementation
        for _ in range(iterations):
            # Adding 1 in quantum is a cascade of controlled-not gates
            # Starting from the least significant qubit
            circuit.append(cirq.X(qubits[0]))
            for i in range(num_qubits - 1):
                # If the previous bit is set to 1, flip the next bit
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        
        # Extract result
        result_bits = [int(result.final_state_vector[i] != 0) for i in range(2**num_qubits)]
        result_decimal = sum(b * (2**i) for i, b in enumerate(result_bits))
        
        return result_decimal

    def _ekadhikena_purvena_hybrid(self, x, iterations, context):
        """Hybrid implementation of ekadhikena_purvena"""
        # For hybrid mode, use classical for large iterations and quantum for small
        threshold = 5  # Arbitrary threshold based on quantum efficiency
        
        if iterations <= threshold:
            return self._ekadhikena_purvena_quantum(x, iterations, context)
        else:
            # Split into quantum and classical parts
            quantum_part = self._ekadhikena_purvena_quantum(x, threshold, context)
            return self._ekadhikena_purvena_classical(quantum_part, iterations - threshold, context)
    
    def _ekadhikena_purvena_classical(self, x, iterations, context):
        """Classical implementation of ekadhikena_purvena"""
        result = x
        for _ in range(iterations):
            result = result + 1
        return result

    def nikhilam_navatashcaramam_dashatah(self, x: Union[float, np.ndarray, torch.Tensor], 
                                         base: Optional[float] = None,
                                         ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 2: Nikhilam Navatashcaramam Dashatah - "All from 9 and the last from 10"
        
        Mathematical logic: Calculate complement with respect to a base value.
        For decimal base (traditional): The 9's complement for all digits except the last, 
        which is the 10's complement.
        
        Generalized formula: base - x
        
        Classical applications:
        - Complement-based number representation
        - Simplifying subtraction operations
        - Numerical stability in iterative methods
        - Error correction in data transmission
        
        Quantum applications:
        - Quantum state inversion (X gates)
        - Phase inversion for amplitude amplification
        - Quantum error correction via state complementation
        - Uncomputation in oracle implementations
        
        Args:
            x: Input value or array
            base: Base for complement (default from context)
            ctx: Optional execution context override
            
        Returns:
            Complement of x with respect to base
        """
        context = ctx or self.context
        base_value = base if base is not None else context.base
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            base_device = self._to_device(base_value)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._nikhilam_quantum(x, base_value, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._nikhilam_hybrid(x, base_value, context)
            
            # Classical implementation (default)
            if isinstance(x_device, torch.Tensor):
                result = base_device - x_device
            elif isinstance(x_device, np.ndarray):
                result = base_value - x_device
            else:
                # Handle scalar case
                result = base_value - x_device
            
            # Convert back to original type
            result = self._from_device(result, original_type)

            end_time = time.time()
            self._record_performance(
                "nikhilam_navatashcaramam_dashatah",
                start_time,
                end_time,
                True,
                data_size,
            )
            return result

        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(
                f"Error in nikhilam_navatashcaramam_dashatah: {error_msg}"
            )
            self._record_performance(
                "nikhilam_navatashcaramam_dashatah",
                start_time,
                end_time,
                False,
                data_size,
                error_msg,
            )
            raise

    def paravartya_yojayet(
        self,
        x: Union[float, np.ndarray, torch.Tensor],
        divisor: Union[float, np.ndarray, torch.Tensor],
        ctx: Optional[SutraContext] = None,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 3: Paravartya Yojayet - "Transpose and Apply"
        
        Mathematical logic: Implements division through transposition and recursive application.
        For x/divisor, transforms into x * (1/divisor) with strategic inversions.
        
        Classical applications:
        - Efficient polynomial division
        - Matrix inversion techniques
        - Transform-domain calculations
        - Numerical stability in division operations
        
        Quantum applications:
        - Quantum Fourier transforms
        - Phase estimation circuits
        - Quantum state normalization
        - Controlled unitary inversions
        
        Args:
            x: Numerator (value or array)
            divisor: Denominator (value or array)
            ctx: Optional execution context override
            
        Returns:
            Result of division operation with transpose-apply methodology
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            divisor_device = self._to_device(divisor)
            
            # Prevent division by zero
            epsilon = self._to_device(context.epsilon)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._paravartya_yojayet_quantum(x, divisor, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._paravartya_yojayet_hybrid(x, divisor, context)
            
            # Classical implementation (default)
            if isinstance(x_device, torch.Tensor):
                # Safe division for tensors
                safe_divisor = torch.where(
                    torch.abs(divisor_device) > epsilon,
                    divisor_device,
                    torch.sign(divisor_device) * epsilon
                )
                result = x_device / safe_divisor
            elif isinstance(x_device, np.ndarray):
                # Safe division for arrays
                safe_divisor = np.where(
                    np.abs(divisor_device) > context.epsilon,
                    divisor_device,
                    np.sign(divisor_device) * context.epsilon
                )
                result = x_device / safe_divisor
            else:
                # Handle scalar case with safety check
                if abs(divisor_device) < context.epsilon:
                    safe_divisor = context.epsilon if divisor_device >= 0 else -context.epsilon
                else:
                    safe_divisor = divisor_device
                result = x_device / safe_divisor
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("paravartya_yojayet", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in paravartya_yojayet: {error_msg}")
            self._record_performance("paravartya_yojayet", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _paravartya_yojayet_quantum(self, x, divisor, context):
        """Quantum implementation of paravartya_yojayet using Cirq"""
        # Implementation for scalar division using quantum phase estimation
        # This is applicable for scalar division approximation
        
        # Determine precision parameters
        precision_qubits = 8  # Adjust based on desired precision
        
        # Create quantum register
        qubits = [cirq.LineQubit(i) for i in range(precision_qubits + 1)]
        
        # Target qubit for division result
        target = qubits[-1]
        
        # Create circuit
        circuit = cirq.Circuit()
        
        # Initialize target in |1⟩ state
        circuit.append(cirq.X(target))
        
        # Apply Hadamard gates to create superposition
        for i in range(precision_qubits):
            circuit.append(cirq.H(qubits[i]))
        
        # Calculate rotation angle based on divisor
        theta = 1.0 / divisor if divisor != 0 else 0
        
        # Apply controlled rotations
        for i in range(precision_qubits):
            # Each qubit controls a rotation by theta * 2^i
            power = 2 ** i
            circuit.append(cirq.ControlledGate(cirq.Rz(power * theta * 2 * np.pi))(qubits[i], target))
        
        # Apply inverse QFT
        for i in range(precision_qubits // 2):
            circuit.append(cirq.SWAP(qubits[i], qubits[precision_qubits - i - 1]))
            
        for i in range(precision_qubits):
            circuit.append(cirq.H(qubits[i]))
            for j in range(i):
                phase = -2 * np.pi / (2 ** (i - j))
                circuit.append(cirq.CZ(qubits[j], qubits[i]) ** (phase / np.pi))
        
        # Measure qubits
        circuit.append(cirq.measure(*qubits[:-1], key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Get most frequent measurement outcome
        result_bits = result.data['result'].value_counts().index[0]
        
        # Convert to decimal
        result_decimal = sum(int(bit) * (2**i) for i, bit in enumerate(result_bits))
        
        # Scale by x
        return x * result_decimal / (2**precision_qubits)
    
    def _paravartya_yojayet_hybrid(self, x, divisor, context):
        """Hybrid implementation of paravartya_yojayet"""
        # For hybrid mode, calculate reciprocal quantum-mechanically,
        # then perform multiplication classically
        if isinstance(x, (int, float)) and isinstance(divisor, (int, float)):
            reciprocal = self._quantum_reciprocal(divisor, context)
            return x * reciprocal
        else:
            # For arrays, use classical implementation
            return self._paravartya_yojayet_classical(x, divisor, context)
    
    def _quantum_reciprocal(self, value, context):
        """Calculate reciprocal using quantum phase estimation"""
        # Create CUDAQ kernel for reciprocal calculation
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(8)  # 8 qubits for precision
        
        # Initialize in superposition
        kernel.h(q)
        
        # Apply phase rotations based on value
        angle = 1.0 / value if value != 0 else 0
        for i in range(8):
            kernel.rz(q[i], 2 * np.pi * angle * (2**i))
        
        # Apply inverse QFT
        cudaq.inverseFQFT(kernel, q)
        
        # Measure
        kernel.mz(q)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Get most frequent outcome
        top_result = result.most_probable()
        
        # Convert to decimal
        result_decimal = int(top_result, 2)
        
        # Scale to [0,1] and return reciprocal
        return result_decimal / (2**8)
    
    def _paravartya_yojayet_classical(self, x, divisor, context):
        """Classical implementation of paravartya_yojayet"""
        # Implement regular division with safety checks
        epsilon = context.epsilon
        
        if isinstance(divisor, np.ndarray):
            safe_divisor = np.where(
                np.abs(divisor) > epsilon,
                divisor,
                np.sign(divisor) * epsilon
            )
            return x / safe_divisor
        else:
            if abs(divisor) < epsilon:
                safe_divisor = epsilon if divisor >= 0 else -epsilon
            else:
                safe_divisor = divisor
            return x / safe_divisor

    def shunyam_samyasamuccaye(self, a: Union[float, np.ndarray, torch.Tensor],
                              b: Union[float, np.ndarray, torch.Tensor],
                              ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 4: Shunyam Samyasamuccaye - "When the sum is the same, it is zero"
        
        Mathematical logic: Identifies and handles cases where sums or differences
        approach zero, with special consideration for numerical stability.
        
        Classical applications:
        - Detecting cancellations in numerical calculations
        - Eliminating noise in signal processing
        - Identifying equilibrium states in dynamical systems
        - Balance equations in chemical or economic models
        
        Quantum applications:
        - Quantum interference detection
        - Phase cancellation in quantum walks
        - Quantum error correction for phase flip errors
        - Identifying decoherence-free subspaces
        
        Args:
            a: First value or array
            b: Second value or array
            ctx: Optional execution context override
            
        Returns:
            Zero where sums approach zero, otherwise returns a + b
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._shunyam_samyasamuccaye_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._shunyam_samyasamuccaye_hybrid(a, b, context)
            
            # Classical implementation (default)
            if isinstance(a_device, torch.Tensor):
                # Calculate sum
                sum_result = a_device + b_device
                
                # Create mask for values close to zero
                zero_mask = torch.abs(sum_result) < context.epsilon
                
                # Apply zero where sum is close to zero
                result = torch.where(zero_mask, torch.zeros_like(sum_result), sum_result)
                
            elif isinstance(a_device, np.ndarray):
                # Calculate sum
                sum_result = a_device + b_device
                
                # Create mask for values close to zero
                zero_mask = np.abs(sum_result) < context.epsilon
                
                # Apply zero where sum is close to zero
                result = np.where(zero_mask, np.zeros_like(sum_result), sum_result)
                
            else:
                # Handle scalar case
                sum_result = a_device + b_device
                if abs(sum_result) < context.epsilon:
                    result = 0
                else:
                    result = sum_result
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("shunyam_samyasamuccaye", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in shunyam_samyasamuccaye: {error_msg}")
            self._record_performance("shunyam_samyasamuccaye", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _shunyam_samyasamuccaye_quantum(self, a, b, context):
        """Quantum implementation of shunyam_samyasamuccaye using Cirq"""
        # Simple interference-based implementation for scalar inputs
        
        # This implementation works for scalar values
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._shunyam_samyasamuccaye_classical(a, b, context)
        
        # Normalize inputs to range [0, 1] for encoding as quantum amplitudes
        max_val = max(abs(a), abs(b)) * 2
        if max_val < context.epsilon:
            return 0
            
        norm_a = a / max_val
        norm_b = b / max_val
        
        # Create quantum circuit with one qubit
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit()
        
        # Prepare superposition state
        circuit.append(cirq.H(q))
        
        # Apply phase rotations based on inputs
        circuit.append(cirq.ZPowGate(exponent=norm_a)(q))
        circuit.append(cirq.X(q))
        circuit.append(cirq.ZPowGate(exponent=norm_b)(q))
        circuit.append(cirq.X(q))
        
        # Apply Hadamard to observe interference
        circuit.append(cirq.H(q))
        
        # Measure
        circuit.append(cirq.measure(q, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Analyze measurements
        counts = result.histogram(key='result')
        
        # If interference leads to significant bias toward |0⟩ or |1⟩,
        # then a and b are likely cancelling each other
        threshold = 0.8  # Arbitrary threshold for determining interference
        total_shots = sum(counts.values())
        
        if counts.get(0, 0) / total_shots > threshold or counts.get(1, 0) / total_shots > threshold:
            # Strong interference detected, likely zero sum
            return 0
        else:
            # No strong interference, return actual sum
            return a + b
    
    def _shunyam_samyasamuccaye_hybrid(self, a, b, context):
        """Hybrid implementation of shunyam_samyasamuccaye"""
        # For small arrays, use quantum interference checking
        if (isinstance(a, np.ndarray) and a.size <= 4) or isinstance(a, (int, float)):
            if isinstance(a, np.ndarray):
                # Process each element through quantum check
                result = np.zeros_like(a)
                for i in range(a.size):
                    result.flat[i] = self._shunyam_samyasamuccaye_quantum(
                        a.flat[i], b.flat[i] if isinstance(b, np.ndarray) else b, context
                    )
                return result
            else:
                # Single value case
                return self._shunyam_samyasamuccaye_quantum(a, b, context)
        else:
            # For larger arrays, use classical implementation
            return self._shunyam_samyasamuccaye_classical(a, b, context)
    
    def _shunyam_samyasamuccaye_classical(self, a, b, context):
        """Classical implementation of shunyam_samyasamuccaye"""
        # Calculate sum
        sum_result = a + b
        
        # Check if result is close to zero
        if isinstance(sum_result, np.ndarray):
            zero_mask = np.abs(sum_result) < context.epsilon
            return np.where(zero_mask, np.zeros_like(sum_result), sum_result)
        else:
            if abs(sum_result) < context.epsilon:
                return 0
            else:
                return sum_result

    def vyashtisamanstih(self, whole: Union[float, np.ndarray, torch.Tensor],
                        parts: Union[List, np.ndarray, torch.Tensor],
                        ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 5: Vyashtisamanstih - "Part and Whole"
        
        Mathematical logic: Establishes relationship between a whole and its 
        constituent parts, enabling transformations between these representations.
        
        Classical applications:
        - Decomposition of complex systems into components
        - Mereology in data structures and algorithms
        - Hierarchical clustering and segmentation
        - Multi-resolution analysis in signal processing
        
        Quantum applications:
        - Quantum state decomposition into basis states
        - Tensor network factorization
        - Entanglement analysis between subsystems
        - Quantum circuit partitioning and optimization
        
        Args:
            whole: The complete entity (value or array)
            parts: The constituent components (list, array, or tensor)
            ctx: Optional execution context override
            
        Returns:
            Reconciled representation of part-whole relationship
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(whole)
        data_size = np.size(whole) if hasattr(whole, 'size') else 1
        
        try:
            # Convert to device if using GPU
            whole_device = self._to_device(whole)
            
            # Handle parts conversion based on type
            if isinstance(parts, list):
                parts_device = [self._to_device(p) for p in parts]
            else:
                parts_device = self._to_device(parts)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._vyashtisamanstih_quantum(whole, parts, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._vyashtisamanstih_hybrid(whole, parts, context)
            
            # Classical implementation (default)
            if isinstance(whole_device, torch.Tensor):
                # Check if whole equals sum of parts
                if isinstance(parts_device, list):
                    parts_sum = sum(parts_device)
                else:
                    # Assume it's a tensor with parts along first dimension
                    parts_sum = torch.sum(parts_device, dim=0)
                
                # Compare whole with sum of parts
                diff = whole_device - parts_sum
                
                # If difference is small, return whole; otherwise return reconstructed whole
                if torch.all(torch.abs(diff) < context.epsilon):
                    result = whole_device
                else:
                    result = parts_sum
                    
            elif isinstance(whole_device, np.ndarray):
                # Check if whole equals sum of parts
                if isinstance(parts_device, list):
                    parts_sum = sum(parts_device)
                else:
                    # Assume it's an array with parts along first dimension
                    parts_sum = np.sum(parts_device, axis=0)
                
                # Compare whole with sum of parts
                diff = whole_device - parts_sum
                
                # If difference is small, return whole; otherwise return reconstructed whole
                if np.all(np.abs(diff) < context.epsilon):
                    result = whole_device
                else:
                    result = parts_sum
                    
            else:
                # Handle scalar case
                if isinstance(parts_device, list):
                    parts_sum = sum(parts_device)
                else:
                    parts_sum = np.sum(parts_device)
                
                # Compare whole with sum of parts
                diff = whole_device - parts_sum
                
                # If difference is small, return whole; otherwise return reconstructed whole
                if abs(diff) < context.epsilon:
                    result = whole_device
                else:
                    result = parts_sum
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("vyashtisamanstih", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in vyashtisamanstih: {error_msg}")
            self._record_performance("vyashtisamanstih", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _vyashtisamanstih_quantum(self, whole, parts, context):
        """Quantum implementation of vyashtisamanstih using Cirq"""
        # This implementation demonstrates tensor decomposition using quantum SVD
        # For simplicity, we'll handle the scalar or small vector case
        
        if isinstance(whole, (int, float)) and all(isinstance(p, (int, float)) for p in parts):
            # For scalar whole and parts, use CUDAQ for decomposition verification
            kernel = cudaq.make_kernel()
            q = kernel.qalloc(len(parts) + 1)
            
            # Encode whole into amplitude of first qubit
            theta_whole = 2 * np.arcsin(min(1.0, abs(whole) / 10.0))  # Normalize
            kernel.ry(q[0], theta_whole)
            
            # Encode parts into amplitudes of remaining qubits
            for i, part in enumerate(parts):
                theta_part = 2 * np.arcsin(min(1.0, abs(part) / 10.0))  # Normalize
                kernel.ry(q[i+1], theta_part)
            
            # Create entanglement to check part-whole relationship
            for i in range(len(parts)):
                kernel.cx(q[0], q[i+1])
            
            # Measure
            kernel.mz(q)
            
            # Execute
            result = cudaq.sample(kernel)
            
            # Check measurement outcomes
            # If whole = sum(parts), measurements should show high correlation
            top_results = result.most_probable(5)
            
            # If measurements show correlation, return whole; otherwise return sum of parts
            correlation_threshold = 0.6  # Arbitrary threshold
            if self._check_quantum_correlation(top_results, correlation_threshold):
                return whole
            else:
                return sum(parts)
        else:
            # For more complex cases, fall back to classical implementation
            return self._vyashtisamanstih_classical(whole, parts, context)
    
    def _check_quantum_correlation(self, results, threshold):
        """Helper function to check for quantum correlation in measurement outcomes"""
        # This is a simplified approach to check if measurement outcomes
        # indicate correlation between whole and parts
        total_prob = sum(results.values())
        correlated_prob = 0
        
        for bitstring, count in results.items():
            # In a correlated outcome, if first bit is 1, most other bits should also be 1
            first_bit = bitstring[0]
            if first_bit == '1':
                # Count how many other bits match the first bit
                matches = sum(1 for bit in bitstring[1:] if bit == '1')
                if matches > len(bitstring[1:]) / 2:
                    correlated_prob += count
            
        return correlated_prob / total_prob > threshold
    
    def _vyashtisamanstih_hybrid(self, whole, parts, context):
        """Hybrid implementation of vyashtisamanstih"""
        # For scalar or small vector cases, use quantum implementation
        if (isinstance(whole, (int, float)) or 
            (isinstance(whole, np.ndarray) and whole.size <= 4)) and len(parts) <= 4:
            return self._vyashtisamanstih_quantum(whole, parts, context)
        else:
            # For larger cases, use classical implementation
            return self._vyashtisamanstih_classical(whole, parts, context)
    
    def _vyashtisamanstih_classical(self, whole, parts, context):
        """Classical implementation of vyashtisamanstih"""
        # Calculate sum of parts
        if isinstance(parts, list):
            parts_sum = sum(parts)
        elif isinstance(parts, np.ndarray):
            parts_sum = np.sum(parts, axis=0)
        else:
            parts_sum = torch.sum(parts, dim=0)
        
        # Compare with whole
        diff = whole - parts_sum
        
        # Check if difference is small
        if isinstance(diff, np.ndarray):
            if np.all(np.abs(diff) < context.epsilon):
                return whole
            else:
                return parts_sum
        elif isinstance(diff, torch.Tensor):
            if torch.all(torch.abs(diff) < context.epsilon):
                return whole
            else:
                return parts_sum
        else:
            if abs(diff) < context.epsilon:
                return whole
            else:
                return parts_sum

    def chalana_kalana(self, x: Union[float, np.ndarray, torch.Tensor],
                      steps: int = 1,
                      direction: int = 1,
                      ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 6: Chalana-Kalana - "Sequential Operations"
        
        Mathematical logic: Implements sequential transformations and iterative processes,
        enabling step-by-step evolution of values or systems.
        
        Classical applications:
        - Iterative numerical methods
        - Time-series forecasting
        - Game theory sequential moves
        - Stepwise optimization procedures
        
        Quantum applications:
        - Quantum walk implementations
        - Sequential quantum gates
        - Progressive quantum annealing
        - Quantum trajectory analysis
        
        Args:
            x: Input value or array
            steps: Number of sequential steps to perform
            direction: Direction of operation (1 for forward, -1 for backward)
            ctx: Optional execution context override
            
        Returns:
            Result after applying sequential operations
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._chalana_kalana_quantum(x, steps, direction, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._chalana_kalana_hybrid(x, steps, direction, context)
            
            # Classical implementation (default)
            result = x_device
            step_size = direction  # Basic step size
            
            for _ in range(steps):
                if isinstance(result, torch.Tensor):
                    result = result + step_size
                elif isinstance(result, np.ndarray):
                    result = result + step_size
                else:
                    result = result + step_size
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("chalana_kalana", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in chalana_kalana: {error_msg}")
            self._record_performance("chalana_kalana", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _chalana_kalana_quantum(self, x, steps, direction, context):
        """Quantum implementation of chalana_kalana using Cirq"""
        # This implements a quantum walk for sequential operations
        
        # Determine bit width needed for the operation
        if isinstance(x, (np.ndarray, list)):
            max_val = max(np.max(x) + steps, 0)
        else:
            max_val = max(x + steps, 0)
            
        num_qubits = max(1, int(np.ceil(np.log2(max_val + 1))))
        
        # Create quantum circuit
        position_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        # Coin qubit to determine direction
        coin_qubit = cirq.LineQubit(num_qubits)
        
        circuit = cirq.Circuit()
        
        # Encode initial value
        if isinstance(x, (int, float)):
            binary = bin(int(x))[2:].zfill(num_qubits)
            for i, bit in enumerate(reversed(binary)):
                if bit == '1':
                    circuit.append(cirq.X(position_qubits[i]))
        
        # Initialize coin qubit based on direction
        if direction > 0:
            circuit.append(cirq.X(coin_qubit))
        
        # Perform quantum walk for each step
        for _ in range(steps):
            # Apply Hadamard to coin qubit
            circuit.append(cirq.H(coin_qubit))
            
            # Controlled shift based on coin state
            # If coin is |1⟩, increment position
            for i in range(num_qubits):
                circuit.append(cirq.CNOT(position_qubits[i], position_qubits[(i+1) % num_qubits]).controlled_by(coin_qubit))
            
            # If coin is |0⟩, decrement position
            circuit.append(cirq.X(coin_qubit))
            for i in range(num_qubits-1, -1, -1):
                circuit.append(cirq.CNOT(position_qubits[i], position_qubits[(i-1) % num_qubits]).controlled_by(coin_qubit))
            circuit.append(cirq.X(coin_qubit))
        
        # Measure position qubits
        circuit.append(cirq.measure(*position_qubits, key='position'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Get most frequent position
        position_counts = result.histogram(key='position')
        most_frequent_position = max(position_counts.items(), key=lambda x: x[1])[0]
        
        return most_frequent_position
    
    def _chalana_kalana_hybrid(self, x, steps, direction, context):
        """Hybrid implementation of chalana_kalana"""
        # Split steps between quantum and classical
        quantum_steps = min(steps, 5)  # Limit quantum steps for efficiency
        classical_steps = steps - quantum_steps
        
        # Apply quantum steps first
        if quantum_steps > 0:
            intermediate = self._chalana_kalana_quantum(x, quantum_steps, direction, context)
        else:
            intermediate = x
        
        # Then apply classical steps
        if classical_steps > 0:
            return self._chalana_kalana_classical(intermediate, classical_steps, direction, context)
        else:
            return intermediate
    
    def _chalana_kalana_classical(self, x, steps, direction, context):
        """Classical implementation of chalana_kalana"""
        result = x
        step_size = direction  # Basic step size
        
        for _ in range(steps):
            result = result + step_size
            
        return result

    def sankalana_vyavakalanabhyam(self, a: Union[float, np.ndarray, torch.Tensor],
                                 b: Union[float, np.ndarray, torch.Tensor],
                                 operation: str = 'add',
                                 ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 7: Sankalana-Vyavakalanabhyam - "By Addition and Subtraction"
        
        Mathematical logic: Provides a unified approach to addition and subtraction operations,
        with optimizations for numerical stability and computational efficiency.
        
        Classical applications:
        - Stabilized numerical addition/subtraction
        - Parallel computation of sum and difference
        - Conservation law enforcement in simulations
        - Financial transaction balancing
        
        Quantum applications:
        - Quantum adder/subtractor circuits
        - Quantum interference manipulation
        - Phase addition/subtraction in quantum algorithms
        - Quantum state preparation via superposition
        
        Args:
            a: First value or array
            b: Second value or array
            operation: Type of operation ('add', 'subtract', or 'both')
            ctx: Optional execution context override
            
        Returns:
            Result of addition, subtraction, or both operations
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._sankalana_vyavakalanabhyam_quantum(a, b, operation, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sankalana_vyavakalanabhyam_hybrid(a, b, operation, context)
            
            # Classical implementation (default)
            if operation == 'add':
                result = a_device + b_device
            elif operation == 'subtract':
                result = a_device - b_device
            elif operation == 'both':
                # Return tuple of both results
                if isinstance(a_device, torch.Tensor):
                    result = (a_device + b_device, a_device - b_device)
                elif isinstance(a_device, np.ndarray):
                    result = (a_device + b_device, a_device - b_device)
                else:
                    result = (a_device + b_device, a_device - b_device)
            else:
                raise ValueError(f"Unknown operation: {operation}. Use 'add', 'subtract', or 'both'.")
            
            # Convert back to original type (except for 'both' which returns a tuple)
            if operation != 'both':
                result = self._from_device(result, original_type)
            else:
                result = (self._from_device(result[0], original_type), 
                         self._from_device(result[1], original_type))
            
            end_time = time.time()
            self._record_performance("sankalana_vyavakalanabhyam", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sankalana_vyavakalanabhyam: {error_msg}")
            self._record_performance("sankalana_vyavakalanabhyam", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _sankalana_vyavakalanabhyam_quantum(self, a, b, operation, context):
        """Quantum implementation of sankalana_vyavakalanabhyam using Cirq"""
        # This implements a quantum adder/subtractor circuit
        # For simplicity, we'll handle scalar values
        
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._sankalana_vyavakalanabhyam_classical(a, b, operation, context)
        
        # Determine bit width needed for the operation
        max_val = max(abs(a), abs(b)) * 2
        num_qubits = max(1, int(np.ceil(np.log2(max_val + 1))))
        
        # Create quantum circuit
        a_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        b_qubits = [cirq.LineQubit(i + num_qubits) for i in range(num_qubits)]
        # Additional qubit for carry bit
        carry_qubit = cirq.LineQubit(2 * num_qubits)
        
        circuit = cirq.Circuit()
        
        # Encode 'a' value
        a_int = int(a)
        a_binary = bin(a_int if a_int >= 0 else (1 << num_qubits) + a_int)[2:].zfill(num_qubits)
        for i, bit in enumerate(reversed(a_binary)):
            if bit == '1':
                circuit.append(cirq.X(a_qubits[i]))
        
        # Encode 'b' value
        b_int = int(b)
        b_binary = bin(b_int if b_int >= 0 else (1 << num_qubits) + b_int)[2:].zfill(num_qubits)
        for i, bit in enumerate(reversed(b_binary)):
            if bit == '1':
                circuit.append(cirq.X(b_qubits[i]))
        
        # Perform quantum addition or subtraction
        if operation == 'add' or operation == 'both':
            # Quantum addition circuit using CARRY operations
            for i in range(num_qubits):
                # Compute carry bit using Toffoli gates
                if i == 0:
                    circuit.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
                    circuit.append(cirq.CNOT(a_qubits[i], carry_qubit))
                else:
                    circuit.append(cirq.TOFFOLI(a_qubits[i-1], b_qubits[i-1], carry_qubit))
                    circuit.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
                    circuit.append(cirq.CNOT(carry_qubit, b_qubits[i]))
                    # Uncompute carry bit for next iteration
                    circuit.append(cirq.TOFFOLI(a_qubits[i-1], b_qubits[i-1], carry_qubit))
        
        if operation == 'subtract' or operation == 'both':
            # For subtraction, perform two's complement on b before addition
            for i in range(num_qubits):
                circuit.append(cirq.X(b_qubits[i]))
            
            # Add 1 to complete two's complement
            circuit.append(cirq.X(carry_qubit))
            
            # Then perform addition as before
            for i in range(num_qubits):
                # Compute carry bit using Toffoli gates
                if i == 0:
                    circuit.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
                    circuit.append(cirq.CNOT(a_qubits[i], carry_qubit))
                else:
                    circuit.append(cirq.TOFFOLI(a_qubits[i-1], b_qubits[i-1], carry_qubit))
                    circuit.append(cirq.CNOT(a_qubits[i], b_qubits[i]))
                    circuit.append(cirq.CNOT(carry_qubit, b_qubits[i]))
                    # Uncompute carry bit for next iteration
                    circuit.append(cirq.TOFFOLI(a_qubits[i-1], b_qubits[i-1], carry_qubit))
        
        # Measure results
        if operation == 'add':
            circuit.append(cirq.measure(*b_qubits, key='result'))
            # Simulate
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1)
            # Extract result
            result_bits = result.measurements['result'][0]
            result_int = sum(bit * (2**i) for i, bit in enumerate(result_bits))
            return result_int
            
        elif operation == 'subtract':
            circuit.append(cirq.measure(*b_qubits, key='result'))
            # Simulate
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1)
            # Extract result
            result_bits = result.measurements['result'][0]
            result_int = sum(bit * (2**i) for i, bit in enumerate(result_bits))
            # Convert from two's complement if needed
            if result_bits[-1] == 1:  # Negative number
                result_int = result_int - (1 << num_qubits)
            return result_int
            
        else:  # 'both'
            # For 'both', we need to run two separate circuits
            add_result = self._sankalana_vyavakalanabhyam_quantum(a, b, 'add', context)
            sub_result = self._sankalana_vyavakalanabhyam_quantum(a, b, 'subtract', context)
            return (add_result, sub_result)
    
    def _sankalana_vyavakalanabhyam_hybrid(self, a, b, operation, context):
        """Hybrid implementation of sankalana_vyavakalanabhyam"""
        # For scalar values, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._sankalana_vyavakalanabhyam_quantum(a, b, operation, context)
        # For small arrays, use quantum for some elements and classical for others
        elif isinstance(a, np.ndarray) and a.size <= 4:
            # Process each element
            if operation == 'add' or operation == 'subtract':
                result = np.zeros_like(a)
                for i in range(a.size):
                    a_val = a.flat[i]
                    b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                    result.flat[i] = self._sankalana_vyavakalanabhyam_quantum(
                        a_val, b_val, operation, context
                    )
                return result
            else:  # 'both'
                result_add = np.zeros_like(a)
                result_sub = np.zeros_like(a)
                for i in range(a.size):
                    a_val = a.flat[i]
                    b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                    add_val, sub_val = self._sankalana_vyavakalanabhyam_quantum(
                        a_val, b_val, 'both', context
                    )
                    result_add.flat[i] = add_val
                    result_sub.flat[i] = sub_val
                return (result_add, result_sub)
        else:
            # For larger arrays, use classical implementation
            return self._sankalana_vyavakalanabhyam_classical(a, b, operation, context)
    
    def _sankalana_vyavakalanabhyam_classical(self, a, b, operation, context):
        """Classical implementation of sankalana_vyavakalanabhyam"""
        if operation == 'add':
            return a + b
        elif operation == 'subtract':
            return a - b
        elif operation == 'both':
            return (a + b, a - b)
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'add', 'subtract', or 'both'.")

    def purna_apurna_bhyam(self, x: Union[float, np.ndarray, torch.Tensor],
                          threshold: float = 0.5,
                          ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 8: Purna-Apurna Bhyam - "By the Completion or Non-Completion"
        
        Mathematical logic: Handles boundary conditions and completeness checks,
        determining whether values satisfy specific thresholds or criteria.
        
        Classical applications:
        - Rounding and quantization
        - Threshold-based classification
        - Convergence testing in numerical methods
        - Binary decision boundaries
        
        Quantum applications:
        - Quantum state preparation verification
        - Quantum measurement thresholding
        - Quantum error detection
        - Quantum classifier decision boundaries
        
        Args:
            x: Input value or array
            threshold: Completeness threshold (default 0.5)
            ctx: Optional execution context override
            
        Returns:
            Binary result indicating whether each element exceeds the threshold
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            threshold_device = self._to_device(threshold)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._purna_apurna_bhyam_quantum(x, threshold, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._purna_apurna_bhyam_hybrid(x, threshold, context)
            
            # Classical implementation (default)
            if isinstance(x_device, torch.Tensor):
                result = torch.where(x_device >= threshold_device, 
                                    torch.ones_like(x_device), 
                                    torch.zeros_like(x_device))
            elif isinstance(x_device, np.ndarray):
                result = np.where(x_device >= threshold, 1.0, 0.0)
            else:
                result = 1.0 if x_device >= threshold else 0.0
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("purna_apurna_bhyam", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in purna_apurna_bhyam: {error_msg}")
            self._record_performance("purna_apurna_bhyam", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _purna_apurna_bhyam_quantum(self, x, threshold, context):
        """Quantum implementation of purna_apurna_bhyam using Cirq"""
        # For scalar inputs, implement quantum thresholding circuit
        
        if isinstance(x, (int, float)):
            # Normalize input to [0,1] range for amplitude encoding
            x_norm = min(max(x, 0), 1)  # Clamp to [0,1]
            
            # Create quantum circuit
            q = cirq.LineQubit(0)
            circuit = cirq.Circuit()
            
            # Encode value as amplitude
            theta = 2 * np.arcsin(np.sqrt(x_norm))
            circuit.append(cirq.ry(theta)(q))
            
            # Apply threshold check through measurement
            circuit.append(cirq.measure(q, key='result'))
            
            # Simulate multiple times to get probabilistic outcome
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1000)
            
            # Count '1' outcomes
            counts = result.histogram(key='result')
            probability_one = counts.get(1, 0) / 1000
            
            # Compare with threshold
            return 1.0 if probability_one >= threshold else 0.0
            
        elif isinstance(x, np.ndarray) and x.size <= 4:
            # For small arrays, process each element
            result = np.zeros_like(x)
            for i in range(x.size):
                result.flat[i] = self._purna_apurna_bhyam_quantum(x.flat[i], threshold, context)
            return result
            
        else:
            # For larger arrays, fall back to classical implementation
            return self._purna_apurna_bhyam_classical(x, threshold, context)
    
    def _purna_apurna_bhyam_hybrid(self, x, threshold, context):
        """Hybrid implementation of purna_apurna_bhyam"""
        # For scalar or small vectors, use quantum circuit
        if isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and x.size <= 4):
            return self._purna_apurna_bhyam_quantum(x, threshold, context)
        else:
            # For larger arrays, use classical with GPU acceleration if available
            return self._purna_apurna_bhyam_classical(x, threshold, context)
    
    def _purna_apurna_bhyam_classical(self, x, threshold, context):
        """Classical implementation of purna_apurna_bhyam"""
        if isinstance(x, torch.Tensor):
            return torch.where(x >= threshold, torch.ones_like(x), torch.zeros_like(x))
        elif isinstance(x, np.ndarray):
            return np.where(x >= threshold, 1.0, 0.0)
        else:
            return 1.0 if x >= threshold else 0.0

    def sesanyankena_caramena(self, coefficients: Union[List, np.ndarray, torch.Tensor],
                             x: Union[float, np.ndarray, torch.Tensor],
                             ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 9: Sesanyankena Caramena - "By the Remainder and the Last Digit"
        
        Mathematical logic: Enables efficient polynomial evaluation and processing
        of expressions where the last term has special significance.
        
        Classical applications:
        - Horner's method for polynomial evaluation
        - Modular arithmetic calculations
        - Checksum verification
        - Digit-by-digit processing algorithms
        
        Quantum applications:
        - Quantum polynomial state preparation
        - Quantum phase estimation refinement
        - Quantum modular arithmetic
        - Iterative quantum amplitude amplification
        
        Args:
            coefficients: List of polynomial coefficients [a0, a1, a2, ...]
            x: Value(s) at which to evaluate the polynomial
            ctx: Optional execution context override
            
        Returns:
            Result of polynomial evaluation using efficient nested multiplication
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            
            # Convert coefficients to appropriate format
            if self.context.use_gpu:
                if isinstance(coefficients, list):
                    coeffs_device = torch.tensor(coefficients, device=self.context.device)
                elif isinstance(coefficients, np.ndarray):
                    coeffs_device = torch.tensor(coefficients, device=self.context.device)
                else:  # Assume it's already a tensor
                    coeffs_device = coefficients.to(self.context.device)
            else:
                if isinstance(coefficients, list):
                    coeffs_device = np.array(coefficients)
                elif isinstance(coefficients, torch.Tensor):
                    coeffs_device = coefficients.cpu().numpy()
                else:  # Assume it's already a numpy array
                    coeffs_device = coefficients
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._sesanyankena_caramena_quantum(coeffs_device, x, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sesanyankena_caramena_hybrid(coeffs_device, x, context)
            
            # Classical implementation (default)
            # Use Horner's method for polynomial evaluation
            if isinstance(x_device, torch.Tensor):
                # Handle scalar or array evaluation
                if x_device.ndim == 0:  # Scalar case
                    result = coeffs_device[-1]
                    for coef in reversed(coeffs_device[:-1]):
                        result = result * x_device + coef
                else:  # Array case
                    result = torch.full_like(x_device, coeffs_device[-1])
                    for coef in reversed(coeffs_device[:-1]):
                        result = result * x_device + coef
            elif isinstance(x_device, np.ndarray):
                # Handle scalar or array evaluation
                if x_device.ndim == 0:  # Scalar case
                    result = coeffs_device[-1]
                    for coef in reversed(coeffs_device[:-1]):
                        result = result * x_device + coef
                else:  # Array case
                    result = np.full_like(x_device, coeffs_device[-1])
                    for coef in reversed(coeffs_device[:-1]):
                        result = result * x_device + coef
            else:
                # Scalar case
                result = coeffs_device[-1]
                for coef in reversed(coeffs_device[:-1]):
                    result = result * x_device + coef
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("sesanyankena_caramena", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sesanyankena_caramena: {error_msg}")
            self._record_performance("sesanyankena_caramena", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _sesanyankena_caramena_quantum(self, coefficients, x, context):
        """Quantum implementation of sesanyankena_caramena using CUDAQ"""
        # This implements a quantum circuit for polynomial evaluation
        # using phase estimation techniques
        
        # For simplicity, we'll handle the case of polynomial degree <= 3
        # and scalar x values
        
        if len(coefficients) > 4 or not isinstance(x, (int, float)):
            # Fall back to classical for higher-degree polynomials or non-scalar inputs
            return self._sesanyankena_caramena_classical(coefficients, x, context)
        
        # Create CUDAQ kernel for polynomial evaluation
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(3)  # Use 3 qubits for evaluation
        
        # Initialize in superposition
        kernel.h(q)
        
        # Apply phase rotations based on coefficients
        # This encodes the polynomial evaluation in the phase
        for i, coef in enumerate(coefficients):
            angle = coef * (x ** i) / np.sum(np.abs(coefficients))
            kernel.rz(q[i % 3], 2 * np.pi * angle)
        
        # Apply inverse QFT to extract result
        cudaq.inverseFQFT(kernel, q)
        
        # Measure
        kernel.mz(q)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Get most frequent outcome
        top_result = result.most_probable()
        
        # Convert to decimal
        result_decimal = int(top_result, 2)
        
        # Scale result based on coefficients
        scaling_factor = np.sum(np.abs(coefficients))
        scaled_result = result_decimal * scaling_factor / (2**3)
        
        return scaled_result
    
    def _sesanyankena_caramena_hybrid(self, coefficients, x, context):
        """Hybrid implementation of sesanyankena_caramena"""
        # Split polynomial into low and high degree terms
        # Evaluate high degree terms classically and low degree terms with quantum circuit
        
        if len(coefficients) <= 4 and isinstance(x, (int, float)):
            # Small polynomial with scalar x can be fully evaluated quantum-mechanically
            return self._sesanyankena_caramena_quantum(coefficients, x, context)
        elif len(coefficients) > 4:
            # Split into low and high degree parts
            low_degree = coefficients[:3]
            high_degree = coefficients[3:]
            
            # Evaluate high degree terms classically
            high_result = self._sesanyankena_caramena_classical(high_degree, x, context)
            
            # Evaluate low degree terms quantum-mechanically
            low_result = self._sesanyankena_caramena_quantum(low_degree, x, context)
            
            # Combine results (high_result * x^3 + low_result)
            return high_result * (x ** 3) + low_result
        else:
            # For array inputs or other cases, use classical implementation
            return self._sesanyankena_caramena_classical(coefficients, x, context)
    
    def _sesanyankena_caramena_classical(self, coefficients, x, context):
        """Classical implementation of sesanyankena_caramena"""
        # Use Horner's method for polynomial evaluation
        if isinstance(coefficients, torch.Tensor):
            coeffs = coefficients
        elif isinstance(coefficients, np.ndarray):
            coeffs = coefficients
        else:
            coeffs = np.array(coefficients)
            
        if isinstance(x, torch.Tensor):
            # Tensor implementation
            result = torch.full_like(x, coeffs[-1])
            for coef in reversed(coeffs[:-1]):
                result = result * x + coef
        elif isinstance(x, np.ndarray):
            # NumPy implementation
            result = np.full_like(x, coeffs[-1])
            for coef in reversed(coeffs[:-1]):
                result = result * x + coef
        else:
            # Scalar implementation
            result = coeffs[-1]
            for coef in reversed(coeffs[:-1]):
                result = result * x + coef
                
        return result

    def ekanyunena_purvena(self, x: Union[float, np.ndarray, torch.Tensor],
                          base: float = 10.0,
                          ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 10: Ekanyunena Purvena - "By one less than the previous one"
        
        Mathematical logic: Implements decremental recursion, complementary to
        Ekadhikena Purvena but with subtraction instead of addition.
        
        Classical applications:
        - Decremental series generation
        - Countdown algorithms
        - Resource allocation with decreasing constraints
        - Step-wise reduction in optimization
        
        Quantum applications:
        - Quantum annealing cool-down procedures
        - Quantum amplitude deamplification
        - Iterative phase unwinding in quantum algorithms
        - Quantum gate decomposition methods
        
        Args:
            x: Input value or array
            base: Base value for complement calculations
            ctx: Optional execution context override
            
        Returns:
            Result after recursive decrementation
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._ekanyunena_purvena_quantum(x, base, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._ekanyunena_purvena_hybrid(x, base, context)
            
            # Classical implementation (default)
            if isinstance(x_device, torch.Tensor):
                result = x_device - 1
            elif isinstance(x_device, np.ndarray):
                result = x_device - 1
            else:
                result = x_device - 1
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("ekanyunena_purvena", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in ekanyunena_purvena: {error_msg}")
            self._record_performance("ekanyunena_purvena", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _ekanyunena_purvena_quantum(self, x, base, context):
        """Quantum implementation of ekanyunena_purvena using Cirq"""
        # This implements a quantum decrementor circuit
        
        # Determine bit width needed for the operation
        if isinstance(x, (np.ndarray, list)):
            max_val = max(np.max(x), 0)
        else:
            max_val = max(x, 0)
            
        num_qubits = max(1, int(np.ceil(np.log2(max_val + 1))))
        
        # Create quantum circuit
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        
        # Encode initial value
        if isinstance(x, (int, float)):
            binary = bin(int(x))[2:].zfill(num_qubits)
            for i, bit in enumerate(reversed(binary)):
                if bit == '1':
                    circuit.append(cirq.X(qubits[i]))
        
        # Subtract 1 using quantum decrementer
        # Start with NOT on all qubits
        for q in qubits:
            circuit.append(cirq.X(q))
            
        # Apply Toffoli gates for borrow propagation
        for i in range(1, num_qubits):
            circuit.append(cirq.TOFFOLI(qubits[i-1], qubits[i], qubits[i]))
            
        # Flip the least significant qubit
        circuit.append(cirq.X(qubits[0]))
        
        # Undo NOTs
        for i in range(1, num_qubits):
            circuit.append(cirq.X(qubits[i]))
        
        # Measure qubits
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # Extract result
        result_bits = result.measurements['result'][0]
        result_decimal = sum(bit * (2**i) for i, bit in enumerate(result_bits))
        
        return result_decimal
    
    def _ekanyunena_purvena_hybrid(self, x, base, context):
        """Hybrid implementation of ekanyunena_purvena"""
        # For scalar values, use quantum circuit
        if isinstance(x, (int, float)):
            return self._ekanyunena_purvena_quantum(x, base, context)
        # For small arrays, use quantum for some elements and classical for others
        elif isinstance(x, np.ndarray) and x.size <= 4:
            result = np.zeros_like(x)
            for i in range(x.size):
                result.flat[i] = self._ekanyunena_purvena_quantum(x.flat[i], base, context)
            return result
        else:
            # For larger arrays, use classical implementation
            return self._ekanyunena_purvena_classical(x, base, context)
    
    def _ekanyunena_purvena_classical(self, x, base, context):
        """Classical implementation of ekanyunena_purvena"""
        if isinstance(x, torch.Tensor):
            return x - 1
        elif isinstance(x, np.ndarray):
            return x - 1
        else:
            return x - 1

    def anurupyena(self, a: Union[float, np.ndarray, torch.Tensor],
                  b: Union[float, np.ndarray, torch.Tensor],
                  ratio: float = 0.618,
                  ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 11: Anurupyena - "Proportionality"
        
        Mathematical logic: Establishes proportional relationships between values,
        with optional use of the golden ratio (0.618) as a natural scaling factor.
        
        Classical applications:
        - Golden section search in optimization
        - Proportional scaling in transformations
        - Aesthetic proportioning in design
        - Progressive refinement in search algorithms
        
        Quantum applications:
        - Quantum amplitude re-scaling
        - Phase proportion adjustments
        - Entanglement distribution optimization
        - Golden ratio-based quantum walks
        
        Args:
            a: First value or array
            b: Second value or array
            ratio: Proportionality ratio (default: golden ratio)
            ctx: Optional execution context override
            
        Returns:
            Proportionally combined result
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            ratio_device = self._to_device(ratio)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._anurupyena_quantum(a, b, ratio, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._anurupyena_hybrid(a, b, ratio, context)
            
            # Classical implementation (default)
            if isinstance(a_device, torch.Tensor):
                result = a_device + ratio_device * (b_device - a_device)
            elif isinstance(a_device, np.ndarray):
                result = a_device + ratio * (b_device - a_device)
            else:
                result = a_device + ratio * (b_device - a_device)
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("anurupyena", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in anurupyena: {error_msg}")
            self._record_performance("anurupyena", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _anurupyena_quantum(self, a, b, ratio, context):
        """Quantum implementation of anurupyena using CUDAQ"""
        # For scalar values, implement quantum proportional mixing
        
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._anurupyena_classical(a, b, ratio, context)
        
        # Create CUDAQ kernel for proportional mixing
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(1)  # One qubit for mixing
        
        # Calculate angle for proportional mixing
        # theta = 2 * arcsin(sqrt(ratio))
        theta = 2 * np.arcsin(np.sqrt(ratio))
        
        # Apply rotation to create superposition according to ratio
        kernel.ry(q[0], theta)
        
        # Measure to collapse state
        kernel.mz(q)
        
        # Execute multiple times to get probability distribution
        result = cudaq.sample(kernel, shots=1000)
        counts = result.get_counts()
        
        # Calculate weighted average based on measurement statistics
        prob_0 = counts.get('0', 0) / 1000
        prob_1 = counts.get('1', 0) / 1000
        
        # Combine a and b according to measured probabilities
        return a * prob_0 + b * prob_1
    
    def _anurupyena_hybrid(self, a, b, ratio, context):
        """Hybrid implementation of anurupyena"""
        # For scalar values, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._anurupyena_quantum(a, b, ratio, context)
        # For small arrays, use quantum for some elements and classical for others
        elif isinstance(a, np.ndarray) and a.size <= 4:
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = a.flat[i]
                b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                result.flat[i] = self._anurupyena_quantum(a_val, b_val, ratio, context)
            return result
        else:
            # For larger arrays, use classical implementation
            return self._anurupyena_classical(a, b, ratio, context)
    
    def _anurupyena_classical(self, a, b, ratio, context):
        """Classical implementation of anurupyena"""
        return a + ratio * (b - a)

    def sunyam_samya_samuccaye(self, a: Union[float, np.ndarray, torch.Tensor],
                              b: Union[float, np.ndarray, torch.Tensor],
                              epsilon: Optional[float] = None,
                              ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 12: Sunyam Samya Samuccaye - "If one is in ratio, the other is zero"
        
        Mathematical logic: Identifies and resolves cases where values satisfy 
        specific ratio relationships, setting appropriate values to zero.
        
        Classical applications:
        - Balance equations in physical systems
        - Economic equilibrium modeling
        - Zero-sum game strategies
        - Feedback control systems with equilibrium states
        
        Quantum applications:
        - Quantum state normalization
        - Interference pattern identification
        - Phase cancellation detection
        - Quantum error syndrome diagnosis
        
        Args:
            a: First value or array
            b: Second value or array
            epsilon: Tolerance for zero detection (default from context)
            ctx: Optional execution context override
            
        Returns:
            Result with appropriate values set to zero based on ratio relationships
        """
        context = ctx or self.context
        eps = epsilon if epsilon is not None else context.epsilon
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            eps_device = self._to_device(eps)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._sunyam_samya_samuccaye_quantum(a, b, eps, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._sunyam_samya_samuccaye_hybrid(a, b, eps, context)
            
            # Classical implementation (default)
            if isinstance(a_device, torch.Tensor):
                # Calculate sum and ratio
                sum_val = a_device + b_device
                ratio_condition = torch.abs(a_device - b_device) < eps_device
                
                # Apply zero where ratio condition is met
                result = torch.where(ratio_condition, torch.zeros_like(sum_val), sum_val)
                
            elif isinstance(a_device, np.ndarray):
                # Calculate sum and ratio
                sum_val = a_device + b_device
                ratio_condition = np.abs(a_device - b_device) < eps
                
                # Apply zero where ratio condition is met
                result = np.where(ratio_condition, np.zeros_like(sum_val), sum_val)
                
            else:
                # Handle scalar case
                sum_val = a_device + b_device
                if abs(a_device - b_device) < eps:
                    result = 0
                else:
                    result = sum_val
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("sunyam_samya_samuccaye", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in sunyam_samya_samuccaye: {error_msg}")
            self._record_performance("sunyam_samya_samuccaye", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _sunyam_samya_samuccaye_quantum(self, a, b, epsilon, context):
        """Quantum implementation of sunyam_samya_samuccaye using Cirq"""
        # This implements quantum interference to detect ratio relationships
        
        # Simple implementation for scalar inputs
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._sunyam_samya_samuccaye_classical(a, b, epsilon, context)
        
        # Normalize inputs to range [0, 1] for encoding as quantum amplitudes
        max_val = max(abs(a), abs(b)) * 2
        if max_val < epsilon:
            return 0
            
        norm_a = a / max_val
        norm_b = b / max_val
        
        # Create quantum circuit with one qubit
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit()
        
        # Prepare superposition state
        circuit.append(cirq.H(q))
        
        # Apply phase rotations based on inputs
        circuit.append(cirq.ZPowGate(exponent=norm_a)(q))
        circuit.append(cirq.X(q))
        circuit.append(cirq.ZPowGate(exponent=norm_b)(q))
        circuit.append(cirq.X(q))
        
        # Apply Hadamard to observe interference
        circuit.append(cirq.H(q))
        
        # Measure
        circuit.append(cirq.measure(q, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Analyze measurements
        counts = result.histogram(key='result')
        
        # If interference leads to significant bias toward |0⟩ or |1⟩,
        # then a and b are likely in ratio
        threshold = 0.8  # Arbitrary threshold for determining ratio relationship
        total_shots = sum(counts.values())
        
        if counts.get(0, 0) / total_shots > threshold or counts.get(1, 0) / total_shots > threshold:
            # Strong interference detected, likely in ratio
            return 0
        else:
            # No strong ratio relationship detected
            return a + b
    
    def _sunyam_samya_samuccaye_hybrid(self, a, b, epsilon, context):
        """Hybrid implementation of sunyam_samya_samuccaye"""
        # For small arrays, use quantum interference checking
        if (isinstance(a, np.ndarray) and a.size <= 4) or isinstance(a, (int, float)):
            if isinstance(a, np.ndarray):
                # Process each element through quantum check
                result = np.zeros_like(a)
                for i in range(a.size):
                    result.flat[i] = self._sunyam_samya_samuccaye_quantum(
                        a.flat[i], b.flat[i] if isinstance(b, np.ndarray) else b, epsilon, context
                    )
                return result
            else:
                # Single value case
                return self._sunyam_samya_samuccaye_quantum(a, b, epsilon, context)
        else:
            # For larger arrays, use classical implementation
            return self._sunyam_samya_samuccaye_classical(a, b, epsilon, context)
    
    def _sunyam_samya_samuccaye_classical(self, a, b, epsilon, context):
        """Classical implementation of sunyam_samya_samuccaye"""
        # Calculate sum
        sum_val = a + b
        
        # Check if a and b are in ratio (approximately equal)
        if isinstance(sum_val, np.ndarray):
            ratio_condition = np.abs(a - b) < epsilon
            return np.where(ratio_condition, np.zeros_like(sum_val), sum_val)
        elif isinstance(sum_val, torch.Tensor):
            ratio_condition = torch.abs(a - b) < epsilon
            return torch.where(ratio_condition, torch.zeros_like(sum_val), sum_val)
        else:
            if abs(a - b) < epsilon:
                return 0
            else:
                return sum_val

    def gunitasamuccayah(self, multiplicand: Union[float, np.ndarray, torch.Tensor],
                        multiplier: Union[float, np.ndarray, torch.Tensor],
                        ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 13: Gunitasamuccayah - "The product of the sum is equal to the sum of the products"
        
        Mathematical logic: Implements distributive property in multiplication,
        with optimizations for parallel computation and numerical stability.
        
        Classical applications:
        - Optimized polynomial multiplication
        - Distributed computation of products
        - Matrix multiplication algorithms
        - Statistical moment calculations
        
        Quantum applications:
        - Quantum multiplier circuits
        - Superposition-based parallel multiplication
        - Quantum polynomial evaluation
        - State preparation for quantum machine learning
        
        Args:
            multiplicand: First value or array to multiply
            multiplier: Second value or array to multiply
            ctx: Optional execution context override
            
        Returns:
            Product using distributive optimizations
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(multiplicand)
        data_size = np.size(multiplicand) if hasattr(multiplicand, 'size') else 1
        
        try:
            # Convert to device if using GPU
            multiplicand_device = self._to_device(multiplicand)
            multiplier_device = self._to_device(multiplier)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._gunitasamuccayah_quantum(multiplicand, multiplier, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._gunitasamuccayah_hybrid(multiplicand, multiplier, context)
            
            # Classical implementation (default)
            if isinstance(multiplicand_device, torch.Tensor):
                result = multiplicand_device * multiplier_device
            elif isinstance(multiplicand_device, np.ndarray):
                result = multiplicand_device * multiplier_device
            else:
                result = multiplicand_device * multiplier_device
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("gunitasamuccayah", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in gunitasamuccayah: {error_msg}")
            self._record_performance("gunitasamuccayah", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _gunitasamuccayah_quantum(self, multiplicand, multiplier, context):
        """Quantum implementation of gunitasamuccayah using CUDAQ"""
        # This implements a quantum multiplier circuit
        
        # For scalar values, implement quantum multiplier
        if not isinstance(multiplicand, (int, float)) or not isinstance(multiplier, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._gunitasamuccayah_classical(multiplicand, multiplier, context)
        
        # Limit to small integers for quantum implementation
        if abs(multiplicand) > 8 or abs(multiplier) > 8:
            return self._gunitasamuccayah_classical(multiplicand, multiplier, context)
        
        # Create CUDAQ kernel for multiplication
        kernel = cudaq.make_kernel()
        a = kernel.qalloc(3)  # 3 qubits for first number (up to 8)
        b = kernel.qalloc(3)  # 3 qubits for second number (up to 8)
        result_reg = kernel.qalloc(6)  # 6 qubits for result (up to 64)
        
        # Encode multiplicand into a register
        a_int = int(abs(multiplicand))
        a_binary = bin(a_int)[2:].zfill(3)
        for i, bit in enumerate(reversed(a_binary)):
            if bit == '1':
                kernel.x(a[i])
        
        # Encode multiplier into b register
        b_int = int(abs(multiplier))
        b_binary = bin(b_int)[2:].zfill(3)
        for i, bit in enumerate(reversed(b_binary)):
            if bit == '1':
                kernel.x(b[i])
        
        # Implement quantum multiplication
        # For each bit in a, conditionally add shifted b to result
        for i in range(3):
            # If a[i] is 1, add b << i to result
            for j in range(3):
                # Controlled addition of b[j] to result[i+j]
                kernel.cx(a[i], result_reg[i+j])
                kernel.cx(b[j], result_reg[i+j]).controlled_by(a[i])
        
        # Measure result register
        kernel.mz(result_reg)
        
        # Execute
        result = cudaq.sample(kernel)
        
        # Get most frequent outcome
        top_result = result.most_probable()
        
        # Convert to decimal
        result_decimal = int(top_result, 2)
        
        # Apply sign based on input signs
        if (multiplicand < 0 and multiplier > 0) or (multiplicand > 0 and multiplier < 0):
            result_decimal = -result_decimal
            
        return result_decimal
    
    def _gunitasamuccayah_hybrid(self, multiplicand, multiplier, context):
        """Hybrid implementation of gunitasamuccayah"""
        # For small scalar values, use quantum circuit
        if (isinstance(multiplicand, (int, float)) and isinstance(multiplier, (int, float)) and
           abs(multiplicand) <= 8 and abs(multiplier) <= 8):
            return self._gunitasamuccayah_quantum(multiplicand, multiplier, context)
        # For small arrays with small values, use quantum for element-wise multiplication
        elif (isinstance(multiplicand, np.ndarray) and multiplicand.size <= 4 and
             np.all(np.abs(multiplicand) <= 8) and np.all(np.abs(multiplier) <= 8)):
            result = np.zeros_like(multiplicand)
            for i in range(multiplicand.size):
                a_val = multiplicand.flat[i]
                b_val = multiplier.flat[i] if isinstance(multiplier, np.ndarray) else multiplier
                result.flat[i] = self._gunitasamuccayah_quantum(a_val, b_val, context)
            return result
        else:
            # For larger or more complex cases, use classical implementation
            return self._gunitasamuccayah_classical(multiplicand, multiplier, context)
    
    def _gunitasamuccayah_classical(self, multiplicand, multiplier, context):
        """Classical implementation of gunitasamuccayah"""
        return multiplicand * multiplier

    def yavadunam(self, x: Union[float, np.ndarray, torch.Tensor],
                 base: float = 10.0,
                 ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 14: Yavadunam - "Whatever the extent of its deficiency"
        
        Mathematical logic: Calculates complement or deficiency with respect to a base,
        with applications in number representation and computational optimization.
        
        Classical applications:
        - One's or two's complement arithmetic
        - Deficit-based optimization algorithms
        - Gap analysis in numerical sequences
        - Numerical representation transformations
        
        Quantum applications:
        - Quantum state inversion operations
        - Phase complement calculations
        - Quantum error detection via deficiency measures
        - Entanglement deficit quantification
        
        Args:
            x: Input value or array
            base: Base value for deficiency calculation
            ctx: Optional execution context override
            
        Returns:
            Deficiency of x with respect to base
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(x)
        data_size = np.size(x) if hasattr(x, 'size') else 1
        
        try:
            # Convert to device if using GPU
            x_device = self._to_device(x)
            base_device = self._to_device(base)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._yavadunam_quantum(x, base, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._yavadunam_hybrid(x, base, context)
            
            # Classical implementation (default)
            if isinstance(x_device, torch.Tensor):
                result = base_device - x_device
            elif isinstance(x_device, np.ndarray):
                result = base - x_device
            else:
                result = base - x_device
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("yavadunam", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in yavadunam: {error_msg}")
            self._record_performance("yavadunam", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _yavadunam_quantum(self, x, base, context):
        """Quantum implementation of yavadunam using Cirq"""
        # This implements a quantum circuit for complement calculation
        
        # For scalar values, implement quantum complementation
        if not isinstance(x, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._yavadunam_classical(x, base, context)
        
        # Determine bit width needed for the operation
        max_val = max(x, base) * 2
        num_qubits = max(1, int(np.ceil(np.log2(max_val + 1))))
        
        # Create quantum circuit
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
        circuit = cirq.Circuit()
        
        # Encode the value x
        x_int = int(x)
        x_binary = bin(x_int)[2:].zfill(num_qubits)
        for i, bit in enumerate(reversed(x_binary)):
            if bit == '1':
                circuit.append(cirq.X(qubits[i]))
        
        # Apply X gates to all qubits to compute one's complement
        for q in qubits:
            circuit.append(cirq.X(q))
        
        # Add base value to the complemented value
        # This is done by adding base-1 to the one's complement
        base_minus_one = int(base) - 1
        base_binary = bin(base_minus_one)[2:].zfill(num_qubits)
        
        # Implement quantum adder for base-1
        carry = cirq.LineQubit(num_qubits)
        circuit.append(cirq.X(carry))  # Initialize carry
        
        for i, bit in enumerate(reversed(base_binary)):
            if bit == '1':
                circuit.append(cirq.CNOT(carry, qubits[i]))
            
            # Propagate carry
            if i < num_qubits - 1:
                circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
                circuit.append(cirq.CNOT(carry, qubits[i+1]))
        
        # Measure result
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # Extract result
        result_bits = result.measurements['result'][0]
        result_decimal = sum(bit * (2**i) for i, bit in enumerate(result_bits))
        
        return result_decimal
    
    def _yavadunam_hybrid(self, x, base, context):
        """Hybrid implementation of yavadunam"""
        # For scalar values, use quantum circuit
        if isinstance(x, (int, float)):
            return self._yavadunam_quantum(x, base, context)
        # For small arrays, use quantum for some elements and classical for others
        elif isinstance(x, np.ndarray) and x.size <= 4:
            result = np.zeros_like(x)
            for i in range(x.size):
                result.flat[i] = self._yavadunam_quantum(x.flat[i], base, context)
            return result
        else:
            # For larger arrays, use classical implementation
            return self._yavadunam_classical(x, base, context)
    
    def _yavadunam_classical(self, x, base, context):
        """Classical implementation of yavadunam"""
        return base - x

    def samuccayagunitah(
        self,
        a: Union[float, np.ndarray, torch.Tensor],
        b: Union[float, np.ndarray, torch.Tensor],
        operation: str = 'product_sum',
        ctx: Optional[SutraContext] = None,
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 15: Samuccayagunitah - "The product of the sum is equal to the sum of the products"
        
        Mathematical logic: Implements algebraic distributive property, with applications
        in polynomial multiplication and algebraic transformations.
        
        Classical applications:
        - Algebraic expansion simplification
        - Polynomial multiplication optimization
        - Statistical product-moment calculations
        - Parallel computation of aggregate products
        
        Quantum applications:
        - Quantum state superposition preparation
        - Entangled state analysis
        - Quantum circuit optimization
        - Quantum polynomial state encoding
        
        Args:
            a: First value or array
            b: Second value or array
            operation: Type of operation ('product_sum' or 'sum_product')
            ctx: Optional execution context override
            
        Returns:
            Result of distributive operation
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._samuccayagunitah_quantum(a, b, operation, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._samuccayagunitah_hybrid(a, b, operation, context)
            
            # Classical implementation (default)
            if operation == 'product_sum':
                # (a + b) * (a + b) = a*a + a*b + b*a + b*b
                if isinstance(a_device, torch.Tensor):
                    sum_ab = a_device + b_device
                    result = sum_ab * sum_ab
                elif isinstance(a_device, np.ndarray):
                    sum_ab = a_device + b_device
                    result = sum_ab * sum_ab
                else:
                    sum_ab = a_device + b_device
                    result = sum_ab * sum_ab
            elif operation == 'sum_product':
                # a*a + b*b = (a + b)*(a + b) - 2*a*b
                if isinstance(a_device, torch.Tensor):
                    result = a_device * a_device + b_device * b_device
                elif isinstance(a_device, np.ndarray):
                    result = a_device * a_device + b_device * b_device
                else:
                    result = a_device * a_device + b_device * b_device
            else:
                raise ValueError(f"Unknown operation: {operation}. Use 'product_sum' or 'sum_product'.")
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("samuccayagunitah", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in samuccayagunitah: {error_msg}")
            self._record_performance("samuccayagunitah", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _samuccayagunitah_quantum(self, a, b, operation, context):
        """Quantum implementation of samuccayagunitah using Cirq"""
        # This implements a quantum circuit for distributive property
        
        # For scalar values, implement quantum circuit
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._samuccayagunitah_classical(a, b, operation, context)
        
        # Normalize inputs to range [0, 1] for encoding as quantum amplitudes
        max_val = max(abs(a), abs(b)) * 2
        if max_val < context.epsilon:
            return 0
            
        norm_a = a / max_val
        norm_b = b / max_val
        
        if operation == 'product_sum':
            # Create CUDAQ kernel for product of sum
            kernel = cudaq.make_kernel()
            q = kernel.qalloc(2)  # Two qubits for the two values
            
            # Encode a and b as rotation angles
            theta_a = 2 * np.arcsin(np.sqrt(abs(norm_a)))
            theta_b = 2 * np.arcsin(np.sqrt(abs(norm_b)))
            
            # Apply rotations to create superposition
            kernel.ry(q[0], theta_a)
            kernel.ry(q[1], theta_b)
            
            # Create entanglement to model multiplication
            kernel.cx(q[0], q[1])
            
            # Apply phase kickback based on sign
            if a < 0:
                kernel.z(q[0])
            if b < 0:
                kernel.z(q[1])
            
            # Measure
            kernel.mz(q)
            
            # Execute
            result = cudaq.sample(kernel)
            
            # Get measurement probabilities
            counts = result.get_counts()
            
            # Calculate weighted result
            weighted_sum = 0
            for bitstring, count in counts.items():
                value = int(bitstring, 2)
                weighted_sum += value * (count / 1000)
            
            # Scale back to original range
            return weighted_sum * max_val * max_val
            
        elif operation == 'sum_product':
            # Create Cirq circuit for sum of products
            qubits = [cirq.LineQubit(i) for i in range(2)]
            circuit = cirq.Circuit()
            
            # Encode a and b as rotation angles
            theta_a = 2 * np.arcsin(np.sqrt(abs(norm_a)))
            theta_b = 2 * np.arcsin(np.sqrt(abs(norm_b)))
            
            # Apply rotations to create superposition
            circuit.append(cirq.ry(theta_a)(qubits[0]))
            circuit.append(cirq.ry(theta_b)(qubits[1]))
            
            # Apply phase kickback based on sign
            if a < 0:
                circuit.append(cirq.Z(qubits[0]))
            if b < 0:
                circuit.append(cirq.Z(qubits[1]))
            
            # Measure
            circuit.append(cirq.measure(*qubits, key='result'))
            
            # Simulate
            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1000)
            
            # Get measurement probabilities
            counts = result.histogram(key='result')
            
            # Calculate a*a + b*b based on measurement statistics
            weighted_sum = 0
            total_counts = sum(counts.values())
            
            # Weights for different outcomes:
            # |00⟩: no contribution
            # |01⟩: b*b
            # |10⟩: a*a
            # |11⟩: a*a + b*b + 2*a*b
            weighted_sum += counts.get(1, 0) * (b**2) / total_counts
            weighted_sum += counts.get(2, 0) * (a**2) / total_counts
            weighted_sum += counts.get(3, 0) * ((a+b)**2) / total_counts
            
            return weighted_sum * max_val * max_val
        
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'product_sum' or 'sum_product'.")
    
    def _samuccayagunitah_hybrid(self, a, b, operation, context):
        """Hybrid implementation of samuccayagunitah"""
        # For scalar values, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._samuccayagunitah_quantum(a, b, operation, context)
        # For small arrays, use quantum for some elements and classical for others
        elif isinstance(a, np.ndarray) and a.size <= 4:
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = a.flat[i]
                b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                result.flat[i] = self._samuccayagunitah_quantum(
                    a_val, b_val, operation, context
                )
            return result
        else:
            # For larger arrays, use classical implementation
            return self._samuccayagunitah_classical(a, b, operation, context)
    
    def _samuccayagunitah_classical(self, a, b, operation, context):
        """Classical implementation of samuccayagunitah"""
        if operation == 'product_sum':
            sum_ab = a + b
            return sum_ab * sum_ab
        elif operation == 'sum_product':
            return a * a + b * b
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'product_sum' or 'sum_product'.")

    def gunakasamuccayah(self, a: Union[float, np.ndarray, torch.Tensor],
                        b: Union[float, np.ndarray, torch.Tensor],
                        ctx: Optional[SutraContext] = None) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Sutra 16: Gunakasamuccayah - "The factors of the sum are equal to the sum of the factors"
        
        Mathematical logic: Provides factorization techniques for algebraic expressions,
        with applications in equation solving and algebraic manipulation.
        
        Classical applications:
        - Polynomial factorization
        - Algebraic simplification
        - Solving quadratic and cubic equations
        - Number theory factor decomposition
        
        Quantum applications:
        - Quantum factoring algorithms
        - Entanglement decomposition
        - Quantum circuit factorization
        - Quantum error correction syndrome factorization
        
        Args:
            a: First value or array
            b: Second value or array
            ctx: Optional execution context override
            
        Returns:
            Result of factorization operation
        """
        context = ctx or self.context
        start_time = time.time()
        original_type = type(a)
        data_size = np.size(a) if hasattr(a, 'size') else 1
        
        try:
            # Convert to device if using GPU
            a_device = self._to_device(a)
            b_device = self._to_device(b)
            
            # Quantum implementation
            if context.mode == SutraMode.QUANTUM:
                return self._gunakasamuccayah_quantum(a, b, context)
            
            # Hybrid implementation
            elif context.mode == SutraMode.HYBRID:
                return self._gunakasamuccayah_hybrid(a, b, context)
            
            # Classical implementation (default)
            # For this sutra, we're factoring a^2 - b^2 = (a+b)(a-b)
            if isinstance(a_device, torch.Tensor):
                result = (a_device + b_device) * (a_device - b_device)
            elif isinstance(a_device, np.ndarray):
                result = (a_device + b_device) * (a_device - b_device)
            else:
                result = (a_device + b_device) * (a_device - b_device)
            
            # Convert back to original type
            result = self._from_device(result, original_type)
            
            end_time = time.time()
            self._record_performance("gunakasamuccayah", start_time, end_time, 
                                    True, data_size)
            return result
            
        except Exception as e:
            end_time = time.time()
            error_msg = str(e)
            logger.error(f"Error in gunakasamuccayah: {error_msg}")
            self._record_performance("gunakasamuccayah", start_time, end_time, 
                                   False, data_size, error_msg)
            raise
    
    def _gunakasamuccayah_quantum(self, a, b, context):
        """Quantum implementation of gunakasamuccayah using Cirq"""
        # This implements a quantum circuit for factorization
        
        # For scalar values, implement quantum circuit
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            # Fall back to classical for non-scalar inputs
            return self._gunakasamuccayah_classical(a, b, context)
        
        # Create Cirq circuit for factorization
        qubits = [cirq.LineQubit(i) for i in range(2)]
        circuit = cirq.Circuit()
        
        # Encode a and b using rotation gates
        # We use the fact that a^2 - b^2 = (a+b)(a-b)
        
        # Normalize inputs to prevent overflow
        max_val = max(abs(a), abs(b)) * 2
        if max_val < context.epsilon:
            return 0
            
        norm_a = a / max_val
        norm_b = b / max_val
        
        # Calculate rotation angles for (a+b) and (a-b)
        theta_sum = np.arcsin(min(1.0, abs(norm_a + norm_b)))
        theta_diff = np.arcsin(min(1.0, abs(norm_a - norm_b)))
        
        # Apply rotations to create superposition
        circuit.append(cirq.ry(2 * theta_sum)(qubits[0]))
        circuit.append(cirq.ry(2 * theta_diff)(qubits[1]))
        
        # Create entanglement to model multiplication
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        
        # Apply phase kickback based on sign
        if (norm_a + norm_b) < 0:
            circuit.append(cirq.Z(qubits[0]))
        if (norm_a - norm_b) < 0:
            circuit.append(cirq.Z(qubits[1]))
        
        # Measure
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        
        # Analyze measurements
        counts = result.histogram(key='result')
        
        # Calculate result based on measurement statistics
        # The probability of measuring |11⟩ corresponds to the product (a+b)(a-b)
        p_11 = counts.get(3, 0) / 1000
        
        # Scale back to original range
        return p_11 * (max_val ** 2)
    
    def _gunakasamuccayah_hybrid(self, a, b, context):
        """Hybrid implementation of gunakasamuccayah"""
        # For scalar values, use quantum circuit
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return self._gunakasamuccayah_quantum(a, b, context)
        # For small arrays, use quantum for some elements and classical for others
        elif isinstance(a, np.ndarray) and a.size <= 4:
            result = np.zeros_like(a)
            for i in range(a.size):
                a_val = a.flat[i]
                b_val = b.flat[i] if isinstance(b, np.ndarray) else b
                result.flat[i] = self._gunakasamuccayah_quantum(a_val, b_val, context)
            return result
        else:
            # For larger arrays, use classical implementation
            return self._gunakasamuccayah_classical(a, b, context)
    
    def _gunakasamuccayah_classical(self, a, b, context):
        """Classical implementation of gunakasamuccayah"""
        return (a + b) * (a - b)
