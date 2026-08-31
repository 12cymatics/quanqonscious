from typing import Any
import numpy as np
import cirq
import cudaq
import torch
import matplotlib.pyplot as plt
import scipy.linalg as la

import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from fractions import Fraction
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import sympy as sp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VedicSutras")


def _ripple_increment(circuit, qubits) -> None:
    """Append |v> -> |v + 1 mod 2^n| to `circuit`.

    Bit i flips exactly when every lower bit is set, so the flip is controlled
    by all of them; and the sweep runs high to low so each control is read
    before it is modified. Both details were wrong in the hand-written copies
    this replaces: they used `CNOT(q[i], q[i+1])` -- a carry from one bit
    rather than from all lower bits -- and swept low to high.
    """
    for i in range(len(qubits) - 1, 0, -1):
        circuit.append(cirq.X(qubits[i]).controlled_by(*qubits[:i]))
    circuit.append(cirq.X(qubits[0]))


def _ripple_decrement(circuit, qubits) -> None:
    """Append |v> -> |v - 1 mod 2^n| to `circuit`.

    Decrement is increment conjugated by a full bitwise complement. The
    complement must be applied to EVERY qubit on both sides; one copy of this
    in the file complemented all n and then un-complemented only qubits
    1..n-1, leaving the low bit inverted.
    """
    for q in qubits:
        circuit.append(cirq.X(q))
    _ripple_increment(circuit, qubits)
    for q in qubits:
        circuit.append(cirq.X(q))

def _ripple_add_constant(circuit, qubits, addend: int) -> None:
    """Append |v> -> |v + addend mod 2^n| for a non-negative classical addend.

    Adding 2^j to a little-endian register is exactly one increment of the
    sub-register starting at bit j, because `qubits[j:]` holds floor(v / 2^j).
    So a classical addend is applied one set bit at a time, and every carry is
    a real ripple carry through `_ripple_increment`. Nothing is approximated
    and no bit is dropped inside the register.
    """
    for j in range(len(qubits)):
        if (addend >> j) & 1:
            _ripple_increment(circuit, qubits[j:])


def _ripple_sub_constant(circuit, qubits, subtrahend: int) -> None:
    """Append |v> -> |v - subtrahend mod 2^n|, the mirror of the above."""
    for j in range(len(qubits)):
        if (subtrahend >> j) & 1:
            _ripple_decrement(circuit, qubits[j:])


# A state vector is 2**n amplitudes, so simulating the register costs memory
# exponential in its width: 24 qubits is 0.12 GiB, 31 qubits is 16 GiB. This is
# a property of simulating the circuit, not of the arithmetic.
#
# Past this width the honest options are to compute the value exactly or to
# refuse; they do not include quietly computing it a different way. So
# `_exact_via_circuit` RAISES here rather than falling through to the classical
# body. A caller that wants the answer for operands this large should ask for
# CLASSICAL mode, which is a decision the caller makes and can see, not one
# this function makes silently on their behalf.
_MAX_SIMULABLE_QUBITS = 24


def _exact_via_circuit(num_qubits: int, initial: int, ops) -> int:
    """Run a sequence of exact register operations and measure the result.

    `ops` is a sequence of ('add', k) / ('sub', k) pairs with classical k.
    The register is sized by the caller to hold every intermediate value, so
    nothing wraps: this returns the exact integer, not a residue.
    """
    if num_qubits > _MAX_SIMULABLE_QUBITS:
        raise ArithmeticError(
            f"register needs {num_qubits} qubits; simulating it would take "
            f"{8 * (2 ** num_qubits) / 2 ** 30:.1f} GiB of state vector, above "
            f"the {_MAX_SIMULABLE_QUBITS}-qubit limit. The value is computable "
            f"-- ask for SutraMode.CLASSICAL, which evaluates it directly. "
            f"This refuses rather than substituting the classical answer, so "
            f"the choice of algorithm stays with the caller."
        )
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = cirq.Circuit()
    for j in range(num_qubits):
        if (initial >> j) & 1:
            circuit.append(cirq.X(qubits[j]))
    for op, operand in ops:
        if op == 'add':
            _ripple_add_constant(circuit, qubits, operand)
        elif op == 'sub':
            _ripple_sub_constant(circuit, qubits, operand)
        else:
            raise ValueError(f"unknown register operation {op!r}")
    circuit.append(cirq.measure(*qubits, key='r'))
    bits = cirq.Simulator().run(circuit, repetitions=1).measurements['r'][0]
    return sum(int(b) * (2 ** i) for i, b in enumerate(bits))


def _width_for(magnitude: int) -> int:
    """Register width that holds `magnitude` without wrapping."""
    return max(1, int(magnitude).bit_length())


def _integral_scalars(*values) -> bool:
    """True when every value has an exact register encoding.

    The circuits below encode integers into a fixed-width register. Arrays,
    tensors and non-integral reals have no such encoding, so the callers hand
    those to the classical body instead. That is a domain guard, not a
    fallback: the circuit is undefined on those inputs, it is not merely
    slower or less convenient.
    """
    for value in values:
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return False
        if float(value) != int(value):
            return False
    return True


def _quantum_sum(a: int, b: int) -> int:
    """a + b, by ripple carry. Exact for any signed integer pair."""
    if a + b < 0:
        return -_quantum_sum(-a, -b)
    width = _width_for(abs(a) + abs(b)) + 1
    if a >= 0 and b >= 0:
        return _exact_via_circuit(width, a, [('add', b)])
    if a < 0:
        a, b = b, a
    return _exact_via_circuit(width, a, [('sub', -b)])


def _quantum_product(a: int, b: int) -> int:
    """a * b as a shift-add over the multiplier's set bits.

    For each set bit j of |b| the circuit adds |a| << j, which is the standard
    shift-and-add multiplier; the carries are real. The sign is applied
    afterwards, because the register is unsigned -- that is arithmetic on the
    result, not an approximation of it.
    """
    sign = -1 if (a < 0) != (b < 0) else 1
    a_mag, b_mag = abs(int(a)), abs(int(b))
    if a_mag == 0 or b_mag == 0:
        return 0
    width = _width_for(a_mag * b_mag) + 1
    ops = [('add', a_mag << j) for j in range(b_mag.bit_length()) if (b_mag >> j) & 1]
    return sign * _exact_via_circuit(width, 0, ops)


def _quantum_divmod(n: int, d: int):
    """Exact integer quotient and remainder, with the arithmetic in the register.

    Binary long division. The *schedule* -- which shifted subtractions happen --
    is classical because both operands are, exactly as the multiplier's set bits
    are in `_quantum_product`; the arithmetic itself is done by the verified
    ripple primitives, and the invariant `n = q*d + r` with `0 <= r < |d|` is
    checked against them rather than assumed.

    This replaces a quantum phase estimation reciprocal that ran the answer
    through an 8-qubit register, quantising every result to a multiple of
    1/256. That is an approximation, and an approximation is the one thing this
    file is not allowed to contain.
    """
    if d == 0:
        raise ZeroDivisionError("paravartya_yojayet: division by zero has no quotient")

    a, b = abs(int(n)), abs(int(d))
    quotient_bits, remainder = [], 0
    for i in reversed(range(max(1, a.bit_length()))):
        remainder = 2 * remainder + ((a >> i) & 1)
        if remainder >= b:
            remainder -= b
            quotient_bits.append(i)

    q_mag = (_exact_via_circuit(_width_for(a) + 1, 0,
                                [('add', 1 << i) for i in quotient_bits])
             if quotient_bits else 0)
    r_mag = _quantum_sum(a, -_quantum_product(q_mag, b))
    if not 0 <= r_mag < b:
        raise ArithmeticError(
            f"divmod invariant violated: {a} = {q_mag}*{b} + {r_mag}")
    return q_mag, r_mag


def _quantum_polynomial(coefficients, x: int) -> int:
    """Sum(c_i * x**i) accumulated in one register.

    Every term is a classical constant, so each is added by ripple carry into
    a register wide enough for the running total. Negative coefficients are
    subtracted rather than added; the accumulator is offset by the total of
    the negative terms so the running value never goes below zero, since the
    register is unsigned.
    """
    terms = [int(c) * (int(x) ** i) for i, c in enumerate(coefficients)]
    offset = sum(-t for t in terms if t < 0)
    width = _width_for(sum(abs(t) for t in terms) + offset) + 1
    ops = [('add', t) if t >= 0 else ('sub', -t) for t in terms]
    return _exact_via_circuit(width, offset, ops) - offset


def _complement_via_circuit(x_int: int, base_int: int) -> int:
    """`base - x` computed as a Cirq circuit, exactly.

    Shared by the quantum paths of nikhilam and yavadunam, whose classical
    implementations are both literally `base - x`: nikhilam takes the
    complement to a base, yavadunam the deficiency from one. They had two
    separate hand-written circuits, and yavadunam's propagated carries with
    bare CNOTs -- `CNOT(q[i], q[i+1])` flips the next bit whenever this one is
    set, rather than when every lower bit is set -- so it returned 0 where the
    sutra gives 2.

    Construction: X on every qubit of an n-bit register takes |x> to the one's
    complement |(2^n - 1) - x>; d ripple decrements then land on `base - x`,
    where d = 2^n - 1 - base is non-negative by the choice of n. Verified
    exact against `base - x` for every x in range for bases 10 and 100.
    """
    num_qubits = max(1, int(np.ceil(np.log2(base_int + 1))))
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    circuit = cirq.Circuit()

    for i, bit in enumerate(reversed(bin(x_int)[2:].zfill(num_qubits))):
        if bit == '1':
            circuit.append(cirq.X(qubits[i]))

    for q in qubits:
        circuit.append(cirq.X(q))

    for _ in range(2 ** num_qubits - 1 - base_int):
        _ripple_decrement(circuit, qubits)

    circuit.append(cirq.measure(*qubits, key='complement'))
    bits = cirq.Simulator().run(circuit, repetitions=1).measurements['complement'][0]
    return sum(int(b) * (2 ** i) for i, b in enumerate(bits))

def _enforce_heavy_dependencies() -> None:
    required = {
        "numpy": np,
        "pandas": pd,
        "sympy": sp,
        "cirq": cirq,
        "cudaq": cudaq,
        "torch": torch,
        "matplotlib.pyplot": plt,
        "scipy.linalg": la,
    }
    missing = [name for name, module in required.items() if not isinstance(module, ModuleType)]
    if missing:
        deps = ", ".join(missing)
        raise ImportError(
            f"Missing required heavy dependencies for VedicSutras runtime: {deps}. "
            "Install and configure all dependencies before executing simulations."
        )


_enforce_heavy_dependencies()

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
    Implementation of the 16 primary Vedic sutras, with full mathematical
    logic, quantum integration, and inter-sutra interactions.

    Sixteen, not twenty-nine. This line read "all 29 Vedic sutras (16
    primary + 13 sub-sutras)"; the class defines 16 methods and no
    sub-sutras, and `tests/test_primarysutra_modes.py` pins that surface.
    The 29 live in `vedic_trainer/vedic/kernel/sutras_canonical.py`.
    """
    
    def __init__(self, context: Optional[SutraContext] = None):
        """
        Initialize the Vedic Sutras system with the specified context.
        
        Args:
            context: Configuration context for sutra execution
        """
        self.context = context if context else SutraContext()
        
        # Initialize GPU if requested and available
        if self.context.use_gpu and torch.cuda.is_available():
            self.context.device = torch.device("cuda")
            logger.info(
                f"Using GPU device: {torch.cuda.get_device_name(0)}"
            )
        else:
            self.context.use_gpu = False
            self.context.device = torch.device("cpu")
            logger.info("Using CPU for computations")
            
        # Initialize quantum backend if in quantum or hybrid mode
        if self.context.mode in [SutraMode.QUANTUM, SutraMode.HYBRID]:
            if self.context.quantum_backend is None:
                # Default to the CUDA-Q simulator target.
                #
                # This read `cudaq.get_platform()`, which does not exist in
                # CUDA-Q and never has -- the accessor is `get_target()`, and
                # `Target.name` is an attribute, not a method. So constructing
                # `VedicSutras(SutraContext(mode=SutraMode.QUANTUM))` raised
                # AttributeError before any sutra ran. It went unseen because
                # the branch needs `quantum_backend is None`, and every test
                # and example built the engine in the default CLASSICAL mode
                # and passed the mode per call instead.
                self.quantum_platform = cudaq.get_target()
                logger.info(
                    f"Using CUDAQ target: {self.quantum_platform.name}"
                )
            else:
                self.quantum_platform = self.context.quantum_backend
        else:
            self.quantum_platform = None
        
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
        
        # Perform incrementation.
        #
        # This was a CNOT cascade -- X on the low qubit, then CNOT(q[i], q[i+1])
        # up the register -- which is not an increment: a carry into bit i+1
        # happens only when every lower bit is set, so the flip must be
        # controlled by all of them, not by the single bit below.
        for _ in range(iterations):
            _ripple_increment(circuit, qubits)

        # Measure and read the register back.
        #
        # The extraction was
        #     result_bits = [int(result.final_state_vector[i] != 0)
        #                    for i in range(2**num_qubits)]
        # which walks the 2**n amplitudes of the STATE VECTOR and treats each
        # as a bit of the answer. For x=7, iterations=2 that returned 256
        # where the sutra gives 9. Measuring the register reads the n qubits
        # that actually hold the value, in a defined order.
        circuit.append(cirq.measure(*qubits, key='ekadhikena'))
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        bits = result.measurements['ekadhikena'][0]

        return sum(int(b) * (2 ** i) for i, b in enumerate(bits))

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

    def _nikhilam_classical(self, x, base_value, context):
        """Classical implementation of nikhilam: the complement `base - x`."""
        return base_value - x

    def _nikhilam_quantum(self, x, base_value, context):
        """Quantum implementation of nikhilam using Cirq.

        `nikhilam_navatashcaramam_dashatah` dispatched to this method and to
        `_nikhilam_hybrid` in QUANTUM and HYBRID mode, and NEITHER EXISTED --
        both raised `AttributeError` on every call. Sutra 2 had no working
        quantum path at all.

        The construction is the one the sutra's own docstring names under
        "Quantum applications: quantum state inversion (X gates)". For an
        n-qubit register, applying X to every qubit takes |x> to the one's
        complement |(2^n - 1) - x>. The requested base is generally smaller
        than 2^n, so the remainder is removed by d applications of a ripple
        decrementer, where d = 2^n - 1 - base is non-negative by the choice of
        n. The result is exactly `base - x`.

        Falls through to the classical complement for arrays and for
        non-integral or out-of-range inputs, which have no register encoding.
        """
        if not isinstance(x, (int, float, np.integer, np.floating)):
            return self._nikhilam_classical(x, base_value, context)
        if float(x) != int(x) or float(base_value) != int(base_value):
            return self._nikhilam_classical(x, base_value, context)

        xi, base_i = int(x), int(base_value)
        if xi < 0 or base_i <= 0 or xi > base_i:
            return self._nikhilam_classical(x, base_value, context)

        return _complement_via_circuit(xi, base_i)

    def _nikhilam_hybrid(self, x, base_value, context):
        """Hybrid implementation of nikhilam.

        Delegates to the quantum path, which already refuses every input it
        has no register encoding for (non-scalar, non-integral, out of range)
        and complements classically there instead.

        An earlier version of this method carried a `hybrid_base_limit = 1024`
        above which it called the classical complement directly. That was a
        silent fallback: it swapped the algorithm on the magnitude of the base
        with nothing logged and nothing returned to say so, and it changed the
        return type across the boundary -- `int` from the circuit below 1024,
        `float` from the classical body above it. It was also on this path
        only; `_yavadunam_hybrid`, which drives the same circuit through the
        same `_complement_via_circuit`, never had it. The cap is gone: wide
        bases cost circuit time, which is the honest price of computing them.
        """
        return self._nikhilam_quantum(x, base_value, context)

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
        """x / divisor, exactly: integer quotient and remainder, recombined.

        Division is the one sutra here whose true answer is generally not an
        integer, so a fixed-width register cannot hold it. It does not follow
        that the result must be approximated. `_quantum_divmod` computes q and
        r exactly in the register, and `x/d = q + r/d` is then an exact
        rational -- converted to float only at the return boundary, to match
        the type the classical body returns, and that conversion is correctly
        rounded so it agrees with `x / d` bit for bit.

        What this replaces was an approximation twice over: an 8-qubit phase
        estimation reciprocal, quantising every answer to a multiple of 1/256,
        read out through a hand-rolled inverse QFT that recovered the phase in
        only 4 of 16 cases. It returned 1.5 or 4.5 at random where 12/4 is 3.
        """
        if not _integral_scalars(x, divisor) or int(divisor) == 0:
            return self._paravartya_yojayet_classical(x, divisor, context)
        n, d = int(x), int(divisor)
        q, r = _quantum_divmod(n, d)
        magnitude = Fraction(q) + Fraction(r, abs(d))
        exact = -magnitude if (n < 0) != (d < 0) else magnitude
        return float(exact)
    def _paravartya_yojayet_hybrid(self, x, divisor, context):
        """Delegates to the quantum path, which guards its own domain.

        This used to compute a reciprocal by quantum phase estimation and then
        multiply classically. That is where HYBRID\'s 0.046875 and 11.95 for
        12 / 4 came from: the reciprocal was quantised to a multiple of 1/256
        and read out through an inverse QFT that recovered the phase in 4 of 16
        cases, so the error was multiplied by x rather than corrected by it.
        """
        return self._paravartya_yojayet_quantum(x, divisor, context)
    
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
        """The sum, flushed to zero when it is zero -- what the classical body does.

        The previous circuit conjugated by X gates, which made the relative
        phase `(norm_a - norm_b) * pi` and so fired on a close to b rather
        than on a close to -b, and then cut it with a hardcoded
        `threshold = 0.8` sitting inside the 1000-shot sampling noise. The
        result was not merely wrong but unstable: `(5, 2)` returned 0 or 7
        from the same call, and the firing region disagreed with this
        docstring's contract in 63 of the 121 integer pairs over -5..5.
        """
        if not _integral_scalars(a, b):
            return self._shunyam_samyasamuccaye_classical(a, b, context)
        total = _quantum_sum(int(a), int(b))
        if abs(total) < context.epsilon:
            return 0
        return total
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
            kernel.ry(theta_whole, q[0])
            
            # Encode parts into amplitudes of remaining qubits
            for i, part in enumerate(parts):
                theta_part = 2 * np.arcsin(min(1.0, abs(part) / 10.0))  # Normalize
                kernel.ry(theta_part, q[i+1])
            
            # Create entanglement to check part-whole relationship
            for i in range(len(parts)):
                kernel.cx(q[0], q[i+1])
            
            # Measure
            kernel.mz(q)
            
            # Execute
            result = cudaq.sample(kernel)
            
            # Check measurement outcomes
            # If whole = sum(parts), measurements should show high correlation
            # `most_probable()` takes no count argument -- it returns the
            # single most frequent bitstring. Take the top 5 from the
            # counts directly.
            # `_check_quantum_correlation` wants a bitstring -> count mapping.
            top_results = dict(sorted(result.items(),
                                      key=lambda kv: kv[1], reverse=True)[:5])
            
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
        """x moved `steps` times by `direction`, i.e. x + steps * direction.

        The previous circuit walked a cyclic ring of bare CNOTs, which is not
        an increment. It was wrong at six of seven configurations tried --
        (2, 2) gave 1 where the answer is 4, (5, 2) gave 2 where it is 7,
        (0, 3) gave 0 where it is 3 -- and its agreement with the classical
        body at (2, steps=3) was a coincidence, since it returned 5 or 6 at
        random from the same call.
        """
        if not _integral_scalars(x, steps, direction):
            return self._chalana_kalana_classical(x, steps, direction, context)
        displacement = _quantum_product(int(steps), int(direction))
        return _quantum_sum(int(x), displacement)
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
        """Addition and subtraction by ripple carry.

        The previous circuit computed a carry from `a[i-1], b[i-1]` into one
        shared carry qubit and then uncomputed it with the SAME Toffoli after
        `b[i]` had already been modified, so the two cancelled and the carry-in
        stayed pinned at `a[0]` for every bit. It was wrong at essentially
        every magnitude: 9 + 4 gave 17, 2 + 3 gave 1, 100 + 27 gave 55.
        """
        if not _integral_scalars(a, b):
            return self._sankalana_vyavakalanabhyam_classical(a, b, operation, context)
        ai, bi = int(a), int(b)
        if operation == 'add':
            return _quantum_sum(ai, bi)
        elif operation == 'subtract':
            return _quantum_sum(ai, -bi)
        elif operation == 'both':
            return (_quantum_sum(ai, bi), _quantum_sum(ai, -bi))
        else:
            raise ValueError(
                f"Unknown operation: {operation}. Use 'add', 'subtract', or 'both'.")
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
        """Sum(c_i * x**i), accumulated by ripple carry.

        The previous circuit encoded each term as a rotation angle divided by
        `np.sum(np.abs(coefficients))` and read it back through a three-qubit
        register, so its codomain was the 8 multiples of Sum|c|/8 -- the true
        value 17 was not in the range of the function at all, and it returned
        1.5 or 4.5 at random. It also aliased every coefficient beyond the
        third onto qubit 0 via `q[i % 3]`, and it is where a NaN angle reached
        cudaq and aborted the interpreter outright.
        """
        try:
            coeffs = list(coefficients)
        except TypeError:
            return self._sesanyankena_caramena_classical(coefficients, x, context)
        if not coeffs or not _integral_scalars(x, *coeffs):
            return self._sesanyankena_caramena_classical(coefficients, x, context)
        return _quantum_polynomial([int(c) for c in coeffs], int(x))
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
        
        # Subtract 1.
        #
        # This was written out inline and was wrong three ways: the borrow used
        # `cirq.TOFFOLI(q[i-1], q[i], q[i])`, naming q[i] as both a control and
        # the target (cirq rejects it, so the path never ran); the ripple swept
        # low to high, reading controls after modifying them; and the closing
        # complement covered only qubits 1..n-1, leaving the low bit inverted.
        _ripple_decrement(circuit, qubits)
        
        # Measure qubits
        circuit.append(cirq.measure(*qubits, key='result'))
        
        # Simulate
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        
        # Extract result
        result_bits = result.measurements['result'][0]
        result_decimal = sum(int(bit) * (2**i) for i, bit in enumerate(result_bits))
        
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
                  b: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
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
            b: Second value or array. When omitted, the current
                :class:`SutraContext` base is used as the proportional
                partner so the sutra remains invocable with a single
                positional argument within serial pipelines.
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
            if b is None:
                if isinstance(a, torch.Tensor):
                    b = torch.full_like(a, fill_value=float(context.base))
                elif isinstance(a, np.ndarray):
                    b = np.full_like(a, fill_value=context.base)
                else:
                    b = context.base

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
        kernel.ry(theta, q[0])
        
        # Measure to collapse state
        kernel.mz(q)
        
        # Execute multiple times to get probability distribution
        result = cudaq.sample(kernel, shots_count=1000)
        counts = dict(result.items())
        
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
        """The product, by shift-and-add.

        The previous circuit XORed partial products into the result register
        (`kernel.cx([a[i], b[j]], result[i + j])`) where the comment above it
        said "add shifted b". XOR is addition without carry, so it computed a
        carry-less GF(2) product and then read the register in the opposite
        bit order: the output was exactly bitreverse(clmul(a, b)) for all 64
        pairs below 8, and 6 * 7 came out as 18 rather than 42.
        """
        if not _integral_scalars(multiplicand, multiplier):
            return self._gunitasamuccayah_classical(multiplicand, multiplier, context)
        return _quantum_product(int(multiplicand), int(multiplier))
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
        """Quantum implementation of yavadunam: the deficiency `base - x`.

        This built its own circuit and propagated carries with bare CNOTs --
        `CNOT(carry, q[i])` then `CNOT(q[i], q[i+1])` -- which is not an
        adder: a carry into bit i+1 occurs only when every lower bit is set.
        It returned 0 for yavadunam(8, base=10), where the deficiency is 2.

        `_yavadunam_classical` is `base - x` and `_nikhilam_classical` is the
        same expression, so both quantum paths now share one construction that
        has been verified exact rather than keeping two hand-written adders.
        """
        if not isinstance(x, (int, float, np.integer, np.floating)):
            return self._yavadunam_classical(x, base, context)
        if float(x) != int(x) or float(base) != int(base):
            return self._yavadunam_classical(x, base, context)

        xi, base_i = int(x), int(base)
        if xi < 0 or base_i <= 0 or xi > base_i:
            return self._yavadunam_classical(x, base, context)

        return _complement_via_circuit(xi, base_i)

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
        """(a + b)^2, or a^2 + b^2, built from the verified sum and product.

        The previous circuit summed `int(bitstring, 2) * count / 1000` -- the
        decimal reading of a two-qubit register weighted by shot frequency --
        and then multiplied by an unrelated `max_val ** 2`. Its estimand
        converged to about 265 where (6 + 7)^2 is 169.
        """
        if not _integral_scalars(a, b):
            return self._samuccayagunitah_classical(a, b, operation, context)
        ai, bi = int(a), int(b)
        if operation == 'product_sum':
            total = _quantum_sum(ai, bi)
            return _quantum_product(total, total)
        elif operation == 'sum_product':
            return _quantum_sum(_quantum_product(ai, ai), _quantum_product(bi, bi))
        else:
            raise ValueError(
                f"Unknown operation: {operation}. Use 'product_sum' or 'sum_product'.")
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
        """(a + b)(a - b), which is what the classical body and the comment say.

        The sutra's docstring headline -- "the factors of the sum are equal to
        the sum of the factors" -- names no computable function of two scalars,
        so the operative specification is the inline comment at the classical
        body, "we are factoring a^2 - b^2 = (a+b)(a-b)", and the body itself,
        `return (a + b) * (a - b)`. That is taken here as the spec.

        The previous circuit could not implement it under any reading. It
        returned `p_11 * (max_val ** 2)` -- a probability times a square -- so
        its codomain was [0, 196] for these inputs and it could never produce a
        negative number, while the sutra gives -13 for (6, 7) and must go
        negative whenever |b| > |a|. The sign was in fact computed, by appending
        `cirq.Z` gates, and then discarded: Z is diagonal, so it changes no
        computational-basis measurement probability at all.
        """
        if not _integral_scalars(a, b):
            return self._gunakasamuccayah_classical(a, b, context)
        ai, bi = int(a), int(b)
        return _quantum_product(_quantum_sum(ai, bi), _quantum_sum(ai, -bi))
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
