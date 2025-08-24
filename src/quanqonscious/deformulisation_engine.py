import numpy as np
import sympy as sp
import scipy.linalg
import torch
import cirq
try:
    import cudaq
except ImportError:
    cudaq = None

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import logging

logger = logging.getLogger(__name__)

# Exported symbols for external modules
__all__ = ["DeformulisationContext", "EquationDeformulisationEngine"]

@dataclass
class DeformulisationContext:
    epsilon: float = 1e-6
    max_recursion_depth: int = 10
    quantum_backend: str = "cirq"
    maya_phase_keys: Optional[List[int]] = field(default_factory=list)
    sulba_constraints: Optional[Dict[str, float]] = field(default_factory=dict)

class EquationDeformulisationEngine:
    def __init__(self, context: DeformulisationContext):
        self.context = context
        self._vedic_engine: Optional[Dict[str, Callable]] = None

    def _deformulise_classical(self, equation: Any, target_form: str,
                             constraints: Optional[Dict[str, Any]] = None) -> Any:
        """Classical algebraic deformulisation"""
        logger.debug(f"Classical deformulisation to {target_form}")
        if target_form == "polynomial":
            return self._to_polynomial_form(equation, constraints)
        elif target_form == "matrix":
            return self._to_matrix_form(equation, constraints)
        elif target_form == "differential":
            return self._to_differential_form(equation, constraints)
        elif target_form == "integral":
            return self._to_integral_form(equation, constraints)
        elif target_form == "series":
            return self._to_series_expansion(equation, constraints)
        else:
            if isinstance(equation, sp.Expr):
                return {
                    'simplified': sp.simplify(equation),
                    'expanded': sp.expand(equation),
                    'factored': sp.factor(equation) if equation.is_polynomial() else equation,
                    'canonical': equation
                }
            else:
                return equation

    def _to_polynomial_form(self, equation: Any,
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert equation to polynomial representation"""
        if isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            if variables:
                var = variables[0]
                poly = sp.Poly(equation, var)
                return {
                    'coefficients': poly.all_coeffs(),
                    'degree': poly.degree(),
                    'roots': sp.roots(equation, var),
                    'variable': var,
                    'factorization': sp.factor(equation)
                }
            else:
                return {
                    'coefficients': [equation],
                    'degree': 0,
                    'roots': {},
                    'variable': None,
                    'factorization': equation
                }
        elif isinstance(equation, (int, float, complex)):
            return {
                'coefficients': [equation],
                'degree': 0,
                'roots': {},
                'variable': None,
                'factorization': equation
            }
        else:
            raise ValueError(f"Cannot convert {type(equation)} to polynomial form")

    def _to_matrix_form(self, equation: Any,
                       constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convert equation to matrix representation"""
        if isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            n_vars = len(variables)
            if n_vars == 0:
                return np.array([[float(equation)]])
            elif n_vars == 1:
                poly = sp.Poly(equation, variables[0])
                coeffs = poly.all_coeffs()
                n = len(coeffs) - 1
                if n == 0:
                    return np.array([[float(coeffs[0])]])
                matrix = np.zeros((n, n))
                for i in range(n-1):
                    matrix[i+1, i] = 1
                for i in range(n):
                    matrix[i, n-1] = -float(coeffs[n-i]) / float(coeffs[0])
                return matrix
            else:
                matrix = np.zeros((n_vars, n_vars))
                for i, var_i in enumerate(variables):
                    for j, var_j in enumerate(variables):
                        coeff = equation.coeff(var_i * var_j)
                        if coeff != 0:
                            matrix[i, j] = float(coeff)
                return matrix
        elif isinstance(equation, (list, tuple)):
            return np.array(equation)
        elif isinstance(equation, np.ndarray):
            return equation
        elif isinstance(equation, torch.Tensor):
            return equation.cpu().numpy()
        else:
            return np.array([[float(equation)]])

    def _to_differential_form(self, equation: Any,
                            constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert equation to differential equation form"""
        if isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            if variables:
                t = sp.Symbol('t')
                y = sp.Function('y')
                diff_eq = equation.subs(variables[0], y(t))
                if equation.is_polynomial():
                    degree = sp.degree(equation, variables[0])
                    char_eq = equation
                    terms = []
                    for i in range(degree + 1):
                        coeff = char_eq.coeff(variables[0], i)
                        if i == 0:
                            terms.append(coeff * y(t))
                        else:
                            terms.append(coeff * sp.Derivative(y(t), t, i))
                    diff_eq = sum(terms)
                return {
                    'equation': diff_eq,
                    'order': degree if equation.is_polynomial() else 1,
                    'linear': True,
                    'homogeneous': equation.subs(variables[0], 0) == 0,
                    'characteristic_equation': equation
                }
        return {
            'equation': equation,
            'order': 0,
            'linear': True,
            'homogeneous': True,
            'characteristic_equation': None
        }

    def _to_integral_form(self, equation: Any,
                         constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert equation to integral representation"""
        if isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            if variables:
                var = variables[0]
                integral = sp.integrate(equation, var)
                try:
                    definite = sp.integrate(equation, (var, -sp.pi, sp.pi))
                except Exception:
                    definite = None
                greens_kernel = sp.exp(-sp.Abs(var - sp.Symbol('x_prime')))
                return {
                    'indefinite': integral,
                    'definite': definite,
                    'variable': var,
                    'greens_function': greens_kernel,
                    'convolution_form': sp.Integral(
                        equation.subs(var, sp.Symbol('tau')) *
                        sp.Symbol('f')(var - sp.Symbol('tau')),
                        (sp.Symbol('tau'), -sp.oo, sp.oo)
                    )
                }
        return {
            'indefinite': equation,
            'definite': equation,
            'variable': None,
            'greens_function': None,
            'convolution_form': equation
        }

    def _to_series_expansion(self, equation: Any,
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert equation to series expansion"""
        if isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            if variables:
                var = variables[0]
                n_terms = constraints.get('series_terms', 10) if constraints else 10
                taylor = equation.series(var, 0, n_terms).removeO()
                try:
                    fourier_coeffs = []
                    for n in range(n_terms):
                        an = sp.integrate(equation * sp.cos(n * var), (var, -sp.pi, sp.pi)) / sp.pi
                        bn = sp.integrate(equation * sp.sin(n * var), (var, -sp.pi, sp.pi)) / sp.pi
                        fourier_coeffs.append((an, bn))
                except Exception:
                    fourier_coeffs = None
                try:
                    laurent = equation.series(var, 0, n_terms)
                except Exception:
                    laurent = taylor
                return {
                    'taylor': taylor,
                    'fourier': fourier_coeffs,
                    'laurent': laurent,
                    'convergence_radius': self._estimate_convergence_radius(equation, var),
                    'variable': var,
                    'order': n_terms
                }
        return {
            'taylor': equation,
            'fourier': None,
            'laurent': equation,
            'convergence_radius': float('inf'),
            'variable': None,
            'order': 0
        }

    def _estimate_convergence_radius(self, expr: sp.Expr, var: sp.Symbol) -> float:
        """Estimate radius of convergence for series expansion"""
        try:
            coeffs = []
            for n in range(20):
                coeff = expr.diff(var, n).subs(var, 0) / sp.factorial(n)
                coeffs.append(abs(complex(coeff)))
            ratios = []
            for i in range(1, len(coeffs)-1):
                if coeffs[i] != 0 and coeffs[i+1] != 0:
                    ratios.append(coeffs[i] / coeffs[i+1])
            if ratios:
                return float(np.mean(ratios))
            return float('inf')
        except Exception:
            return 1.0

    def _deformulise_quantum(self, equation: Any, target_form: str,
                           constraints: Optional[Dict[str, Any]] = None) -> Any:
        """Quantum superposition deformulisation"""
        logger.debug(f"Quantum deformulisation to {target_form}")
        if target_form == "quantum_circuit":
            return self._equation_to_quantum_circuit(equation, constraints)
        elif target_form == "quantum_state":
            return self._equation_to_quantum_state(equation, constraints)
        elif target_form == "density_matrix":
            return self._equation_to_density_matrix(equation, constraints)
        elif target_form == "hamiltonian":
            return self._equation_to_hamiltonian(equation, constraints)
        elif target_form == "unitary":
            return self._equation_to_unitary(equation, constraints)
        else:
            return self._equation_to_quantum_state(equation, constraints)

    def _equation_to_quantum_circuit(self, equation: Any,
                                   constraints: Optional[Dict[str, Any]] = None) -> Union[cirq.Circuit, object]:
        """Convert equation to quantum circuit"""
        if isinstance(equation, (int, float)):
            n_qubits = max(3, int(np.ceil(np.log2(abs(equation) + 1))))
        elif isinstance(equation, sp.Expr):
            n_qubits = max(3, len(equation.free_symbols) + 2)
        else:
            n_qubits = 4
        if self.context.quantum_backend == "cirq" or cudaq is None:
            qubits = cirq.LineQubit.range(n_qubits)
            circuit = cirq.Circuit()
            if isinstance(equation, (int, float)):
                angle = float(equation) % (2 * np.pi)
                circuit.append(cirq.ry(angle).on(qubits[0]))
                circuit.append(cirq.rx(angle/2).on(qubits[1]))
                for i in range(n_qubits - 1):
                    circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
            elif isinstance(equation, sp.Expr):
                variables = list(equation.free_symbols)
                for q in qubits:
                    circuit.append(cirq.H(q))
                for i, var in enumerate(variables[:n_qubits]):
                    angle = float(equation.coeff(var)) % (2 * np.pi)
                    if i < n_qubits:
                        circuit.append(cirq.ry(angle).on(qubits[i]))
                for i in range(n_qubits - 1):
                    circuit.append(cirq.CZ(qubits[i], qubits[i+1]))
            circuit.append(cirq.measure(*qubits, key='result'))
            return circuit
        else:
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(n_qubits)
            if isinstance(equation, (int, float)):
                angle = float(equation) % (2 * np.pi)
                kernel.ry(qubits[0], angle)
                kernel.rx(qubits[1], angle/2)
                for i in range(n_qubits - 1):
                    kernel.cx(qubits[i], qubits[i+1])
            elif isinstance(equation, sp.Expr):
                variables = list(equation.free_symbols)
                for i in range(n_qubits):
                    kernel.h(qubits[i])
                for i, var in enumerate(variables[:n_qubits]):
                    angle = float(equation.coeff(var)) % (2 * np.pi)
                    if i < n_qubits:
                        kernel.ry(qubits[i], angle)
                for i in range(n_qubits - 1):
                    kernel.cz(qubits[i], qubits[i+1])
            kernel.mz(qubits)
            return kernel

    def _equation_to_quantum_state(self, equation: Any,
                                 constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convert equation to quantum state vector"""
        if isinstance(equation, (int, float)):
            dim = 2 ** max(3, int(np.ceil(np.log2(abs(equation) + 1))))
        elif isinstance(equation, sp.Expr):
            dim = 2 ** max(3, len(equation.free_symbols) + 2)
        elif isinstance(equation, (np.ndarray, torch.Tensor)):
            dim = 2 ** int(np.ceil(np.log2(equation.size)))
        else:
            dim = 8
        state = np.zeros(dim, dtype=complex)
        if isinstance(equation, (int, float)):
            theta = float(equation) % (2 * np.pi)
            phi = (float(equation) * np.pi) % (2 * np.pi)
            for i in range(dim):
                if bin(i).count('1') == 0:
                    state[i] = np.cos(theta/2)
                elif bin(i).count('1') == bin(dim-1).count('1'):
                    state[i] = np.sin(theta/2) * np.exp(1j * phi)
                else:
                    weight = 1.0 / (1 + bin(i).count('1'))
                    state[i] = weight * np.exp(1j * phi * i / dim)
        elif isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            for i in range(min(dim, 2**len(variables))):
                binary = format(i, f'0{len(variables)}b')
                subs = {var: int(bit) for var, bit in zip(variables, binary)}
                try:
                    value = complex(equation.subs(subs))
                    state[i] = value
                except Exception:
                    state[i] = 1.0 / np.sqrt(dim)
        elif isinstance(equation, (np.ndarray, torch.Tensor)):
            data = equation.cpu().numpy() if isinstance(equation, torch.Tensor) else equation
            flat_data = data.flatten()
            n_elements = min(len(flat_data), dim)
            state[:n_elements] = flat_data[:n_elements]
        norm = np.linalg.norm(state)
        if norm > self.context.epsilon:
            state = state / norm
        else:
            state = np.ones(dim, dtype=complex) / np.sqrt(dim)
        return state

    def _equation_to_density_matrix(self, equation: Any,
                                  constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convert equation to density matrix representation"""
        state = self._equation_to_quantum_state(equation, constraints)
        purity = constraints.get('purity', 1.0) if constraints else 1.0
        if purity >= 1.0 - self.context.epsilon:
            density = np.outer(state, np.conj(state))
        else:
            dim = len(state)
            pure_density = np.outer(state, np.conj(state))
            mixed_density = np.eye(dim) / dim
            density = purity * pure_density + (1 - purity) * mixed_density
        return density

    def _equation_to_hamiltonian(self, equation: Any,
                               constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convert equation to Hamiltonian operator"""
        if isinstance(equation, (int, float)):
            dim = 4
        elif isinstance(equation, sp.Expr):
            dim = 2 ** min(len(equation.free_symbols) + 1, 4)
        else:
            dim = 4
        H = np.zeros((dim, dim), dtype=complex)
        if isinstance(equation, (int, float)):
            value = float(equation)
            for i in range(dim):
                H[i, i] = value * (i - dim/2)
            coupling = value / dim
            for i in range(dim - 1):
                H[i, i+1] = coupling
                H[i+1, i] = coupling
        elif isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
            pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
            H = np.eye(dim, dtype=complex) * float(equation.subs([(v, 0) for v in variables]))
            if len(variables) >= 1 and dim >= 2:
                coeff = float(equation.coeff(variables[0]))
                H[:2, :2] += coeff * pauli_z
            if len(variables) >= 2 and dim >= 2:
                coeff = float(equation.coeff(variables[1]))
                H[:2, :2] += coeff * pauli_x
            if len(variables) >= 3 and dim >= 2:
                coeff = float(equation.coeff(variables[2]))
                H[:2, :2] += coeff * pauli_y
        H = (H + H.conj().T) / 2
        return H

    def _equation_to_unitary(self, equation: Any,
                           constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convert equation to unitary evolution operator"""
        H = self._equation_to_hamiltonian(equation, constraints)
        t = constraints.get('evolution_time', 1.0) if constraints else 1.0
        U = scipy.linalg.expm(-1j * H * t)
        return U

    def _deformulise_vedic(self, equation: Any, target_form: str,
                         constraints: Optional[Dict[str, Any]] = None) -> Any:
        """Vedic recursive deformulisation using sutras"""
        logger.debug(f"Vedic deformulisation to {target_form}")
        if not hasattr(self, '_vedic_engine') or self._vedic_engine is None:
            self._vedic_engine = self._initialize_vedic_engine()
        if target_form == "vedic_decomposition":
            return self._vedic_decompose(equation, constraints)
        elif target_form == "sutra_sequence":
            return self._apply_sutra_sequence(equation, constraints)
        elif target_form == "recursive_pattern":
            return self._extract_recursive_pattern(equation, constraints)
        elif target_form == "vedic_matrix":
            return self._to_vedic_matrix(equation, constraints)
        else:
            return self._vedic_decompose(equation, constraints)

    def _initialize_vedic_engine(self) -> Dict[str, Callable]:
        """Initialize Vedic mathematics engine with sutras"""
        sutras = {
            'ekadhikena_purvena': lambda x: x + 1,
            'nikhilam': lambda x, base: base - x,
            'urdhva_tiryak': lambda a, b: a * b,
            'paravartya': lambda x: 1/x if x != 0 else 0,
            'sunyam_samya': lambda x, y: x if x == y else x - y,
            'anurupye': lambda x, ratio: x * ratio,
            'sankalana_vyavakalanabhyam': lambda x, y: (x + y, x - y),
            'puranapuranabhyam': lambda x, y, base: (base - x) * (base - y),
            'chalana_kalanabhyam': lambda f, x, dx: f(x + dx) - f(x),
            'yavadunam': lambda x, base: x if x < base else x - base,
            'vyashtisamanstih': lambda whole, parts: whole == sum(parts),
            'shesanyankena': lambda x, y: x % y,
            'sopantyadvayamantyam': lambda x: 2 * x[-1] + x[-2] if len(x) > 1 else x,
            'ekanyunena_purvena': lambda x: x - 1,
            'gunitasamuchyah': lambda factors: np.prod(factors),
            'gunaka_samuchyah': lambda x, y: x * y == (x + y) ** 2 - (x - y) ** 2,
        }
        return sutras

    def _vedic_decompose(self, equation: Any,
                       constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Decompose equation using Vedic principles"""
        decomposition = {
            'sutras_applied': [],
            'recursive_depth': 0,
            'components': [],
            'vedic_form': None,
            'convergence': False
        }
        current = equation
        max_depth = self.context.max_recursion_depth
        for depth in range(max_depth):
            sutra_name, transformation = self._select_vedic_sutra(current)
            if sutra_name:
                decomposition['sutras_applied'].append(sutra_name)
                try:
                    if sutra_name == 'urdhva_tiryak' and isinstance(current, sp.Expr):
                        result = self._urdhva_tiryak_symbolic(current)
                    elif sutra_name == 'nikhilam' and isinstance(current, (int, float)):
                        base = self._find_nearest_base(current)
                        result = transformation(current, base)
                    else:
                        result = transformation(current)
                    decomposition['components'].append({
                        'depth': depth,
                        'sutra': sutra_name,
                        'input': current,
                        'output': result
                    })
                    if self._check_vedic_convergence(current, result):
                        decomposition['convergence'] = True
                        decomposition['vedic_form'] = result
                        break
                    current = result
                except Exception as e:
                    logger.warning(f"Sutra {sutra_name} failed: {e}")
                    break
            else:
                break
        decomposition['recursive_depth'] = depth + 1
        if not decomposition['convergence']:
            decomposition['vedic_form'] = current
        return decomposition

    def _select_vedic_sutra(self, equation: Any) -> Tuple[Optional[str], Optional[Callable]]:
        """Select appropriate Vedic sutra based on equation characteristics"""
        if isinstance(equation, (int, float)):
            value = float(equation)
            if abs(value - self._find_nearest_base(value)) < value * 0.1:
                return 'nikhilam', self._vedic_engine['nikhilam']
            if abs(value) > 10:
                return 'urdhva_tiryak', self._vedic_engine['urdhva_tiryak']
            if 0 < abs(value) < 1:
                return 'paravartya', self._vedic_engine['paravartya']
        elif isinstance(equation, sp.Expr):
            if equation.is_polynomial():
                return 'urdhva_tiryak', self._vedic_engine['urdhva_tiryak']
            if equation.has(sp.sin, sp.cos, sp.exp):
                return 'chalana_kalanabhyam', self._vedic_engine['chalana_kalanabhyam']
        elif isinstance(equation, (list, tuple, np.ndarray)):
            return 'vyashtisamanstih', self._vedic_engine['vyashtisamanstih']
        return 'ekadhikena_purvena', self._vedic_engine['ekadhikena_purvena']

    def _find_nearest_base(self, value: float) -> int:
        """Find nearest power of 10 base for Vedic calculations"""
        abs_value = abs(value)
        if abs_value < 1:
            return 1
        power = int(np.log10(abs_value))
        base = 10 ** power
        if abs_value / base > 5:
            base *= 10
        return base

    def _urdhva_tiryak_symbolic(self, expr: sp.Expr) -> sp.Expr:
        """Apply Urdhva-Tiryak to symbolic expression"""
        if expr.is_Mul:
            factors = expr.as_ordered_factors()
            if len(factors) == 2:
                a, b = factors
                if a.is_Add and b.is_Add:
                    a_terms = a.as_ordered_terms()
                    b_terms = b.as_ordered_terms()
                    result = 0
                    for i, a_term in enumerate(a_terms):
                        for j, b_term in enumerate(b_terms):
                            weight = 1 / (1 + abs(i - j))
                            result += weight * a_term * b_term
                    return result
        return expr

    def _check_vedic_convergence(self, previous: Any, current: Any) -> bool:
        """Check if Vedic transformation has converged"""
        if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
            return abs(previous - current) < self.context.epsilon
        elif isinstance(previous, sp.Expr) and isinstance(current, sp.Expr):
            return previous.equals(current)
        elif isinstance(previous, np.ndarray) and isinstance(current, np.ndarray):
            return np.allclose(previous, current, atol=self.context.epsilon)
        else:
            return previous == current

    def _apply_sutra_sequence(self, equation: Any,
                            constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Apply sequence of Vedic sutras"""
        if isinstance(equation, (int, float)):
            if abs(equation) > 100:
                sequence = ['nikhilam', 'urdhva_tiryak', 'anurupye']
            else:
                sequence = ['ekadhikena_purvena', 'sankalana_vyavakalanabhyam']
        elif isinstance(equation, sp.Expr):
            if equation.is_polynomial():
                sequence = ['urdhva_tiryak', 'sopantyadvayamantyam', 'gunitasamuchyah']
            else:
                sequence = ['chalana_kalanabhyam', 'paravartya', 'sunyam_samya']
        else:
            sequence = ['vyashtisamanstih', 'anurupye', 'gunaka_samuchyah']
        results = []
        current = equation
        for sutra_name in sequence:
            if sutra_name in self._vedic_engine:
                transformation = self._vedic_engine[sutra_name]
                try:
                    if sutra_name == 'nikhilam':
                        base = self._find_nearest_base(current)
                        result = transformation(current, base)
                    elif sutra_name in ['sankalana_vyavakalanabhyam', 'urdhva_tiryak']:
                        result = transformation(current, current)
                    else:
                        result = transformation(current)
                    results.append({
                        'sutra': sutra_name,
                        'input': current,
                        'output': result,
                        'description': self._get_sutra_description(sutra_name)
                    })
                    current = result
                except Exception as e:
                    logger.warning(f"Failed to apply {sutra_name}: {e}")
        return results

    def _extract_recursive_pattern(self, equation: Any,
                                 constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract recursive patterns using Vedic principles"""
        pattern = {
            'type': 'unknown',
            'parameters': {},
            'recurrence': None,
            'generating_function': None,
            'vedic_representation': None
        }
        if isinstance(equation, (list, tuple, np.ndarray)):
            seq = np.array(equation) if not isinstance(equation, np.ndarray) else equation.flatten()
            if len(seq) > 2:
                diffs = np.diff(seq)
                if np.allclose(diffs, diffs[0], atol=self.context.epsilon):
                    pattern['type'] = 'arithmetic'
                    pattern['parameters'] = {
                        'first_term': seq[0],
                        'common_difference': diffs[0]
                    }
                    pattern['recurrence'] = f"a_n = a_{{n-1}} + {diffs[0]}"
                if not np.any(seq == 0):
                    ratios = seq[1:] / seq[:-1]
                    if np.allclose(ratios, ratios[0], atol=self.context.epsilon):
                        pattern['type'] = 'geometric'
                        pattern['parameters'] = {
                            'first_term': seq[0],
                            'common_ratio': ratios[0]
                        }
                        pattern['recurrence'] = f"a_n = a_{{n-1}} * {ratios[0]}"
                if len(seq) > 3:
                    fib_check = seq[2:] - (seq[1:-1] + seq[:-2])
                    if np.allclose(fib_check, 0, atol=self.context.epsilon):
                        pattern['type'] = 'fibonacci'
                        pattern['parameters'] = {
                            'initial_values': [seq[0], seq[1]]
                        }
                        pattern['recurrence'] = "a_n = a_{n-1} + a_{n-2}"
        elif isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            if variables:
                var = variables[0]
                if equation.is_polynomial(var):
                    poly = sp.Poly(equation, var)
                    pattern['type'] = 'polynomial'
                    pattern['parameters'] = {
                        'degree': poly.degree(),
                        'coefficients': poly.all_coeffs()
                    }
                if equation.has(sp.exp):
                    pattern['type'] = 'exponential'
                if equation.has(sp.sin, sp.cos):
                    pattern['type'] = 'trigonometric'
        pattern['vedic_representation'] = self._generate_vedic_notation(pattern)
        return pattern

    def _to_vedic_matrix(self, equation: Any,
                       constraints: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Convert equation to Vedic matrix representation"""
        if isinstance(equation, (int, float)):
            n = 3
            value = float(equation)
            base_square = np.array([[2, 7, 6],
                                   [9, 5, 1],
                                   [4, 3, 8]])
            factor = value / 15
            vedic_matrix = base_square * factor
        elif isinstance(equation, sp.Expr):
            variables = list(equation.free_symbols)
            n = len(variables)
            if n == 0:
                vedic_matrix = np.array([[float(equation)]])
            else:
                vedic_matrix = np.zeros((n, n))
                for i, var_i in enumerate(variables):
                    for j, var_j in enumerate(variables):
                        if i == j:
                            coeff = equation.coeff(var_i, 1)
                        else:
                            coeff = equation.coeff(var_i * var_j, 1)
                        vedic_matrix[i, j] = float(coeff) if coeff else 0
        elif isinstance(equation, (list, tuple, np.ndarray)):
            data = np.array(equation) if not isinstance(equation, np.ndarray) else equation
            size = int(np.sqrt(data.size))
            if size * size == data.size:
                vedic_matrix = data.reshape(size, size)
            else:
                new_size = size + 1
                padded = np.zeros(new_size * new_size)
                padded[:data.size] = data.flatten()
                vedic_matrix = padded.reshape(new_size, new_size)
        else:
            vedic_matrix = np.array([[float(equation)]])
        return vedic_matrix

    def _get_sutra_description(self, sutra_name: str) -> str:
        """Get description of Vedic sutra"""
        descriptions = {
            'ekadhikena_purvena': "By one more than the previous one",
            'nikhilam': "All from 9 and the last from 10",
            'urdhva_tiryak': "Vertically and crosswise",
            'paravartya': "Transpose and apply",
            'sunyam_samya': "If the same then zero",
            'anurupye': "If one is in ratio, the other is zero",
            'sankalana_vyavakalanabhyam': "By addition and by subtraction",
            'puranapuranabhyam': "By the completion or non-completion",
            'chalana_kalanabhyam': "Differences and similarities",
            'yavadunam': "Whatever the deficiency",
            'vyashtisamanstih': "Part and whole",
            'shesanyankena': "The remainder by the last digit",
            'sopantyadvayamantyam': "The ultimate and twice the penultimate",
            'ekanyunena_purvena': "By one less than the previous one",
            'gunitasamuchyah': "The product of the sum is equal to the sum of the product",
            'gunaka_samuchyah': "The factors of the sum is equal to the sum of the factors"
        }
        return descriptions.get(sutra_name, "Unknown sutra")

    def _generate_vedic_notation(self, pattern: Dict[str, Any]) -> str:
        """Generate Vedic mathematical notation for pattern"""
        if pattern['type'] == 'arithmetic':
            a0 = pattern['parameters']['first_term']
            d = pattern['parameters']['common_difference']
            return f"॥ a_n = {a0} + (n-1)×{d} ॥"
        elif pattern['type'] == 'geometric':
            a0 = pattern['parameters']['first_term']
            r = pattern['parameters']['common_ratio']
            return f"॥ a_n = {a0} × {r}^(n-1) ॥"
        elif pattern['type'] == 'fibonacci':
            return "॥ a_n = a_{n-1} + a_{n-2} ॥"
        elif pattern['type'] == 'polynomial':
            degree = pattern['parameters']['degree']
            return f"॥ P_n(x) = Σ[k=0 to {degree}] a_k × x^k ॥"
        else:
            return f"॥ {pattern['type']} pattern ॥"
