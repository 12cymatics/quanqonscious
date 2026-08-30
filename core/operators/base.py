"""
Operator Base Classes (CODEX 4.1, 4.2)

Implements the common operator interface for all sutra and field operators.

All operators must:
- Implement apply(state, context) -> state'
- Be composable (pipeline)
- Support exact arithmetic mode
- Log actions in structured trace for replay and proofs
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from fractions import Fraction
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union
)
import hashlib
import json
import copy

from ..state import FieldState, FieldStateSnapshot, ArithmeticMode, state_digest


class OperatorCategory(Enum):
    """
    Operator categories per CODEX 4.2.

    1) Arithmetic transforms (exact integer/rational transforms)
    2) Indexing/permutation transforms (lattice remaps, R4 adjacency rewires)
    3) Series/product transforms (factorizations and controlled expansions)
    4) Constraint/suppression transforms (stability envelopes, boundedness gates)
    5) Field evolution (GRVQ ansatz, wave propagation)
    6) Coupling transforms (MSTVQ, R4 coupling)
    """
    ARITHMETIC = "arithmetic"
    INDEXING = "indexing"
    SERIES = "series"
    CONSTRAINT = "constraint"
    FIELD = "field"
    COUPLING = "coupling"
    COMPOSITE = "composite"


@dataclass
class OperatorContext:
    """
    Context passed to operators during execution.

    Contains:
    - Global parameters and configuration
    - Current timestep and evolution state
    - Trace recorder for logging
    - Random seed for reproducibility
    """

    # Time evolution
    timestep: int = 0
    dt: Fraction = Fraction(1, 100)

    # Global parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # MSTVQ knobs (CODEX 6.1)
    h_m: Fraction = Fraction(1)  # Magnetic stress-tension global scale

    # Reproducibility seed
    seed: int = 42

    # Trace log
    trace: Optional['OperatorTrace'] = None

    # Arithmetic mode
    mode: ArithmeticMode = ArithmeticMode.EXACT

    # Parent state hash for determinism verification
    parent_hash: Optional[str] = None

    def with_timestep(self, t: int) -> 'OperatorContext':
        """Create new context with updated timestep."""
        new_ctx = copy.copy(self)
        new_ctx.timestep = t
        return new_ctx

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter with optional default."""
        return self.parameters.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """Set a parameter."""
        self.parameters[key] = value


@dataclass
class TraceEntry:
    """Single entry in the operator trace log."""
    operator_name: str
    operator_id: str
    category: OperatorCategory
    timestep: int
    timestamp: str
    input_hash: str
    output_hash: str
    parameters: Dict[str, Any]
    delta_summary: Dict[str, Any]
    invariants_checked: List[str]
    invariants_passed: bool
    # 0 for an operator the caller applied; 1 for a child of a composite, and
    # so on. A composite logs itself *and* its children, so a replay that
    # re-applied every entry would apply the children twice.
    depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'operator_name': self.operator_name,
            'operator_id': self.operator_id,
            'category': self.category.value,
            'timestep': self.timestep,
            'timestamp': self.timestamp,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'parameters': {k: str(v) if isinstance(v, Fraction) else v
                          for k, v in self.parameters.items()},
            'delta_summary': self.delta_summary,
            'invariants_checked': self.invariants_checked,
            'invariants_passed': self.invariants_passed,
            'depth': self.depth,
        }


@dataclass
class OperatorTrace:
    """
    Structured trace log for operator replay and proofs (CODEX 7.2).

    Records all operator applications with:
    - Input/output state hashes
    - Parameter values
    - Delta summaries
    - Invariant check results
    """

    entries: List[TraceEntry] = field(default_factory=list)
    initial_state_hash: Optional[str] = None
    # How many operator applications are currently in flight. A composite is
    # still on the stack while its children run, so a child logs at depth 1.
    _depth: int = 0

    def enter(self) -> int:
        """Mark an operator application as started; return its nesting depth."""
        depth = self._depth
        self._depth += 1
        return depth

    def leave(self) -> None:
        """Mark the innermost in-flight operator application as finished."""
        self._depth -= 1

    def log(self,
            operator: 'Operator',
            context: OperatorContext,
            input_state: FieldState,
            output_state: FieldState,
            delta_summary: Dict[str, Any],
            invariants: List[str],
            passed: bool,
            depth: int = 0) -> None:
        """Log an operator application."""
        entry = TraceEntry(
            operator_name=operator.name,
            operator_id=operator.operator_id,
            category=operator.category,
            timestep=context.timestep,
            timestamp=datetime.now().isoformat(),
            input_hash=self._state_hash(input_state),
            output_hash=self._state_hash(output_state),
            parameters=dict(context.parameters),
            delta_summary=delta_summary,
            invariants_checked=invariants,
            invariants_passed=passed,
            depth=depth
        )
        self.entries.append(entry)

    def _state_hash(self, state: FieldState) -> str:
        """Hash of the field state, for determinism verification and replay.

        Delegates to the one canonical digest in `core.state`. This used to
        hash a 1-in-(N/100) sample of sites and truncate to 16 characters --
        cheaper, but it disagreed with `StateCheckpoint._compute_hash` on
        every state, so `TraceReplayer.replay`, which compares the two, could
        not verify any evolution at all. It was also blind to a change at any
        site outside the sample.
        """
        return state_digest(state)

    def verify_determinism(self, other: 'OperatorTrace') -> bool:
        """Check if two traces are identical (determinism test)."""
        if len(self.entries) != len(other.entries):
            return False
        for e1, e2 in zip(self.entries, other.entries):
            if e1.input_hash != e2.input_hash or e1.output_hash != e2.output_hash:
                return False
        return True

    def to_json(self) -> str:
        """Serialize trace to JSON."""
        return json.dumps({
            'initial_state_hash': self.initial_state_hash,
            'entries': [e.to_dict() for e in self.entries]
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'OperatorTrace':
        """Deserialize trace from JSON."""
        data = json.loads(json_str)
        trace = cls(initial_state_hash=data['initial_state_hash'])
        for e in data['entries']:
            entry = TraceEntry(
                operator_name=e['operator_name'],
                operator_id=e['operator_id'],
                category=OperatorCategory(e['category']),
                timestep=e['timestep'],
                timestamp=e['timestamp'],
                input_hash=e['input_hash'],
                output_hash=e['output_hash'],
                parameters=e['parameters'],
                delta_summary=e['delta_summary'],
                invariants_checked=e['invariants_checked'],
                invariants_passed=e['invariants_passed']
            )
            trace.entries.append(entry)
        return trace


class Operator(ABC):
    """
    Abstract base class for all operators (CODEX 4.1).

    All operators must implement:
        apply(state, context) -> state'

    And are:
    - Composable via >> operator
    - Support exact arithmetic mode
    - Log actions in trace for replay
    """

    def __init__(self, name: str, category: OperatorCategory):
        self._name = name
        self._category = category
        self._operator_id = f"{name}_{id(self) % 10000:04d}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> OperatorCategory:
        return self._category

    @property
    def operator_id(self) -> str:
        return self._operator_id

    @abstractmethod
    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply this operator to the field state.

        Args:
            state: Current field state
            context: Operator context with parameters, timestep, etc.

        Returns:
            New field state (may be mutated or copied)
        """
        pass

    def check_invariants(self, state: FieldState, context: OperatorContext) -> Tuple[List[str], bool]:
        """
        Check invariants after operator application.

        Override in subclasses for specific invariant checks.

        Returns:
            (list of invariant names checked, all passed)
        """
        invariants = []
        passed = True

        # Check toroidal closure
        invariants.append("toroidal_closure")
        for point in state.lattice.iterate_all():
            if not state.lattice.validate_closure(point):
                passed = False
                break

        return invariants, passed

    @staticmethod
    def _safe_scalar_text(value: Any) -> str:
        if isinstance(value, Fraction):
            if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
                return f"{float(value):.12e}"
        try:
            return str(value)
        except ValueError:
            if isinstance(value, Fraction):
                return f"{float(value):.12e}"
            return repr(value)

    def __call__(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply operator with logging and invariant checking.

        This is the primary entry point for operator application.
        """
        # Capture input state info
        input_norm = state.total_norm_squared()

        # Apply the operator. The depth is taken before `apply` runs, so a
        # composite records 0 and the children it drives record 1 -- which is
        # what lets a replay re-apply only what the caller actually applied.
        depth = context.trace.enter() if context.trace is not None else 0
        try:
            output_state = self.apply(state, context)
        finally:
            if context.trace is not None:
                context.trace.leave()

        # Check invariants
        invariants, passed = self.check_invariants(output_state, context)

        if not passed:
            raise RuntimeError(
                f"Invariant check failed for operator {self.name} at timestep {context.timestep}"
            )

        # Log to trace if available
        if context.trace is not None:
            output_norm = output_state.total_norm_squared()
            delta_summary = {
                'input_norm_sq': self._safe_scalar_text(input_norm),
                'output_norm_sq': self._safe_scalar_text(output_norm),
                'delta_norm_sq': self._safe_scalar_text(output_norm - input_norm),
            }
            context.trace.log(
                self, context, state, output_state,
                delta_summary, invariants, passed, depth
            )

        return output_state

    def __rshift__(self, other: 'Operator') -> 'CompositeOperator':
        """Compose operators: op1 >> op2 applies op1 then op2."""
        return CompositeOperator([self, other])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, category={self.category.value})"


class CompositeOperator(Operator):
    """
    Composite operator that applies a sequence of operators.

    Supports pipeline composition via >> operator.
    """

    def __init__(self, operators: List[Operator]):
        super().__init__(
            name="Composite[" + ",".join(op.name for op in operators) + "]",
            category=OperatorCategory.COMPOSITE
        )
        self._operators = list(operators)

    @property
    def operators(self) -> List[Operator]:
        return self._operators

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """Apply all operators in sequence."""
        current_state = state
        for op in self._operators:
            current_state = op(current_state, context)
        return current_state

    def check_invariants(self, state: FieldState, context: OperatorContext) -> Tuple[List[str], bool]:
        """Check invariants (handled by individual operators)."""
        return ["composite_pass"], True

    def __rshift__(self, other: Operator) -> 'CompositeOperator':
        """Extend the composite with another operator."""
        return CompositeOperator(self._operators + [other])


class IdentityOperator(Operator):
    """Identity operator that returns state unchanged."""

    def __init__(self):
        super().__init__(name="Identity", category=OperatorCategory.ARITHMETIC)

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        return state


class FunctionOperator(Operator):
    """
    Operator constructed from a function.

    Convenience class for simple transformations.
    """

    def __init__(self,
                 name: str,
                 category: OperatorCategory,
                 func: Callable[[FieldState, OperatorContext], FieldState]):
        super().__init__(name, category)
        self._func = func

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        return self._func(state, context)


# Operator factories

def make_pointwise_operator(
    name: str,
    transform: Callable[[Tuple[int, ...], Any, OperatorContext], Any],
    category: OperatorCategory = OperatorCategory.ARITHMETIC
) -> Operator:
    """
    Create an operator that applies a transform to each lattice point.

    Args:
        name: Operator name
        transform: Function (coords, value, context) -> new_value
        category: Operator category

    Returns:
        Operator that applies transform pointwise
    """
    def apply_func(state: FieldState, context: OperatorContext) -> FieldState:
        new_state = state.copy()
        new_state.apply_func(lambda c, v: transform(c, v, context))
        return new_state

    return FunctionOperator(name, category, apply_func)


def make_stencil_operator(
    name: str,
    stencil: Callable[[FieldState, Tuple[int, ...], OperatorContext], Any],
    category: OperatorCategory = OperatorCategory.FIELD
) -> Operator:
    """
    Create an operator that applies a stencil (local neighborhood) operation.

    Args:
        name: Operator name
        stencil: Function (state, coords, context) -> new_value at coords
        category: Operator category

    Returns:
        Operator that applies stencil to all points
    """
    def apply_func(state: FieldState, context: OperatorContext) -> FieldState:
        from ..state import RationalComplex
        new_state = state.copy()
        new_values = {}

        for point in state.lattice.iterate_all():
            new_val = stencil(state, point.coords, context)
            if isinstance(new_val, RationalComplex):
                new_values[point.coords] = new_val
            else:
                new_values[point.coords] = RationalComplex.from_real(new_val)

        for coords, val in new_values.items():
            new_state._psi[coords] = val
        new_state._derived.invalidate()

        return new_state

    return FunctionOperator(name, category, apply_func)


# Self-test
def _self_test():
    """Run basic operator tests."""
    from ..lattice import create_3d_lattice
    from ..state import create_zero_field, RationalComplex

    lattice = create_3d_lattice(4, 4, 4)
    state = create_zero_field(lattice)
    context = OperatorContext()

    # Test identity operator
    identity = IdentityOperator()
    result = identity(state, context)
    assert result.total_norm_squared() == 0

    # Test pointwise operator
    def add_one(coords, val, ctx):
        return val + RationalComplex.from_int(1)

    add_op = make_pointwise_operator("AddOne", add_one)
    result = add_op(state, context)
    assert result.total_norm_squared() == lattice.total_sites

    # Test composition
    double = add_op >> add_op
    result = double(state, context)
    expected_norm_sq = 4 * lattice.total_sites  # (0+1+1)² = 4 per point
    assert result.total_norm_squared() == expected_norm_sq

    # Test trace
    trace = OperatorTrace()
    context.trace = trace
    _ = identity(state, context)
    assert len(trace.entries) == 1
    assert trace.entries[0].operator_name == "Identity"


if __name__ == "__main__":
    _self_test()
