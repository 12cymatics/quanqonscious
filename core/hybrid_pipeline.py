"""
Two-Lane Hybrid Execution Pipeline (CODEX 3)

Implements the required two-lane pipeline:

Lane A — Classical/HPC (authoritative evolution):
- Discrete lattice evolution steps
- Applies sutra operators, MSTVQ stress-tension transforms, boundary/toroidal wraps
- Produces Ψ_{t+Δt} and derived observables

Lane B — Quantum assist (bounded role):
- Produces auxiliary objects only (coefficient proposals, symmetry sector tags,
  small entanglement kernels, phase templates)
- Must be reproducible (seeded) and auditable
- Must not claim physical "quantum advantage"

Merge rule:
Lane B may adjust parameters α_j, select subsets of S_j, or produce T_R4 micro-kernels,
but Lane A remains the evolution authority.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import math
import random

from .state import FieldState, RationalComplex
from .lattice import ToroidalHypercube
from .operators.base import Operator, OperatorContext, OperatorTrace, CompositeOperator
from .operators.grvq_ansatz import GRVQAnsatzOperator, ShapeFunction
from .operators.mstvq import MSTVQCompositeOperator, MSTVQConfig
from .operators.r4_coupling import R4CompositeOperator, R4CouplingConfig
from .operators.sutra_ops import get_all_sutras, create_sutra_pipeline
from .observables import ObservableSet, create_standard_observables, InvariantChecker, create_standard_invariants
from .trace import EvolutionTrace, StateCheckpoint


class LaneType(Enum):
    """Execution lane type."""
    CLASSICAL = "classical"  # Lane A
    QUANTUM = "quantum"      # Lane B


@dataclass
class QuantumAssistOutput:
    """
    Output from Lane B (quantum assist).

    Contains auxiliary objects that Lane A can use:
    - Coefficient proposals (α_j adjustments)
    - Shape function selections
    - Phase templates
    - Entanglement micro-kernels
    """
    # Coefficient adjustments (Sutra/GRVQ)
    coefficient_proposals: Dict[str, Fraction] = field(default_factory=dict)

    # Shape function selection (which modes to activate)
    active_modes: List[int] = field(default_factory=list)

    # Phase template (suggested phase pattern)
    phase_template: Optional[Dict[Tuple[int, ...], float]] = None

    # Entanglement kernel (small R4 coupling adjustments)
    r4_kernel_adjustments: Dict[Tuple[int, int], Fraction] = field(default_factory=dict)

    # Metadata for auditing
    seed: int = 0
    num_shards: int = 1  # "Quantum parallel" shards
    merge_cost: float = 0.0  # Cost of merging results


@dataclass
class QuantumAssistConfig:
    """
    Configuration for Lane B (quantum assist).

    Defines:
    - What auxiliary computations to perform
    - Reproducibility seed
    - Resource bounds
    """
    enabled: bool = True
    seed: int = 42

    # Which assists to enable
    enable_coefficient_tuning: bool = True
    enable_mode_selection: bool = True
    enable_phase_template: bool = True
    enable_r4_kernel: bool = False

    # Resource bounds (CODEX 3.2: operational definition)
    max_shards: int = 8       # Maximum parallel "circuits"
    max_kernel_size: int = 16  # Maximum R4 kernel adjustments
    coherence_time: float = 0.0  # Not physical, just a bound

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'seed': self.seed,
            'max_shards': self.max_shards,
            'max_kernel_size': self.max_kernel_size,
        }


class QuantumAssistLane:
    """
    Lane B: Quantum Assist (CODEX 3.1).

    NOT a real quantum computer - this is a classical simulator that:
    - Produces structured auxiliary outputs
    - Uses seeded random for reproducibility
    - Provides candidate adjustments for Lane A

    "Quantum-parallel" here means (CODEX 3.2):
    - number of shards/circuits: how many independent computations
    - merge cost: how results are combined
    - latency: time to produce output
    """

    def __init__(self, config: QuantumAssistConfig = None):
        self.config = config or QuantumAssistConfig()
        self._rng = random.Random(self.config.seed)

    def reset_seed(self, seed: int) -> None:
        """Reset random state for reproducibility."""
        self._rng = random.Random(seed)

    def compute(self, state: FieldState, context: OperatorContext) -> QuantumAssistOutput:
        """
        Compute auxiliary outputs for Lane A.

        This is NOT quantum computation - it's a structured classical
        heuristic that provides suggestions for parameter tuning.
        """
        if not self.config.enabled:
            return QuantumAssistOutput(seed=self.config.seed)

        output = QuantumAssistOutput(seed=self.config.seed)

        # Coefficient tuning: analyze field and suggest adjustments
        if self.config.enable_coefficient_tuning:
            output.coefficient_proposals = self._propose_coefficients(state, context)

        # Mode selection: choose active shape functions
        if self.config.enable_mode_selection:
            output.active_modes = self._select_modes(state, context)

        # Phase template: generate phase pattern suggestion
        if self.config.enable_phase_template:
            output.phase_template = self._generate_phase_template(state, context)

        # R4 kernel adjustments
        if self.config.enable_r4_kernel:
            output.r4_kernel_adjustments = self._compute_r4_kernel(state, context)

        # Record "quantum-parallel" metrics (CODEX 3.2)
        output.num_shards = min(self.config.max_shards, 4)  # Simulated shards
        output.merge_cost = output.num_shards * 0.1  # Simple merge cost model

        return output

    def _propose_coefficients(self, state: FieldState,
                              context: OperatorContext) -> Dict[str, Fraction]:
        """Propose coefficient adjustments based on field analysis."""
        proposals = {}

        # Analyze field intensity distribution
        intensities = []
        for point in list(state.lattice.iterate_all())[:100]:  # Sample
            intensities.append(float(state.intensity(point)))

        if intensities:
            mean_intensity = sum(intensities) / len(intensities)
            std_intensity = math.sqrt(sum((i - mean_intensity)**2 for i in intensities) / len(intensities))

            # Propose GRVQ coefficient based on intensity pattern
            if std_intensity > mean_intensity:
                # High variance: suggest stronger damping
                proposals['grvq_damping'] = Fraction(1, 5)
            else:
                # Low variance: suggest enhancement
                proposals['grvq_enhancement'] = Fraction(1, 10)

            # Random exploration (seeded)
            noise = self._rng.gauss(0, 0.1)
            proposals['exploration_noise'] = Fraction(noise).limit_denominator(1000)

        return proposals

    def _select_modes(self, state: FieldState,
                      context: OperatorContext) -> List[int]:
        """Select which shape function modes should be active."""
        # Analyze field for mode content
        # This is a simplified selection based on field statistics

        active = []
        total_norm = float(state.total_norm_squared())

        if total_norm > 0:
            # Select modes based on field characteristics
            # Low norm: activate more modes
            # High norm: fewer, more focused modes
            num_modes = max(1, min(5, int(10 / (total_norm + 1))))
            active = list(range(num_modes))

            # Add randomness (seeded)
            if self._rng.random() > 0.5:
                active.append(self._rng.randint(0, 7))

        return sorted(set(active))

    def _generate_phase_template(self, state: FieldState,
                                 context: OperatorContext) -> Dict[Tuple[int, ...], float]:
        """Generate suggested phase pattern."""
        template = {}

        # Create a simple wave-like phase template
        for point in list(state.lattice.iterate_all())[:50]:  # Sample
            coords = point.coords
            # Phase based on position (standing wave pattern)
            phase = sum(c * math.pi / n for c, n in zip(coords, state.lattice.shape))
            phase += self._rng.gauss(0, 0.1)  # Small noise
            template[coords] = phase % (2 * math.pi)

        return template

    def _compute_r4_kernel(self, state: FieldState,
                           context: OperatorContext) -> Dict[Tuple[int, int], Fraction]:
        """Compute R4 coupling kernel adjustments."""
        adjustments = {}

        # Limit to config bound
        max_adj = self.config.max_kernel_size

        # Sample random pairs
        points = list(state.lattice.iterate_all())[:20]

        for i in range(min(max_adj, len(points) * (len(points) - 1) // 2)):
            p1_idx = self._rng.randint(0, len(points) - 1)
            p2_idx = self._rng.randint(0, len(points) - 1)
            if p1_idx != p2_idx:
                # Suggest adjustment based on value correlation
                v1 = state.get(points[p1_idx])
                v2 = state.get(points[p2_idx])
                correlation = float((v1.conjugate() * v2).real)
                adjustment = Fraction(correlation / 10).limit_denominator(100)
                adjustments[(p1_idx, p2_idx)] = adjustment

        return adjustments


@dataclass
class ClassicalEvolutionConfig:
    """Configuration for Lane A (classical evolution)."""
    dt: Fraction = Fraction(1, 100)
    use_grvq: bool = True
    use_mstvq: bool = True
    use_r4: bool = True
    use_sutras: bool = True
    sutra_sequence: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])

    # MSTVQ parameters (CODEX 6.1)
    h_m: Fraction = Fraction(1)
    stress_coupling: Fraction = Fraction(1, 10)

    # Bounds for stability
    max_field_bound: Fraction = Fraction(1000)


class ClassicalEvolutionLane:
    """
    Lane A: Classical/HPC Evolution (CODEX 3.1).

    The authoritative evolution lane that:
    - Performs discrete lattice evolution steps
    - Applies sutra operators, MSTVQ, GRVQ, R4 coupling
    - Produces Ψ_{t+Δt} and observables
    """

    def __init__(self, config: ClassicalEvolutionConfig = None):
        self.config = config or ClassicalEvolutionConfig()
        self._build_operators()

    def _build_operators(self) -> None:
        """Build operator pipeline."""
        operators = []

        # GRVQ ansatz
        if self.config.use_grvq:
            self.grvq = GRVQAnsatzOperator()
            operators.append(self.grvq)

        # MSTVQ
        if self.config.use_mstvq:
            mstvq_config = MSTVQConfig(
                h_m=self.config.h_m,
                stress_coupling=self.config.stress_coupling
            )
            self.mstvq = MSTVQCompositeOperator(mstvq_config)
            operators.append(self.mstvq)

        # R4 coupling
        if self.config.use_r4:
            self.r4 = R4CompositeOperator()
            operators.append(self.r4)

        # Sutra pipeline
        if self.config.use_sutras:
            self.sutras = create_sutra_pipeline(self.config.sutra_sequence)
            operators.append(self.sutras)

        self.pipeline = CompositeOperator(operators) if operators else None

    def apply_quantum_adjustments(self, context: OperatorContext,
                                   quantum_output: QuantumAssistOutput) -> None:
        """
        Apply adjustments from Lane B to evolution parameters.

        This is the merge step where Lane B suggestions are incorporated.
        Lane A remains authoritative - it decides what to accept.
        """
        # Apply coefficient proposals
        for key, value in quantum_output.coefficient_proposals.items():
            if key == 'grvq_damping' and hasattr(self, 'grvq'):
                context.set_param('grvq_damping', value)
            elif key == 'grvq_enhancement':
                context.set_param('grvq_enhancement', value)

        # Apply mode selection (not directly modifying GRVQ here)
        context.set_param('active_modes', quantum_output.active_modes)

        # Phase template is available but not directly applied
        # (Lane A decides whether to use it)
        if quantum_output.phase_template:
            context.set_param('phase_template_available', True)

    def evolve(self, state: FieldState, context: OperatorContext,
               quantum_output: Optional[QuantumAssistOutput] = None) -> FieldState:
        """
        Perform one evolution step.

        Args:
            state: Current field state
            context: Operator context
            quantum_output: Optional Lane B output for parameter adjustment

        Returns:
            Evolved field state
        """
        # Apply quantum adjustments if provided
        if quantum_output is not None:
            self.apply_quantum_adjustments(context, quantum_output)

        # Apply operator pipeline
        if self.pipeline is not None:
            state = self.pipeline(state, context)

        return state


@dataclass
class HybridPipelineConfig:
    """Configuration for the hybrid two-lane pipeline."""
    classical: ClassicalEvolutionConfig = field(default_factory=ClassicalEvolutionConfig)
    quantum: QuantumAssistConfig = field(default_factory=QuantumAssistConfig)

    # Evolution parameters
    num_steps: int = 100
    checkpoint_interval: int = 10

    # Observable computation
    compute_observables: bool = True

    # Invariant checking
    check_invariants: bool = True
    fail_fast: bool = True  # Stop on invariant failure


class HybridPipeline:
    """
    Two-Lane Hybrid Execution Pipeline (CODEX 3).

    Coordinates:
    - Lane A: Classical evolution (authoritative)
    - Lane B: Quantum assist (auxiliary)

    With:
    - Observable computation
    - Invariant checking
    - Trace recording
    """

    def __init__(self, config: HybridPipelineConfig = None):
        self.config = config or HybridPipelineConfig()

        # Initialize lanes
        self.classical = ClassicalEvolutionLane(self.config.classical)
        self.quantum = QuantumAssistLane(self.config.quantum)

        # Observables and invariants
        self.observables = create_standard_observables()
        self.invariants = create_standard_invariants()

    def run(self, initial_state: FieldState,
            context: Optional[OperatorContext] = None) -> Tuple[FieldState, EvolutionTrace, List[Dict[str, Any]]]:
        """
        Run the hybrid evolution pipeline.

        Args:
            initial_state: Starting field state
            context: Optional initial context

        Returns:
            (final_state, trace, observable_history)
        """
        if context is None:
            context = OperatorContext(dt=self.config.classical.dt)

        # Initialize trace
        trace = EvolutionTrace(checkpoint_interval=self.config.checkpoint_interval)
        trace.start(initial_state)
        context.trace = trace.operator_trace

        # Initialize observable history
        observable_history = []

        # Record initial norm for conservation check
        context.set_param('initial_norm_sq', float(initial_state.total_norm_squared()))
        context.set_param('max_field_bound', self.config.classical.max_field_bound)

        # Evolution loop
        state = initial_state.copy()

        for t in range(self.config.num_steps):
            context = context.with_timestep(t)

            # Lane B: Quantum assist (compute suggestions)
            quantum_output = self.quantum.compute(state, context)

            # Lane A: Classical evolution (authoritative)
            state = self.classical.evolve(state, context, quantum_output)

            # Record checkpoint
            trace.record_step(t, state)

            # Compute observables
            if self.config.compute_observables:
                obs = self.observables.compute_all(state, context)
                obs['timestep'] = t
                obs['quantum_shards'] = quantum_output.num_shards
                obs['quantum_merge_cost'] = quantum_output.merge_cost
                observable_history.append(obs)

            # Check invariants
            if self.config.check_invariants:
                all_passed, results = self.invariants.verify_all(state, context)
                if not all_passed and self.config.fail_fast:
                    failures = [f"{name}: {msg}" for name, (passed, msg) in results.items() if not passed]
                    raise RuntimeError(f"Invariant failure at step {t}: {failures}")

        # Finalize trace
        trace.finish(state)

        return state, trace, observable_history


def create_standard_pipeline() -> HybridPipeline:
    """Create a standard hybrid pipeline with default configuration."""
    return HybridPipeline()


def create_classical_only_pipeline() -> HybridPipeline:
    """Create a pipeline with quantum assist disabled."""
    config = HybridPipelineConfig()
    config.quantum.enabled = False
    return HybridPipeline(config)


def create_sutra_pipeline_hybrid(sutra_numbers: List[int]) -> HybridPipeline:
    """Create a pipeline with specific sutra sequence."""
    config = HybridPipelineConfig()
    config.classical.sutra_sequence = sutra_numbers
    return HybridPipeline(config)


# Self-test
def _self_test():
    """Run basic hybrid pipeline tests."""
    from .lattice import create_3d_lattice
    from .state import create_gaussian_field

    # Create test environment
    lattice = create_3d_lattice(8, 8, 8)
    state = create_gaussian_field(lattice, (4, 4, 4), sigma=1.5, amplitude=1.0)

    # Test quantum assist
    quantum = QuantumAssistLane()
    context = OperatorContext()
    output = quantum.compute(state, context)
    assert isinstance(output, QuantumAssistOutput)
    assert output.seed == 42

    # Test classical evolution
    classical = ClassicalEvolutionLane()
    evolved = classical.evolve(state.copy(), context)
    assert evolved.validate_bounded(Fraction(10000))

    # Test hybrid pipeline (small number of steps)
    config = HybridPipelineConfig()
    config.num_steps = 3
    config.classical.sutra_sequence = [1, 3]

    pipeline = HybridPipeline(config)
    final, trace, history = pipeline.run(state)

    assert final.validate_bounded(Fraction(10000))
    assert trace.final_checkpoint is not None
    assert len(history) == 3


_self_test()
