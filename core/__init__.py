"""
GRVQ/MSTVQ/TGCR Cymatic Simulation Core (CODEX Compliant)

This module implements the complete cymatic field simulation engine with:
- Vedic Sutra operator stack (29-sutra library as first-class operators)
- GRVQ ansatz (wavefunction/field composition layer)
- MSTVQ (magnetic stress-tension knobs)
- TGCR topology (toroidal hypercube with R4 coupling)
- Two-lane hybrid execution (Classical + Quantum assist)

All implementations follow the CODEX specification:
- No placeholder code
- Exact arithmetic (rationals) by default
- Operator trace for replay
- Invariant checking per step
"""

from .lattice import (
    ToroidalHypercube,
    LatticePoint,
    R4AdjacencyKernel,
    create_3d_lattice,
    create_4d_hypercube,
    create_cubic_lattice,
)

from .state import (
    FieldState,
    FieldStateSnapshot,
    RationalComplex,
    ArithmeticMode,
    create_zero_field,
    create_uniform_field,
    create_gaussian_field,
)

from .operators import (
    Operator,
    OperatorContext,
    OperatorCategory,
    OperatorTrace,
    CompositeOperator,
    IdentityOperator,
)

from .operators.grvq_ansatz import (
    GRVQAnsatzOperator,
    GRVQEvolutionOperator,
    ShapeFunction,
    RadialSuppression,
    VedicCarrier,
    create_cymatic_ansatz,
    create_bessel_ansatz,
)

from .operators.mstvq import (
    MSTVQConfig,
    MSTVQCompositeOperator,
    MSTVQStressOperator,
    MSTVQTensionOperator,
    StressTensorField,
    create_mstvq_pipeline,
)

from .operators.r4_coupling import (
    R4CouplingConfig,
    R4CompositeOperator,
    R4CouplingOperator,
    R4EnergyOperator,
    R4CoherenceOperator,
    create_r4_pipeline,
)

from .operators.sutra_ops import (
    SutraOperator,
    get_all_sutras,
    get_sutra_by_number,
    get_sutras_by_category,
    create_sutra_pipeline,
)

from .observables import (
    Observable,
    ObservableSet,
    InvariantChecker,
    create_standard_observables,
    create_full_observables,
    create_standard_invariants,
    TotalNormSquared,
    TotalR4Energy,
    PhaseCoherence,
)

from .trace import (
    EvolutionTrace,
    StateCheckpoint,
    TraceReplayer,
    DeterminismVerifier,
    TraceProof,
    create_traced_evolution,
)

from .hybrid_pipeline import (
    HybridPipeline,
    HybridPipelineConfig,
    ClassicalEvolutionLane,
    QuantumAssistLane,
    QuantumAssistOutput,
    create_standard_pipeline,
    create_classical_only_pipeline,
)

__version__ = "1.0.0"
__codex_version__ = "CODEX_GRVQ_MSTVQ_TGCR_v1"

__all__ = [
    # Lattice
    'ToroidalHypercube',
    'LatticePoint',
    'R4AdjacencyKernel',
    'create_3d_lattice',
    'create_4d_hypercube',
    'create_cubic_lattice',

    # State
    'FieldState',
    'FieldStateSnapshot',
    'RationalComplex',
    'ArithmeticMode',
    'create_zero_field',
    'create_uniform_field',
    'create_gaussian_field',

    # Operators
    'Operator',
    'OperatorContext',
    'OperatorCategory',
    'OperatorTrace',
    'CompositeOperator',
    'IdentityOperator',

    # GRVQ
    'GRVQAnsatzOperator',
    'GRVQEvolutionOperator',
    'ShapeFunction',
    'RadialSuppression',
    'VedicCarrier',
    'create_cymatic_ansatz',
    'create_bessel_ansatz',

    # MSTVQ
    'MSTVQConfig',
    'MSTVQCompositeOperator',
    'MSTVQStressOperator',
    'MSTVQTensionOperator',
    'StressTensorField',
    'create_mstvq_pipeline',

    # R4
    'R4CouplingConfig',
    'R4CompositeOperator',
    'R4CouplingOperator',
    'R4EnergyOperator',
    'R4CoherenceOperator',
    'create_r4_pipeline',

    # Sutras
    'SutraOperator',
    'get_all_sutras',
    'get_sutra_by_number',
    'get_sutras_by_category',
    'create_sutra_pipeline',

    # Observables
    'Observable',
    'ObservableSet',
    'InvariantChecker',
    'create_standard_observables',
    'create_full_observables',
    'create_standard_invariants',

    # Trace
    'EvolutionTrace',
    'StateCheckpoint',
    'TraceReplayer',
    'DeterminismVerifier',
    'TraceProof',
    'create_traced_evolution',

    # Pipeline
    'HybridPipeline',
    'HybridPipelineConfig',
    'ClassicalEvolutionLane',
    'QuantumAssistLane',
    'QuantumAssistOutput',
    'create_standard_pipeline',
    'create_classical_only_pipeline',
]
