"""
R4 Coupling/Topology Operator (CODEX 5.2)

Implements the R4 adjacency kernel and cross-lattice coupling:

    T_R4(x) = f(N_R4(x), Ψ)

Where N_R4(x) ⊆ Ω is the R4 neighborhood and the coupling function
creates entanglement-like correlations in the classical evolution.

Features:
- Explicit adjacency list with exact rational weights
- Deterministic neighborhood ordering
- Configurable coupling shells (nearest, face diagonal, body diagonal)
- Energy proxy computation for observables
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from typing import Dict, Tuple, List, Optional, Callable, Any

# CRITICAL: math module FORBIDDEN - violates exact arithmetic
# R4 coupling operations must use ONLY Vedic sutra functions and rational arithmetic

from .base import Operator, OperatorCategory, OperatorContext
from ..state import FieldState, RationalComplex
from ..lattice import ToroidalHypercube, LatticePoint, R4AdjacencyKernel


@dataclass
class R4CouplingConfig:
    """
    R4 coupling configuration.

    Defines coupling strengths for different neighborhood shells
    and cross-lattice interaction parameters.
    """
    # Shell coupling weights (exact rationals)
    shell_1_weight: Fraction = Fraction(1)      # Nearest neighbors
    shell_2_weight: Fraction = Fraction(1, 2)   # Face diagonals
    shell_3_weight: Fraction = Fraction(1, 4)   # Body diagonals / R4 extension

    # Coupling mode
    coupling_mode: str = "linear"  # "linear", "quadratic", "exponential"

    # Global coupling strength
    coupling_strength: Fraction = Fraction(1, 10)

    # Phase coherence threshold
    coherence_threshold: Fraction = Fraction(1, 100)

    # Maximum coupling radius
    max_radius: int = 3

    # Enable cross-lattice (R4) coupling
    enable_r4: bool = True


class R4CouplingOperator(Operator):
    """
    R4 coupling operator for cross-lattice correlations.

    Implements the T_R4(x) term in the GRVQ ansatz:
    - Computes weighted average of neighbor field values
    - Applies coupling function based on configuration
    - Creates entanglement-like correlations across the lattice
    """

    def __init__(self, config: R4CouplingConfig = None):
        super().__init__(name="R4Coupling", category=OperatorCategory.COUPLING)
        self.config = config or R4CouplingConfig()

    def compute_coupling(self, center_val: RationalComplex,
                         neighbor_vals: List[Tuple[RationalComplex, Fraction]],
                         context: OperatorContext) -> RationalComplex:
        """
        Compute R4 coupling contribution from neighbors.

        Args:
            center_val: Field value at center point
            neighbor_vals: List of (neighbor_value, weight) pairs
            context: Operator context

        Returns:
            Coupling contribution to add to field
        """
        if not neighbor_vals:
            return RationalComplex.zero()

        # Compute weighted average of neighbors
        total = RationalComplex.zero()
        total_weight = Fraction(0)

        for val, weight in neighbor_vals:
            total = total + val * RationalComplex.from_real(weight)
            total_weight += weight

        if total_weight == 0:
            return RationalComplex.zero()

        avg = total * RationalComplex.from_real(Fraction(1) / total_weight)

        # Apply coupling function
        if self.config.coupling_mode == "linear":
            # Linear coupling: γ * (avg - center)
            coupling = (avg - center_val) * RationalComplex.from_real(self.config.coupling_strength)

        elif self.config.coupling_mode == "quadratic":
            # Quadratic coupling: γ * |avg - center|² * (avg - center)
            diff = avg - center_val
            magnitude_sq = diff.norm_squared()
            coupling = diff * RationalComplex.from_real(self.config.coupling_strength * magnitude_sq)

        elif self.config.coupling_mode == "exponential":
            # FORBIDDEN: Exponential requires exp() and norm() which violate exact arithmetic
            # TODO: Reimplement using ONLY Vedic sutra functions
            # - Use rational polynomial approximations for decay: 1/(1+x)^n
            # - Use norm_squared() instead of norm()
            # - Use nikhilam sutra for complement operations
            # OLD CODE (FORBIDDEN):
            # decay = math.exp(-diff.norm() / 0.5)
            diff = avg - center_val
            # Rational polynomial decay instead: 1/(1 + |diff|²)
            decay_denom = Fraction(1) + diff.norm_squared()
            decay = Fraction(1) / decay_denom
            coupling = diff * RationalComplex.from_real(self.config.coupling_strength * decay)

        else:
            coupling = RationalComplex.zero()

        return coupling

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply R4 coupling to evolve field.

        Each point receives contributions from its R4 neighborhood.
        """
        new_state = state.copy()

        for point in state.lattice.iterate_all():
            center_val = state.get(point)

            if not self.config.enable_r4:
                continue

            # Gather neighbor values with weights
            neighbor_vals = []
            for neighbor, weight in state.lattice.neighbors(point):
                neighbor_val = state.get(neighbor)
                neighbor_vals.append((neighbor_val, weight))

            # Compute coupling contribution
            coupling = self.compute_coupling(center_val, neighbor_vals, context)

            # Update field
            new_val = center_val + coupling
            new_state.set(point, new_val)

        return new_state


class R4EnergyOperator(Operator):
    """
    R4 coupling energy computation operator.

    Computes the R4 coupling energy proxy for observables (CODEX 7.1):
        E_R4 = Σ_x Σ_{y∈N_R4(x)} w(x,y) * |Ψ(x) - Ψ(y)|²

    This measures the "frustration" or "entanglement" in the field.
    """

    def __init__(self, config: R4CouplingConfig = None):
        super().__init__(name="R4Energy", category=OperatorCategory.FIELD)
        self.config = config or R4CouplingConfig()

    def compute_energy(self, state: FieldState) -> Fraction:
        """
        Compute total R4 coupling energy.

        Returns exact rational energy value.
        """
        total_energy = Fraction(0)

        for point in state.lattice.iterate_all():
            center_val = state.get(point)

            for neighbor, weight in state.lattice.neighbors(point):
                neighbor_val = state.get(neighbor)

                # |Ψ(x) - Ψ(y)|²
                diff = center_val - neighbor_val
                diff_sq = diff.norm_squared()

                total_energy += weight * diff_sq

        # Divide by 2 to avoid double-counting
        return total_energy / 2

    def compute_local_energy(self, state: FieldState, point: LatticePoint) -> Fraction:
        """Compute R4 energy at a single point."""
        center_val = state.get(point)
        local_energy = Fraction(0)

        for neighbor, weight in state.lattice.neighbors(point):
            neighbor_val = state.get(neighbor)
            diff = center_val - neighbor_val
            local_energy += weight * diff.norm_squared()

        return local_energy

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Compute R4 energy and store in context.

        Does not modify the state, only computes observable.
        """
        total_energy = self.compute_energy(state)
        context.set_param('r4_energy', total_energy)

        # Also compute local energy map
        energy_map = {}
        for point in state.lattice.iterate_all():
            energy_map[point.coords] = self.compute_local_energy(state, point)

        context.set_param('r4_energy_map', energy_map)

        return state


class R4CoherenceOperator(Operator):
    """
    R4 phase coherence operator.

    Computes and enforces phase coherence across the R4 neighborhood.
    High coherence means neighbors have aligned phases.
    """

    def __init__(self, config: R4CouplingConfig = None):
        super().__init__(name="R4Coherence", category=OperatorCategory.CONSTRAINT)
        self.config = config or R4CouplingConfig()

    def compute_local_coherence(self, state: FieldState, point: LatticePoint) -> float:
        """
        Compute local phase coherence at a point.

        Returns value in [0, 1] where 1 = perfect coherence.
        """
        # FORBIDDEN: Phase coherence requires atan2 and math.pi which violate exact arithmetic
        # TODO: Reimplement using ONLY Vedic sutra functions and exact arithmetic
        # - Use amplitude (norm_squared) coherence instead of phase coherence
        # - Use vyashtisamanstih sutra for part/whole relationships
        # - Return exact Fraction instead of float
        # OLD CODE (FORBIDDEN):
        # center_phase = state.phase(point) - uses atan2
        # Phase operations cannot exist in exact rational arithmetic

        # Use amplitude coherence instead (exact)
        center_intensity = state.intensity(point)
        if center_intensity == 0:
            return Fraction(1)

        intensity_diffs = []
        for neighbor, _ in state.lattice.neighbors(point):
            neighbor_intensity = state.intensity(neighbor)
            diff_sq = abs(neighbor_intensity - center_intensity)
            intensity_diffs.append(diff_sq)

        if not intensity_diffs:
            return Fraction(1)

        avg_diff = sum(intensity_diffs) / len(intensity_diffs)
        # Normalize to [0, 1] - higher coherence when diffs are small
        coherence = Fraction(1) / (Fraction(1) + avg_diff)

        return coherence

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply coherence enforcement.

        Points with low coherence are damped.
        """
        new_state = state.copy()

        for point in state.lattice.iterate_all():
            psi = state.get(point)
            coherence = self.compute_local_coherence(state, point)

            # Damp incoherent regions
            threshold = float(self.config.coherence_threshold)
            if coherence < threshold:
                damping = coherence / threshold
                new_psi = psi * RationalComplex.from_real(Fraction(damping).limit_denominator(10000))
                new_state.set(point, new_psi)

        return new_state


class R4EntanglementOperator(Operator):
    """
    R4 entanglement-like correlation operator.

    Creates non-local correlations across the R4 adjacency structure.
    This is NOT quantum entanglement - it's a classical analog that
    creates structured correlations in the field.
    """

    def __init__(self, config: R4CouplingConfig = None):
        super().__init__(name="R4Entanglement", category=OperatorCategory.COUPLING)
        self.config = config or R4CouplingConfig()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply entanglement-like correlation step.

        Creates correlations between R4 neighbors:
        - Product state: neighbors are independently modulated
        - Correlated state: neighbors evolve together
        """
        new_state = state.copy()
        entanglement_strength = context.get_param('r4_entanglement', Fraction(1, 100))

        for point in state.lattice.iterate_all():
            center_val = state.get(point)

            # Compute "entanglement" contribution
            # This mixes the center with a nonlinear function of neighbors
            neighbor_product = RationalComplex.one()
            for neighbor, weight in state.lattice.neighbors(point):
                neighbor_val = state.get(neighbor)
                if not neighbor_val.is_zero():
                    # FORBIDDEN: Normalization requires norm() which uses sqrt
                    # TODO: Use unnormalized values or sutra-based scaling
                    # - Use norm_squared() for threshold checks
                    # - Avoid division by magnitude (introduces floats)
                    # OLD CODE (FORBIDDEN):
                    # norm = neighbor_val.norm()
                    # normalized = neighbor_val / norm

                    # Use unnormalized contribution (exact)
                    contribution = RationalComplex.one() + neighbor_val * RationalComplex.from_real(weight * entanglement_strength)
                    neighbor_product = neighbor_product * contribution

            # Apply to center
            new_val = center_val * neighbor_product
            new_state.set(point, new_val)

        return new_state


class R4CompositeOperator(Operator):
    """
    Complete R4 topology operator.

    Combines all R4 operations:
    1. R4 coupling (diffusive mixing)
    2. Coherence enforcement
    3. Entanglement-like correlations
    4. Energy computation
    """

    def __init__(self, config: R4CouplingConfig = None):
        super().__init__(name="R4Composite", category=OperatorCategory.COUPLING)
        self.config = config or R4CouplingConfig()

        self.coupling_op = R4CouplingOperator(config)
        self.coherence_op = R4CoherenceOperator(config)
        self.entanglement_op = R4EntanglementOperator(config)
        self.energy_op = R4EnergyOperator(config)

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """Apply complete R4 topology update."""
        # Compute initial energy
        initial_energy = self.energy_op.compute_energy(state)

        # Apply R4 operations
        state = self.coupling_op.apply(state, context)
        state = self.coherence_op.apply(state, context)
        state = self.entanglement_op.apply(state, context)

        # Compute final energy
        state = self.energy_op.apply(state, context)
        final_energy = context.get_param('r4_energy')

        # Log energy delta
        context.set_param('r4_energy_delta', float(final_energy - initial_energy))

        return state

    def check_invariants(self, state: FieldState, context: OperatorContext) -> Tuple[List[str], bool]:
        """Check R4 invariants."""
        invariants = ["coupling_stable", "energy_bounded"]
        passed = True

        # Check field stability
        max_bound = context.get_param('r4_max_bound', Fraction(1000))
        if not state.validate_bounded(max_bound):
            passed = False

        # Check energy is finite
        r4_energy = context.get_param('r4_energy')
        if r4_energy is not None and r4_energy > Fraction(10**10):
            passed = False

        return invariants, passed


# Utility functions

def compute_r4_correlation_matrix(state: FieldState,
                                  sample_points: List[LatticePoint] = None) -> Dict[Tuple[int, int], float]:
    """
    Compute correlation matrix between sample points.

    Returns dict mapping (i, j) -> correlation coefficient.
    """
    if sample_points is None:
        # Sample uniformly
        sample_points = list(state.lattice.iterate_all())[:100]

    n = len(sample_points)
    correlations = {}

    for i in range(n):
        for j in range(i, n):
            psi_i = state.get(sample_points[i])
            psi_j = state.get(sample_points[j])

            # FORBIDDEN: Correlation with norm() division violates exact arithmetic
            # TODO: Use norm_squared() based correlation
            # - Correlation = Re(ψ_i* ψ_j)² / (|ψ_i|² |ψ_j|²) - all exact
            # OLD CODE (FORBIDDEN):
            # norm_i = psi_i.norm() - uses sqrt
            # corr = product.real / (norm_i * norm_j) - float division

            # Exact correlation using norm_squared
            product = psi_i.conjugate() * psi_j
            norm_sq_i = psi_i.norm_squared()
            norm_sq_j = psi_j.norm_squared()

            threshold = Fraction(1, 1000000)  # 0.001²
            if norm_sq_i > threshold and norm_sq_j > threshold:
                # Exact: Re(ψ_i* ψ_j) / sqrt(|ψ_i|² |ψ_j|²)
                # But sqrt violates exactness, so use squared version:
                # corr² = Re(ψ_i* ψ_j)² / (|ψ_i|² |ψ_j|²)
                corr_sq = (product.real * product.real) / (norm_sq_i * norm_sq_j)
                corr = float(corr_sq)  # For now, convert only at output
            else:
                corr = 0.0

            correlations[(i, j)] = corr
            correlations[(j, i)] = corr

    return correlations


# Factory functions

def create_r4_pipeline(coupling_strength: Fraction = Fraction(1, 10),
                       enable_entanglement: bool = True) -> R4CompositeOperator:
    """
    Create standard R4 topology pipeline.
    """
    config = R4CouplingConfig(
        coupling_strength=coupling_strength,
        enable_r4=True
    )
    return R4CompositeOperator(config)


# Self-test
def _self_test():
    """Run basic R4 coupling tests."""
    from ..lattice import create_3d_lattice
    from ..state import create_gaussian_field

    # Create test lattice and field
    lattice = create_3d_lattice(8, 8, 8)
    center = (4, 4, 4)
    state = create_gaussian_field(lattice, center, sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    # Test R4 coupling operator
    config = R4CouplingConfig()
    coupling_op = R4CouplingOperator(config)
    result = coupling_op(state, context)

    # Verify result is valid
    assert result.validate_bounded(Fraction(1000))

    # Test R4 energy computation
    energy_op = R4EnergyOperator(config)
    _ = energy_op(state, context)
    energy = context.get_param('r4_energy')
    assert energy is not None
    assert energy >= 0

    # Test composite operator
    r4 = create_r4_pipeline()
    result = r4(state, context)
    assert result.validate_bounded(Fraction(1000))

    # Verify energy was computed
    final_energy = context.get_param('r4_energy')
    assert final_energy is not None


_self_test()
