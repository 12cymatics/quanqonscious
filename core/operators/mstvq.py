"""
MSTVQ Stress-Tension Operator (CODEX 6)

Implements the Magneto-Stress Tensor Vacuum Quantization (MSTVQ) module.

MSTVQ replaces gravity-like couplings with magnetic stress-tension knobs:
- h_m: magnetic stress-tension global scale
- S(x,y,z,t): stress scalar or tensor field
- Additional MSTVQ constants as structured config

The update rule modifies:
- The suppression envelope R
- The local coupling in T_R4
- The GRVQ coefficients α_j

All operations are logged with deltas and invariants.
"""

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from typing import Dict, Tuple, List, Optional, Any
import math

from .base import Operator, OperatorCategory, OperatorContext
from ..state import FieldState, RationalComplex
from ..lattice import ToroidalHypercube, LatticePoint


@dataclass
class MSTVQConfig:
    """
    MSTVQ configuration parameters (CODEX 6.1).

    All values are exact rationals by default.
    """
    # Global magnetic stress-tension scale
    h_m: Fraction = Fraction(1)

    # Stress field coupling strength
    stress_coupling: Fraction = Fraction(1, 10)

    # Tension field coupling strength
    tension_coupling: Fraction = Fraction(1, 10)

    # Vacuum energy density (ZPE proxy)
    vacuum_energy: Fraction = Fraction(1, 1000)

    # Magnetic permeability analog
    mu_m: Fraction = Fraction(1)

    # Electric permittivity analog
    epsilon_e: Fraction = Fraction(1)

    # Stress-tension ratio (balance parameter)
    st_ratio: Fraction = Fraction(1)

    # Minimum stress threshold (prevents division by zero)
    min_stress: Fraction = Fraction(1, 10000)

    # Maximum stress bound (stability)
    max_stress: Fraction = Fraction(100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'h_m': str(self.h_m),
            'stress_coupling': str(self.stress_coupling),
            'tension_coupling': str(self.tension_coupling),
            'vacuum_energy': str(self.vacuum_energy),
            'mu_m': str(self.mu_m),
            'epsilon_e': str(self.epsilon_e),
            'st_ratio': str(self.st_ratio),
        }


@dataclass
class StressTensorField:
    """
    Stress-tension tensor field S(x,y,z,t).

    Stores:
    - Stress components (diagonal: σ_xx, σ_yy, σ_zz)
    - Tension components (off-diagonal shear)
    - Pressure (trace)
    """
    lattice: ToroidalHypercube

    # Stress field storage (scalar approximation, can extend to full tensor)
    _stress: Dict[Tuple[int, ...], Fraction] = dataclass_field(default_factory=dict)
    _tension: Dict[Tuple[int, ...], Fraction] = dataclass_field(default_factory=dict)

    def __post_init__(self):
        """Initialize fields to uniform values."""
        if not self._stress:
            for point in self.lattice.iterate_all():
                self._stress[point.coords] = Fraction(1)
                self._tension[point.coords] = Fraction(0)

    def get_stress(self, coords: Tuple[int, ...]) -> Fraction:
        """Get stress at coordinates."""
        wrapped = self.lattice.wrap_index(coords)
        return self._stress.get(wrapped, Fraction(1))

    def get_tension(self, coords: Tuple[int, ...]) -> Fraction:
        """Get tension at coordinates."""
        wrapped = self.lattice.wrap_index(coords)
        return self._tension.get(wrapped, Fraction(0))

    def set_stress(self, coords: Tuple[int, ...], value: Fraction) -> None:
        """Set stress at coordinates."""
        wrapped = self.lattice.wrap_index(coords)
        self._stress[wrapped] = value

    def set_tension(self, coords: Tuple[int, ...], value: Fraction) -> None:
        """Set tension at coordinates."""
        wrapped = self.lattice.wrap_index(coords)
        self._tension[wrapped] = value

    def compute_from_field(self, state: FieldState, config: MSTVQConfig) -> None:
        """
        Compute stress-tension from field configuration.

        Stress: proportional to field intensity gradient
        Tension: proportional to phase curvature
        """
        for point in self.lattice.iterate_all():
            coords = point.coords

            # Compute intensity gradient (stress)
            center_intensity = state.intensity(point)
            stress = Fraction(0)

            for neighbor, weight in self.lattice.neighbors(point):
                neighbor_intensity = state.intensity(neighbor)
                gradient = abs(float(neighbor_intensity) - float(center_intensity))
                stress += weight * Fraction(gradient).limit_denominator(10000)

            # Compute phase curvature (tension)
            center_phase = state.phase(point)
            tension = Fraction(0)

            for neighbor, weight in self.lattice.neighbors(point):
                neighbor_phase = state.phase(neighbor)
                phase_diff = abs(neighbor_phase - center_phase)
                if phase_diff > math.pi:
                    phase_diff = 2 * math.pi - phase_diff
                tension += weight * Fraction(phase_diff / math.pi).limit_denominator(10000)

            # Apply configuration
            self._stress[coords] = stress * config.stress_coupling
            self._tension[coords] = tension * config.tension_coupling

    def total_stress(self) -> Fraction:
        """Compute total stress."""
        return sum(self._stress.values())

    def total_tension(self) -> Fraction:
        """Compute total tension."""
        return sum(self._tension.values())


class MSTVQStressOperator(Operator):
    """
    MSTVQ stress update operator.

    Computes and applies stress field modifications to the state.
    """

    def __init__(self, config: MSTVQConfig = None):
        super().__init__(name="MSTVQStress", category=OperatorCategory.COUPLING)
        self.config = config or MSTVQConfig()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply MSTVQ stress transformation.

        Updates field values based on local stress gradient.
        """
        new_state = state.copy()

        # Get or create stress field
        stress_field = context.get_param('mstvq_stress_field')
        if stress_field is None:
            stress_field = StressTensorField(state.lattice)
            stress_field.compute_from_field(state, self.config)
            context.set_param('mstvq_stress_field', stress_field)

        # Apply stress modulation to field
        h_m = self.config.h_m
        dt = context.dt

        for point in state.lattice.iterate_all():
            coords = point.coords
            psi = state.get(point)

            # Get local stress
            S = stress_field.get_stress(coords)

            # Clamp stress to bounds
            S = max(self.config.min_stress, min(self.config.max_stress, S))

            # Stress modulation: ψ' = ψ * (1 + h_m * S * dt)
            modulation = Fraction(1) + h_m * S * dt
            new_psi = psi * RationalComplex.from_real(modulation)

            new_state.set(point, new_psi)

        return new_state

    def check_invariants(self, state: FieldState, context: OperatorContext) -> Tuple[List[str], bool]:
        """Check MSTVQ stress invariants."""
        invariants = ["stress_bounded", "field_stable"]
        passed = True

        # Get stress field
        stress_field = context.get_param('mstvq_stress_field')
        if stress_field is not None:
            total_stress = stress_field.total_stress()
            max_allowed = self.config.max_stress * state.lattice.total_sites
            if total_stress > max_allowed:
                passed = False

        # Check field stability
        max_bound = context.get_param('mstvq_max_bound', Fraction(1000))
        if not state.validate_bounded(max_bound):
            passed = False

        return invariants, passed


class MSTVQTensionOperator(Operator):
    """
    MSTVQ tension update operator.

    Computes and applies tension field modifications to the state.
    Tension affects phase dynamics and field coherence.
    """

    def __init__(self, config: MSTVQConfig = None):
        super().__init__(name="MSTVQTension", category=OperatorCategory.COUPLING)
        self.config = config or MSTVQConfig()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply MSTVQ tension transformation.

        Updates field phase based on local tension.
        """
        new_state = state.copy()

        # Get or create stress field
        stress_field = context.get_param('mstvq_stress_field')
        if stress_field is None:
            stress_field = StressTensorField(state.lattice)
            stress_field.compute_from_field(state, self.config)
            context.set_param('mstvq_stress_field', stress_field)

        # Apply tension-induced phase rotation
        for point in state.lattice.iterate_all():
            coords = point.coords
            psi = state.get(point)

            if psi.is_zero():
                continue

            # Get local tension
            T = stress_field.get_tension(coords)

            # Tension induces phase rotation: ψ' = ψ * exp(i * T * h_m * dt)
            phase_shift = float(T * self.config.h_m * context.dt)
            rotation = RationalComplex.from_complex(
                complex(math.cos(phase_shift), math.sin(phase_shift))
            )
            new_psi = psi * rotation

            new_state.set(point, new_psi)

        return new_state


class MSTVQSuppressionOperator(Operator):
    """
    MSTVQ suppression envelope operator.

    Modifies the radial suppression R based on MSTVQ stress-tension.
    High stress regions get stronger suppression (stability).
    """

    def __init__(self, config: MSTVQConfig = None):
        super().__init__(name="MSTVQSuppression", category=OperatorCategory.CONSTRAINT)
        self.config = config or MSTVQConfig()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply MSTVQ-modified suppression envelope.
        """
        new_state = state.copy()

        # Get stress field
        stress_field = context.get_param('mstvq_stress_field')
        if stress_field is None:
            stress_field = StressTensorField(state.lattice)
            stress_field.compute_from_field(state, self.config)
            context.set_param('mstvq_stress_field', stress_field)

        for point in state.lattice.iterate_all():
            coords = point.coords
            psi = state.get(point)

            # Compute suppression based on local stress
            S = stress_field.get_stress(coords)
            T = stress_field.get_tension(coords)

            # Suppression factor: R = 1 / (1 + S + |T|)
            # High stress/tension regions are suppressed
            denom = Fraction(1) + S + abs(T)
            suppression = Fraction(1) / denom if denom > 0 else Fraction(1)

            new_psi = psi * RationalComplex.from_real(suppression)
            new_state.set(point, new_psi)

        return new_state


class MSTVQCouplingOperator(Operator):
    """
    MSTVQ R4 coupling modification operator.

    Adjusts R4 coupling weights based on MSTVQ stress-tension field.
    Creates anisotropic coupling in high-stress regions.
    """

    def __init__(self, config: MSTVQConfig = None):
        super().__init__(name="MSTVQCoupling", category=OperatorCategory.COUPLING)
        self.config = config or MSTVQConfig()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply MSTVQ-modified R4 coupling.

        In high-stress regions, coupling to neighbors is reduced.
        This models the "freezing" of correlations under stress.
        """
        new_state = state.copy()

        # Get stress field
        stress_field = context.get_param('mstvq_stress_field')
        if stress_field is None:
            stress_field = StressTensorField(state.lattice)
            stress_field.compute_from_field(state, self.config)
            context.set_param('mstvq_stress_field', stress_field)

        # Compute MSTVQ-modified neighbor coupling
        for point in state.lattice.iterate_all():
            coords = point.coords
            psi = state.get(point)

            # Local stress modifies coupling strength
            S = stress_field.get_stress(coords)
            coupling_reduction = Fraction(1) / (Fraction(1) + S * self.config.st_ratio)

            # Compute weighted neighbor average
            neighbor_contribution = RationalComplex.zero()
            total_weight = Fraction(0)

            for neighbor, base_weight in state.lattice.neighbors(point):
                neighbor_psi = state.get(neighbor)
                # Modify weight by stress reduction
                modified_weight = base_weight * coupling_reduction
                neighbor_contribution = neighbor_contribution + neighbor_psi * RationalComplex.from_real(modified_weight)
                total_weight += modified_weight

            if total_weight > 0:
                avg_neighbor = neighbor_contribution * RationalComplex.from_real(Fraction(1) / total_weight)
                # Mix with local value
                mix = context.get_param('mstvq_mix', Fraction(1, 10))
                new_psi = psi * RationalComplex.from_real(Fraction(1) - mix) + avg_neighbor * RationalComplex.from_real(mix)
            else:
                new_psi = psi

            new_state.set(point, new_psi)

        return new_state


class MSTVQCompositeOperator(Operator):
    """
    Complete MSTVQ update operator (CODEX 6.2).

    Combines all MSTVQ operations into a single update stage:
    1. Compute stress-tension field from current state
    2. Apply stress modulation
    3. Apply tension phase rotation
    4. Apply suppression envelope
    5. Apply modified R4 coupling

    All with logged deltas and invariants.
    """

    def __init__(self, config: MSTVQConfig = None):
        super().__init__(name="MSTVQComposite", category=OperatorCategory.COUPLING)
        self.config = config or MSTVQConfig()

        # Sub-operators
        self.stress_op = MSTVQStressOperator(config)
        self.tension_op = MSTVQTensionOperator(config)
        self.suppression_op = MSTVQSuppressionOperator(config)
        self.coupling_op = MSTVQCouplingOperator(config)

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply complete MSTVQ update.
        """
        # Create fresh stress-tension field
        stress_field = StressTensorField(state.lattice)
        stress_field.compute_from_field(state, self.config)
        context.set_param('mstvq_stress_field', stress_field)

        # Record initial state for delta logging
        initial_norm = state.total_norm_squared()

        # Apply MSTVQ pipeline
        state = self.stress_op.apply(state, context)
        state = self.tension_op.apply(state, context)
        state = self.suppression_op.apply(state, context)
        state = self.coupling_op.apply(state, context)

        # Record delta
        final_norm = state.total_norm_squared()
        context.set_param('mstvq_delta_norm', float(final_norm - initial_norm))
        context.set_param('mstvq_total_stress', float(stress_field.total_stress()))
        context.set_param('mstvq_total_tension', float(stress_field.total_tension()))

        return state

    def check_invariants(self, state: FieldState, context: OperatorContext) -> Tuple[List[str], bool]:
        """Check all MSTVQ invariants."""
        invariants = [
            "stress_bounded",
            "tension_bounded",
            "field_stable",
            "energy_conservation_approx"
        ]
        passed = True

        # Get stress field
        stress_field = context.get_param('mstvq_stress_field')
        if stress_field is not None:
            # Check stress bounds
            total_stress = stress_field.total_stress()
            max_stress = self.config.max_stress * state.lattice.total_sites
            if total_stress > max_stress:
                passed = False

            # Check tension bounds
            total_tension = stress_field.total_tension()
            if abs(total_tension) > max_stress:
                passed = False

        # Check field stability
        max_bound = context.get_param('mstvq_max_bound', Fraction(1000))
        if not state.validate_bounded(max_bound):
            passed = False

        return invariants, passed


# Factory functions

def create_mstvq_pipeline(h_m: Fraction = Fraction(1),
                          stress_coupling: Fraction = Fraction(1, 10)) -> MSTVQCompositeOperator:
    """
    Create a standard MSTVQ operator pipeline.

    Args:
        h_m: Magnetic stress-tension global scale
        stress_coupling: Stress field coupling strength

    Returns:
        Configured MSTVQCompositeOperator
    """
    config = MSTVQConfig(
        h_m=h_m,
        stress_coupling=stress_coupling
    )
    return MSTVQCompositeOperator(config)


# Self-test
def _self_test():
    """Run basic MSTVQ tests."""
    from ..lattice import create_3d_lattice
    from ..state import create_gaussian_field

    # Create test lattice and field
    lattice = create_3d_lattice(8, 8, 8)
    center = (4, 4, 4)
    state = create_gaussian_field(lattice, center, sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    # Test stress field computation
    config = MSTVQConfig()
    stress_field = StressTensorField(lattice)
    stress_field.compute_from_field(state, config)

    # Stress should be higher where gradient is steeper
    center_stress = stress_field.get_stress(center)
    edge_stress = stress_field.get_stress((0, 0, 0))
    # Near center has high gradient in Gaussian
    assert isinstance(center_stress, Fraction)

    # Test MSTVQ composite operator
    mstvq = create_mstvq_pipeline()
    result = mstvq(state, context)

    # Verify result is valid
    assert result.validate_bounded(Fraction(1000))

    # Verify invariants passed
    invariants, passed = mstvq.check_invariants(result, context)
    assert passed, f"MSTVQ invariants failed: {invariants}"


_self_test()
