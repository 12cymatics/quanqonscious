"""
GRVQ Ansatz Operator (CODEX 2.2)

Implements the GRVQ-style composition as a product of shape perturbations
times a Vedic carrier and a radial/topological suppression:

    Ψ(x) = ( ∏_{j=1..J} [1 + α_j * S_j(x)] ) * R(x) * f_Vedic(x) * T_R4(x)

Where:
- S_j(x): shape functions (cymatic / Chladni / Bessel / lattice harmonics)
- α_j: coefficients (exact rational by default)
- R(x): radial / boundary suppression (no-singularity / boundedness envelope)
- f_Vedic(x): Vedic carrier wave (explicit trig/poly combo)
- T_R4(x): R4 coupling/topology factor
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Callable
import math

from .base import Operator, OperatorCategory, OperatorContext
from ..state import FieldState, RationalComplex
from ..lattice import ToroidalHypercube, LatticePoint


@dataclass
class ShapeFunction:
    """
    Shape function S_j(x) for GRVQ composition.

    Each shape function represents a mode of the cymatic field:
    - Chladni patterns (plate modes)
    - Bessel functions (circular membrane)
    - Lattice harmonics (discrete Fourier modes)
    """
    name: str
    mode_numbers: Tuple[int, ...]  # (m, n, ...) mode indices
    shape_type: str  # "chladni", "bessel", "harmonic", "radial"

    def evaluate(self, coords: Tuple[int, ...], lattice: ToroidalHypercube,
                 context: OperatorContext) -> RationalComplex:
        """Evaluate shape function at given coordinates."""
        # Normalize coordinates to [0, 1] per axis
        norm_coords = tuple(c / n for c, n in zip(coords, lattice.shape))

        if self.shape_type == "chladni":
            return self._chladni(norm_coords)
        elif self.shape_type == "bessel":
            return self._bessel(norm_coords)
        elif self.shape_type == "harmonic":
            return self._harmonic(norm_coords)
        elif self.shape_type == "radial":
            return self._radial(norm_coords, lattice)
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")

    def _chladni(self, norm_coords: Tuple[float, ...]) -> RationalComplex:
        """
        Chladni plate mode: sin(mπx)sin(nπy) + sin(nπx)sin(mπy)

        Creates symmetric nodal patterns characteristic of vibrating plates.
        """
        if len(norm_coords) < 2:
            return RationalComplex.from_real(0)

        x, y = norm_coords[0], norm_coords[1]
        m, n = self.mode_numbers[0], self.mode_numbers[1] if len(self.mode_numbers) > 1 else 1

        # sin(mπx)sin(nπy) + sin(nπx)sin(mπy)
        val = (math.sin(m * math.pi * x) * math.sin(n * math.pi * y) +
               math.sin(n * math.pi * x) * math.sin(m * math.pi * y))

        return RationalComplex.from_complex(complex(val, 0))

    def _bessel(self, norm_coords: Tuple[float, ...]) -> RationalComplex:
        """
        Bessel function mode (approximation for circular membrane).

        Uses the radial distance and angular position.
        """
        if len(norm_coords) < 2:
            return RationalComplex.from_real(0)

        # Center and compute radial distance
        x, y = norm_coords[0] - 0.5, norm_coords[1] - 0.5
        r = math.sqrt(x*x + y*y) * 2  # Scale to [0, 1]
        theta = math.atan2(y, x)

        m = self.mode_numbers[0] if self.mode_numbers else 1
        n = self.mode_numbers[1] if len(self.mode_numbers) > 1 else 0

        # Approximate Bessel J_n (Taylor expansion for small argument)
        # J_n(x) ≈ (x/2)^n / n! for small x
        if r < 1e-10:
            bessel_val = 1.0 if n == 0 else 0.0
        else:
            kr = m * math.pi * r
            # Use recurrence relation approximation
            bessel_val = math.cos(kr - n * math.pi / 2) / (max(kr, 0.1) ** 0.5)

        # Angular part
        angular = math.cos(n * theta)

        val = bessel_val * angular
        return RationalComplex.from_complex(complex(val, 0))

    def _harmonic(self, norm_coords: Tuple[float, ...]) -> RationalComplex:
        """
        Lattice harmonic mode (discrete Fourier mode).

        exp(2πi * (k · x)) for wave vector k = mode_numbers
        """
        if not self.mode_numbers:
            return RationalComplex.one()

        # k · x = sum of k_i * x_i
        phase = 0.0
        for k, x in zip(self.mode_numbers, norm_coords):
            phase += 2 * math.pi * k * x

        return RationalComplex.from_complex(complex(math.cos(phase), math.sin(phase)))

    def _radial(self, norm_coords: Tuple[float, ...], lattice: ToroidalHypercube) -> RationalComplex:
        """
        Radial mode (distance from center).

        Creates concentric ring patterns.
        """
        # Distance from center (normalized)
        center = tuple(0.5 for _ in norm_coords)
        r_sq = sum((x - c)**2 for x, c in zip(norm_coords, center))
        r = math.sqrt(r_sq)

        m = self.mode_numbers[0] if self.mode_numbers else 1

        # Radial oscillation: cos(mπr)
        val = math.cos(m * math.pi * r)
        return RationalComplex.from_complex(complex(val, 0))


@dataclass
class RadialSuppression:
    """
    Radial/boundary suppression R(x) for boundedness.

    Ensures field remains finite by suppressing near boundaries or singularities.
    The suppression factor is:
        R(x) = (1 - exp(-r²/σ²)) * boundary_factor

    This prevents:
    - Singularities at the origin
    - Unbounded growth at boundaries
    """
    sigma: Fraction = Fraction(1, 10)  # Suppression width
    boundary_width: Fraction = Fraction(1, 20)  # Boundary layer width

    def evaluate(self, coords: Tuple[int, ...], lattice: ToroidalHypercube,
                 context: OperatorContext) -> Fraction:
        """Evaluate suppression factor at given coordinates."""
        # Normalize coordinates
        norm_coords = tuple(Fraction(c, n) for c, n in zip(coords, lattice.shape))

        # Distance from center
        center = tuple(Fraction(1, 2) for _ in norm_coords)
        r_sq = sum((x - c) * (x - c) for x, c in zip(norm_coords, center))

        # Origin suppression: 1 - exp(-r²/σ²)
        # Use rational approximation for small r
        if r_sq < self.sigma * self.sigma / 100:
            # Taylor expansion: 1 - exp(-x) ≈ x for small x
            origin_factor = r_sq / (self.sigma * self.sigma)
        else:
            # For larger r, use clamped value
            origin_factor = min(Fraction(1), r_sq / (self.sigma * self.sigma))

        # Boundary suppression: smooth falloff near edges
        # For each dimension, check distance to boundary
        boundary_factor = Fraction(1)
        for x, n in zip(coords, lattice.shape):
            # Distance to nearest boundary (periodic)
            dist_to_edge = min(x, n - 1 - x)
            dist_frac = Fraction(dist_to_edge, n)

            if dist_frac < self.boundary_width:
                # Smooth falloff
                boundary_factor *= dist_frac / self.boundary_width

        # Combine factors
        return min(origin_factor, Fraction(1)) * boundary_factor


@dataclass
class VedicCarrier:
    """
    Vedic carrier wave f_Vedic(x).

    Explicit trig/polynomial combination based on Vedic mathematics.
    The carrier encodes:
    - Primary frequency (related to Schumann resonance)
    - Harmonic structure (based on Vedic ratios)
    - Phase modulation (sutra-derived)
    """
    base_frequency: Fraction = Fraction(783, 100)  # 7.83 Hz (Schumann)
    harmonic_ratios: Tuple[Fraction, ...] = (
        Fraction(1),
        Fraction(3, 2),  # Perfect fifth
        Fraction(5, 4),  # Major third
        Fraction(7, 4),  # Harmonic seventh
    )

    def evaluate(self, coords: Tuple[int, ...], lattice: ToroidalHypercube,
                 context: OperatorContext) -> RationalComplex:
        """Evaluate Vedic carrier at given coordinates."""
        # Normalize coordinates
        norm_coords = tuple(c / n for c, n in zip(coords, lattice.shape))

        # Base spatial frequency
        t = context.timestep * float(context.dt)
        omega = 2 * math.pi * float(self.base_frequency)

        # Sum of harmonics
        val = 0.0
        for i, ratio in enumerate(self.harmonic_ratios):
            amplitude = 1.0 / (i + 1)  # Decreasing amplitude for higher harmonics
            freq = omega * float(ratio)

            # Spatial variation: standing wave pattern
            spatial = sum(math.cos(2 * math.pi * x) for x in norm_coords)
            spatial /= len(norm_coords)

            val += amplitude * math.cos(freq * t + spatial * math.pi)

        # Normalize
        val /= len(self.harmonic_ratios)

        return RationalComplex.from_complex(complex(val, 0))


class GRVQAnsatzOperator(Operator):
    """
    GRVQ Ansatz composition operator (CODEX 2.2).

    Computes the field as:
        Ψ(x) = ( ∏_{j=1..J} [1 + α_j * S_j(x)] ) * R(x) * f_Vedic(x) * T_R4(x)

    Where the product term represents shape perturbations around unity.
    """

    def __init__(self,
                 shape_functions: List[ShapeFunction] = None,
                 coefficients: List[Fraction] = None,
                 radial_suppression: RadialSuppression = None,
                 vedic_carrier: VedicCarrier = None,
                 r4_coupling_enabled: bool = True):
        super().__init__(name="GRVQAnsatz", category=OperatorCategory.FIELD)

        # Default shape functions: first few Chladni modes
        if shape_functions is None:
            shape_functions = [
                ShapeFunction("S1", (1, 1), "chladni"),
                ShapeFunction("S2", (2, 1), "chladni"),
                ShapeFunction("S3", (2, 2), "chladni"),
            ]
        self.shape_functions = shape_functions

        # Default coefficients: small perturbations
        if coefficients is None:
            coefficients = [Fraction(1, 10) for _ in shape_functions]
        self.coefficients = coefficients

        # Components
        self.radial_suppression = radial_suppression or RadialSuppression()
        self.vedic_carrier = vedic_carrier or VedicCarrier()
        self.r4_coupling_enabled = r4_coupling_enabled

        # Verify coefficient count matches
        if len(self.coefficients) != len(self.shape_functions):
            raise ValueError(
                f"Coefficient count ({len(self.coefficients)}) must match "
                f"shape function count ({len(self.shape_functions)})"
            )

    def compute_product_term(self, coords: Tuple[int, ...],
                             lattice: ToroidalHypercube,
                             context: OperatorContext) -> RationalComplex:
        """
        Compute the product term: ∏_{j=1..J} [1 + α_j * S_j(x)]
        """
        result = RationalComplex.one()

        for alpha, S in zip(self.coefficients, self.shape_functions):
            S_val = S.evaluate(coords, lattice, context)
            # 1 + α * S(x)
            term = RationalComplex.one() + RationalComplex.from_real(alpha) * S_val
            result = result * term

        return result

    def compute_r4_topology(self, coords: Tuple[int, ...],
                            state: FieldState,
                            context: OperatorContext) -> RationalComplex:
        """
        Compute R4 topology factor T_R4(x).

        This creates entanglement-like correlations via the R4 adjacency kernel.
        """
        if not self.r4_coupling_enabled:
            return RationalComplex.one()

        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.neighbors(point)

        # T_R4 is an average of neighbor contributions
        total = RationalComplex.zero()
        total_weight = Fraction(0)

        for neighbor, weight in neighbors:
            val = state.get(neighbor)
            total = total + val * RationalComplex.from_real(weight)
            total_weight += weight

        if total_weight > 0:
            # Normalize
            factor = RationalComplex.from_real(Fraction(1) / total_weight)
            avg = total * factor
            # Return 1 + small coupling to average
            coupling = context.get_param('r4_coupling', Fraction(1, 20))
            return RationalComplex.one() + avg * RationalComplex.from_real(coupling)
        else:
            return RationalComplex.one()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply GRVQ ansatz to compute new field values.
        """
        new_state = state.copy()
        new_state.timestep = context.timestep

        for point in state.lattice.iterate_all():
            coords = point.coords

            # Product term: ∏[1 + α_j * S_j(x)]
            product = self.compute_product_term(coords, state.lattice, context)

            # Radial suppression: R(x)
            R = self.radial_suppression.evaluate(coords, state.lattice, context)

            # Vedic carrier: f_Vedic(x)
            f_vedic = self.vedic_carrier.evaluate(coords, state.lattice, context)

            # R4 topology: T_R4(x)
            T_r4 = self.compute_r4_topology(coords, state, context)

            # Full GRVQ composition
            psi = product * RationalComplex.from_real(R) * f_vedic * T_r4

            new_state.set(point, psi)

        return new_state

    def check_invariants(self, state: FieldState, context: OperatorContext) -> Tuple[List[str], bool]:
        """Check GRVQ-specific invariants."""
        invariants = ["boundedness", "toroidal_closure"]
        passed = True

        # Check boundedness (field should be bounded by suppression)
        max_bound = context.get_param('grvq_max_bound', Fraction(100))
        if not state.validate_bounded(max_bound):
            passed = False

        # Check toroidal closure
        for point in state.lattice.iterate_all():
            if not state.lattice.validate_closure(point):
                passed = False
                break

        return invariants, passed


class GRVQEvolutionOperator(Operator):
    """
    GRVQ field evolution operator.

    Evolves the field according to a GRVQ-modified wave equation,
    preserving the ansatz structure while allowing time evolution.
    """

    def __init__(self, ansatz: GRVQAnsatzOperator = None):
        super().__init__(name="GRVQEvolution", category=OperatorCategory.FIELD)
        self.ansatz = ansatz or GRVQAnsatzOperator()

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """
        Apply evolution step.

        Uses operator splitting:
        1. Apply GRVQ ansatz (shape modulation)
        2. Apply diffusive relaxation (smoothing)
        3. Apply nonlinear feedback (self-interaction)
        """
        # Step 1: GRVQ modulation
        state = self.ansatz(state, context)

        # Step 2: Diffusive relaxation (Laplacian)
        state = self._laplacian_step(state, context)

        # Step 3: Nonlinear feedback
        state = self._nonlinear_step(state, context)

        return state

    def _laplacian_step(self, state: FieldState, context: OperatorContext) -> FieldState:
        """Apply discrete Laplacian for diffusion/wave propagation."""
        new_state = state.copy()
        dt = context.dt
        diffusion = context.get_param('grvq_diffusion', Fraction(1, 100))

        for point in state.lattice.iterate_all():
            # Compute Laplacian: sum of neighbors - 2*d*center
            center = state.get(point)
            lap = RationalComplex.zero()

            for neighbor, weight in state.lattice.neighbors(point):
                lap = lap + (state.get(neighbor) - center) * RationalComplex.from_real(weight)

            # Update: ψ' = ψ + dt * D * ∇²ψ
            update = lap * RationalComplex.from_real(dt * diffusion)
            new_state.set(point, center + update)

        return new_state

    def _nonlinear_step(self, state: FieldState, context: OperatorContext) -> FieldState:
        """Apply nonlinear self-interaction term."""
        new_state = state.copy()
        nonlinear = context.get_param('grvq_nonlinear', Fraction(1, 1000))

        for point in state.lattice.iterate_all():
            psi = state.get(point)
            # Nonlinear term: -λ|ψ|²ψ (focusing/defocusing)
            norm_sq = psi.norm_squared()
            correction = psi * RationalComplex.from_real(-nonlinear * norm_sq)
            new_state.set(point, psi + correction)

        return new_state


# Factory functions

def create_cymatic_ansatz(modes: List[Tuple[int, int]],
                          coefficients: List[Fraction] = None) -> GRVQAnsatzOperator:
    """
    Create a GRVQ ansatz with Chladni modes.

    Args:
        modes: List of (m, n) mode number pairs
        coefficients: Optional list of mode coefficients

    Returns:
        GRVQAnsatzOperator configured with specified modes
    """
    shape_funcs = [
        ShapeFunction(f"Chladni_{m}_{n}", (m, n), "chladni")
        for m, n in modes
    ]

    if coefficients is None:
        # Default: decreasing coefficients for higher modes
        coefficients = [
            Fraction(1, 10 * (m + n))
            for m, n in modes
        ]

    return GRVQAnsatzOperator(
        shape_functions=shape_funcs,
        coefficients=coefficients
    )


def create_bessel_ansatz(modes: List[Tuple[int, int]],
                         coefficients: List[Fraction] = None) -> GRVQAnsatzOperator:
    """
    Create a GRVQ ansatz with Bessel function modes.

    Args:
        modes: List of (radial_mode, angular_mode) pairs
        coefficients: Optional list of mode coefficients

    Returns:
        GRVQAnsatzOperator configured with Bessel modes
    """
    shape_funcs = [
        ShapeFunction(f"Bessel_{m}_{n}", (m, n), "bessel")
        for m, n in modes
    ]

    if coefficients is None:
        coefficients = [Fraction(1, 10) for _ in modes]

    return GRVQAnsatzOperator(
        shape_functions=shape_funcs,
        coefficients=coefficients
    )


# Self-test
def _self_test():
    """Run basic GRVQ ansatz tests."""
    from ..lattice import create_3d_lattice
    from ..state import create_uniform_field

    # Create small test lattice
    lattice = create_3d_lattice(8, 8, 8)
    state = create_uniform_field(lattice, 1)
    context = OperatorContext()

    # Test shape function evaluation
    sf = ShapeFunction("test", (1, 2), "chladni")
    val = sf.evaluate((2, 3, 4), lattice, context)
    assert isinstance(val, RationalComplex)

    # Test radial suppression
    rs = RadialSuppression()
    R = rs.evaluate((4, 4, 4), lattice, context)  # Near center
    assert R <= 1
    R_edge = rs.evaluate((0, 0, 0), lattice, context)  # Corner/edge
    assert R_edge <= 1  # Should be within bounds

    # Test GRVQ ansatz operator
    ansatz = create_cymatic_ansatz([(1, 1), (2, 1)])
    result = ansatz(state, context)

    # Verify result is valid
    assert result.validate_bounded(Fraction(1000))

    # Check that field has non-trivial structure
    vals = [result.amplitude(lattice.point(i, i, i)) for i in range(8)]
    assert len(set(vals)) > 1, "Field should have spatial variation"


_self_test()
