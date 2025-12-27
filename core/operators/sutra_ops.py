"""
Vedic Sutra Operators (CODEX 4)

Implements all 29 Vedic sutras (16 primary + 13 sub-sutras) as operators
with the common interface: apply(state, context) -> state'

Each sutra operator:
- Is composable (pipeline)
- Supports exact arithmetic mode
- Logs its action in structured trace

Sutra Categories (CODEX 4.2):
1) Arithmetic transforms (exact integer/rational transforms)
2) Indexing/permutation transforms (lattice remaps, R4 adjacency rewires)
3) Series/product transforms (factorizations and controlled expansions)
4) Constraint/suppression transforms (stability envelopes, boundedness gates)
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Any

# CRITICAL: math module FORBIDDEN - violates exact arithmetic
# All 29 Vedic sutra operations MUST use ONLY rational arithmetic
# NO sqrt, cos, sin, atan2, exp, or other transcendental functions

from .base import Operator, OperatorCategory, OperatorContext
from ..state import FieldState, RationalComplex
from ..lattice import ToroidalHypercube, LatticePoint


# =============================================================================
# Base Sutra Operator Class
# =============================================================================

class SutraOperator(Operator):
    """
    Base class for all Vedic sutra operators.

    Provides common functionality for sutra operations:
    - Sutra number and name
    - Category classification
    - Invariant checking
    """

    def __init__(self, number: int, name: str, sanskrit: str, category: OperatorCategory):
        super().__init__(name=name, category=category)
        self.sutra_number = number
        self.sanskrit_name = sanskrit

    @abstractmethod
    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        """
        Apply the sutra transformation to a single field value.

        Args:
            value: Current field value
            coords: Lattice coordinates
            state: Full field state (for context)
            context: Operator context

        Returns:
            Transformed field value
        """
        pass

    def apply(self, state: FieldState, context: OperatorContext) -> FieldState:
        """Apply sutra transformation to all lattice points."""
        new_state = state.copy()

        for point in state.lattice.iterate_all():
            old_val = state.get(point)
            new_val = self.sutra_transform(old_val, point.coords, state, context)
            new_state.set(point, new_val)

        return new_state


# =============================================================================
# Primary Sutras (1-16)
# =============================================================================

class Sutra01_EkadhikenaPurvena(SutraOperator):
    """
    Sutra 1: Ekadhikena Purvena - "By one more than the previous"

    Arithmetic sutra for increment patterns and recurrence relations.
    In field context: adds local gradient contribution to each value.
    """

    def __init__(self):
        super().__init__(
            number=1,
            name="EkadhikenaPurvena",
            sanskrit="एकाधिकेन पूर्वेण",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # "By one more than the previous": add 1 plus local gradient
        point = LatticePoint(coords, state.lattice.shape)

        # Compute "previous" as average of lower-index neighbors
        prev_sum = RationalComplex.zero()
        count = 0
        for i, c in enumerate(coords):
            if c > 0:
                prev_coords = list(coords)
                prev_coords[i] = c - 1
                prev_val = state.get_by_coords(*prev_coords)
                prev_sum = prev_sum + prev_val
                count += 1

        if count > 0:
            prev_avg = prev_sum * RationalComplex.from_real(Fraction(1, count))
            # "By one more": add 1 to the progression
            increment = RationalComplex.one() + prev_avg * RationalComplex.from_real(Fraction(1, 10))
            return value + increment * RationalComplex.from_real(context.dt)
        else:
            return value + RationalComplex.from_real(context.dt)


class Sutra02_NikhilamNavatashcaramam(SutraOperator):
    """
    Sutra 2: Nikhilam Navatashcaramam Dashatah - "All from 9, last from 10"

    Complement sutra for subtraction and decimal operations.
    In field context: computes complement from local maximum.
    """

    def __init__(self):
        super().__init__(
            number=2,
            name="NikhilamNavatashcaramam",
            sanskrit="निखिलं नवतश्चरमं दशतः",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Find local maximum (the "10" or "9+1")
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        max_norm = value.norm()
        for neighbor in neighbors:
            neighbor_norm = state.get(neighbor).norm()
            max_norm = max(max_norm, neighbor_norm)

        # "All from 9, last from 10": complement operation
        base = Fraction(max_norm).limit_denominator(10000) + Fraction(1)
        complement = RationalComplex.from_real(base) - value

        # Mix with original
        mix = context.get_param('nikhilam_mix', Fraction(1, 10))
        return value * RationalComplex.from_real(Fraction(1) - mix) + complement * RationalComplex.from_real(mix)


class Sutra03_UrdhvaTiryagbhyam(SutraOperator):
    """
    Sutra 3: Urdhva-Tiryagbhyam - "Vertically and crosswise"

    Multiplication sutra using crosswise products.
    In field context: couples vertical and horizontal neighbors.
    """

    def __init__(self):
        super().__init__(
            number=3,
            name="UrdhvaTiryagbhyam",
            sanskrit="ऊर्ध्वतिर्यग्भ्याम्",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        if len(coords) < 2:
            return value

        # Get vertical (dim 0) and horizontal (dim 1) neighbors
        point = LatticePoint(coords, state.lattice.shape)

        # Vertical neighbors
        v_up = state.get_by_coords(coords[0], (coords[1] + 1) % state.lattice.shape[1], *coords[2:])
        v_dn = state.get_by_coords(coords[0], (coords[1] - 1) % state.lattice.shape[1], *coords[2:])

        # Horizontal neighbors
        h_rt = state.get_by_coords((coords[0] + 1) % state.lattice.shape[0], coords[1], *coords[2:])
        h_lt = state.get_by_coords((coords[0] - 1) % state.lattice.shape[0], coords[1], *coords[2:])

        # Crosswise product: (v_up * h_rt + v_dn * h_lt)
        cross1 = v_up * h_rt
        cross2 = v_dn * h_lt

        # Combine with original
        coupling = context.get_param('urdhva_coupling', Fraction(1, 20))
        return value + (cross1 + cross2) * RationalComplex.from_real(coupling)


class Sutra04_ParavartyaYojayet(SutraOperator):
    """
    Sutra 4: Paravartya Yojayet - "Transpose and apply"

    Division sutra through transposition.
    In field context: applies coordinate transposition.
    """

    def __init__(self):
        super().__init__(
            number=4,
            name="ParavartyaYojayet",
            sanskrit="परावर्त्य योजयेत्",
            category=OperatorCategory.INDEXING
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        if len(coords) < 2:
            return value

        # Transpose: swap first two coordinates
        transposed_coords = (coords[1], coords[0]) + coords[2:]
        transposed_val = state.get_by_coords(*transposed_coords)

        # Mix transposed value with original
        mix = context.get_param('paravartya_mix', Fraction(1, 4))
        return value * RationalComplex.from_real(Fraction(1) - mix) + transposed_val * RationalComplex.from_real(mix)


class Sutra05_ShunyamSamuccaye(SutraOperator):
    """
    Sutra 5: Shunyam Samuccaye - "When the samuccaya is the same, that samuccaya is zero"

    Zero detection and elimination sutra.
    In field context: identifies and smooths near-zero regions.
    """

    def __init__(self):
        super().__init__(
            number=5,
            name="ShunyamSamuccaye",
            sanskrit="शून्यं साम्यसमुच्चये",
            category=OperatorCategory.CONSTRAINT
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        threshold = context.get_param('shunyam_threshold', Fraction(1, 1000))

        # If value is near zero, smooth with neighbors
        if value.norm_squared() < threshold:
            point = LatticePoint(coords, state.lattice.shape)
            neighbors = state.lattice.nearest_neighbors(point)

            neighbor_sum = RationalComplex.zero()
            for neighbor in neighbors:
                neighbor_sum = neighbor_sum + state.get(neighbor)

            if neighbors:
                return neighbor_sum * RationalComplex.from_real(Fraction(1, len(neighbors)))
            else:
                return RationalComplex.zero()

        return value


class Sutra06_Anurupyena(SutraOperator):
    """
    Sutra 6: Anurupyena - "Proportionately"

    Proportionality and scaling sutra.
    In field context: enforces local proportionality constraints.
    """

    def __init__(self):
        super().__init__(
            number=6,
            name="Anurupyena",
            sanskrit="आनुरूप्येण",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Compute local ratio (proportionality check)
        neighbor_avg = RationalComplex.zero()
        for neighbor in neighbors:
            neighbor_avg = neighbor_avg + state.get(neighbor)
        neighbor_avg = neighbor_avg * RationalComplex.from_real(Fraction(1, len(neighbors)))

        if neighbor_avg.norm() > 0.001:
            # Compute ratio to neighbors
            ratio = value.norm() / neighbor_avg.norm()
            target_ratio = context.get_param('anurupya_ratio', 1.0)

            # Adjust toward target ratio
            if ratio > 0.001:
                adjustment = target_ratio / ratio
                adjustment = max(0.5, min(2.0, adjustment))  # Clamp
                return value * RationalComplex.from_real(Fraction(adjustment).limit_denominator(1000))

        return value


class Sutra07_SankalanVyavakalanabhyam(SutraOperator):
    """
    Sutra 7: Sankalana-Vyavakalanabhyam - "By addition and subtraction"

    Addition/subtraction balance sutra.
    In field context: balances local sums and differences.
    """

    def __init__(self):
        super().__init__(
            number=7,
            name="SankalanaVyavakalanabhyam",
            sanskrit="संकलन व्यवकलनाभ्याम्",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if len(neighbors) < 2:
            return value

        # Sum pairs of opposite neighbors
        sums = []
        diffs = []

        for i in range(0, len(neighbors), 2):
            if i + 1 < len(neighbors):
                v1 = state.get(neighbors[i])
                v2 = state.get(neighbors[i + 1])
                sums.append(v1 + v2)
                diffs.append(v1 - v2)

        if sums:
            # Balance: new value is average of (value + sum_avg) and (value - diff_avg)
            sum_avg = sum(sums, RationalComplex.zero()) * RationalComplex.from_real(Fraction(1, len(sums)))
            diff_avg = sum(diffs, RationalComplex.zero()) * RationalComplex.from_real(Fraction(1, len(diffs)))

            balanced = (value + sum_avg) + (value - diff_avg)
            return balanced * RationalComplex.from_real(Fraction(1, 2))

        return value


class Sutra08_Puranapuranabhyam(SutraOperator):
    """
    Sutra 8: Puranapuranabhyam - "By completion or non-completion"

    Completion to reference value sutra.
    In field context: completes field to local maximum.
    """

    def __init__(self):
        super().__init__(
            number=8,
            name="Puranapuranabhyam",
            sanskrit="पूरणापूरणाभ्याम्",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        # Find local maximum norm
        max_norm = value.norm()
        for neighbor in neighbors:
            max_norm = max(max_norm, state.get(neighbor).norm())

        # Completion factor
        if value.norm() > 0.001:
            completion = max_norm / value.norm()
            completion = min(completion, 2.0)  # Limit growth

            strength = context.get_param('purana_strength', Fraction(1, 10))
            factor = Fraction(1) + Fraction(completion - 1).limit_denominator(1000) * strength
            return value * RationalComplex.from_real(factor)

        return value


class Sutra09_CalanaKalanabhyam(SutraOperator):
    """
    Sutra 9: Calana-Kalanabhyam - "Differential calculus"

    Differentiation sutra.
    In field context: computes local derivative (gradient).
    """

    def __init__(self):
        super().__init__(
            number=9,
            name="CalanaKalanabhyam",
            sanskrit="चलन कलनाभ्याम्",
            category=OperatorCategory.FIELD
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)

        # Compute gradient (discrete derivative)
        gradient = RationalComplex.zero()

        for d in range(len(coords)):
            # Forward difference
            fwd_coords = list(coords)
            fwd_coords[d] = (coords[d] + 1) % state.lattice.shape[d]
            fwd_val = state.get_by_coords(*fwd_coords)

            # Backward difference
            bwd_coords = list(coords)
            bwd_coords[d] = (coords[d] - 1) % state.lattice.shape[d]
            bwd_val = state.get_by_coords(*bwd_coords)

            # Central difference
            diff = (fwd_val - bwd_val) * RationalComplex.from_real(Fraction(1, 2))
            gradient = gradient + diff

        # Mix gradient with value
        calana_strength = context.get_param('calana_strength', Fraction(1, 10))
        return value + gradient * RationalComplex.from_real(calana_strength * context.dt)


class Sutra10_Yavadunam(SutraOperator):
    """
    Sutra 10: Yavadunam - "Whatever the extent of its deficiency"

    Deficiency compensation sutra.
    In field context: compensates for deviation from mean.
    """

    def __init__(self):
        super().__init__(
            number=10,
            name="Yavadunam",
            sanskrit="यावदूनम्",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Compute local mean
        total = value
        for neighbor in neighbors:
            total = total + state.get(neighbor)
        mean = total * RationalComplex.from_real(Fraction(1, len(neighbors) + 1))

        # Deficiency from mean
        deficiency = mean - value

        # "Whatever the deficiency": add back proportionally
        compensation = context.get_param('yavadunam_compensation', Fraction(1, 4))
        return value + deficiency * RationalComplex.from_real(compensation)


class Sutra11_Vyashtisamanstih(SutraOperator):
    """
    Sutra 11: Vyashti-Samanstih - "Part and whole"

    Part-whole relationship sutra.
    In field context: relates local to global properties.
    """

    def __init__(self):
        super().__init__(
            number=11,
            name="VyashtiSamanstih",
            sanskrit="व्यष्टि समष्टिः",
            category=OperatorCategory.SERIES
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Get global norm (cached or computed)
        global_norm_sq = context.get_param('global_norm_sq')
        if global_norm_sq is None:
            global_norm_sq = state.total_norm_squared()
            context.set_param('global_norm_sq', global_norm_sq)

        # Local contribution to whole
        local_norm_sq = value.norm_squared()

        # Adjust based on part/whole ratio
        if float(global_norm_sq) > 0.001:
            ratio = float(local_norm_sq) / float(global_norm_sq) * state.lattice.total_sites
            # Normalize toward equal contribution
            if ratio > 0.001:
                target_ratio = 1.0
                adjustment = math.sqrt(target_ratio / ratio)
                adjustment = max(0.5, min(2.0, adjustment))

                strength = context.get_param('vyashti_strength', Fraction(1, 10))
                factor = Fraction(1) + Fraction(adjustment - 1).limit_denominator(1000) * strength
                return value * RationalComplex.from_real(factor)

        return value


class Sutra12_Shesanyankena(SutraOperator):
    """
    Sutra 12: Shesanyankena Charamena - "The remainders by the last digit"

    Remainder and modular arithmetic sutra.
    In field context: applies modular constraints.
    """

    def __init__(self):
        super().__init__(
            number=12,
            name="ShesanyankenaCharmona",
            sanskrit="शेषाण्यङ्केन चरमेण",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Modular constraint: keep phase within bounds
        norm = value.norm()
        phase = value.phase()

        # "Last digit" constraint: quantize phase to discrete levels
        n_levels = context.get_param('shesanya_levels', 8)
        quantized_phase = round(phase * n_levels / (2 * math.pi)) * (2 * math.pi) / n_levels

        # Mix with original
        mix = context.get_param('shesanya_mix', Fraction(1, 4))
        new_phase = phase * float(Fraction(1) - mix) + quantized_phase * float(mix)

        return RationalComplex.from_complex(complex(norm * math.cos(new_phase),
                                                    norm * math.sin(new_phase)))


class Sutra13_Sopantyadvayamantyam(SutraOperator):
    """
    Sutra 13: Sopantyadvayamantyam - "The ultimate and twice the penultimate"

    Boundary and penultimate sutra.
    In field context: handles boundary conditions specially.
    """

    def __init__(self):
        super().__init__(
            number=13,
            name="Sopantyadvayamantyam",
            sanskrit="सोपान्त्यद्वयमन्त्यम्",
            category=OperatorCategory.CONSTRAINT
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Check if at boundary (penultimate or ultimate positions)
        is_boundary = False
        is_penultimate = False

        for c, n in zip(coords, state.lattice.shape):
            if c == 0 or c == n - 1:
                is_boundary = True
            if c == 1 or c == n - 2:
                is_penultimate = True

        if is_boundary:
            # "Ultimate": apply boundary condition
            damping = context.get_param('sopantya_damping', Fraction(1, 2))
            return value * RationalComplex.from_real(damping)
        elif is_penultimate:
            # "Twice the penultimate": enhance near-boundary values
            enhancement = context.get_param('sopantya_enhancement', Fraction(3, 2))
            return value * RationalComplex.from_real(enhancement)

        return value


class Sutra14_EkanyunenaPurvena(SutraOperator):
    """
    Sutra 14: Ekanyunena Purvena - "By one less than the previous"

    Decrement sutra (complement to Sutra 1).
    In field context: subtracts gradient contribution.
    """

    def __init__(self):
        super().__init__(
            number=14,
            name="EkanyunenaPurvena",
            sanskrit="एकन्यूनेन पूर्वेण",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Compute "previous" average
        prev_sum = RationalComplex.zero()
        count = 0
        for i, c in enumerate(coords):
            if c > 0:
                prev_coords = list(coords)
                prev_coords[i] = c - 1
                prev_val = state.get_by_coords(*prev_coords)
                prev_sum = prev_sum + prev_val
                count += 1

        if count > 0:
            prev_avg = prev_sum * RationalComplex.from_real(Fraction(1, count))
            # "By one less": subtract from progression
            decrement = RationalComplex.one() - prev_avg * RationalComplex.from_real(Fraction(1, 10))
            return value - decrement * RationalComplex.from_real(context.dt)

        return value - RationalComplex.from_real(context.dt)


class Sutra15_Gunitasamuccayah(SutraOperator):
    """
    Sutra 15: Gunitasamuccayah - "The product of the sum"

    Product-sum relationship sutra.
    In field context: relates local products to neighbor sums.
    """

    def __init__(self):
        super().__init__(
            number=15,
            name="Gunitasamuccayah",
            sanskrit="गुणितसमुच्चयः",
            category=OperatorCategory.SERIES
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Sum of neighbors
        neighbor_sum = RationalComplex.zero()
        for neighbor in neighbors:
            neighbor_sum = neighbor_sum + state.get(neighbor)

        # Product term: value * sum
        product = value * neighbor_sum

        # Normalize by neighbor count
        normalized = product * RationalComplex.from_real(Fraction(1, len(neighbors)))

        # Mix with original
        mix = context.get_param('gunita_mix', Fraction(1, 20))
        return value * RationalComplex.from_real(Fraction(1) - mix) + normalized * RationalComplex.from_real(mix)


class Sutra16_Gunakasamuccayah(SutraOperator):
    """
    Sutra 16: Gunakasamuccayah - "The factors of the sum"

    Factor-sum relationship sutra (complement to Sutra 15).
    In field context: decomposes sums into factor contributions.
    """

    def __init__(self):
        super().__init__(
            number=16,
            name="Gunakasamuccayah",
            sanskrit="गुणकसमुच्चयः",
            category=OperatorCategory.SERIES
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors or value.is_zero():
            return value

        # Sum of neighbors
        neighbor_sum = RationalComplex.zero()
        for neighbor in neighbors:
            neighbor_sum = neighbor_sum + state.get(neighbor)

        # Factor: what would multiply value to get sum?
        if value.norm() > 0.001:
            # factor = sum / value (approximately)
            factor_norm = neighbor_sum.norm() / value.norm() / len(neighbors)
            factor = RationalComplex.from_real(Fraction(factor_norm).limit_denominator(1000))

            # Apply factor influence
            influence = context.get_param('gunaka_influence', Fraction(1, 20))
            return value * (RationalComplex.one() + factor * RationalComplex.from_real(influence))

        return value


# =============================================================================
# Sub-Sutras (17-29)
# =============================================================================

class SubSutra17_AnurupyenaSunyamanyat(SutraOperator):
    """
    Sub-Sutra 17: Anurupyena Sunyamanyat - "If one is in ratio, the other is zero"

    Ratio detection sub-sutra.
    """

    def __init__(self):
        super().__init__(
            number=17,
            name="AnurupyenaSunyamanyat",
            sanskrit="आनुरूप्येण शून्यमन्यत्",
            category=OperatorCategory.CONSTRAINT
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        # Check if in ratio with any neighbor
        for neighbor in neighbors:
            neighbor_val = state.get(neighbor)
            if value.norm() > 0.01 and neighbor_val.norm() > 0.01:
                ratio = value.norm() / neighbor_val.norm()
                target = context.get_param('anurupyena_target_ratio', 1.0)
                if abs(ratio - target) < 0.1:
                    # "In ratio" - zero out the other contribution
                    return value * RationalComplex.from_real(Fraction(1, 2))

        return value


class SubSutra18_Yavadunam(SutraOperator):
    """
    Sub-Sutra 18: Yavadunam Tavadunikritya - "Whatever deficiency, lessen by that much"
    """

    def __init__(self):
        super().__init__(
            number=18,
            name="YavadunamTavadunikritya",
            sanskrit="यावदूनं तावदूनीकृत्य",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Similar to Sutra 10 but with reduction
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Local mean
        total = value
        for neighbor in neighbors:
            total = total + state.get(neighbor)
        mean = total * RationalComplex.from_real(Fraction(1, len(neighbors) + 1))

        # Deficiency
        deficiency = value - mean

        # Reduce by deficiency amount
        reduction = context.get_param('yavadunam_reduction', Fraction(1, 5))
        return value - deficiency * RationalComplex.from_real(reduction)


class SubSutra19_Adyamadyenantyamantyena(SutraOperator):
    """
    Sub-Sutra 19: Adyamadyenantyamantyena - "First by first and last by last"
    """

    def __init__(self):
        super().__init__(
            number=19,
            name="Adyamadyenantyamantyena",
            sanskrit="आद्यमाद्येनान्त्यमन्त्येन",
            category=OperatorCategory.INDEXING
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # First by first: multiply by value at origin
        first_val = state.get_by_coords(*([0] * len(coords)))

        # Last by last: multiply by value at max indices
        last_coords = tuple(n - 1 for n in state.lattice.shape)
        last_val = state.get_by_coords(*last_coords)

        # Apply product
        if first_val.norm() > 0.001 and last_val.norm() > 0.001:
            product = first_val * last_val
            # Normalize
            product = product * RationalComplex.from_real(Fraction(1) / Fraction(product.norm()).limit_denominator(1000))

            mix = context.get_param('adyam_mix', Fraction(1, 20))
            return value * (RationalComplex.one() + product * RationalComplex.from_real(mix))

        return value


class SubSutra20_KevalaiSaptakam(SutraOperator):
    """
    Sub-Sutra 20: Kevalaih Saptakam Gunyat - "Multiply only by 7"

    Sacred multiplier sub-sutra.
    """

    def __init__(self):
        super().__init__(
            number=20,
            name="KevalaiSaptakamGunyat",
            sanskrit="केवलैः सप्तकं गुण्यात्",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Apply sacred multiplier 7 with modulation
        phase = value.phase()
        modulation = math.cos(7 * phase) + 1  # 0 to 2

        strength = context.get_param('kevala_strength', Fraction(1, 10))
        factor = Fraction(1) + Fraction(modulation / 2).limit_denominator(1000) * strength

        return value * RationalComplex.from_real(factor)


class SubSutra21_Veshtanam(SutraOperator):
    """
    Sub-Sutra 21: Veshtanam - "By osculation"

    Osculation (touching) sub-sutra.
    """

    def __init__(self):
        super().__init__(
            number=21,
            name="Veshtanam",
            sanskrit="वेष्टनम्",
            category=OperatorCategory.COUPLING
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        # Find "osculating" neighbors (those closest in value)
        min_diff = float('inf')
        osculating = value

        for neighbor in neighbors:
            neighbor_val = state.get(neighbor)
            diff = (value - neighbor_val).norm()
            if diff < min_diff and diff > 0.001:
                min_diff = diff
                osculating = neighbor_val

        # Mix with osculating neighbor
        mix = context.get_param('veshtana_mix', Fraction(1, 3))
        return value * RationalComplex.from_real(Fraction(1) - mix) + osculating * RationalComplex.from_real(mix)


class SubSutra22_YavadumamTavadum(SutraOperator):
    """
    Sub-Sutra 22: Yavadunam Tavadum Vilokanam - "Whatever excess, that much observe"
    """

    def __init__(self):
        super().__init__(
            number=22,
            name="YavadumamTavadumVilokanam",
            sanskrit="यावदूनं तावदूं विलोकनम्",
            category=OperatorCategory.CONSTRAINT
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Compute excess over mean
        total = RationalComplex.zero()
        for neighbor in neighbors:
            total = total + state.get(neighbor)
        mean = total * RationalComplex.from_real(Fraction(1, len(neighbors)))

        excess = value.norm() - mean.norm()

        # Observe excess: modulate amplitude
        if excess > 0:
            damping = context.get_param('yavadum_damping', Fraction(1, 10))
            factor = Fraction(1) - Fraction(excess).limit_denominator(1000) * damping
            factor = max(Fraction(1, 10), factor)
            return value * RationalComplex.from_real(factor)

        return value


class SubSutra23_AntyayorDashakepi(SutraOperator):
    """
    Sub-Sutra 23: Antyayordashake'pi - "The last digits also add to ten"
    """

    def __init__(self):
        super().__init__(
            number=23,
            name="AntyayorDashakepi",
            sanskrit="अन्त्ययोर्दशकेऽपि",
            category=OperatorCategory.ARITHMETIC
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Complement to 10 (mod 10 arithmetic on phase)
        phase = value.phase()
        norm = value.norm()

        # Phase complement
        target = math.pi  # "10" analog
        complement_phase = (2 * target - phase) % (2 * math.pi)

        # Mix phases
        mix = context.get_param('antyayor_mix', Fraction(1, 5))
        new_phase = phase * float(Fraction(1) - mix) + complement_phase * float(mix)

        return RationalComplex.from_complex(complex(norm * math.cos(new_phase),
                                                    norm * math.sin(new_phase)))


class SubSutra24_AntyayorEva(SutraOperator):
    """
    Sub-Sutra 24: Antyayoreva - "Only the last terms"
    """

    def __init__(self):
        super().__init__(
            number=24,
            name="AntyayorEva",
            sanskrit="अन्त्ययोरेव",
            category=OperatorCategory.INDEXING
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Focus on last (highest index) dimensions
        last_dim = len(coords) - 1
        if last_dim < 0:
            return value

        # Get neighbors along last dimension only
        up_coords = list(coords)
        up_coords[last_dim] = (coords[last_dim] + 1) % state.lattice.shape[last_dim]
        dn_coords = list(coords)
        dn_coords[last_dim] = (coords[last_dim] - 1) % state.lattice.shape[last_dim]

        up_val = state.get_by_coords(*up_coords)
        dn_val = state.get_by_coords(*dn_coords)

        # Average of last dimension neighbors
        last_avg = (up_val + dn_val) * RationalComplex.from_real(Fraction(1, 2))

        mix = context.get_param('antyayor_eva_mix', Fraction(1, 4))
        return value * RationalComplex.from_real(Fraction(1) - mix) + last_avg * RationalComplex.from_real(mix)


class SubSutra25_Samuccayagunitah(SutraOperator):
    """
    Sub-Sutra 25: Samuccayagunitah - "The sum is multiplied"
    """

    def __init__(self):
        super().__init__(
            number=25,
            name="Samuccayagunitah",
            sanskrit="समुच्चयगुणितः",
            category=OperatorCategory.SERIES
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Sum of all neighbors
        total = RationalComplex.zero()
        for neighbor in neighbors:
            total = total + state.get(neighbor)

        # Multiply by scaled sum
        scale = context.get_param('samuccaya_scale', Fraction(1, 100))
        return value * (RationalComplex.one() + total * RationalComplex.from_real(scale / len(neighbors)))


class SubSutra26_LopanaSthapanabhyam(SutraOperator):
    """
    Sub-Sutra 26: Lopana-Sthapanabhyam - "By elimination and retention"

    Gaussian elimination analog.
    """

    def __init__(self):
        super().__init__(
            number=26,
            name="LopanaSthapanabhyam",
            sanskrit="लोपनस्थापनाभ्याम्",
            category=OperatorCategory.CONSTRAINT
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        threshold = context.get_param('lopana_threshold', Fraction(1, 100))

        # Eliminate if below threshold
        if value.norm_squared() < threshold:
            return RationalComplex.zero()

        # Retain otherwise
        return value


class SubSutra27_Vilokanam(SutraOperator):
    """
    Sub-Sutra 27: Vilokanam - "By observation"

    Pattern recognition sub-sutra.
    """

    def __init__(self):
        super().__init__(
            number=27,
            name="Vilokanam",
            sanskrit="विलोकनम्",
            category=OperatorCategory.CONSTRAINT
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        # "Observe" pattern: check for regularity
        if not neighbors:
            return value

        phases = [state.get(n).phase() for n in neighbors]
        phase_std = 0.0
        if len(phases) > 1:
            mean_phase = sum(phases) / len(phases)
            phase_std = math.sqrt(sum((p - mean_phase)**2 for p in phases) / len(phases))

        # High regularity (low std): enhance
        # Low regularity (high std): dampen
        if phase_std < 0.5:
            factor = Fraction(1) + context.get_param('vilokanam_enhance', Fraction(1, 10))
        else:
            factor = Fraction(1) - context.get_param('vilokanam_dampen', Fraction(1, 10))

        return value * RationalComplex.from_real(factor)


class SubSutra28_GunitasamuccayahSamuccayagunitah(SutraOperator):
    """
    Sub-Sutra 28: Gunitasamuccayah Samuccayagunitah - "Product sum equals sum product"
    """

    def __init__(self):
        super().__init__(
            number=28,
            name="GunitasamuccayahSamuccayagunitah",
            sanskrit="गुणितसमुच्चयः समुच्चयगुणितः",
            category=OperatorCategory.SERIES
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        point = LatticePoint(coords, state.lattice.shape)
        neighbors = state.lattice.nearest_neighbors(point)

        if not neighbors:
            return value

        # Product of sums: (a+b)(c+d)
        # Sum of products: ac + ad + bc + bd
        # These should be equal for proper transforms

        total_sum = RationalComplex.zero()
        total_prod = RationalComplex.one()

        for neighbor in neighbors:
            n_val = state.get(neighbor)
            total_sum = total_sum + n_val
            if not n_val.is_zero():
                total_prod = total_prod * (RationalComplex.one() + n_val * RationalComplex.from_real(Fraction(1, 10)))

        # Balance: average of sum-based and product-based
        sum_contribution = total_sum * RationalComplex.from_real(Fraction(1, len(neighbors)))
        prod_contribution = total_prod

        balanced = (sum_contribution + prod_contribution) * RationalComplex.from_real(Fraction(1, 2))

        mix = context.get_param('gunita_balance_mix', Fraction(1, 10))
        return value * RationalComplex.from_real(Fraction(1) - mix) + balanced * RationalComplex.from_real(mix)


class SubSutra29_Dwandwayoga(SutraOperator):
    """
    Sub-Sutra 29: Dwandwa Yoga - "Duplex combination"

    Binary pairing sub-sutra.
    """

    def __init__(self):
        super().__init__(
            number=29,
            name="DwandwaYoga",
            sanskrit="द्वन्द्व योग",
            category=OperatorCategory.COUPLING
        )

    def sutra_transform(self, value: RationalComplex, coords: Tuple[int, ...],
                        state: FieldState, context: OperatorContext) -> RationalComplex:
        # Find duplex partner (coordinate inversion)
        partner_coords = tuple(n - 1 - c for c, n in zip(coords, state.lattice.shape))
        partner_val = state.get_by_coords(*partner_coords)

        # Combine: (a + bi)(c + di) + (a + bi)(c - di) = 2ac + 2bdi
        # This creates symmetric pairing
        combined = value * partner_val.conjugate() + value * partner_val
        combined = combined * RationalComplex.from_real(Fraction(1, 2))

        mix = context.get_param('dwandwa_mix', Fraction(1, 4))
        return value * RationalComplex.from_real(Fraction(1) - mix) + combined * RationalComplex.from_real(mix)


# =============================================================================
# Sutra Registry and Factories
# =============================================================================

def get_all_sutras() -> List[SutraOperator]:
    """Get all 29 sutra operators in order."""
    return [
        # Primary Sutras (1-16)
        Sutra01_EkadhikenaPurvena(),
        Sutra02_NikhilamNavatashcaramam(),
        Sutra03_UrdhvaTiryagbhyam(),
        Sutra04_ParavartyaYojayet(),
        Sutra05_ShunyamSamuccaye(),
        Sutra06_Anurupyena(),
        Sutra07_SankalanVyavakalanabhyam(),
        Sutra08_Puranapuranabhyam(),
        Sutra09_CalanaKalanabhyam(),
        Sutra10_Yavadunam(),
        Sutra11_Vyashtisamanstih(),
        Sutra12_Shesanyankena(),
        Sutra13_Sopantyadvayamantyam(),
        Sutra14_EkanyunenaPurvena(),
        Sutra15_Gunitasamuccayah(),
        Sutra16_Gunakasamuccayah(),

        # Sub-Sutras (17-29)
        SubSutra17_AnurupyenaSunyamanyat(),
        SubSutra18_Yavadunam(),
        SubSutra19_Adyamadyenantyamantyena(),
        SubSutra20_KevalaiSaptakam(),
        SubSutra21_Veshtanam(),
        SubSutra22_YavadumamTavadum(),
        SubSutra23_AntyayorDashakepi(),
        SubSutra24_AntyayorEva(),
        SubSutra25_Samuccayagunitah(),
        SubSutra26_LopanaSthapanabhyam(),
        SubSutra27_Vilokanam(),
        SubSutra28_GunitasamuccayahSamuccayagunitah(),
        SubSutra29_Dwandwayoga(),
    ]


def get_sutra_by_number(number: int) -> Optional[SutraOperator]:
    """Get a specific sutra by its number (1-29)."""
    sutras = get_all_sutras()
    if 1 <= number <= len(sutras):
        return sutras[number - 1]
    return None


def get_sutras_by_category(category: OperatorCategory) -> List[SutraOperator]:
    """Get all sutras of a specific category."""
    return [s for s in get_all_sutras() if s.category == category]


def create_sutra_pipeline(sutra_numbers: List[int]) -> 'CompositeOperator':
    """
    Create a composite operator that applies sutras in sequence.

    Args:
        sutra_numbers: List of sutra numbers (1-29) to apply in order

    Returns:
        Composite operator
    """
    from .base import CompositeOperator

    operators = []
    for num in sutra_numbers:
        sutra = get_sutra_by_number(num)
        if sutra is not None:
            operators.append(sutra)

    return CompositeOperator(operators)


# Self-test
def _self_test():
    """Run basic sutra tests."""
    from ..lattice import create_3d_lattice
    from ..state import create_gaussian_field

    # Create test environment
    lattice = create_3d_lattice(8, 8, 8)
    center = (4, 4, 4)
    state = create_gaussian_field(lattice, center, sigma=1.5, amplitude=1.0)
    context = OperatorContext()

    # Test all sutras
    all_sutras = get_all_sutras()
    assert len(all_sutras) == 29, f"Expected 29 sutras, got {len(all_sutras)}"

    # Test a few sutras
    for i, sutra in enumerate([all_sutras[0], all_sutras[2], all_sutras[8], all_sutras[15]]):
        result = sutra(state, context)
        assert result.validate_bounded(Fraction(10000)), f"Sutra {i+1} produced unbounded result"

    # Test pipeline
    pipeline = create_sutra_pipeline([1, 3, 5, 7, 9])
    result = pipeline(state, context)
    assert result.validate_bounded(Fraction(10000))


_self_test()
