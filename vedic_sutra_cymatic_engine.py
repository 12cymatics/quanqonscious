#!/usr/bin/env python3
"""
Vedic Sutra Cymatic Engine - Uses EXACT methods from existing codebase

NO NORMALIZATION - NO FLATTENING - RAW SUTRA VALUES PRESERVED

This engine generates cymatic patterns using:
- GRVQ field solver with proper singularity suppression (from grvqsutraws.py)
- Sub-sutras for interference and calculations (from primarysutraaws2.py)
- Sulba methods for geometric constructions (from sulbasutraws.py)
- Maya illusion transforms for phase modulation (from mayasutraaws.py)

Chakra frequencies: 396, 417, 528, 639, 741, 852, 963 Hz
Schumann resonances: 7.83, 14.3, 20.8, 27.3, 33.8, 39, 45 Hz
"""

import math
import os
from fractions import Fraction
from typing import List, Tuple, Dict

# Chakra frequencies (Solfeggio)
CHAKRA_FREQUENCIES = {
    'Root': 396,
    'Sacral': 417,
    'Solar': 528,
    'Heart': 639,
    'Throat': 741,
    'Third_Eye': 852,
    'Crown': 963
}

# Schumann resonances (Earth's electromagnetic resonance)
SCHUMANN_RESONANCES = [7.83, 14.3, 20.8, 27.3, 33.8, 39.0, 45.0]

# ============================================================================
# GRVQ FIELD SOLVER - Exact implementation from grvqsutraws.py
# ============================================================================

def grvq_singularity_suppression(r: float, r0_squared: float = 1.0) -> float:
    """
    GRVQ radial singularity suppression term.
    From grvqsutraws.py: radial_term = 1.0 - r*r / (r*r + r0_squared)
    """
    return 1.0 - (r * r) / (r * r + r0_squared)


def grvq_shape_S1(theta: float, phi: float, r: float) -> float:
    """
    Shape function S1: Spherical harmonic-inspired.
    From grvqsutraws.py: S1 = sin(theta) * cos(phi) * exp(-0.1 * r)
    """
    return math.sin(theta) * math.cos(phi) * math.exp(-0.1 * r)


def grvq_shape_S2(theta: float, phi: float, r: float) -> float:
    """
    Shape function S2: Toroidal function-inspired.
    From grvqsutraws.py: S2 = cos(theta) * sin(phi) * exp(-0.05 * r * r)
    """
    return math.cos(theta) * math.sin(phi) * math.exp(-0.05 * r * r)


def grvq_vedic_wave(r: float, theta: float, phi: float) -> float:
    """
    Vedic wave function.
    From grvqsutraws.py: f_vedic = sin(r + theta + phi) + 0.5 * cos(2 * (r + theta + phi))
    """
    sum_coord = r + theta + phi
    return math.sin(sum_coord) + 0.5 * math.cos(2 * sum_coord)


def grvq_product_terms(S1: float, S2: float, epsilon: float = 1e-8) -> Tuple[float, float]:
    """
    GRVQ product terms for singularity avoidance.
    From grvqsutraws.py:
      product_term1 = 1.0 - 1.0 / (|S1| + epsilon)
      product_term2 = 1.0 - 2.0 / (|S2| + epsilon)
    """
    product_term1 = 1.0 - 1.0 / (abs(S1) + epsilon)
    product_term2 = 1.0 - 2.0 / (abs(S2) + epsilon)
    return product_term1, product_term2


def grvq_turyavrtti_modulation(r: float, theta: float, phi: float,
                                turyavrtti_factor: float = 0.5) -> float:
    """
    Turyavrtti modulation for quantum-like oscillatory behavior.
    From grvqsutraws.py: turyavrtti_modulation = 1.0 + turyavrtti_factor * sin(π * r * theta * phi)
    """
    return 1.0 + turyavrtti_factor * math.sin(math.pi * r * theta * phi)


def grvq_field_solver(r: float, theta: float, phi: float,
                      turyavrtti_factor: float = 0.5) -> float:
    """
    Complete GRVQ Field Solver - Exact implementation from grvqsutraws.py

    GRVQ wavefunction ansatz:
    Ψ(r,θ,φ) = ∏ⱼ₌₁ⁿ(1-j/Sⱼ(r,θ,φ))(1-r²/r₀²)fVedic(r,θ,φ)
    """
    epsilon = 1e-8
    r0_squared = 1.0

    # Radial suppression (singularity-free)
    radial_term = grvq_singularity_suppression(r, r0_squared)

    # Shape functions
    S1 = grvq_shape_S1(theta, phi, r)
    S2 = grvq_shape_S2(theta, phi, r)

    # Vedic wave function
    f_vedic = grvq_vedic_wave(r, theta, phi)

    # Product terms for singularity avoidance
    product_term1, product_term2 = grvq_product_terms(S1, S2, epsilon)

    # Turyavrtti modulation
    turyavrtti_mod = grvq_turyavrtti_modulation(r, theta, phi, turyavrtti_factor)

    # Final GRVQ field calculation
    grvq_field = product_term1 * product_term2 * radial_term * f_vedic * turyavrtti_mod

    return grvq_field


# ============================================================================
# SUB-SUTRAS - Exact implementations from primarysutraaws2.py
# ============================================================================

def anurupye_sunyamanyat(a: float, b: float, epsilon: float = 1e-8) -> Tuple[float, float]:
    """
    Sub-Sutra 1: "If one is in ratio, the other is zero"
    Evaluates proportional relationships, zeroing terms when ratio conditions are met.
    From primarysutraaws2.py
    """
    if abs(b) < epsilon:
        if abs(a) < epsilon:
            return (0.0, 0.0)
        return (a, b)

    ratio = a / b
    # Check if ratio is close to 1 or -1
    if abs(abs(ratio) - 1.0) < epsilon:
        return (0.0, 0.0)
    return (a, b)


def sisyate_sesasamjnah(a: float, b: float, epsilon: float = 1e-8) -> float:
    """
    Sub-Sutra 2: "The remainder remains constant"
    Modular arithmetic - identifies and preserves remainder terms.
    From primarysutraaws2.py
    """
    if abs(b) < epsilon:
        return a
    quotient = int(a // b)
    return a - quotient * b


def yavadunam_tavadunam(a: float, b: float, base: float = 10.0) -> float:
    """
    Sub-Sutra 6: "Transfer the deficiency to the next level"
    Deficiency-based calculation.
    From primarysutraaws2.py:
    result = (base - a_def - b_def) * base + (a_def * b_def)
    """
    a_deficiency = base - a
    b_deficiency = base - b
    return (base - a_deficiency - b_deficiency) * base + (a_deficiency * b_deficiency)


def samuccayagunitah_subsutras(a: float, b: float) -> float:
    """
    Sub-Sutra 7: "Sum of products of sums"
    (a+1)(b+1) = ab + a + b + 1
    From primarysutraaws2.py
    """
    return a * b + a + b + 1


# ============================================================================
# SULBA SUTRAS - Exact implementations from sulbasutraws.py
# ============================================================================

def sulba_pi() -> float:
    """
    Sulba approximation for π: √10 ≈ 3.162...
    From sulbasutraws.py: pi_sulba = np.sqrt(10)
    """
    return math.sqrt(10)


def sulba_geometric_mean(a: float, b: float) -> float:
    """
    Sulba geometric mean: √(a×b)
    From sulbasutraws.py
    """
    return math.sqrt(abs(a * b))


def sulba_circle_area(radius: float) -> float:
    """
    Circle area using Sulba π approximation.
    From sulbasutraws.py: area = pi_sulba * radius * radius
    """
    return sulba_pi() * radius * radius


def sulba_pythagorean_construction(m: int, n: int) -> Tuple[int, int, int]:
    """
    Generate Pythagorean triple using Sulba method.
    From sulbasutraws.py:
    a = m² - n², b = 2mn, c = m² + n²
    """
    a = m * m - n * n
    b = 2 * m * n
    c = m * m + n * n
    return (min(a, b), max(a, b), c)


# ============================================================================
# MAYA ILLUSION SUTRAS - Exact implementations from mayasutraaws.py
# ============================================================================

def maya_illusion_transform(x: float, phase_factor: float = 0.5,
                           frequency: float = 1.0) -> float:
    """
    Maya Illusion Transform: Phase-modulated transformation.
    From mayasutraaws.py: result = x * (1 + phase_factor * sin(frequency * π * x))
    """
    return x * (1 + phase_factor * math.sin(frequency * math.pi * x))


def maya_illusion_multi_layer(x: float, phase_factors: List[float],
                              frequencies: List[float]) -> float:
    """
    Multi-layer Maya illusion transformation.
    From mayasutraaws.py: applies multiple phase modulations
    """
    result = x
    n_layers = min(len(phase_factors), len(frequencies))

    for i in range(n_layers):
        result = result * (1 + phase_factors[i] * math.sin(frequencies[i] * math.pi * result))

    return result


def maya_phase_cancellation(x: float, phase_factor: float = 0.5,
                           frequency: float = 1.0, threshold: float = 1e-8) -> float:
    """
    Maya phase cancellation - eliminates specific phase patterns.
    From mayasutraaws.py
    """
    phase_component = phase_factor * math.sin(frequency * math.pi * x)

    if abs(phase_component) > threshold:
        return x / (1 + phase_component)
    return x


# ============================================================================
# RAW VALUE TO COLOR - NO NORMALIZATION
# ============================================================================

def raw_value_to_rgb_no_normalize(value: float, mode: str = 'vedic') -> Tuple[int, int, int]:
    """
    Convert raw sutra value to RGB WITHOUT any normalization.
    Uses modular arithmetic to preserve variations.
    FORBIDS all normalization and flattening.
    """
    # Use modular arithmetic to wrap values into color space
    # This preserves ALL variations without flattening

    # Extract different frequency components from the raw value
    # Using sisyate_sesasamjnah (remainder) to preserve cyclic structure
    val_abs = abs(value)

    # Multiple frequency extraction - NO normalization
    low_freq = sisyate_sesasamjnah(val_abs * 100, 256)
    mid_freq = sisyate_sesasamjnah(val_abs * 317, 256)  # Using Sulba π * 100
    high_freq = sisyate_sesasamjnah(val_abs * 528, 256)  # Heart chakra frequency

    # Sign affects color channel assignment
    if value >= 0:
        r_base = low_freq
        g_base = mid_freq
        b_base = high_freq
    else:
        r_base = high_freq
        g_base = low_freq
        b_base = mid_freq

    if mode == 'vedic':
        # Gold/amber to deep purple - sacred colors
        r = int(abs(r_base) % 256)
        g = int(abs(g_base * 0.7) % 256)
        b = int(abs(b_base * 0.9) % 256)
    elif mode == 'chakra':
        # Full spectrum chakra colors
        r = int(abs(r_base) % 256)
        g = int(abs(g_base) % 256)
        b = int(abs(b_base) % 256)
    elif mode == 'schumann':
        # Earth tones
        r = int(abs(r_base * 0.6 + 80) % 256)
        g = int(abs(g_base * 0.8 + 60) % 256)
        b = int(abs(b_base * 0.5 + 100) % 256)
    else:
        # Direct mapping
        r = int(abs(r_base) % 256)
        g = int(abs(g_base) % 256)
        b = int(abs(b_base) % 256)

    return (r, g, b)


def multi_component_to_rgb(grvq: float, sulba: float, maya: float) -> Tuple[int, int, int]:
    """
    Convert multiple sutra components to RGB directly.
    Each sutra type drives a different color channel.
    NO NORMALIZATION.
    """
    # GRVQ drives red channel - use remainder to preserve structure
    r_val = sisyate_sesasamjnah(abs(grvq) * 1000, 256)

    # Sulba drives green channel
    g_val = sisyate_sesasamjnah(abs(sulba) * 1000, 256)

    # Maya drives blue channel
    b_val = sisyate_sesasamjnah(abs(maya) * 1000, 256)

    # Apply sign-based modulation
    if grvq < 0:
        r_val = 255 - r_val
    if sulba < 0:
        g_val = 255 - g_val
    if maya < 0:
        b_val = 255 - b_val

    return (int(r_val) % 256, int(g_val) % 256, int(b_val) % 256)


# ============================================================================
# CYMATIC FIELD GENERATOR - Uses all Vedic Sutra methods
# NO NORMALIZATION - PRESERVES ALL VARIATIONS
# ============================================================================

class VedicCymaticEngine:
    """
    Cymatic visualization engine using actual Vedic Sutra methods.
    All wave patterns are generated by the sutra computations.
    NO NORMALIZATION - ALL VARIATIONS PRESERVED.
    """

    def __init__(self, resolution: int = 800):
        self.resolution = resolution
        # Store raw values - no normalization
        self.field_grvq = [[0.0] * resolution for _ in range(resolution)]
        self.field_sulba = [[0.0] * resolution for _ in range(resolution)]
        self.field_maya = [[0.0] * resolution for _ in range(resolution)]

    def compute_grvq_cymatic_field(self, frequency: float,
                                   schumann: float,
                                   turyavrtti: float = 0.5) -> None:
        """
        Generate cymatic pattern using GRVQ field solver.
        The pattern is entirely driven by sutra computations.
        NO NORMALIZATION.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2

        for y in range(self.resolution):
            for x in range(self.resolution):
                # Convert to polar coordinates centered on grid
                dx = x - center
                dy = y - center
                r_raw = math.sqrt(dx * dx + dy * dy)

                # Scale r differently for different radial zones
                # This preserves structure at all scales
                r = (r_raw / max_r) * 10  # Scale r to [0, 10]

                if r < 0.001:
                    r = 0.001  # Avoid exact zero

                theta = math.atan2(dy, dx)

                # Phi modulated by frequency - creates unique patterns per frequency
                phi = (frequency / 100.0) * theta + (schumann / 10.0) * r

                # GRVQ field solver with full singularity suppression
                grvq_value = grvq_field_solver(r, theta, phi, turyavrtti)

                # Chakra frequency modulation - direct multiplication, no normalization
                chakra_wave = math.sin(2 * math.pi * (frequency / 50.0) * r)

                # Schumann resonance - additive modulation
                schumann_wave = math.sin(2 * math.pi * (schumann / 5.0) * r)

                # Turyavrtti spatial modulation
                turyavrtti_spatial = grvq_turyavrtti_modulation(r, theta, phi, turyavrtti)

                # Combined field - raw multiplication, no normalization
                self.field_grvq[y][x] = grvq_value * (1 + chakra_wave) * (1 + 0.5 * schumann_wave) * turyavrtti_spatial

    def compute_sulba_geometric_field(self, frequency: float) -> None:
        """
        Generate cymatic pattern using Sulba geometric constructions.
        NO NORMALIZATION.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2
        pi_sulba = sulba_pi()

        for y in range(self.resolution):
            for x in range(self.resolution):
                dx = x - center
                dy = y - center
                r_raw = math.sqrt(dx * dx + dy * dy)
                r = r_raw / max_r if max_r > 0 else 0.001
                theta = math.atan2(dy, dx)

                if r < 0.001:
                    r = 0.001

                # Sulba circle - area varies with radius
                circle_val = sulba_circle_area(r)

                # Sulba geometric mean creates interference
                geo_mean = sulba_geometric_mean(
                    abs(math.sin(theta * 3)) + 0.1,
                    abs(math.cos(theta * 5)) + 0.1
                )

                # Pythagorean harmonics (3-4-5 triple)
                pyth_a, pyth_b, pyth_c = sulba_pythagorean_construction(2, 1)

                # Wave using Sulba π and Pythagorean ratios
                wave1 = math.sin(pi_sulba * (frequency / 50.0) * r * pyth_a)
                wave2 = math.cos(pi_sulba * (frequency / 50.0) * r * pyth_b / pyth_c)

                # Angular harmonics from Pythagorean ratios
                angular = math.sin(pyth_a * theta) * math.cos(pyth_b * theta)

                # Raw combination - no normalization
                self.field_sulba[y][x] = (wave1 + wave2) * geo_mean * angular * circle_val

    def compute_maya_illusion_field(self, frequency: float,
                                    schumann: float) -> None:
        """
        Generate cymatic pattern using Maya illusion transforms.
        NO NORMALIZATION.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2

        # Phase factors from Schumann harmonics - raw values
        phase_factors = [s / 30.0 for s in SCHUMANN_RESONANCES[:4]]
        frequencies_list = [frequency / 100.0, frequency / 150.0,
                           frequency / 200.0, frequency / 250.0]

        for y in range(self.resolution):
            for x in range(self.resolution):
                dx = x - center
                dy = y - center
                r_raw = math.sqrt(dx * dx + dy * dy)
                r = r_raw / max_r if max_r > 0 else 0.001
                theta = math.atan2(dy, dx)

                if r < 0.001:
                    r = 0.001

                # Base radial wave - raw, not normalized
                base_radial = math.sin(2 * math.pi * (frequency / 30.0) * r)

                # Angular component
                base_angular = math.cos(3 * theta) + 0.5 * math.sin(5 * theta)

                # Combined base - raw product
                base_val = base_radial * base_angular + r

                # Multi-layer Maya illusion - cascaded transforms
                maya_multi = maya_illusion_multi_layer(
                    base_val + 1.0,  # Offset to avoid zero
                    phase_factors,
                    frequencies_list
                )

                # Single-layer transform for angular modulation
                maya_angular = maya_illusion_transform(
                    theta / math.pi + 0.5,
                    schumann / 50.0,
                    frequency / 200.0
                )

                # Phase cancellation creates interference nodes
                cancelled = maya_phase_cancellation(
                    maya_multi,
                    0.4,
                    schumann / 15.0
                )

                # Raw combination
                self.field_maya[y][x] = cancelled * maya_angular

    def compute_unified_vedic_field(self, chakra_freq: float,
                                    schumann: float) -> None:
        """
        Unified cymatic field combining all Vedic sutra methods.
        NO NORMALIZATION - ALL VARIATIONS PRESERVED.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2
        pi_sulba = sulba_pi()

        for y in range(self.resolution):
            for x in range(self.resolution):
                dx = x - center
                dy = y - center
                r_raw = math.sqrt(dx * dx + dy * dy)
                r = (r_raw / max_r) * 10 if max_r > 0 else 0.01

                if r < 0.01:
                    r = 0.01

                theta = math.atan2(dy, dx)
                phi = (chakra_freq / 100.0) * math.pi + theta * 0.5

                # =====================================================
                # 1. GRVQ field with full singularity suppression
                # =====================================================
                grvq_val = grvq_field_solver(r, theta, phi, 0.6)

                # =====================================================
                # 2. Sub-sutra modulations - RAW VALUES
                # =====================================================

                # Anurupye - ratio detection on wave components
                wave_a = math.sin(chakra_freq / 50.0 * r)
                wave_b = math.cos(schumann / 5.0 * r)
                filtered_a, filtered_b = anurupye_sunyamanyat(wave_a, wave_b)

                # Sisyate - cyclical remainder for periodic patterns
                cycle_val = sisyate_sesasamjnah(r * chakra_freq / 10.0, schumann)

                # Yavadunam - deficiency transfer between components
                deficiency_val = yavadunam_tavadunam(
                    abs(grvq_val * 5) + 1,
                    abs(cycle_val) + 1,
                    10.0
                )

                # Samuccayagunitah - sum-product expansion
                samuccaya_val = samuccayagunitah_subsutras(
                    abs(filtered_a) + 0.1,
                    abs(filtered_b) + 0.1
                )

                # =====================================================
                # 3. Sulba geometric constructions
                # =====================================================
                geo_mean = sulba_geometric_mean(
                    abs(grvq_val) + 0.01,
                    abs(deficiency_val / 50) + 0.01
                )

                # Sulba wave with exact π
                sulba_wave = math.sin(pi_sulba * r * 2) * math.cos(pi_sulba * theta)

                # =====================================================
                # 4. Maya illusion transforms
                # =====================================================
                maya_val = maya_illusion_transform(
                    geo_mean + 0.5,
                    0.5,
                    chakra_freq / 300.0
                )

                maya_angular = maya_illusion_transform(
                    theta / math.pi + 0.5,
                    schumann / 80.0,
                    2.0
                )

                # =====================================================
                # 5. Final combination - NO NORMALIZATION
                # Each component contributes with its raw magnitude
                # =====================================================
                self.field_grvq[y][x] = grvq_val
                self.field_sulba[y][x] = sulba_wave * geo_mean
                self.field_maya[y][x] = maya_val * maya_angular

                # Unified field stores combined value
                # Using addition and multiplication to preserve all variations

    def generate_image_raw(self, filename: str, mode: str = 'unified') -> bool:
        """
        Generate PNG image from field data WITHOUT NORMALIZATION.
        Uses modular arithmetic to map raw values to colors.
        """
        try:
            from PIL import Image
        except ImportError:
            print("PIL not available - cannot generate images")
            return False

        # Create RGB image
        img = Image.new('RGB', (self.resolution, self.resolution))
        pixels = img.load()

        for y in range(self.resolution):
            for x in range(self.resolution):
                if mode == 'grvq':
                    # GRVQ field to color - NO normalization
                    rgb = raw_value_to_rgb_no_normalize(
                        self.field_grvq[y][x], 'vedic'
                    )
                elif mode == 'sulba':
                    rgb = raw_value_to_rgb_no_normalize(
                        self.field_sulba[y][x], 'chakra'
                    )
                elif mode == 'maya':
                    rgb = raw_value_to_rgb_no_normalize(
                        self.field_maya[y][x], 'schumann'
                    )
                else:  # unified
                    # Each sutra field drives one color channel
                    rgb = multi_component_to_rgb(
                        self.field_grvq[y][x],
                        self.field_sulba[y][x],
                        self.field_maya[y][x]
                    )

                pixels[x, y] = rgb

        img.save(filename)
        return True


def main():
    """Generate cymatic visualizations using Vedic Sutra methods."""
    print("=" * 70)
    print("VEDIC SUTRA CYMATIC ENGINE - NO NORMALIZATION")
    print("Using exact methods from: grvqsutraws.py, primarysutraaws2.py,")
    print("                          sulbasutraws.py, mayasutraaws.py")
    print("ALL VARIATIONS PRESERVED - NO FLATTENING")
    print("=" * 70)

    # Create output directory
    output_dir = "vedic_sutra_cymatics"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize engine at high resolution
    engine = VedicCymaticEngine(resolution=1600)

    # Demonstrate the sutra methods are being used
    print("\n[GRVQ FIELD SOLVER DEMONSTRATION]")
    print("-" * 50)
    test_r, test_theta, test_phi = 2.0, math.pi/4, math.pi/3
    print(f"Input: r={test_r}, θ={test_theta:.4f}, φ={test_phi:.4f}")

    radial = grvq_singularity_suppression(test_r)
    print(f"  Singularity suppression: {radial:.6f}")

    S1 = grvq_shape_S1(test_theta, test_phi, test_r)
    S2 = grvq_shape_S2(test_theta, test_phi, test_r)
    print(f"  Shape S1: {S1:.6f}")
    print(f"  Shape S2: {S2:.6f}")

    f_vedic = grvq_vedic_wave(test_r, test_theta, test_phi)
    print(f"  Vedic wave: {f_vedic:.6f}")

    grvq_val = grvq_field_solver(test_r, test_theta, test_phi)
    print(f"  Full GRVQ field: {grvq_val:.6f}")

    print("\n[SUB-SUTRA DEMONSTRATION]")
    print("-" * 50)
    print(f"  Anurupye (5.0, 5.001): {anurupye_sunyamanyat(5.0, 5.001)}")
    print(f"  Sisyate (17, 5): {sisyate_sesasamjnah(17, 5)}")
    print(f"  Yavadunam (94, 98, 100): {yavadunam_tavadunam(94, 98, 100)}")
    print(f"  Samuccayagunitah (3, 4): {samuccayagunitah_subsutras(3, 4)}")

    print("\n[SULBA DEMONSTRATION]")
    print("-" * 50)
    print(f"  Sulba π (√10): {sulba_pi():.6f}")
    print(f"  Geometric mean (4, 9): {sulba_geometric_mean(4, 9):.6f}")
    print(f"  Pythagorean (m=2, n=1): {sulba_pythagorean_construction(2, 1)}")

    print("\n[MAYA ILLUSION DEMONSTRATION]")
    print("-" * 50)
    print(f"  Maya transform (0.5): {maya_illusion_transform(0.5, 0.5, 1.0):.6f}")
    print(f"  Maya multi-layer (0.5): {maya_illusion_multi_layer(0.5, [0.3, 0.5], [1.0, 2.0]):.6f}")

    # Generate images for each chakra frequency
    print("\n[GENERATING CHAKRA CYMATIC IMAGES - NO NORMALIZATION]")
    print("-" * 50)

    for idx, (chakra_name, freq) in enumerate(CHAKRA_FREQUENCIES.items()):
        schumann = SCHUMANN_RESONANCES[idx]

        print(f"\nGenerating: {chakra_name} Chakra ({freq} Hz) + Schumann ({schumann} Hz)")

        # 1. GRVQ-based cymatic
        print("  Computing GRVQ field (raw values)...")
        engine.compute_grvq_cymatic_field(freq, schumann)
        filename = f"{output_dir}/grvq_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image_raw(filename, 'grvq')
        print(f"  Saved: {filename}")

        # 2. Sulba geometric cymatic
        print("  Computing Sulba geometric field (raw values)...")
        engine.compute_sulba_geometric_field(freq)
        filename = f"{output_dir}/sulba_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image_raw(filename, 'sulba')
        print(f"  Saved: {filename}")

        # 3. Maya illusion cymatic
        print("  Computing Maya illusion field (raw values)...")
        engine.compute_maya_illusion_field(freq, schumann)
        filename = f"{output_dir}/maya_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image_raw(filename, 'maya')
        print(f"  Saved: {filename}")

        # 4. Unified Vedic field - all methods combined
        print("  Computing unified Vedic field (raw values)...")
        engine.compute_unified_vedic_field(freq, schumann)
        filename = f"{output_dir}/unified_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image_raw(filename, 'unified')
        print(f"  Saved: {filename}")

    print("\n" + "=" * 70)
    print("VEDIC SUTRA CYMATIC ENGINE - COMPLETE")
    print(f"Generated {len(CHAKRA_FREQUENCIES) * 4} images using actual sutra methods")
    print("NO NORMALIZATION APPLIED - ALL VARIATIONS PRESERVED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    main()
