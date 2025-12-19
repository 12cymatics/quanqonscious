#!/usr/bin/env python3
"""
Vedic Sutra Cymatic Engine - Uses EXACT methods from existing codebase

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
# CYMATIC FIELD GENERATOR - Uses all Vedic Sutra methods
# ============================================================================

class VedicCymaticEngine:
    """
    Cymatic visualization engine using actual Vedic Sutra methods.
    All wave patterns are generated by the sutra computations.
    """

    def __init__(self, resolution: int = 800):
        self.resolution = resolution
        self.field = [[0.0] * resolution for _ in range(resolution)]

    def compute_grvq_cymatic_field(self, frequency: float,
                                   schumann: float,
                                   turyavrtti: float = 0.5) -> List[List[float]]:
        """
        Generate cymatic pattern using GRVQ field solver.
        The pattern is entirely driven by sutra computations.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2

        for y in range(self.resolution):
            for x in range(self.resolution):
                # Convert to polar coordinates centered on grid
                dx = x - center
                dy = y - center
                r = math.sqrt(dx * dx + dy * dy) / max_r * 10  # Scale r to [0, 10]

                if r < 0.001:
                    r = 0.001  # Avoid exact zero

                theta = math.atan2(dy, dx)
                phi = (frequency / 100.0) * math.pi  # Frequency modulates azimuthal angle

                # Apply GRVQ field solver (singularity-suppressed)
                grvq_value = grvq_field_solver(r, theta, phi, turyavrtti)

                # Modulate by chakra frequency wave
                chakra_wave = math.sin(2 * math.pi * (frequency / 100.0) * r)

                # Apply Schumann resonance modulation
                schumann_mod = 1.0 + 0.3 * math.sin(2 * math.pi * (schumann / 10.0) * r)

                # Maya illusion phase transform
                maya_val = maya_illusion_transform(grvq_value, 0.3, frequency / 500.0)

                # Combine all sutra contributions
                self.field[y][x] = maya_val * chakra_wave * schumann_mod

        return self.field

    def compute_sulba_geometric_field(self, frequency: float) -> List[List[float]]:
        """
        Generate cymatic pattern using Sulba geometric constructions.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2
        pi_sulba = sulba_pi()

        for y in range(self.resolution):
            for x in range(self.resolution):
                dx = x - center
                dy = y - center
                r = math.sqrt(dx * dx + dy * dy) / max_r
                theta = math.atan2(dy, dx)

                if r < 0.001:
                    r = 0.001

                # Sulba circle construction
                circle_area = sulba_circle_area(r)

                # Sulba geometric mean of coordinates
                geo_mean = sulba_geometric_mean(abs(dx / max_r) + 0.01,
                                                abs(dy / max_r) + 0.01)

                # Apply frequency modulation with Sulba π
                wave = math.sin(pi_sulba * (frequency / 100.0) * r * 4)

                # Pythagorean harmonic (3-4-5 triple)
                pyth_a, pyth_b, pyth_c = sulba_pythagorean_construction(2, 1)
                pythagorean_wave = math.sin(pyth_a * theta) * math.cos(pyth_b * theta)

                self.field[y][x] = wave * geo_mean * pythagorean_wave * 2

        return self.field

    def compute_maya_illusion_field(self, frequency: float,
                                    schumann: float) -> List[List[float]]:
        """
        Generate cymatic pattern using Maya illusion transforms.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2

        # Phase factors derived from Schumann harmonics
        phase_factors = [s / 50.0 for s in SCHUMANN_RESONANCES[:3]]
        frequencies_list = [frequency / 200.0, frequency / 300.0, frequency / 400.0]

        for y in range(self.resolution):
            for x in range(self.resolution):
                dx = x - center
                dy = y - center
                r = math.sqrt(dx * dx + dy * dy) / max_r
                theta = math.atan2(dy, dx)

                if r < 0.001:
                    r = 0.001

                # Base value for illusion transform
                base_val = math.sin(2 * math.pi * (frequency / 100.0) * r) * r

                # Multi-layer Maya illusion
                maya_multi = maya_illusion_multi_layer(base_val + 0.5,
                                                       phase_factors,
                                                       frequencies_list)

                # Single-layer Maya transform for angular component
                maya_angular = maya_illusion_transform(theta / math.pi,
                                                      schumann / 100.0,
                                                      frequency / 500.0)

                # Phase cancellation for interference patterns
                cancelled = maya_phase_cancellation(maya_multi * maya_angular,
                                                   0.3, schumann / 20.0)

                self.field[y][x] = cancelled

        return self.field

    def compute_unified_vedic_field(self, chakra_freq: float,
                                    schumann: float) -> List[List[float]]:
        """
        Unified cymatic field combining all Vedic sutra methods.
        """
        center = self.resolution // 2
        max_r = self.resolution // 2
        pi_sulba = sulba_pi()

        for y in range(self.resolution):
            for x in range(self.resolution):
                dx = x - center
                dy = y - center
                r = math.sqrt(dx * dx + dy * dy) / max_r * 10

                if r < 0.01:
                    r = 0.01

                theta = math.atan2(dy, dx)
                phi = (chakra_freq / 100.0) * math.pi

                # 1. GRVQ field with singularity suppression
                grvq_val = grvq_field_solver(r, theta, phi, 0.5)

                # 2. Sub-sutra modulations
                # Anurupye - ratio-based filtering
                a_val = math.sin(chakra_freq / 100.0 * r)
                b_val = math.cos(schumann / 10.0 * r)
                filtered_a, filtered_b = anurupye_sunyamanyat(a_val, b_val)

                # Sisyate - cyclical remainder pattern
                cycle_val = sisyate_sesasamjnah(r * chakra_freq, schumann)

                # Yavadunam - deficiency transfer
                deficiency_val = yavadunam_tavadunam(abs(grvq_val) * 10,
                                                    abs(cycle_val) + 1,
                                                    10.0)

                # 3. Sulba geometric construction
                geo_mean = sulba_geometric_mean(abs(grvq_val) + 0.01,
                                               abs(deficiency_val / 100) + 0.01)

                # 4. Maya illusion transform
                maya_val = maya_illusion_transform(geo_mean, 0.4, chakra_freq / 400.0)

                # 5. Final combination with all sutra contributions
                combined = (grvq_val * 0.3 +
                           (filtered_a + filtered_b) * 0.2 +
                           maya_val * 0.3 +
                           math.sin(pi_sulba * r) * 0.2)

                self.field[y][x] = combined

        return self.field

    def generate_image(self, field: List[List[float]],
                       filename: str,
                       colormap: str = 'vedic') -> bool:
        """
        Generate PNG image from field data.
        """
        try:
            from PIL import Image
        except ImportError:
            print("PIL not available - cannot generate images")
            return False

        # Normalize field to 0-255 range
        min_val = min(min(row) for row in field)
        max_val = max(max(row) for row in field)
        val_range = max_val - min_val if max_val != min_val else 1.0

        # Create RGB image
        img = Image.new('RGB', (self.resolution, self.resolution))
        pixels = img.load()

        for y in range(self.resolution):
            for x in range(self.resolution):
                # Normalize value
                norm_val = (field[y][x] - min_val) / val_range

                # Apply colormap
                if colormap == 'vedic':
                    # Gold to deep purple gradient (sacred colors)
                    r = int(255 * (0.7 + 0.3 * norm_val))
                    g = int(255 * (0.3 + 0.4 * norm_val))
                    b = int(255 * (0.2 + 0.6 * (1 - norm_val)))
                elif colormap == 'chakra':
                    # Rainbow chakra colors
                    hue = norm_val * 0.8  # Limit to avoid wrap
                    r, g, b = self._hue_to_rgb(hue)
                elif colormap == 'schumann':
                    # Earth tones (brown, green, blue)
                    r = int(255 * (0.4 + 0.3 * (1 - norm_val)))
                    g = int(255 * (0.5 + 0.3 * norm_val))
                    b = int(255 * (0.3 + 0.5 * norm_val))
                else:  # grayscale
                    gray = int(255 * norm_val)
                    r = g = b = gray

                pixels[x, y] = (max(0, min(255, r)),
                               max(0, min(255, g)),
                               max(0, min(255, b)))

        img.save(filename)
        return True

    def _hue_to_rgb(self, h: float) -> Tuple[int, int, int]:
        """Convert hue (0-1) to RGB."""
        i = int(h * 6)
        f = h * 6 - i

        if i == 0:
            r, g, b = 1, f, 0
        elif i == 1:
            r, g, b = 1-f, 1, 0
        elif i == 2:
            r, g, b = 0, 1, f
        elif i == 3:
            r, g, b = 0, 1-f, 1
        elif i == 4:
            r, g, b = f, 0, 1
        else:
            r, g, b = 1, 0, 1-f

        return int(r * 255), int(g * 255), int(b * 255)


def main():
    """Generate cymatic visualizations using Vedic Sutra methods."""
    print("=" * 70)
    print("VEDIC SUTRA CYMATIC ENGINE")
    print("Using exact methods from: grvqsutraws.py, primarysutraaws2.py,")
    print("                          sulbasutraws.py, mayasutraaws.py")
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
    print("\n[GENERATING CHAKRA CYMATIC IMAGES]")
    print("-" * 50)

    for idx, (chakra_name, freq) in enumerate(CHAKRA_FREQUENCIES.items()):
        schumann = SCHUMANN_RESONANCES[idx]

        print(f"\nGenerating: {chakra_name} Chakra ({freq} Hz) + Schumann ({schumann} Hz)")

        # 1. GRVQ-based cymatic
        print("  Computing GRVQ field...")
        field = engine.compute_grvq_cymatic_field(freq, schumann)
        filename = f"{output_dir}/grvq_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image(field, filename, 'vedic')
        print(f"  Saved: {filename}")

        # 2. Sulba geometric cymatic
        print("  Computing Sulba geometric field...")
        field = engine.compute_sulba_geometric_field(freq)
        filename = f"{output_dir}/sulba_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image(field, filename, 'chakra')
        print(f"  Saved: {filename}")

        # 3. Maya illusion cymatic
        print("  Computing Maya illusion field...")
        field = engine.compute_maya_illusion_field(freq, schumann)
        filename = f"{output_dir}/maya_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image(field, filename, 'schumann')
        print(f"  Saved: {filename}")

        # 4. Unified Vedic field
        print("  Computing unified Vedic field...")
        field = engine.compute_unified_vedic_field(freq, schumann)
        filename = f"{output_dir}/unified_{chakra_name.lower()}_{freq}Hz.png"
        engine.generate_image(field, filename, 'vedic')
        print(f"  Saved: {filename}")

    print("\n" + "=" * 70)
    print("VEDIC SUTRA CYMATIC ENGINE - COMPLETE")
    print(f"Generated {len(CHAKRA_FREQUENCIES) * 4} images using actual sutra methods")
    print("=" * 70)

    return True


if __name__ == "__main__":
    main()
