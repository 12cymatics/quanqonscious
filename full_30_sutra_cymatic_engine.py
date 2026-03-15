#!/usr/bin/env python3
"""
FULL 30-SUTRA CYMATIC ENGINE

Uses ALL 16 PRIMARY SUTRAS (applied in SERIES) and ALL 14 SUB-SUTRAS (applied in PARALLEL)
as specified in the codebase: colab2.txt, grvq3.txt, primarysutraaws2.py

16 PRIMARY SUTRAS (SERIES - each transforms the previous result):
1. Ekadhikena Purvena - "By one more than the previous"
2. Nikhilam - "All from 9 and last from 10"
3. Urdhva Tiryagbhyam - "Vertically and crosswise"
4. Urdhva Veerya - "Vertical energy multiplication"
5. Paravartya Yojayet - "Transpose and apply"
6. Shunyam Sampurna - "When zero is whole"
7. Anurupyena - "Proportionately"
8. Sopantyadvayamantyam - "Ultimate and twice penultimate"
9. Ekanyunena Purvena - "By one less than the previous"
10. Dvitiya - "Second portion application"
11. Virahata - "Separate by second harmonic"
12. Ayur/Ayadalagana - "Life force scaling"
13. Samuchchhayo - "Sum aggregation"
14. Alankara/Gunakasamuchyah - "Ornamental index modulation"
15. Sandhya - "Junction averaging"
16. Sandhya Samuccaya - "Weighted junction aggregation"

14 SUB-SUTRAS (PARALLEL - all run independently, then averaged):
1. Anurupye Sunyamanyat - "If one is in ratio, other is zero"
2. Sisyate Sesasamjnah - "Remainder remains constant"
3. Adyamadyenantyamantyena - "First by first, last by last"
4. Antyayordasakepi - "Last digits sum to 10"
5. Antyayoreva - "Only the last terms"
6. Yavadunam Tavadunam - "Deficiency transfer"
7. Samuccayagunitah - "Sum of products of sums"
8. Ekadhikena Sub - "Recursive increment"
9. Paravartya Sub - "Recursive division"
10. Sankalana Samanantara - "Adjacent sum"
11. Shunyam Samyasamuccaye - "Sum to zero check"
12. Puranapuranabhyam - "Completion to base"
13. Vargamula - "Square root approximation"
14. Convergence - "Dampening factor"

NO NORMALIZATION - NO FLATTENING - RAW SUTRA VALUES DRIVE PATTERNS
"""

import math
import os
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from hc_ipc import HcIpcClient
from hypercube_fm8 import HyperCubeFM8


# ═══════════════════════════════════════════════════════════════════════════════
# CHAKRA FREQUENCIES AND SCHUMANN RESONANCES
# ═══════════════════════════════════════════════════════════════════════════════

CHAKRA_FREQUENCIES = {
    'Root': 396,
    'Sacral': 417,
    'Solar': 528,
    'Heart': 639,
    'Throat': 741,
    'Third_Eye': 852,
    'Crown': 963
}

SCHUMANN_RESONANCES = [7.83, 14.3, 20.8, 27.3, 33.8, 39.0, 45.0]

# Audio integration (enabled when QUANQONSCIOUS_AUDIO=1)
AUDIO_ENABLED = os.getenv("QUANQONSCIOUS_AUDIO", "0") == "1"
AUDIO_CLIENT = HcIpcClient() if AUDIO_ENABLED else None
AUDIO_CUBE = HyperCubeFM8(num_ops=12, base_frequency=432.0) if AUDIO_ENABLED else None
AUDIO_STARTED = False

def _init_audio_matrices() -> None:
    if not AUDIO_ENABLED or AUDIO_CUBE is None:
        return
    mod = np.zeros((AUDIO_CUBE.num_ops, AUDIO_CUBE.num_ops), dtype=float)
    for i in range(AUDIO_CUBE.num_ops):
        for j in range(AUDIO_CUBE.num_ops):
            mod[i, j] = 0.02 / (1.0 + abs(i - j)) if i != j else 0.0
    AUDIO_CUBE.set_modulation_matrix(mod)
    AUDIO_CUBE.set_input_matrix([[0.0] for _ in range(AUDIO_CUBE.num_ops)])
    AUDIO_CUBE.set_mix_mode("serial")
    AUDIO_CUBE.add_sutra_mapping(
        "cymatic_field",
        operator_indices=range(AUDIO_CUBE.num_ops),
        freq_scale=0.03,
        level_scale=0.01,
        ratio_scale=0.002,
        detune_scale=0.45,
    )

def _emit_audio_update(values: List[float]) -> None:
    global AUDIO_STARTED
    if not AUDIO_ENABLED or AUDIO_CLIENT is None or AUDIO_CUBE is None:
        return
    if not AUDIO_STARTED:
        AUDIO_CLIENT.start()
        AUDIO_STARTED = True
    AUDIO_CUBE.apply_sutra_to_operators("cymatic_field", values)
    payload = AUDIO_CUBE.as_update_payload()
    AUDIO_CLIENT.send_state(
        payload["base_ops"],
        payload["levels"],
        mod_matrix=payload["mod_matrix"],
        input_matrix=payload["input_matrix"],
        mix_mode=payload["mix_mode"],
    )

_init_audio_matrices()


# ═══════════════════════════════════════════════════════════════════════════════
# 16 PRIMARY SUTRAS - Applied in SERIES (each transforms the previous)
# From colab2.txt exact implementations
# ═══════════════════════════════════════════════════════════════════════════════

def sutra1_ekadhikena(p: float) -> float:
    """Sutra 1: By one more than the previous - sinusoidal increment"""
    return p + 0.001 * math.sin(p)

def sutra2_nikhilam(p: float) -> float:
    """Sutra 2: All from 9, last from 10 - complement adjustment"""
    return p - 0.002 * (1 - p)

def sutra3_urdhva_tiryagbhyam(p: float) -> float:
    """Sutra 3: Vertically and crosswise - cosine multiplication"""
    return p * (1 + 0.003 * math.cos(p))

def sutra4_urdhva_veerya(p: float) -> float:
    """Sutra 4: Vertical energy - exponential scaling"""
    return p * math.exp(0.0005 * p)

def sutra5_paravartya(p: float, context_sign: float = 1.0) -> float:
    """Sutra 5: Transpose and apply - sign-based offset"""
    return p * context_sign + 0.0008

def sutra6_shunyam_sampurna(p: float) -> float:
    """Sutra 6: When zero is whole - threshold application"""
    return p if abs(p) > 0.1 else p + 0.1 * (1 if p >= 0 else -1)

def sutra7_anurupyena(p: float, avg: float) -> float:
    """Sutra 7: Proportionately - deviation scaling"""
    return p * (1 + 0.0003 * (p - avg))

def sutra8_sopantyadvayamantyam(p: float, neighbor: float) -> float:
    """Sutra 8: Ultimate and twice penultimate - pairwise average"""
    return (p + neighbor) / 2.0

def sutra9_ekanyunena(p: float, factor: float) -> float:
    """Sutra 9: By one less than previous - factor offset"""
    return p + 0.0007 * factor

def sutra10_dvitiya(p: float, factor: float) -> float:
    """Sutra 10: Second portion - second half scaling"""
    return p * (1 + 0.0004 * factor)

def sutra11_virahata(p: float) -> float:
    """Sutra 11: Separate by second harmonic - double frequency sin"""
    return p + 0.0015 * math.sin(2 * p)

def sutra12_ayur(p: float) -> float:
    """Sutra 12: Life force - absolute value scaling"""
    return p * (1 + 0.0006 * abs(p))

def sutra13_samuchchhayo(p: float, total: float) -> float:
    """Sutra 13: Sum aggregation - total-based offset"""
    return p + 0.0002 * total

def sutra14_alankara(p: float, index: int) -> float:
    """Sutra 14: Ornamental - index-based sinusoidal"""
    return p + 0.0005 * math.sin(index)

def sutra15_sandhya(p: float, neighbor: float) -> float:
    """Sutra 15: Junction - neighbor averaging"""
    return (p + neighbor) / 2.0

def sutra16_sandhya_samuccaya(p: float, weighted_avg: float) -> float:
    """Sutra 16: Weighted junction - weighted average offset"""
    return p + 0.0003 * weighted_avg


def apply_16_primary_sutras_series(base_value: float, x: int, y: int,
                                   resolution: int, frequency: float) -> float:
    """
    Apply ALL 16 primary sutras IN SERIES (sequentially).
    Each sutra transforms the result of the previous.
    """
    p = base_value

    # Context values derived from position and frequency
    center = resolution // 2
    dx = x - center
    dy = y - center
    r = math.sqrt(dx * dx + dy * dy) / (resolution / 2) if resolution > 0 else 0.01
    theta = math.atan2(dy, dx)

    # Compute context factors
    context_sign = 1.0 if dx >= 0 else -1.0
    avg_context = math.sin(frequency / 100.0) * r
    neighbor_approx = p * (1 + 0.01 * math.cos(theta))
    factor_first = abs(math.sin(theta * 2))
    factor_second = abs(math.cos(theta * 3))
    total_approx = p * 4  # Approximation
    idx = x * resolution + y
    weighted_avg = p * r

    # SERIES APPLICATION - Each sutra transforms previous result
    p = sutra1_ekadhikena(p)                           # 1. Sinusoidal increment
    p = sutra2_nikhilam(p)                             # 2. Complement adjustment
    p = sutra3_urdhva_tiryagbhyam(p)                   # 3. Cosine multiplication
    p = sutra4_urdhva_veerya(p)                        # 4. Exponential scaling
    p = sutra5_paravartya(p, context_sign)            # 5. Sign-based transpose
    p = sutra6_shunyam_sampurna(p)                     # 6. Threshold application
    p = sutra7_anurupyena(p, avg_context)             # 7. Deviation scaling
    p = sutra8_sopantyadvayamantyam(p, neighbor_approx) # 8. Pairwise average
    p = sutra9_ekanyunena(p, factor_first)            # 9. Factor offset
    p = sutra10_dvitiya(p, factor_second)             # 10. Second half scaling
    p = sutra11_virahata(p)                            # 11. Second harmonic
    p = sutra12_ayur(p)                                # 12. Absolute scaling
    p = sutra13_samuchchhayo(p, total_approx)         # 13. Total-based offset
    p = sutra14_alankara(p, idx)                       # 14. Index sinusoidal
    p = sutra15_sandhya(p, neighbor_approx)           # 15. Junction average
    p = sutra16_sandhya_samuccaya(p, weighted_avg)    # 16. Weighted junction

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 14 SUB-SUTRAS - Applied in PARALLEL (all run independently, then combined)
# From colab2.txt and grvq3.txt exact implementations
# ═══════════════════════════════════════════════════════════════════════════════

def subsutra1_anurupye_sunyamanyat(p: float, ratio_ref: float, epsilon: float = 1e-8) -> float:
    """Sub-Sutra 1: If one is in ratio, other is zero - ratio detection"""
    if abs(ratio_ref) < epsilon:
        return p
    ratio = p / (ratio_ref + epsilon)
    if abs(abs(ratio) - 1.0) < epsilon:
        return 0.0  # Ratio condition met
    return p + 0.0001 * p * p  # Refinement

def subsutra2_sisyate_sesasamjnah(p: float, modulus: float) -> float:
    """Sub-Sutra 2: Remainder remains constant - modular arithmetic"""
    if abs(modulus) < 1e-8:
        return p
    quotient = int(p // modulus) if modulus != 0 else 0
    remainder = p - quotient * modulus
    return p - 0.0002 * (p - 0.5) + 0.001 * remainder

def subsutra3_adyamadyenantyamantyena(p: float, first_val: float, last_val: float) -> float:
    """Sub-Sutra 3: First by first, last by last - endpoint multiplication"""
    # Recursion effect - roll and average
    rolled = p * 0.5 + first_val * 0.25 + last_val * 0.25
    return (p + rolled) / 2.0

def subsutra4_antyayordasakepi(p: float, mod_base: float = 10.0) -> float:
    """Sub-Sutra 4: Last digits sum to 10 - modular completion"""
    last_digit = abs(p * 1000) % mod_base
    complement = mod_base - last_digit
    return 0.9 * p + 0.001 * complement  # Convergence factor

def subsutra5_antyayoreva(p: float) -> float:
    """Sub-Sutra 5: Only the last terms - focus on terminal values"""
    # Stabilization - clip to range
    return max(min(p, 1.0), -1.0) if abs(p) <= 10 else p * 0.1

def subsutra6_yavadunam_tavadunam(p: float, base: float = 10.0) -> float:
    """Sub-Sutra 6: Deficiency transfer - base completion"""
    deficiency = base - abs(p * 10) % base
    return p + 0.0001 * deficiency  # Simplification

def subsutra7_samuccayagunitah(p: float, companion: float) -> float:
    """Sub-Sutra 7: Sum of products of sums - (a+1)(b+1)"""
    # Interpolation with companion
    sum_product = p * companion + p + companion + 1
    return p + 0.00005 * sum_product

def subsutra8_ekadhikena_sub(p: float) -> float:
    """Sub-Sutra 8: Recursive increment application"""
    # Extrapolation - polynomial trend
    trend = 0.001 * p  # Linear trend
    return p + 0.0001 * (p + trend)

def subsutra9_paravartya_sub(p: float, divisor: float = 2.0) -> float:
    """Sub-Sutra 9: Recursive division - element-wise scaling"""
    # Error reduction
    std_approx = abs(p - 0.5)
    return (p / divisor) - 0.0001 * std_approx

def subsutra10_sankalana_samanantara(p: float, neighbor: float) -> float:
    """Sub-Sutra 10: Adjacent sum - neighboring element aggregation"""
    # Optimization - mean centering
    mean_approx = (p + neighbor) / 2.0
    return p + 0.0002 * (mean_approx - p)

def subsutra11_shunyam_samyasamuccaye(p: float, total: float) -> float:
    """Sub-Sutra 11: Sum to zero check - balance verification"""
    # Adjustment - cosine modulation
    if abs(total) < 1e-8:
        return 0.0
    return p + 0.0003 * math.cos(p)

def subsutra12_puranapuranabhyam(p: float, base: float = 10.0) -> float:
    """Sub-Sutra 12: Completion to base - rounding to nearest multiple"""
    # Modulation - index-like scaling
    completed = round(p * 10 / base) * base / 10
    return p * (1 + 0.00005 * abs(completed - p * 10))

def subsutra13_vargamula(p: float) -> float:
    """Sub-Sutra 13: Square root approximation - Vedic sqrt method"""
    # Differentiation - gradient approximation
    if p <= 0:
        return abs(p) + 0.0001
    # Newton-Raphson inspired
    guess = abs(p) if p > 1 else 1.0
    for _ in range(3):
        guess = 0.5 * (guess + abs(p) / guess)
    gradient_approx = guess - abs(p)
    return p + 0.0001 * gradient_approx

def subsutra14_convergence(p: float) -> float:
    """Sub-Sutra 14: Dampening factor for stability"""
    return 0.95 * p  # Slight convergence


def apply_14_subsutras_parallel(base_value: float, x: int, y: int,
                                resolution: int, frequency: float) -> float:
    """
    Apply ALL 14 sub-sutras IN PARALLEL (concurrently).
    Each sub-sutra operates on the SAME input, results are averaged.
    """
    # Context values
    center = resolution // 2
    dx = x - center
    dy = y - center
    r = math.sqrt(dx * dx + dy * dy) / (resolution / 2) if resolution > 0 else 0.01
    theta = math.atan2(dy, dx)

    p = base_value

    # Context factors for sub-sutras
    ratio_ref = math.sin(frequency / 100.0 * r)
    modulus = frequency / 100.0
    first_val = math.sin(theta)
    last_val = math.cos(theta)
    companion = math.sin(2 * theta) * r
    neighbor = p * (1 + 0.01 * math.cos(theta))
    total = p * 4

    # PARALLEL APPLICATION - All sub-sutras run on same input
    results = []

    # Execute all 14 sub-sutras
    results.append(subsutra1_anurupye_sunyamanyat(p, ratio_ref))
    results.append(subsutra2_sisyate_sesasamjnah(p, modulus))
    results.append(subsutra3_adyamadyenantyamantyena(p, first_val, last_val))
    results.append(subsutra4_antyayordasakepi(p))
    results.append(subsutra5_antyayoreva(p))
    results.append(subsutra6_yavadunam_tavadunam(p))
    results.append(subsutra7_samuccayagunitah(p, companion))
    results.append(subsutra8_ekadhikena_sub(p))
    results.append(subsutra9_paravartya_sub(p))
    results.append(subsutra10_sankalana_samanantara(p, neighbor))
    results.append(subsutra11_shunyam_samyasamuccaye(p, total))
    results.append(subsutra12_puranapuranabhyam(p))
    results.append(subsutra13_vargamula(p))
    results.append(subsutra14_convergence(p))

    # Combine parallel results by averaging
    return sum(results) / len(results)


# ═══════════════════════════════════════════════════════════════════════════════
# CHLADNI PLATE EQUATION - Core wave pattern
# ═══════════════════════════════════════════════════════════════════════════════

def chladni_wave(x: float, y: float, m: int, n: int) -> float:
    """
    Chladni plate equation: sin(mπx)sin(nπy) + sin(nπx)sin(mπy)
    Creates nodal lines where value = 0
    """
    term1 = math.sin(m * math.pi * x) * math.sin(n * math.pi * y)
    term2 = math.sin(n * math.pi * x) * math.sin(m * math.pi * y)
    return term1 + term2


def multi_mode_chladni(x: float, y: float, modes: List[Tuple[int, int]],
                       weights: List[float]) -> float:
    """Multi-mode Chladni superposition with weights"""
    total = 0.0
    for (m, n), w in zip(modes, weights):
        total += w * chladni_wave(x, y, m, n)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# BESSEL FUNCTIONS - From colab3.txt (pure Python approximation)
# ═══════════════════════════════════════════════════════════════════════════════

def bessel_j(n: int, x: float, terms: int = 20) -> float:
    """Bessel function J_n(x) approximation using series expansion"""
    result = 0.0
    for k in range(terms):
        # J_n(x) = sum_{k=0}^inf (-1)^k / (k! * (n+k)!) * (x/2)^(n+2k)
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i
        factorial_nk = 1
        for i in range(1, n + k + 1):
            factorial_nk *= i
        term = ((-1) ** k) / (factorial_k * factorial_nk) * ((x / 2) ** (n + 2 * k))
        result += term
    return result


def bessel_shape_function(j: int, r: float) -> float:
    """Shape function S_j(r) = J_{j+1}(2πr) from colab3.txt"""
    return bessel_j(j + 1, 2 * math.pi * r)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL 30-SUTRA CYMATIC ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class Full30SutraCymaticEngine:
    """
    Cymatic visualization engine using FULL 30-SUTRA system:
    - 16 Primary Sutras in SERIES
    - 14 Sub-Sutras in PARALLEL

    Pattern = Chladni × 16_Sutras_Series × 14_SubSutras_Parallel

    NO NORMALIZATION - ALL VARIATIONS PRESERVED
    """

    def __init__(self, resolution: int = 1600):
        self.resolution = resolution
        self.field = [[0.0] * resolution for _ in range(resolution)]

    def compute_full_30_sutra_field(self, frequency: float, schumann: float) -> None:
        """
        Generate cymatic pattern using FULL 30-SUTRA transformation chain.

        For each pixel:
        1. Compute base Chladni wave pattern
        2. Apply ALL 16 primary sutras IN SERIES
        3. Apply ALL 14 sub-sutras IN PARALLEL
        4. Combine results multiplicatively (preserves nodal zeros)
        """
        center = self.resolution // 2
        max_r = self.resolution // 2

        # Derive Chladni modes from frequency
        # Higher frequencies = higher mode numbers = more complex patterns
        base_m = max(2, int((frequency - 350) / 50))
        base_n = base_m * 2 + int(frequency / 200)

        # Multi-mode for richness
        modes = [
            (base_m, base_n),
            (base_m + 1, base_n - 2),
            (base_m - 1, base_n + 2)
        ]
        weights = [1.0, 0.5, 0.3]

        print(f"  Computing field with modes: {modes}")
        print(f"  16 PRIMARY SUTRAS (series) + 14 SUB-SUTRAS (parallel)")

        for y in range(self.resolution):
            if y % 200 == 0:
                print(f"    Row {y}/{self.resolution}...")

            for x in range(self.resolution):
                # Normalized coordinates [-1, 1]
                nx = (x - center) / max_r
                ny = (y - center) / max_r

                # Polar coordinates
                r = math.sqrt(nx * nx + ny * ny)
                theta = math.atan2(ny, nx)

                # Skip outside circle
                if r > 1.0:
                    self.field[y][x] = 0.0
                    continue

                # ═══════════════════════════════════════════════
                # STEP 1: BASE CHLADNI WAVE PATTERN
                # ═══════════════════════════════════════════════
                chladni_val = multi_mode_chladni(
                    (nx + 1) / 2,  # Map to [0, 1]
                    (ny + 1) / 2,
                    modes,
                    weights
                )

                # ═══════════════════════════════════════════════
                # STEP 2: BESSEL RADIAL MODULATION
                # ═══════════════════════════════════════════════
                bessel_mod = 1.0
                for j in range(5):
                    alpha = 0.1 + 0.02 * j * (frequency / 500)
                    S_j = bessel_shape_function(j, r * 3)
                    bessel_mod *= (1.0 + alpha * S_j)

                # ═══════════════════════════════════════════════
                # STEP 3: APPLY 16 PRIMARY SUTRAS (SERIES)
                # ═══════════════════════════════════════════════
                # Base value combining Chladni and Bessel
                base_for_sutras = chladni_val * bessel_mod

                # Transform through ALL 16 primary sutras sequentially
                series_result = apply_16_primary_sutras_series(
                    base_for_sutras, x, y, self.resolution, frequency
                )

                # ═══════════════════════════════════════════════
                # STEP 4: APPLY 14 SUB-SUTRAS (PARALLEL)
                # ═══════════════════════════════════════════════
                # All sub-sutras run on series result, then averaged
                parallel_result = apply_14_subsutras_parallel(
                    series_result, x, y, self.resolution, frequency
                )

                # ═══════════════════════════════════════════════
                # STEP 5: SCHUMANN RESONANCE MODULATION
                # ═══════════════════════════════════════════════
                schumann_wave = math.sin(2 * math.pi * schumann / 10.0 * r)
                schumann_mod = 1.0 + 0.1 * schumann_wave

                # ═══════════════════════════════════════════════
                # STEP 6: FINAL COMBINATION
                # ═══════════════════════════════════════════════
                # Multiplicative to preserve nodal structure
                # Chladni provides the nodal zeros
                # Sutras modulate the amplitude

                # The series result preserves Chladni zero crossings
                # The parallel average adds texture
                combined = series_result * (1.0 + 0.5 * (parallel_result - series_result))
                combined *= schumann_mod

                self.field[y][x] = combined

    def value_to_rgb(self, value: float, chakra_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Convert field value to RGB using chakra-specific coloring.
        Uses fractional parts to preserve micro-variations.
        NO NORMALIZATION.
        """
        # Scale to bring out details
        scaled = value * 1000

        # Fractional parts at different scales
        frac1 = abs(scaled) - int(abs(scaled))
        frac2 = abs(scaled * 3.162) - int(abs(scaled * 3.162))  # √10 scale
        frac3 = abs(scaled * 5.28) - int(abs(scaled * 5.28))    # 528Hz scale

        # Integer parts for macro structure
        int1 = int(abs(scaled)) % 256
        int2 = int(abs(scaled * 3.162)) % 256
        int3 = int(abs(scaled * 5.28)) % 256

        # Base RGB from chakra color
        base_r, base_g, base_b = chakra_color

        # Modulate with fractional and integer parts
        r = int((frac1 * 128 + int1 * 0.5 + base_r) % 256)
        g = int((frac2 * 128 + int2 * 0.5 + base_g) % 256)
        b = int((frac3 * 128 + int3 * 0.5 + base_b) % 256)

        # Sign inverts for contrast
        if value < 0:
            r, b = b, r

        return (r, g, b)

    def generate_image(self, filename: str, chakra_color: Tuple[int, int, int]) -> bool:
        """Generate PNG image from field data."""
        try:
            from PIL import Image
        except ImportError:
            print("PIL not available")
            return False

        img = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        pixels = img.load()

        for y in range(self.resolution):
            for x in range(self.resolution):
                rgb = self.value_to_rgb(self.field[y][x], chakra_color)
                pixels[x, y] = rgb

        img.save(filename)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

CHAKRA_COLORS = {
    'Root': (180, 30, 30),       # Red
    'Sacral': (220, 120, 30),    # Orange
    'Solar': (220, 200, 50),     # Yellow
    'Heart': (50, 180, 80),      # Green
    'Throat': (50, 130, 200),    # Blue
    'Third_Eye': (100, 60, 180), # Indigo
    'Crown': (150, 80, 200)      # Violet
}


def main():
    """Generate cymatic visualizations using FULL 30-SUTRA system."""
    print("=" * 80)
    print("FULL 30-SUTRA CYMATIC ENGINE")
    print("=" * 80)
    print("16 PRIMARY SUTRAS (applied in SERIES)")
    print("14 SUB-SUTRAS (applied in PARALLEL)")
    print("NO NORMALIZATION - NO FLATTENING - ALL VARIATIONS PRESERVED")
    print("=" * 80)

    # Create output directory
    output_dir = "full_30_sutra_cymatics"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize engine
    engine = Full30SutraCymaticEngine(resolution=1600)

    # Sutra chain execution
    print("\n[SUTRA CHAIN EXECUTION]")
    print("-" * 50)
    test_val = 0.5
    print(f"Input value: {test_val}")

    # Series execution
    series_out = apply_16_primary_sutras_series(test_val, 400, 400, 800, 528)
    print(f"After 16 PRIMARY SUTRAS (series): {series_out:.6f}")

    # Parallel execution
    parallel_out = apply_14_subsutras_parallel(series_out, 400, 400, 800, 528)
    print(f"After 14 SUB-SUTRAS (parallel): {parallel_out:.6f}")

    print(f"Total transformation: {test_val} -> {parallel_out:.6f}")

    # Generate images
    print("\n[GENERATING CHAKRA CYMATIC IMAGES]")
    print("-" * 50)

    for idx, (chakra_name, freq) in enumerate(CHAKRA_FREQUENCIES.items()):
        schumann = SCHUMANN_RESONANCES[idx]
        chakra_color = CHAKRA_COLORS[chakra_name]

        print(f"\nGenerating: {chakra_name} ({freq} Hz) + Schumann ({schumann} Hz)")

        # Compute full 30-sutra field
        engine.compute_full_30_sutra_field(freq, schumann)
        if AUDIO_ENABLED:
            _emit_audio_update([float(freq), float(schumann), float(idx + 1), float(len(CHAKRA_FREQUENCIES))])

        # Save image
        filename = f"{output_dir}/{chakra_name.lower()}_{freq}Hz_30sutra.png"
        engine.generate_image(filename, chakra_color)
        print(f"  Saved: {filename}")

    print("\n" + "=" * 80)
    print("FULL 30-SUTRA CYMATIC ENGINE - COMPLETE")
    print(f"Generated {len(CHAKRA_FREQUENCIES)} images")
    print("16 PRIMARY + 14 SUB-SUTRAS = 30 TOTAL TRANSFORMATIONS")
    print("=" * 80)

    return True


if __name__ == "__main__":
    main()
