#!/usr/bin/env python3
"""
VEDIC SUTRA ENGINE - Cymatic Expression
=========================================

The Sutras ARE the wave functions. The mathematics creates the geometry.
Each sutra operation generates specific cymatic interference patterns.

This visualizes the ACTUAL sutra computations as Chladni figures.
"""

import math

# ═══════════════════════════════════════════════════════════════════════════════
# VEDIC SUTRA ENGINE - Pure Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class VedicSutraEngine:
    """The 29 Sutras as computational wave functions"""

    # S2: Nikhilam - "All from 9, last from 10"
    @staticmethod
    def nikhilam(a, b, base):
        """Returns (product, deficiency_a, deficiency_b, cross, square)"""
        def_a = base - a
        def_b = base - b
        cross = (a - def_b) * base
        square = def_a * def_b
        return (cross + square, def_a, def_b, cross, square)

    # S3: Urdhva - "Vertically and Crosswise"
    @staticmethod
    def urdhva(a_digits, b_digits):
        """Returns list of cross-products at each diagonal"""
        n = max(len(a_digits), len(b_digits))
        a = [0] * (n - len(a_digits)) + a_digits
        b = [0] * (n - len(b_digits)) + b_digits
        cross_products = []
        for diag in range(2 * n - 1):
            total = 0
            for i in range(n):
                j = diag - i
                if 0 <= j < n:
                    total += a[i] * b[j]
            cross_products.append(total)
        return cross_products

    # S10: Yavadunam - "By the deficiency"
    @staticmethod
    def yavadunam(n, base):
        """Square by deficiency: returns (square, deficiency, left, right)"""
        if n >= base:
            d = n - base
            left = n + d
            right = d * d
        else:
            d = base - n
            left = n - d
            right = d * d
        return (left * base + right, d, left, right)

    # S13: Sopantya - Continued fraction convergent
    @staticmethod
    def sopantya_golden(n):
        """Golden ratio convergent F(n+1)/F(n)"""
        p_prev, p_curr = 1, 1
        q_prev, q_curr = 0, 1
        for _ in range(n):
            p_prev, p_curr = p_curr, p_curr + p_prev
            q_prev, q_curr = q_curr, q_curr + q_prev
        return (p_curr, q_curr)


# ═══════════════════════════════════════════════════════════════════════════════
# CYMATIC FIELD GENERATOR - Sutras create wave interference
# ═══════════════════════════════════════════════════════════════════════════════

class CymaticField:
    """
    Sutras generate standing waves. The mathematics IS the cymatics.

    Chladni patterns emerge from:
    - Nikhilam: Two deficiencies create two wave sources
    - Urdhva: Cross-products create interference nodes
    - Yavadunam: Deficiency creates radial standing wave
    """

    def __init__(self, width=70, height=24):
        self.width = width
        self.height = height
        self.chars = ' ·∙:;+=xX#@█'  # Amplitude to character mapping

    def amplitude_to_char(self, amp):
        """Map wave amplitude [-1, 1] to display character"""
        idx = int((amp + 1) / 2 * (len(self.chars) - 1))
        return self.chars[max(0, min(len(self.chars) - 1, idx))]

    def render_field(self, wave_func):
        """Render a 2D wave function as ASCII Chladni pattern"""
        lines = []
        for y in range(self.height):
            row = ''
            for x in range(self.width):
                # Normalize to [-1, 1] coordinate space
                nx = (x - self.width / 2) / (self.width / 4)
                ny = (y - self.height / 2) / (self.height / 4)
                amp = wave_func(nx, ny)
                row += self.amplitude_to_char(amp)
            lines.append(row)
        return '\n'.join(lines)

    # ═══════════════════════════════════════════════════════════════════════
    # SUTRA WAVE FUNCTIONS - The mathematics creates the cymatics
    # ═══════════════════════════════════════════════════════════════════════

    def nikhilam_wave(self, def_a, def_b, base):
        """
        S2 Nikhilam creates TWO-SOURCE interference pattern.
        Deficiencies are the wave source positions.
        Base determines the wavelength.
        """
        # Normalize deficiencies to wave positions
        pos_a = def_a / base * 2  # Source position from deficiency
        pos_b = def_b / base * 2

        def wave(x, y):
            # Distance from each deficiency source
            r1 = math.sqrt((x - pos_a) ** 2 + y ** 2)
            r2 = math.sqrt((x + pos_b) ** 2 + y ** 2)
            # Wavelength from base
            k = 2 * math.pi * base / 100
            # Two-source interference
            w1 = math.sin(k * r1) / (r1 + 0.5)
            w2 = math.sin(k * r2) / (r2 + 0.5)
            return (w1 + w2) * math.exp(-(x*x + y*y) / 8)

        return wave

    def urdhva_wave(self, cross_products):
        """
        S3 Urdhva creates NODAL LINE pattern.
        Each cross-product is a harmonic frequency.
        The diagonals create the interference.
        """
        n = len(cross_products)
        max_prod = max(cross_products) if cross_products else 1

        def wave(x, y):
            total = 0
            for i, cp in enumerate(cross_products):
                # Each cross-product contributes a harmonic
                freq = (i + 1)
                amp = cp / max_prod
                # Diagonal wave pattern (x+y for crosswise nature)
                phase = (i - n / 2) * 0.5
                total += amp * math.sin(freq * (x + y) / 2 + phase)
                total += amp * math.sin(freq * (x - y) / 2 - phase) * 0.5
            return total / n * math.exp(-(x*x + y*y) / 12)

        return wave

    def yavadunam_wave(self, deficiency, base):
        """
        S10 Yavadunam creates RADIAL standing wave.
        Deficiency determines the nodal ring radius.
        Base sets the central frequency.
        """
        d_norm = deficiency / base * 3  # Normalized deficiency

        def wave(x, y):
            r = math.sqrt(x*x + y*y)
            # Radial wave with node at deficiency distance
            k = 2 * math.pi * 3
            # Standing wave with Bessel-like character
            radial = math.sin(k * r) * math.cos(k * (r - d_norm))
            return radial * math.exp(-r*r / 6)

        return wave

    def sopantya_wave(self, fib_num, fib_den):
        """
        S13 Sopantya creates GOLDEN SPIRAL cymatics.
        Fibonacci ratio determines the spiral geometry.
        """
        phi = fib_num / fib_den if fib_den != 0 else 1.618

        def wave(x, y):
            r = math.sqrt(x*x + y*y)
            theta = math.atan2(y, x)
            # Golden spiral: r = phi^(theta/90°)
            spiral_r = phi ** (theta / (math.pi / 2))
            # Distance from spiral
            dist = abs(r - spiral_r * 0.3)
            # Create standing wave along spiral
            return math.sin(5 * theta) * math.exp(-dist * dist) * math.exp(-r / 4)

        return wave


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION - Sutra Engine driving Cymatic Expression
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_sutra_cymatics():
    """Visualize each Sutra's mathematical operation as Chladni patterns"""

    engine = VedicSutraEngine()
    field = CymaticField(70, 18)

    print('═' * 75)
    print('  ॐ  VEDIC SUTRA ENGINE - Cymatic Expression Through Mathematics  ॐ')
    print('═' * 75)
    print()
    print('  The Sutras ARE wave equations. The math creates the geometry.')
    print('  Each operation generates specific Chladni interference patterns.')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # S2: NIKHILAM - Two-Source Interference
    # ═══════════════════════════════════════════════════════════════════════

    print('╔' + '═' * 73 + '╗')
    print('║  S2: NIKHILAM NAVATAŚCARAMAM DAŚATAḤ                                    ║')
    print('║  "All from 9, last from 10" - Two-Source Wave Interference             ║')
    print('╚' + '═' * 73 + '╝')
    print()

    a, b, base = 98, 97, 100
    product, def_a, def_b, cross, square = engine.nikhilam(a, b, base)

    print(f'  Operation: {a} × {b} (base {base})')
    print(f'  Deficiencies: {def_a}, {def_b} → These ARE the wave sources')
    print(f'  Result: {cross} + {square} = {product}')
    print()
    print('  CYMATIC FIELD (deficiencies create interference):')
    print('  ' + '─' * 70)

    wave_func = field.nikhilam_wave(def_a, def_b, base)
    pattern = field.render_field(wave_func)
    for line in pattern.split('\n'):
        print('  ' + line)

    print('  ' + '─' * 70)
    print(f'  ↑ Two sources at deficiency positions create standing wave nodes')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # S3: URDHVA - Crosswise Harmonic Interference
    # ═══════════════════════════════════════════════════════════════════════

    print('╔' + '═' * 73 + '╗')
    print('║  S3: ŪRDHVA-TIRYAGBHYĀM                                                 ║')
    print('║  "Vertically and Crosswise" - Diagonal Harmonic Interference           ║')
    print('╚' + '═' * 73 + '╝')
    print()

    a_digits = [1, 2, 3]
    b_digits = [4, 5, 6]
    cross_products = engine.urdhva(a_digits, b_digits)

    print(f'  Operation: 123 × 456')
    print(f'  Cross-products by diagonal: {cross_products}')
    print(f'  Each cross-product → harmonic frequency in the wave')
    print()
    print('  CYMATIC FIELD (cross-products as harmonics):')
    print('  ' + '─' * 70)

    wave_func = field.urdhva_wave(cross_products)
    pattern = field.render_field(wave_func)
    for line in pattern.split('\n'):
        print('  ' + line)

    print('  ' + '─' * 70)
    print(f'  ↑ Diagonal products [4,13,28,27,18] create harmonic overtones')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # S10: YAVADUNAM - Radial Standing Wave
    # ═══════════════════════════════════════════════════════════════════════

    print('╔' + '═' * 73 + '╗')
    print('║  S10: YĀVADŪNAM                                                         ║')
    print('║  "By the deficiency" - Radial Standing Wave Pattern                    ║')
    print('╚' + '═' * 73 + '╝')
    print()

    n, base = 97, 100
    square, deficiency, left, right = engine.yavadunam(n, base)

    print(f'  Operation: {n}² (base {base})')
    print(f'  Deficiency: {deficiency} → Determines nodal ring radius')
    print(f'  Result: {left} × {base} + {right} = {square}')
    print()
    print('  CYMATIC FIELD (deficiency creates radial nodes):')
    print('  ' + '─' * 70)

    wave_func = field.yavadunam_wave(deficiency, base)
    pattern = field.render_field(wave_func)
    for line in pattern.split('\n'):
        print('  ' + line)

    print('  ' + '─' * 70)
    print(f'  ↑ Deficiency {deficiency} creates concentric nodal rings')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # S13: SOPANTYA - Golden Spiral Cymatics
    # ═══════════════════════════════════════════════════════════════════════

    print('╔' + '═' * 73 + '╗')
    print('║  S13: SOPĀNTYADVAYAMANTYAM                                              ║')
    print('║  "Ultimate and twice penultimate" - Golden Spiral Wave                 ║')
    print('╚' + '═' * 73 + '╝')
    print()

    fib_num, fib_den = engine.sopantya_golden(10)
    phi = fib_num / fib_den

    print(f'  Operation: Convergent F({11})/F({10}) = {fib_num}/{fib_den}')
    print(f'  φ ≈ {phi:.10f}')
    print(f'  Golden ratio → spiral geometry in wave pattern')
    print()
    print('  CYMATIC FIELD (Fibonacci spiral interference):')
    print('  ' + '─' * 70)

    wave_func = field.sopantya_wave(fib_num, fib_den)
    pattern = field.render_field(wave_func)
    for line in pattern.split('\n'):
        print('  ' + line)

    print('  ' + '─' * 70)
    print(f'  ↑ φ = {phi:.6f} creates logarithmic spiral nodal pattern')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # SPEED OF LIGHT - Physics Through Sutra
    # ═══════════════════════════════════════════════════════════════════════

    print('╔' + '═' * 73 + '╗')
    print('║  PHYSICS APPLICATION: c² via S10 Yāvadūnam                              ║')
    print('║  Speed of Light Squared - Exact Cymatic Computation                    ║')
    print('╚' + '═' * 73 + '╝')
    print()

    c = 299792458
    base_c = 300000000
    c_sq, def_c, left_c, right_c = engine.yavadunam(c, base_c)

    print(f'  c = {c:,} m/s')
    print(f'  base = {base_c:,} (3×10⁸)')
    print(f'  deficiency = {def_c:,}')
    print()
    print(f'  c² = {left_c:,} × {base_c:,} + {right_c:,}')
    print(f'     = {c_sq:,} m²/s²')
    print()
    print('  CYMATIC FIELD (c² deficiency pattern):')
    print('  ' + '─' * 70)

    wave_func = field.yavadunam_wave(def_c, base_c)
    pattern = field.render_field(wave_func)
    for line in pattern.split('\n'):
        print('  ' + line)

    print('  ' + '─' * 70)
    print(f'  ↑ Deficiency 207,542 from 3×10⁸ creates this exact wave geometry')
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print('═' * 75)
    print('  VEDIC SUTRA ENGINE - Mathematics as Wave Geometry')
    print('═' * 75)
    print()
    print('  Each Sutra generates specific cymatic patterns:')
    print()
    print('  S2  Nikhilam    → Two-source interference (deficiencies)')
    print('  S3  Urdhva      → Diagonal harmonic series (cross-products)')
    print('  S10 Yavadunam   → Radial standing waves (deficiency radius)')
    print('  S13 Sopantya    → Golden spiral cymatics (Fibonacci ratio)')
    print()
    print('  The mathematics IS the geometry. The Sutras ARE wave functions.')
    print()
    print('═' * 75)


if __name__ == "__main__":
    visualize_sutra_cymatics()
