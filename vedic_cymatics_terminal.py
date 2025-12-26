#!/usr/bin/env python3
"""
VEDIC CYMATICS - Terminal Real-Time Visualization
Shows geometric patterns of Vedic mathematics with cymatic wave forms
"""

import numpy as np
import time
import sys

def generate_cymatic_frame(t, width=70, height=20):
    """Generate ASCII cymatic standing wave pattern"""
    # Characters from light to dense
    chars = ' ·:;+=xX#@'
    frame = []

    for y in range(height):
        row = ''
        for x in range(width):
            # Normalize to centered coordinates
            nx = (x - width/2) / (width/4)
            ny = (y - height/2) / (height/4) * 2  # Aspect correction

            r = np.sqrt(nx**2 + ny**2)

            # Cymatic wave: Bessel-like pattern
            wave1 = np.sin(3 * np.pi * r - t * 2)
            wave2 = np.cos(2 * np.pi * r - t * 1.5)
            wave = (wave1 + wave2) / 2
            wave *= np.exp(-r**2 / 4)  # Gaussian envelope

            # Map to character
            idx = int((wave + 1) / 2 * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            row += chars[idx]
        frame.append(row)

    return '\n'.join(frame)


def generate_urdhva_visual(t, a_str="123", b_str="456"):
    """Visualize crosswise multiplication step by step"""
    a = [int(d) for d in a_str]
    b = [int(d) for d in b_str]
    n = len(a)

    step = int(t * 2) % (2 * n - 1)

    lines = []
    lines.append(f"    {'  '.join(a_str)}      Ūrdhva-Tiryagbhyām")
    lines.append(f"  × {'  '.join(b_str)}      (Vertical-Crosswise)")
    lines.append("  " + "─" * 20)

    # Show crosswise connections for current step
    cross_products = []
    visual_line = list("     " + " " * 15)

    for i in range(n):
        for j in range(n):
            if i + j == step:
                cross_products.append(f"{a[i]}×{b[j]}={a[i]*b[j]}")

    total = sum(a[i] * b[j] for i in range(n) for j in range(n) if i + j == step)

    lines.append(f"  Step {step+1}: {' + '.join(cross_products)} = {total}")

    # Animation indicator
    arrow_pos = step
    arrow_line = "  " + " " * arrow_pos * 3 + "↓↑"
    lines.append(arrow_line)

    # Final product
    product = int(a_str) * int(b_str)
    lines.append(f"\n  Product: {a_str} × {b_str} = {product}")

    return '\n'.join(lines)


def generate_golden_spiral_ascii(t, size=25):
    """ASCII representation of golden spiral"""
    chars = ' ·∘○◯'
    phi = (1 + np.sqrt(5)) / 2

    canvas = [[' ' for _ in range(size*2)] for _ in range(size)]
    center_x, center_y = size, size // 2

    # Draw spiral points
    for i in range(200):
        theta = i * 0.1 + t
        r = phi ** (theta / (np.pi/2)) * 0.3

        x = int(center_x + r * np.cos(theta))
        y = int(center_y + r * np.sin(theta) * 0.5)

        if 0 <= x < size*2 and 0 <= y < size:
            intensity = min(4, int(i / 50))
            canvas[y][x] = chars[intensity]

    # Add center marker
    canvas[center_y][center_x] = '◉'

    return '\n'.join([''.join(row) for row in canvas])


def generate_nikhilam_visual(a, b, base, t):
    """Visualize Nikhilam multiplication geometry"""
    def_a = base - a
    def_b = base - b
    product = a * b

    # Animated highlight
    highlight = ">>>" if int(t * 4) % 2 == 0 else "   "

    lines = [
        f"  Nikhilam: {a} × {b} (base {base})",
        f"  ══════════════════════════════════",
        f"",
        f"  {highlight} Deficiencies: {def_a}, {def_b}",
        f"",
        f"  ┌─────────────────────────────┐",
        f"  │  {a:3} × {b:3}                    │",
        f"  │  = ({a} - {def_b}) × {base}       │",
        f"  │    + {def_a} × {def_b}              │",
        f"  │  = {a - def_b} × {base} + {def_a * def_b}       │",
        f"  │  = {(a - def_b) * base} + {def_a * def_b}            │",
        f"  │  = {product}                    │",
        f"  └─────────────────────────────┘",
    ]
    return '\n'.join(lines)


def generate_mandala(t, size=21):
    """Generate rotating sacred geometry mandala"""
    canvas = [[' ' for _ in range(size*2)] for _ in range(size)]
    center_x, center_y = size, size // 2

    # Concentric rotating polygons
    for layer in range(1, 6):
        n_sides = 3 + layer
        radius = layer * 2
        rotation = t * (0.5 if layer % 2 == 0 else -0.3)

        for i in range(n_sides):
            angle = 2 * np.pi * i / n_sides + rotation
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle) * 0.5)

            if 0 <= x < size*2 and 0 <= y < size:
                symbols = '·∙○◐●◆★'
                canvas[y][x] = symbols[layer % len(symbols)]

            # Connect to next vertex
            next_angle = 2 * np.pi * ((i + 1) % n_sides) / n_sides + rotation
            x2 = int(center_x + radius * np.cos(next_angle))
            y2 = int(center_y + radius * np.sin(next_angle) * 0.5)

            # Draw line between vertices
            steps = max(abs(x2-x), abs(y2-y), 1)
            for s in range(steps):
                lx = int(x + (x2 - x) * s / steps)
                ly = int(y + (y2 - y) * s / steps)
                if 0 <= lx < size*2 and 0 <= ly < size:
                    if canvas[ly][lx] == ' ':
                        canvas[ly][lx] = '·'

    # Pulsing center
    pulse_char = '◉' if int(t * 5) % 2 == 0 else '◎'
    canvas[center_y][center_x] = pulse_char

    return '\n'.join([''.join(row) for row in canvas])


def run_visualization(duration=10):
    """Run the visualization for specified duration"""
    start_time = time.time()

    frame_num = 0
    while time.time() - start_time < duration:
        t = time.time() - start_time

        # Clear screen
        print('\033[2J\033[H', end='')

        # Header
        print('\033[1;36m' + '═' * 75)
        print('       ॐ  VEDIC CYMATICS - GEOMETRIC VISUALIZATION IN MOTION  ॐ')
        print('═' * 75 + '\033[0m')

        # Cymatic pattern
        print('\n\033[1;33m▼ CYMATIC STANDING WAVE PATTERN\033[0m')
        print('\033[0;37m' + generate_cymatic_frame(t, 70, 12) + '\033[0m')

        # Two-column layout for smaller visualizations
        print('\n\033[1;35m▼ VEDIC MANDALA            \033[1;32m▼ MULTIPLICATION (Ūrdhva)\033[0m')

        mandala_lines = generate_mandala(t, 15).split('\n')
        urdhva_lines = generate_urdhva_visual(t, "123", "456").split('\n')

        # Pad to same length
        max_lines = max(len(mandala_lines), len(urdhva_lines))
        while len(mandala_lines) < max_lines:
            mandala_lines.append(' ' * 30)
        while len(urdhva_lines) < max_lines:
            urdhva_lines.append('')

        for m, u in zip(mandala_lines, urdhva_lines):
            print(f'\033[0;35m{m:<32}\033[0;32m{u}\033[0m')

        # Nikhilam at bottom
        print('\n\033[1;34m▼ NIKHILAM GEOMETRY\033[0m')
        print('\033[0;34m' + generate_nikhilam_visual(98, 97, 100, t) + '\033[0m')

        # Info bar
        phi = (1 + np.sqrt(5)) / 2
        print(f'\n\033[1;37m┌────────────────────────────────────────────────────────────────────────┐')
        print(f'│  φ = {phi:.10f}  │  Frame: {frame_num:4d}  │  Time: {t:.1f}s  │  c² = 89875517873681764  │')
        print(f'└────────────────────────────────────────────────────────────────────────┘\033[0m')

        time.sleep(0.1)
        frame_num += 1

    print('\n\033[1;36mVisualization complete!\033[0m\n')


if __name__ == "__main__":
    duration = 8 if len(sys.argv) < 2 else int(sys.argv[1])
    print('\033[?25l', end='')  # Hide cursor
    try:
        run_visualization(duration)
    finally:
        print('\033[?25h', end='')  # Show cursor
