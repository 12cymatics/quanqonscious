#!/usr/bin/env python3
"""
VEDIC CYMATICS VISUALIZATION
Real-time geometric visualization of Vedic mathematics with cymatic wave patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, RegularPolygon, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import time

# Set up dark theme
plt.style.use('dark_background')

class VedicCymaticsVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('VEDIC CYMATICS - Mathematical Geometry in Motion',
                         fontsize=16, color='cyan', fontweight='bold')

    def cymatic_pattern(self, ax, frequency=3, time_val=0):
        """Generate cymatic standing wave pattern"""
        x = np.linspace(-2, 2, 200)
        y = np.linspace(-2, 2, 200)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        # Bessel-like cymatic pattern with animation
        Z = np.sin(frequency * np.pi * R - time_val) * np.cos(frequency * np.pi * R / 2)
        Z *= np.exp(-R**2 / 4)  # Gaussian envelope

        ax.clear()
        ax.set_facecolor('black')

        # Contour plot for cymatic pattern
        levels = np.linspace(-1, 1, 20)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, levels=levels, colors='cyan', linewidths=0.3, alpha=0.5)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title(f'Cymatic Wave Pattern (f={frequency})', color='white')
        ax.axis('off')

        return Z

    def urdhva_crosswise(self, ax, a_digits, b_digits, step, time_val):
        """Visualize Ūrdhva-Tiryagbhyām crosswise multiplication"""
        ax.clear()
        ax.set_facecolor('black')

        n = len(a_digits)

        # Draw digit circles for number A (top)
        for i, d in enumerate(a_digits):
            x = i - n/2 + 0.5
            circle = Circle((x, 1), 0.3, fill=True, color='cyan', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, 1, str(d), ha='center', va='center', fontsize=14, color='black', fontweight='bold')

        # Draw digit circles for number B (bottom)
        for i, d in enumerate(b_digits):
            x = i - n/2 + 0.5
            circle = Circle((x, -1), 0.3, fill=True, color='magenta', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, -1, str(d), ha='center', va='center', fontsize=14, color='black', fontweight='bold')

        # Animate crosswise connections based on step
        current_step = int(step) % (2 * n - 1)
        colors = plt.cm.rainbow(np.linspace(0, 1, 2*n-1))

        # Draw all cross-products for current diagonal
        products = []
        for i in range(n):
            for j in range(n):
                if i + j == current_step:
                    x1 = i - n/2 + 0.5
                    x2 = j - n/2 + 0.5

                    # Animated line with glow effect
                    alpha = 0.5 + 0.5 * np.sin(time_val * 5)
                    ax.plot([x1, x2], [1, -1], color=colors[current_step],
                           linewidth=3, alpha=alpha)

                    # Show product
                    prod = a_digits[i] * b_digits[j]
                    products.append(prod)
                    mid_x = (x1 + x2) / 2
                    ax.text(mid_x, 0, f'{a_digits[i]}×{b_digits[j]}={prod}',
                           ha='center', va='center', fontsize=10, color='yellow',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_xlim(-3, 3)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(f'Ūrdhva-Tiryagbhyām: Step {current_step+1} (Crosswise)', color='white')
        ax.axis('off')

        # Show running total
        if products:
            ax.text(0, -2, f'Cross-products: {products} = {sum(products)}',
                   ha='center', color='lime', fontsize=11)

    def golden_spiral(self, ax, time_val):
        """Animate golden ratio spiral with Fibonacci"""
        ax.clear()
        ax.set_facecolor('black')

        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # Golden spiral
        theta = np.linspace(0, 4 * np.pi + time_val, 500)
        r = phi ** (theta / (np.pi/2))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Color gradient along spiral
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, len(segments))
        lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2, alpha=0.8)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

        # Draw Fibonacci squares
        fib = [1, 1, 2, 3, 5, 8, 13]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(fib)))

        x_pos, y_pos = 0, 0
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        for i, f in enumerate(fib[:5]):
            dx, dy = directions[i % 4]
            rect = plt.Rectangle((x_pos, y_pos), f * dx if dx else f, f * dy if dy else f,
                                 fill=False, edgecolor=colors[i], linewidth=2, alpha=0.6)
            # ax.add_patch(rect)
            x_pos += f * dx
            y_pos += f * dy

        # Pulsing center
        pulse = 0.5 + 0.3 * np.sin(time_val * 3)
        center = Circle((0, 0), pulse, fill=True, color='gold', alpha=0.5)
        ax.add_patch(center)

        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.set_title(f'Golden Ratio φ = {phi:.10f}', color='gold')
        ax.axis('off')

    def nikhilam_geometry(self, ax, a, b, base, time_val):
        """Visualize Nikhilam multiplication as area geometry"""
        ax.clear()
        ax.set_facecolor('black')

        def_a = base - a
        def_b = base - b

        # Main square (base × base)
        main_square = plt.Rectangle((0, 0), base, base, fill=False,
                                    edgecolor='white', linewidth=2, linestyle='--', alpha=0.5)
        ax.add_patch(main_square)

        # Product rectangle (a × b) - what we want
        pulse = 0.7 + 0.3 * np.sin(time_val * 2)
        product_rect = plt.Rectangle((0, 0), a, b, fill=True,
                                     facecolor='cyan', alpha=pulse * 0.5, edgecolor='cyan', linewidth=2)
        ax.add_patch(product_rect)

        # Deficiency rectangles
        # Top strip: a × def_b
        top_strip = plt.Rectangle((0, b), a, def_b, fill=True,
                                  facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
        ax.add_patch(top_strip)

        # Right strip: def_a × b
        right_strip = plt.Rectangle((a, 0), def_a, b, fill=True,
                                    facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
        ax.add_patch(right_strip)

        # Corner square: def_a × def_b (added back)
        corner = plt.Rectangle((a, b), def_a, def_b, fill=True,
                               facecolor='green', alpha=0.5, edgecolor='lime', linewidth=2)
        ax.add_patch(corner)

        # Labels
        ax.text(a/2, b/2, f'{a}×{b}\n={a*b}', ha='center', va='center',
               fontsize=14, color='white', fontweight='bold')
        ax.text(a + def_a/2, b + def_b/2, f'{def_a}×{def_b}\n={def_a*def_b}',
               ha='center', va='center', fontsize=10, color='lime')

        ax.set_xlim(-5, base + 10)
        ax.set_ylim(-5, base + 10)
        ax.set_aspect('equal')
        ax.set_title(f'Nikhilam: {a}×{b} = ({a}-{def_b})×{base} + {def_a}×{def_b} = {a*b}', color='white')
        ax.axis('off')

    def mandala_vedic(self, ax, time_val):
        """Create animated Vedic mandala with sacred geometry"""
        ax.clear()
        ax.set_facecolor('black')

        # Multiple rotating layers
        for layer in range(5):
            n_points = 6 + layer * 3
            radius = 1 + layer * 0.5
            rotation = time_val * (0.5 if layer % 2 == 0 else -0.3) + layer * np.pi / 6

            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False) + rotation
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)

            # Connect points
            color = plt.cm.rainbow(layer / 5)
            for i in range(n_points):
                for j in range(i+1, n_points):
                    alpha = 0.3 + 0.2 * np.sin(time_val * 2 + layer)
                    ax.plot([x[i], x[j]], [y[i], y[j]], color=color,
                           linewidth=0.5, alpha=alpha)

            # Draw vertices
            ax.scatter(x, y, s=20, color=color, alpha=0.8)

        # Central pulsing circle
        pulse = 0.3 + 0.2 * np.sin(time_val * 4)
        center = Circle((0, 0), pulse, fill=True, color='white', alpha=0.8)
        ax.add_patch(center)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title('Vedic Mandala - Sacred Geometry', color='white')
        ax.axis('off')

    def wave_interference(self, ax, time_val):
        """Show wave interference patterns (cymatics basis)"""
        ax.clear()
        ax.set_facecolor('black')

        x = np.linspace(-3, 3, 300)
        y = np.linspace(-3, 3, 300)
        X, Y = np.meshgrid(x, y)

        # Multiple wave sources
        sources = [(-1.5, 0), (1.5, 0), (0, 1.5), (0, -1.5)]
        Z = np.zeros_like(X)

        for sx, sy in sources:
            R = np.sqrt((X - sx)**2 + (Y - sy)**2)
            Z += np.sin(8 * R - time_val * 5) / (R + 0.5)

        # Plot interference pattern
        ax.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.9)
        ax.contour(X, Y, Z, levels=15, colors='white', linewidths=0.3, alpha=0.3)

        # Mark wave sources
        for sx, sy in sources:
            ax.scatter([sx], [sy], s=100, color='yellow', marker='*', zorder=5)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title('Wave Interference - Cymatics Foundation', color='white')
        ax.axis('off')

    def run_animation(self):
        """Run the full animated visualization"""
        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax1 = self.fig.add_subplot(gs[0, 0])  # Cymatic pattern
        ax2 = self.fig.add_subplot(gs[0, 1])  # Urdhva crosswise
        ax3 = self.fig.add_subplot(gs[0, 2])  # Golden spiral
        ax4 = self.fig.add_subplot(gs[1, 0])  # Nikhilam geometry
        ax5 = self.fig.add_subplot(gs[1, 1])  # Mandala
        ax6 = self.fig.add_subplot(gs[1, 2])  # Wave interference

        # Example numbers for visualization
        a_digits = [1, 2, 3]
        b_digits = [4, 5, 6]

        def animate(frame):
            t = frame * 0.05

            # Update all visualizations
            self.cymatic_pattern(ax1, frequency=3 + np.sin(t/2), time_val=t)
            self.urdhva_crosswise(ax2, a_digits, b_digits, frame/10, t)
            self.golden_spiral(ax3, t)
            self.nikhilam_geometry(ax4, 98, 97, 100, t)
            self.mandala_vedic(ax5, t)
            self.wave_interference(ax6, t)

            return []

        ani = animation.FuncAnimation(self.fig, animate, frames=200,
                                      interval=50, blit=False)
        plt.tight_layout()
        plt.show()


def run_terminal_cymatics():
    """Terminal-based real-time cymatic animation"""
    import sys
    import os

    def clear_screen():
        os.system('cls' if os.name == 'nt' else 'clear')

    def generate_cymatic_frame(t, width=80, height=30):
        """Generate ASCII cymatic pattern"""
        chars = ' ·∙●○◐◑◒◓◔◕⬤'
        frame = []

        for y in range(height):
            row = ''
            for x in range(width):
                # Normalize coordinates
                nx = (x - width/2) / (width/4)
                ny = (y - height/2) / (height/4)
                r = np.sqrt(nx**2 + ny**2)

                # Cymatic wave function
                wave = np.sin(4 * np.pi * r - t) * np.cos(2 * np.pi * r)
                wave *= np.exp(-r**2 / 3)

                # Map to character
                idx = int((wave + 1) / 2 * (len(chars) - 1))
                idx = max(0, min(len(chars) - 1, idx))
                row += chars[idx]
            frame.append(row)

        return '\n'.join(frame)

    def generate_vedic_mandala(t, size=31):
        """Generate rotating Vedic mandala"""
        chars = ' ·+×*◊◆●'
        center = size // 2
        frame = [[' ' for _ in range(size)] for _ in range(size)]

        # Multiple geometric layers
        for layer in range(1, 6):
            n_points = 6 * layer
            radius = layer * 2.5
            rotation = t * (0.3 if layer % 2 == 0 else -0.2)

            for i in range(n_points):
                angle = 2 * np.pi * i / n_points + rotation
                x = int(center + radius * np.cos(angle))
                y = int(center + radius * np.sin(angle) * 0.5)  # Aspect ratio

                if 0 <= x < size and 0 <= y < size:
                    char_idx = (layer + int(t * 2)) % len(chars)
                    frame[y][x] = chars[char_idx]

        # Center pulse
        pulse_size = 1 + int(np.sin(t * 3) * 0.5 + 0.5)
        for dy in range(-pulse_size, pulse_size + 1):
            for dx in range(-pulse_size * 2, pulse_size * 2 + 1):
                x, y = center + dx, center + dy
                if 0 <= x < size and 0 <= y < size:
                    frame[y][x] = '◉'

        return '\n'.join([''.join(row) for row in frame])

    print("\033[?25l")  # Hide cursor

    try:
        t = 0
        while True:
            clear_screen()

            print("\033[1;36m" + "═" * 80)
            print("           VEDIC CYMATICS - REAL-TIME GEOMETRIC VISUALIZATION")
            print("═" * 80 + "\033[0m\n")

            # Cymatic pattern
            print("\033[1;33m▼ CYMATIC WAVE PATTERN (Sound Geometry)\033[0m")
            print(generate_cymatic_frame(t, 60, 15))

            print("\n\033[1;35m▼ VEDIC MANDALA (Sacred Geometry)\033[0m")
            print(generate_vedic_mandala(t, 25))

            # Info panel
            phi = (1 + np.sqrt(5)) / 2
            fib_n = int(10 + 5 * np.sin(t))

            print(f"\n\033[1;32m╔══════════════════════════════════════════════════════════╗")
            print(f"║  Golden Ratio φ = {phi:.10f}                    ║")
            print(f"║  Wave Frequency: {3 + np.sin(t/2):.2f} Hz                            ║")
            print(f"║  Rotation Phase: {(t % (2*np.pi)):.2f} rad                          ║")
            print(f"╚══════════════════════════════════════════════════════════╝\033[0m")

            print("\n\033[1;31mPress Ctrl+C to exit\033[0m")

            time.sleep(0.1)
            t += 0.15

    except KeyboardInterrupt:
        print("\033[?25h")  # Show cursor
        print("\n\033[1;36mVisualization ended.\033[0m")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--terminal':
        run_terminal_cymatics()
    else:
        print("Starting Vedic Cymatics Visualization...")
        print("Close the window or press Ctrl+C to exit.\n")

        viz = VedicCymaticsVisualizer()
        viz.run_animation()
