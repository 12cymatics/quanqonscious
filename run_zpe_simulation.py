#!/usr/bin/env python
"""Simple ZPE field simulation runner"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/quanqonscious')

from zpe_solver import ZPEFieldSolver

def run_zpe_simulation(size=(15, 15, 15), steps=10):
    """Run a ZPE field simulation with a Gaussian initial condition"""

    print(f"Initializing ZPE Field Solver with grid size {size}")
    solver = ZPEFieldSolver(size, dt=0.1, use_gpu=False)

    # Set initial Gaussian pulse at center
    cx, cy, cz = (s//2 for s in size)
    print(f"Setting initial Gaussian pulse at center ({cx}, {cy}, {cz})")

    def gaussian_pulse(X, Y, Z):
        return 1.0 * np.exp(-0.1 * ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2))

    solver.set_initial_field(gaussian_pulse)

    # Get initial field value at center
    initial_field = solver.get_field()
    initial_center = initial_field[cx, cy, cz]
    print(f"Initial field value at center: {initial_center:.6f}")

    # Run simulation
    print(f"\nRunning {steps} simulation steps...")
    solver.step(steps)

    # Get final field
    final_field = solver.get_field()
    final_center = final_field[cx, cy, cz]

    print(f"\nSimulation completed!")
    print(f"Final field value at center: {final_center:.6f}")
    print(f"Field statistics:")
    print(f"  Min: {final_field.min():.6f}")
    print(f"  Max: {final_field.max():.6f}")
    print(f"  Mean: {final_field.mean():.6f}")
    print(f"  Std: {final_field.std():.6f}")

    return solver, final_field

if __name__ == "__main__":
    solver, field = run_zpe_simulation(size=(15, 15, 15), steps=10)
    print("\n✓ ZPE field simulation completed successfully!")
