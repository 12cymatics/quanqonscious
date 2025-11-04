#!/usr/bin/env python3
"""
Enhanced H₂ GRVQ Molecular Dynamics with Explicit Magnetic Stress Tensor
MST-VQ Framework: Magnetic Stress-Energy replaces Gravitational coupling

This version explicitly models:
- Electromagnetic field components (E, H)
- Maxwell stress tensor T^μν
- Magnetic energy density coupling to molecular potential
- GRVQ framework where G → magnetic coupling
"""

import math
import numpy as np
import time
import sys
import hashlib
from numba import njit
import cirq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.fft import fft, fftfreq

print("="*70)
print("H₂ GRVQ Molecular Dynamics Simulation")
print("WITH EXPLICIT MAGNETIC STRESS TENSOR")
print("MST-VQ Framework: Magnetic Stress → Molecular Coupling")
print("="*70)

# Physical constants
c0 = 299792458.0                       # Speed of light (m/s)
mu0 = 4 * math.pi * 1e-7               # Vacuum permeability (N/A²)
epsilon0 = 1.0 / (c0**2 * mu0)         # Vacuum permittivity (F/m)

# GRVQ framework parameters
alpha_const = 1.0
G_equiv = alpha_const * mu0 * 1e36     # Magnetic coupling (REPLACES G!)
kappa = 8 * math.pi * G_equiv / (c0**4)

# Grid settings
NX, NY, NZ = 32, 32, 32                # 32³ = 32,768 points
DX = DY = DZ = 0.01                    # Grid spacing (m)
TIME_STEPS = 29                         # One per Vedic sutra
DT = DX / (2.0 * c0)                   # Courant condition
r_assumed_eq = 1.0

print(f"\n{'─'*70}")
print("ELECTROMAGNETIC FIELD INITIALIZATION")
print(f"{'─'*70}")

# Initialize electromagnetic field arrays
E_x = np.zeros((NX, NY, NZ), dtype=np.float64)
E_y = np.zeros((NX, NY, NZ), dtype=np.float64)
E_z = np.zeros((NX, NY, NZ), dtype=np.float64)
H_x = np.zeros((NX, NY, NZ), dtype=np.float64)
H_y = np.zeros((NX, NY, NZ), dtype=np.float64)
H_z = np.zeros((NX, NY, NZ), dtype=np.float64)

# CRITICAL: Seed magnetic fields at HIGH amplitude (~1.0)
# Electric fields remain LOW (~0.01)
# This establishes magnetic stress dominance!
np.random.seed(42)
E_x[:] = 1e-2 * np.random.randn(NX, NY, NZ)
E_y[:] = 1e-2 * np.random.randn(NX, NY, NZ)
E_z[:] = 1e-2 * np.random.randn(NX, NY, NZ)
H_x[:] = 1.0 * np.random.randn(NX, NY, NZ)   # Magnetic ~1.0
H_y[:] = 1.0 * np.random.randn(NX, NY, NZ)
H_z[:] = 1.0 * np.random.randn(NX, NY, NZ)

print(f"E-field magnitude: ~{np.sqrt(np.mean(E_x**2 + E_y**2 + E_z**2)):.3e} V/m")
print(f"H-field magnitude: ~{np.sqrt(np.mean(H_x**2 + H_y**2 + H_z**2)):.3e} A/m")
print(f"H/E ratio: ~{np.sqrt(np.mean(H_x**2 + H_y**2 + H_z**2)) / np.sqrt(np.mean(E_x**2 + E_y**2 + E_z**2)):.1f}x")
print(f"{'─'*70}\n")

# Metric tensor (4x4 at grid center)
# Start with Minkowski: diag(-1, +1, +1, +1)
metric = np.eye(4, dtype=np.float64)
metric[0, 0] = -1.0

def compute_magnetic_energy_density():
    """
    Compute magnetic field energy density: u_B = (1/2μ₀) B·B
    In our framework, B ≈ μ₀·H
    """
    B_squared = mu0**2 * (H_x**2 + H_y**2 + H_z**2)
    u_mag = 0.5 * B_squared / mu0
    return np.mean(u_mag)

def compute_electric_energy_density():
    """
    Compute electric field energy density: u_E = (ε₀/2) E·E
    """
    E_squared = E_x**2 + E_y**2 + E_z**2
    u_elec = 0.5 * epsilon0 * E_squared
    return np.mean(u_elec)

def compute_maxwell_stress_tensor():
    """
    Compute Maxwell stress tensor T^ij components at grid center

    T^ij = ε₀(E^i E^j - δ^ij E²/2) + (1/μ₀)(B^i B^j - δ^ij B²/2)

    Returns diagonal trace: T^xx + T^yy + T^zz
    """
    cx, cy, cz = NX//2, NY//2, NZ//2

    Ex, Ey, Ez = E_x[cx, cy, cz], E_y[cx, cy, cz], E_z[cx, cy, cz]
    Hx, Hy, Hz = H_x[cx, cy, cz], H_y[cx, cy, cz], H_z[cx, cy, cz]

    # B = μ₀ H
    Bx, By, Bz = mu0 * Hx, mu0 * Hy, mu0 * Hz

    E2 = Ex**2 + Ey**2 + Ez**2
    B2 = Bx**2 + By**2 + Bz**2

    # Diagonal components
    T_xx = epsilon0 * (Ex**2 - E2/2) + (1/mu0) * (Bx**2 - B2/2)
    T_yy = epsilon0 * (Ey**2 - E2/2) + (1/mu0) * (By**2 - B2/2)
    T_zz = epsilon0 * (Ez**2 - E2/2) + (1/mu0) * (Bz**2 - B2/2)

    return T_xx + T_yy + T_zz, (T_xx, T_yy, T_zz)

@njit
def grvq_redistribution(r):
    """Singularity redistribution for r < threshold"""
    threshold = 0.1
    if r < threshold:
        return 1e-3 * math.exp(-r / threshold)
    return 0.0

@njit
def potential_energy(r):
    """
    H₂ potential with GRVQ framework

    KEY: V_attractive uses G_equiv (magnetic coupling) instead of G_Newton!
    """
    r = max(r, 1e-10)

    # Repulsive (proton-proton)
    A_local = G_equiv * math.exp(1.0) / 1.0
    V_repulsive = A_local * math.exp(-1.0 * r)

    # Attractive: MAGNETIC STRESS COUPLING (not gravitational!)
    V_attractive = -G_equiv / r

    # 29 Vedic Sutras
    V_sutra = 0.0
    for i in range(1, 30):
        coeff = G_equiv * (i / 29.0)
        phase = i * (math.pi / 4.0)
        V_sutra += coeff * math.sin((i+1) * math.pi * r / r_assumed_eq + phase) * math.exp(-r / (i+1))

    # ZPE recursive
    V_recursive = 0.0
    for d in range(5, 0, -1):
        V_recursive += math.sin(r) * math.exp(-r / d)

    # GRVQ correction
    V_GRVQ = grvq_redistribution(r)

    return V_repulsive + V_attractive + V_sutra + V_recursive + V_GRVQ

@njit
def effective_potential(r, scale_factor, zpe_offset, mag_coupling=0.0):
    """
    Effective potential with quantum corrections AND magnetic field coupling

    V_eff = scale * V_total + zpe_offset + mag_coupling * u_mag

    The mag_coupling term is NEW - it directly couples magnetic energy density
    to the molecular potential!
    """
    return scale_factor * potential_energy(r) + zpe_offset + mag_coupling

@njit
def effective_potential_derivative(r, scale_factor, zpe_offset, mag_coupling=0.0, h=1e-6):
    """Numerical derivative"""
    return (effective_potential(r + h, scale_factor, zpe_offset, mag_coupling) -
            effective_potential(r - h, scale_factor, zpe_offset, mag_coupling)) / (2*h)

def quantum_refine_cirq(step):
    """8-qubit Cirq quantum feedback"""
    qubits = [cirq.GridQubit(i, 0) for i in range(8)]
    circuit = cirq.Circuit()

    for q in qubits:
        circuit.append(cirq.H(q))

    for i in range(len(qubits) - 1):
        circuit.append(cirq.CZ(qubits[i], qubits[i+1])**0.5)

    angle = min(math.pi, 0.01 * (step + 1))
    for q in qubits:
        circuit.append(cirq.rz(angle).on(q))

    circuit.append(cirq.measure(*qubits, key='m'))

    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=10)
    bits = result.measurements['m'][0]

    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    max_val = (1 << 8) - 1

    feedback_factor = 1.0 + 1e-2 * (val / max_val) * (step + 1)
    zpe_offset_update = 1e-4 * (val / max_val)

    return feedback_factor, zpe_offset_update

def simulate_dynamics_with_magnetic_coupling(r0, v0, scale_factor, zpe_offset):
    """
    Enhanced Verlet integration with EXPLICIT magnetic stress tensor coupling

    At each step:
    1. Compute magnetic energy density u_mag
    2. Compute Maxwell stress tensor T^μν
    3. Couple magnetic stress to molecular potential
    4. Update bond dynamics
    5. Apply quantum feedback
    """
    print(f"\n{'─'*70}")
    print("MOLECULAR DYNAMICS WITH MAGNETIC STRESS TENSOR")
    print(f"{'─'*70}")
    print(f"Initial bond length: {r0:.6f}")
    print(f"Initial velocity: {v0:.6f}\n")

    t_series = np.zeros(TIME_STEPS)
    r_series = np.zeros(TIME_STEPS)
    energy_series = np.zeros(TIME_STEPS)
    mag_energy_series = np.zeros(TIME_STEPS)
    stress_tensor_series = np.zeros(TIME_STEPS)

    r_prev = r0 - v0 * DT
    r_current = r0

    # Initial magnetic coupling
    u_mag = compute_magnetic_energy_density()
    mag_coupling = kappa * u_mag * 1e-10  # Scale for numerical stability

    t_series[0] = 0.0
    r_series[0] = r_current
    energy_series[0] = effective_potential(r_current, scale_factor, zpe_offset, mag_coupling)
    mag_energy_series[0] = u_mag
    stress_trace, _ = compute_maxwell_stress_tensor()
    stress_tensor_series[0] = stress_trace

    print(f"Step   0: t={0:.6e} s")
    print(f"          r={r_current:.6e}, E={energy_series[0]:.6e}")
    print(f"          u_mag={u_mag:.6e} J/m³, T_trace={stress_trace:.6e} N/m²")

    for i in range(1, TIME_STEPS):
        t = i * DT

        # Update magnetic fields (simple evolution for demonstration)
        # In full simulation, these would evolve via Maxwell equations
        H_x[:] *= (1.0 + 1e-4 * np.random.randn(*H_x.shape))
        H_y[:] *= (1.0 + 1e-4 * np.random.randn(*H_y.shape))
        H_z[:] *= (1.0 + 1e-4 * np.random.randn(*H_z.shape))

        # Compute current magnetic energy
        u_mag = compute_magnetic_energy_density()
        mag_coupling = kappa * u_mag * 1e-10

        # Compute Maxwell stress tensor
        stress_trace, stress_components = compute_maxwell_stress_tensor()

        # Acceleration with magnetic coupling
        a = -effective_potential_derivative(r_current, scale_factor, zpe_offset, mag_coupling)

        # Verlet step
        r_next = 2.0 * r_current - r_prev + DT**2 * a

        t_series[i] = t
        r_series[i] = r_next
        energy_series[i] = effective_potential(r_next, scale_factor, zpe_offset, mag_coupling)
        mag_energy_series[i] = u_mag
        stress_tensor_series[i] = stress_trace

        # Quantum feedback
        q_factor, dq_offset = quantum_refine_cirq(i)
        scale_factor *= q_factor
        zpe_offset += dq_offset

        print(f"Step {i:3d}: t={t:.6e} s")
        print(f"          r={r_next:.6e}, E={energy_series[i]:.6e}")
        print(f"          u_mag={u_mag:.6e} J/m³, T_trace={stress_trace:.6e} N/m²")
        print(f"          mag_coupling={mag_coupling:.6e}, qfactor={q_factor:.4f}")

        r_prev = r_current
        r_current = r_next

    print(f"\n{'─'*70}")
    print("SIMULATION COMPLETE")
    print(f"{'─'*70}")
    print(f"Final bond length: {r_series[-1]:.6e}")
    print(f"Final energy: {energy_series[-1]:.6e}")
    print(f"Final mag energy: {mag_energy_series[-1]:.6e}")
    print(f"Final stress trace: {stress_tensor_series[-1]:.6e}")
    print(f"{'─'*70}\n")

    return (t_series, r_series, energy_series, mag_energy_series,
            stress_tensor_series, scale_factor, zpe_offset)

def create_enhanced_dashboard(t_series, r_series, E_series, mag_series, stress_series):
    """
    Enhanced 6-panel dashboard with magnetic stress tensor visualization
    """
    print("Creating enhanced dashboard with magnetic stress tensor...")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Bond Length vs Time", "Total Energy vs Time",
                       "Magnetic Energy Density", "Maxwell Stress Tensor Trace",
                       "Fourier Spectrum", "3D Molecular View"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter3d"}]]
    )

    # 1. Bond length
    fig.add_trace(
        go.Scatter(x=t_series, y=r_series, mode="lines+markers",
                  line=dict(color="cyan"), name="r(t)"),
        row=1, col=1
    )

    # 2. Total energy
    fig.add_trace(
        go.Scatter(x=t_series, y=E_series, mode="lines+markers",
                  line=dict(color="magenta"), name="E_total"),
        row=1, col=2
    )

    # 3. Magnetic energy density (NEW!)
    fig.add_trace(
        go.Scatter(x=t_series, y=mag_series, mode="lines+markers",
                  line=dict(color="red"), name="u_mag"),
        row=2, col=1
    )

    # 4. Maxwell stress tensor trace (NEW!)
    fig.add_trace(
        go.Scatter(x=t_series, y=stress_series, mode="lines+markers",
                  line=dict(color="orange"), name="T_trace"),
        row=2, col=2
    )

    # 5. Fourier spectrum
    r_fft = fft(r_series)
    freqs = fftfreq(len(r_series), DT)
    pos = freqs > 0

    fig.add_trace(
        go.Scatter(x=freqs[pos], y=np.abs(r_fft[pos]), mode="lines",
                  line=dict(color="lime"), name="FFT"),
        row=3, col=1
    )

    # 6. 3D molecular view
    init_r = r_series[0]
    pos1 = (-init_r/2, 0, 0)
    pos2 = (init_r/2, 0, 0)

    fig.add_trace(
        go.Scatter3d(x=[pos1[0], pos2[0]], y=[pos1[1], pos2[1]], z=[pos1[2], pos2[2]],
                    mode="markers+lines",
                    marker=dict(size=10, color=["cyan", "yellow"]),
                    line=dict(color="white", width=4),
                    name="H₂"),
        row=3, col=2
    )

    fig.update_layout(
        title="H₂ MST-VQ GRVQ Simulation (WITH MAGNETIC STRESS TENSOR)",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        height=1200,
        showlegend=True
    )

    fig.update_xaxes(title_text="Time (s)", gridcolor="gray")
    fig.update_yaxes(gridcolor="gray")

    return fig

# ==============================================================================
# MAIN SIMULATION WITH MAGNETIC STRESS TENSOR
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()

    # Initial conditions
    r0 = 1.2
    v0 = 0.0
    initial_scale = 1.0
    initial_zpe = 0.0

    # Initial magnetic diagnostics
    print(f"\n{'='*70}")
    print("INITIAL MAGNETIC FIELD DIAGNOSTICS")
    print(f"{'='*70}")
    u_mag_init = compute_magnetic_energy_density()
    u_elec_init = compute_electric_energy_density()
    stress_trace_init, (Txx, Tyy, Tzz) = compute_maxwell_stress_tensor()

    print(f"Magnetic energy density: {u_mag_init:.6e} J/m³")
    print(f"Electric energy density: {u_elec_init:.6e} J/m³")
    print(f"Magnetic dominance: {u_mag_init/u_elec_init:.2f}x")
    print(f"Maxwell stress T^xx: {Txx:.6e} N/m²")
    print(f"Maxwell stress T^yy: {Tyy:.6e} N/m²")
    print(f"Maxwell stress T^zz: {Tzz:.6e} N/m²")
    print(f"Stress trace: {stress_trace_init:.6e} N/m²")
    print(f"{'='*70}")

    # Run molecular dynamics WITH magnetic coupling
    (t_series, r_series, E_series, mag_series, stress_series,
     final_scale, final_zpe) = simulate_dynamics_with_magnetic_coupling(
        r0, v0, initial_scale, initial_zpe
    )

    # Enhanced dashboard
    dashboard = create_enhanced_dashboard(t_series, r_series, E_series,
                                         mag_series, stress_series)

    output_file = "H2_GRVQ_MagneticStress_Dashboard.html"
    dashboard.write_html(output_file)
    print(f"✓ Enhanced dashboard saved to: {output_file}")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Total simulation time: {elapsed:.2f} seconds")
    print(f"{'='*70}\n")
