"""Run the hybrid quantum-classical simulation defined in the Colab notebook.

This module is a faithful, production-ready port of the notebook labeled
"Untitled6.ipynb".  It recreates the 16 main Vedic sutra transforms applied in
series, the 13 sub-sutra transforms executed in parallel, and the hybrid
ansatz circuit that consumes the transformed parameters.  The implementation
avoids any placeholder code and follows the structure of the source notebook
exactly so that the resulting numerical behaviour matches the documented
outputs.

Running this module as a script executes ``quantum_test_with_full_sutras`` which
prints the intermediate parameter vectors, the final statevector and a textual
diagram of the constructed quantum circuit.
"""

from __future__ import annotations

import concurrent.futures
import math
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 16 Main Vedic Sutras (Series Application)
# ---------------------------------------------------------------------------
def sutra1_Ekadhikena(params: np.ndarray) -> np.ndarray:
    return np.array([p + 0.001 * math.sin(p) for p in params], dtype=float)


def sutra2_Nikhilam(params: np.ndarray) -> np.ndarray:
    return np.array([p - 0.002 * (1 - p) for p in params], dtype=float)


def sutra3_Urdhva_Tiryagbhyam(params: np.ndarray) -> np.ndarray:
    return np.array([p * (1 + 0.003 * math.cos(p)) for p in params], dtype=float)


def sutra4_Urdhva_Veerya(params: np.ndarray) -> np.ndarray:
    return np.array([p * math.exp(0.0005 * p) for p in params], dtype=float)


def sutra5_Paravartya(params: np.ndarray) -> np.ndarray:
    reversed_params = params[::-1]
    return np.array([p + 0.0008 for p in reversed_params], dtype=float)


def sutra6_Shunyam_Sampurna(params: np.ndarray) -> np.ndarray:
    return np.array([p if abs(p) > 0.1 else p + 0.1 for p in params], dtype=float)


def sutra7_Anurupyena(params: np.ndarray) -> np.ndarray:
    avg = float(np.mean(params))
    return np.array([p * (1 + 0.0003 * (p - avg)) for p in params], dtype=float)


def sutra8_Sopantyadvayamantyam(params: np.ndarray) -> np.ndarray:
    new_params = []
    for i in range(0, len(params) - 1, 2):
        avg_pair = (params[i] + params[i + 1]) / 2.0
        new_params.extend([avg_pair, avg_pair])
    if len(params) % 2 != 0:
        new_params.append(params[-1])
    return np.array(new_params, dtype=float)


def sutra9_Ekanyunena(params: np.ndarray) -> np.ndarray:
    half = params[: len(params) // 2]
    factor = float(np.mean(half)) if len(half) else 0.0
    return np.array([p + 0.0007 * factor for p in params], dtype=float)


def sutra10_Dvitiya(params: np.ndarray) -> np.ndarray:
    if len(params) >= 2:
        factor = float(np.mean(params[len(params) // 2 :]))
        return np.array([p * (1 + 0.0004 * factor) for p in params], dtype=float)
    return params.astype(float)


def sutra11_Virahata(params: np.ndarray) -> np.ndarray:
    return np.array([p + 0.0015 * math.sin(2 * p) for p in params], dtype=float)


def sutra12_Ayur(params: np.ndarray) -> np.ndarray:
    return np.array([p * (1 + 0.0006 * abs(p)) for p in params], dtype=float)


def sutra13_Samuchchhayo(params: np.ndarray) -> np.ndarray:
    total = float(np.sum(params))
    return np.array([p + 0.0002 * total for p in params], dtype=float)


def sutra14_Alankara(params: np.ndarray) -> np.ndarray:
    return np.array([p + 0.0005 * math.sin(i) for i, p in enumerate(params)], dtype=float)


def sutra15_Sandhya(params: np.ndarray) -> np.ndarray:
    new_params = [(params[i] + params[i + 1]) / 2.0 for i in range(len(params) - 1)]
    new_params.append(params[-1])
    return np.array(new_params, dtype=float)


def sutra16_Sandhya_Samuccaya(params: np.ndarray) -> np.ndarray:
    indices = np.linspace(1, len(params), len(params))
    weighted_avg = float(np.dot(params, indices) / np.sum(indices))
    return np.array([p + 0.0003 * weighted_avg for p in params], dtype=float)


MAIN_SUTRAS = (
    sutra1_Ekadhikena,
    sutra2_Nikhilam,
    sutra3_Urdhva_Tiryagbhyam,
    sutra4_Urdhva_Veerya,
    sutra5_Paravartya,
    sutra6_Shunyam_Sampurna,
    sutra7_Anurupyena,
    sutra8_Sopantyadvayamantyam,
    sutra9_Ekanyunena,
    sutra10_Dvitiya,
    sutra11_Virahata,
    sutra12_Ayur,
    sutra13_Samuchchhayo,
    sutra14_Alankara,
    sutra15_Sandhya,
    sutra16_Sandhya_Samuccaya,
)


def apply_main_sutras(params: np.ndarray) -> np.ndarray:
    for func in MAIN_SUTRAS:
        params = func(params)
    return params


# ---------------------------------------------------------------------------
# 13 Sub-Sutra Functions (Parallel Application)
# ---------------------------------------------------------------------------
def subsutra1_Refinement(params: np.ndarray) -> np.ndarray:
    return np.array([p + 0.0001 * (p**2) for p in params], dtype=float)


def subsutra2_Correction(params: np.ndarray) -> np.ndarray:
    return np.array([p - 0.0002 * (p - 0.5) for p in params], dtype=float)


def subsutra3_Recursion(params: np.ndarray) -> np.ndarray:
    shifted = np.roll(params, 1)
    return (params + shifted) / 2.0


def subsutra4_Convergence(params: np.ndarray) -> np.ndarray:
    return np.array([0.9 * p for p in params], dtype=float)


def subsutra5_Stabilization(params: np.ndarray) -> np.ndarray:
    return np.clip(params, 0.0, 1.0)


def subsutra6_Simplification(params: np.ndarray) -> np.ndarray:
    return np.array([round(float(p), 4) for p in params], dtype=float)


def subsutra7_Interpolation(params: np.ndarray) -> np.ndarray:
    return np.array([p + 0.00005 for p in params], dtype=float)


def subsutra8_Extrapolation(params: np.ndarray) -> np.ndarray:
    trend = np.polyfit(range(len(params)), params, 1)
    correction = float(np.polyval(trend, len(params)))
    return np.array([p + 0.0001 * correction for p in params], dtype=float)


def subsutra9_ErrorReduction(params: np.ndarray) -> np.ndarray:
    std = float(np.std(params))
    return np.array([p - 0.0001 * std for p in params], dtype=float)


def subsutra10_Optimization(params: np.ndarray) -> np.ndarray:
    mean_val = float(np.mean(params))
    return np.array([p + 0.0002 * (mean_val - p) for p in params], dtype=float)


def subsutra11_Adjustment(params: np.ndarray) -> np.ndarray:
    return np.array([p + 0.0003 * math.cos(p) for p in params], dtype=float)


def subsutra12_Modulation(params: np.ndarray) -> np.ndarray:
    return np.array([p * (1 + 0.00005 * i) for i, p in enumerate(params)], dtype=float)


def subsutra13_Differentiation(params: np.ndarray) -> np.ndarray:
    derivative = np.gradient(params)
    return np.array([p + 0.0001 * d for p, d in zip(params, derivative)], dtype=float)


SUB_SUTRAS = (
    subsutra1_Refinement,
    subsutra2_Correction,
    subsutra3_Recursion,
    subsutra4_Convergence,
    subsutra5_Stabilization,
    subsutra6_Simplification,
    subsutra7_Interpolation,
    subsutra8_Extrapolation,
    subsutra9_ErrorReduction,
    subsutra10_Optimization,
    subsutra11_Adjustment,
    subsutra12_Modulation,
    subsutra13_Differentiation,
)


def apply_subsutras_parallel(params: np.ndarray) -> np.ndarray:
    results: list[np.ndarray] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, params) for func in SUB_SUTRAS]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    stacked = np.stack(results, axis=0)
    return np.mean(stacked, axis=0)


def update_parameters(params: Iterable[float]) -> np.ndarray:
    params_array = np.array(list(params), dtype=float)
    params_series = apply_main_sutras(params_array)
    return apply_subsutras_parallel(params_series)


# ---------------------------------------------------------------------------
# Hybrid GRVQ–Vedic Ansatz Circuit Construction
# ---------------------------------------------------------------------------
def hybrid_ansatz_circuit(updated_params: np.ndarray) -> QuantumCircuit:
    if len(updated_params) < 3:
        raise ValueError("Updated parameter vector must contain at least three entries.")
    qc = QuantumCircuit(3)
    qc.h([0, 1, 2])
    angle0 = float(updated_params[0] % (2 * math.pi))
    angle1 = float(updated_params[1] % (2 * math.pi))
    angle2 = float(updated_params[2] % (2 * math.pi))
    qc.rx(angle0, 0)
    qc.ry(angle1, 1)
    qc.rz(angle2, 2)
    return qc


def quantum_test_with_full_sutras(initial_params: Iterable[float] | None = None) -> None:
    if initial_params is None:
        initial_params = [0.5, 0.6, 0.7, 0.8]
    initial_array = np.array(list(initial_params), dtype=float)
    print("Initial parameters:", initial_array)

    updated_params = update_parameters(initial_array)
    print("Updated parameters after applying 29 sutras:", updated_params)

    qc = hybrid_ansatz_circuit(updated_params)
    qc.global_phase = float(np.sum(updated_params) % (2 * math.pi))

    simulator = AerSimulator(method="statevector")
    executable_circuit = qc.copy()
    executable_circuit.save_statevector()
    compiled = transpile(executable_circuit, simulator)
    result = simulator.run(compiled).result()
    state = result.data(0)["statevector"]

    print("\nFinal statevector from the hybrid ansatz circuit:")
    print(state)
    print("\nQuantum Circuit Diagram:")
    print(qc.draw(output="text"))


if __name__ == "__main__":
    quantum_test_with_full_sutras()

