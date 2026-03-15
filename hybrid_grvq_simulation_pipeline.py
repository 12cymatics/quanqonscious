"""Hybrid GRVQ/TGCR simulation pipeline wired to the 29-sutra system.

This module integrates:
- 29-sutra serial, concurrent, and parallel update paths.
- A deterministic PDE solver that consumes sutra-updated parameters.
- An FCI energy calculation using the GRVQ/TGCR tooling in this repository.
- Optional sutra library execution reports via sutra_simulator.

All components are implemented with explicit numeric operations suitable for
hybrid quantum-classical workflows. No placeholders or pseudo code are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np

from sutra_simulator import HybridQuantumClassicalSimulator
from sutra_repository import SutraContext, SutraMode
from integrated_grvq_tgcr import (
    apply_main_sutras,
    apply_subsutras_parallel,
    build_fci_hamiltonian,
    generate_slater_determinants,
    update_29_sutras,
    _SUBSUTRA_FUNCS,
)


@dataclass(frozen=True)
class GridConfig:
    nx: int
    ny: int
    length_x: float
    length_y: float
    time_step: float
    steps: int


@dataclass(frozen=True)
class FCIConfig:
    n_spin_orbitals: int
    n_electrons: int
    h1_scale: float
    g2_scale: float


@dataclass
class SutraUpdateBundle:
    serial: np.ndarray
    concurrent: np.ndarray
    parallel: np.ndarray


@dataclass
class FCIResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    ground_energy: float
    ground_state: np.ndarray


@dataclass
class PDESnapshot:
    field: np.ndarray
    energy_density: np.ndarray
    step: int


@dataclass
class HybridSimulationOutput:
    sutra_updates: SutraUpdateBundle
    fci_result: FCIResult
    pde_snapshots: List[PDESnapshot]
    sutra_reports: Optional[dict]


def apply_subsutras_serial(params: np.ndarray) -> np.ndarray:
    updated = params
    for func in _SUBSUTRA_FUNCS:
        updated = func(updated)
    return updated


def sutra_update_serial(params: np.ndarray) -> np.ndarray:
    main_updated = apply_main_sutras(params)
    return apply_subsutras_serial(main_updated)


def sutra_update_concurrent(params: np.ndarray, max_workers: int) -> np.ndarray:
    main_updated = apply_main_sutras(params)
    return apply_subsutras_parallel(main_updated, max_workers=max_workers)


def _process_chunk(params: np.ndarray) -> np.ndarray:
    return update_29_sutras(params, max_workers=1)


def sutra_update_parallel(params: np.ndarray, workers: int) -> np.ndarray:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if params.size == 0:
        return params
    chunks = np.array_split(params, workers)
    outputs: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
        for future in futures:
            outputs.append(future.result())
    return np.concatenate(outputs)


def build_grid(config: GridConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    dx = config.length_x / (config.nx - 1)
    dy = config.length_y / (config.ny - 1)
    x = np.linspace(0.0, config.length_x, config.nx, dtype=float)
    y = np.linspace(0.0, config.length_y, config.ny, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt((X - config.length_x / 2.0) ** 2 + (Y - config.length_y / 2.0) ** 2)
    return X, Y, r, x, dx, dy


def compute_laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    laplacian = np.zeros_like(field)
    laplacian[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx * dx)
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dy * dy)
    )
    laplacian[0, :] = laplacian[1, :]
    laplacian[-1, :] = laplacian[-2, :]
    laplacian[:, 0] = laplacian[:, 1]
    laplacian[:, -1] = laplacian[:, -2]
    return laplacian


def compute_shape_product(
    X: np.ndarray,
    Y: np.ndarray,
    r: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    product = np.ones_like(r)
    for idx, coeff in enumerate(alpha, start=1):
        harmonic = np.sin(idx * X) * np.cos(idx * Y) * np.exp(-r / float(idx))
        product *= (1.0 + coeff * harmonic)
    return product


def compute_f_vedic(X: np.ndarray, Y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    f_vedic = np.zeros_like(X)
    degree = max(2, int(math.ceil(len(alpha) ** 0.5)))
    for m in range(degree):
        for n in range(degree):
            coeff = alpha[(m + n) % len(alpha)]
            f_vedic += coeff * (X ** m) * (Y ** n)
    return f_vedic


def compute_grvq_source(
    X: np.ndarray,
    Y: np.ndarray,
    r: np.ndarray,
    alpha: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    product_term = compute_shape_product(X, Y, r, alpha)
    radial_cutoff = 1.0 - (r * r) / (r * r + epsilon * epsilon)
    f_vedic = compute_f_vedic(X, Y, alpha)
    return product_term * radial_cutoff * f_vedic


def evolve_pde(
    config: GridConfig,
    alpha: np.ndarray,
    epsilon: float,
) -> List[PDESnapshot]:
    X, Y, r, _, dx, dy = build_grid(config)
    field = np.zeros((config.nx, config.ny), dtype=float)
    for i in range(config.nx):
        for j in range(config.ny):
            field[i, j] = math.sin(X[i, j]) * math.cos(Y[i, j])

    snapshots: List[PDESnapshot] = []
    for step in range(config.steps):
        laplacian = compute_laplacian(field, dx, dy)
        source = compute_grvq_source(X, Y, r, alpha, epsilon)
        field = field + config.time_step * (laplacian + source)
        grad_x = np.gradient(field, axis=0) / dx
        grad_y = np.gradient(field, axis=1) / dy
        energy_density = 0.5 * (field ** 2 + grad_x ** 2 + grad_y ** 2)
        snapshots.append(PDESnapshot(field=field.copy(), energy_density=energy_density, step=step))
    return snapshots


def build_fci_from_params(params: np.ndarray, config: FCIConfig) -> FCIResult:
    n_spin = config.n_spin_orbitals
    n_elec = config.n_electrons
    slater_dets = generate_slater_determinants(n_spin, n_elec)

    h1 = np.zeros((n_spin, n_spin), dtype=float)
    g2 = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=float)

    for p in range(n_spin):
        for q in range(n_spin):
            factor = params[(p + q) % len(params)]
            h1[p, q] = config.h1_scale * (factor - 0.5 * (p == q))

    for p in range(n_spin):
        for q in range(n_spin):
            for r in range(n_spin):
                for s in range(n_spin):
                    idx = (p + q + r + s) % len(params)
                    g2[p, q, r, s] = config.g2_scale * params[idx]

    h_fci = build_fci_hamiltonian(slater_dets, h1, g2)
    evals, evecs = np.linalg.eigh(h_fci)
    ground_idx = int(np.argmin(evals))
    ground_e = float(evals[ground_idx])
    ground_vec = evecs[:, ground_idx]
    return FCIResult(
        eigenvalues=evals,
        eigenvectors=evecs,
        ground_energy=ground_e,
        ground_state=ground_vec,
    )


def run_hybrid_pipeline(
    params: Sequence[float],
    *,
    grid: GridConfig,
    fci: FCIConfig,
    epsilon: float = 1e-6,
    concurrent_workers: int = 8,
    parallel_workers: int = 2,
    include_sutra_reports: bool = True,
) -> HybridSimulationOutput:
    params_array = np.array(params, dtype=float)
    serial_params = sutra_update_serial(params_array)
    concurrent_params = sutra_update_concurrent(params_array, max_workers=concurrent_workers)
    parallel_params = sutra_update_parallel(params_array, workers=parallel_workers)

    pde_snapshots = evolve_pde(grid, serial_params, epsilon)
    fci_result = build_fci_from_params(concurrent_params, fci)

    sutra_reports = None
    if include_sutra_reports:
        context = SutraContext(mode=SutraMode.HYBRID, precision=64)
        simulator = HybridQuantumClassicalSimulator(context=context, max_workers=concurrent_workers)
        report_serial = simulator.run_serial(float(np.mean(params_array)), mode=SutraMode.CLASSICAL)
        report_concurrent = simulator.run_concurrent(float(np.mean(params_array)), mode=SutraMode.QUANTUM)
        report_parallel = simulator.run_parallel(float(np.mean(params_array)), mode=SutraMode.HYBRID)
        sutra_reports = {
            "serial": report_serial.to_dict(),
            "concurrent": report_concurrent.to_dict(),
            "parallel": report_parallel.to_dict(),
        }

    bundle = SutraUpdateBundle(
        serial=serial_params,
        concurrent=concurrent_params,
        parallel=parallel_params,
    )
    return HybridSimulationOutput(
        sutra_updates=bundle,
        fci_result=fci_result,
        pde_snapshots=pde_snapshots,
        sutra_reports=sutra_reports,
    )


def main() -> None:
    params = [0.31, 0.57, 0.93, 0.22, 0.76, 0.45, 0.68, 0.12, 0.84]
    grid = GridConfig(nx=32, ny=32, length_x=6.0, length_y=6.0, time_step=0.01, steps=10)
    fci = FCIConfig(n_spin_orbitals=4, n_electrons=2, h1_scale=1.2, g2_scale=0.8)

    output = run_hybrid_pipeline(
        params,
        grid=grid,
        fci=fci,
        epsilon=1e-5,
        concurrent_workers=4,
        parallel_workers=2,
        include_sutra_reports=True,
    )

    print("Serial sutra parameters:", output.sutra_updates.serial)
    print("Concurrent sutra parameters:", output.sutra_updates.concurrent)
    print("Parallel sutra parameters:", output.sutra_updates.parallel)
    print("FCI ground energy:", output.fci_result.ground_energy)
    print("PDE final step energy density mean:", float(np.mean(output.pde_snapshots[-1].energy_density)))


if __name__ == "__main__":
    main()
