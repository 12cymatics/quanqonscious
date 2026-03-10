from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.lattice import ToroidalHypercube, LatticePoint
from core.operators.base import OperatorContext
from core.operators.sutra_ops import get_all_sutras
from core.state import ArithmeticMode, FieldState


@dataclass
class SolverConfig:
    grid_shape: Tuple[int, int] = (64, 64)
    time_step: float = 0.005
    spatial_step: float = 0.02
    max_iterations: int = 5000
    tolerance: float = 1e-8
    relaxation: float = 1.6
    parallel_splits: Tuple[int, int] = (2, 2)
    chunk_size: int = 256


def _fraction_from_float(value: float) -> Fraction:
    return Fraction(value).limit_denominator(10**8)


def _field_to_state(field: np.ndarray, mode: ArithmeticMode) -> FieldState:
    lattice = ToroidalHypercube(field.shape)
    state = FieldState(lattice=lattice, mode=mode)
    for j in range(field.shape[0]):
        for i in range(field.shape[1]):
            state.set_by_coords((j, i), float(field[j, i]))
    return state


def _state_to_field(state: FieldState) -> np.ndarray:
    shape = state.lattice.shape
    field = np.zeros(shape, dtype=float)
    for point in state.lattice.iterate_all():
        field[point.coords] = float(state.get(point).real)
    return field


def _apply_sutra_pipeline_serial_array(field: np.ndarray, time_step: float) -> np.ndarray:
    sutras = get_all_sutras()
    context = OperatorContext(dt=_fraction_from_float(time_step), mode=ArithmeticMode.EXACT)
    state = _field_to_state(field, ArithmeticMode.EXACT)
    for sutra in sutras:
        state = sutra(state, context)
    return _state_to_field(state)


def _apply_sutra_parallel_points(
    state: FieldState,
    sutra,
    context: OperatorContext,
    chunk_size: int,
    executor,
) -> FieldState:
    points = list(state.lattice.iterate_all())
    chunks = [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]

    def process(chunk: List[LatticePoint]) -> List[Tuple[Tuple[int, ...], object]]:
        results = []
        for point in chunk:
            old_val = state.get(point)
            new_val = sutra.sutra_transform(old_val, point.coords, state, context)
            results.append((point.coords, new_val))
        return results

    new_state = state.copy()
    futures = [executor.submit(process, chunk) for chunk in chunks]
    for future in futures:
        for coords, value in future.result():
            new_state.set_by_coords(coords, value)

    invariants, passed = sutra.check_invariants(new_state, context)
    if not passed:
        raise RuntimeError(f"Invariant check failed for {sutra.name} in concurrent mode")
    return new_state


def _apply_sutra_pipeline_concurrent_array(
    field: np.ndarray,
    time_step: float,
    chunk_size: int,
) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    sutras = get_all_sutras()
    context = OperatorContext(dt=_fraction_from_float(time_step), mode=ArithmeticMode.EXACT)
    state = _field_to_state(field, ArithmeticMode.EXACT)
    with ThreadPoolExecutor() as executor:
        for sutra in sutras:
            state = _apply_sutra_parallel_points(state, sutra, context, chunk_size, executor)
    return _state_to_field(state)


def _extract_tile(field: np.ndarray, y0: int, y1: int, x0: int, x1: int, halo: int) -> np.ndarray:
    height, width = field.shape
    tile_height = (y1 - y0) + 2 * halo
    tile_width = (x1 - x0) + 2 * halo
    tile = np.zeros((tile_height, tile_width), dtype=float)
    for j in range(y0 - halo, y1 + halo):
        for i in range(x0 - halo, x1 + halo):
            tile[j - (y0 - halo), i - (x0 - halo)] = field[j % height, i % width]
    return tile


def _tile_pipeline_worker(args: Tuple[np.ndarray, float, int]) -> np.ndarray:
    tile, time_step, halo = args
    processed = _apply_sutra_pipeline_serial_array(tile, time_step)
    return processed[halo:-halo, halo:-halo]


def _apply_sutra_pipeline_parallel_array(
    field: np.ndarray,
    time_step: float,
    splits: Tuple[int, int],
    halo: int = 1,
) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor

    height, width = field.shape
    y_splits, x_splits = splits
    y_edges = np.linspace(0, height, y_splits + 1, dtype=int)
    x_edges = np.linspace(0, width, x_splits + 1, dtype=int)

    tiles = []
    tile_meta = []
    for yi in range(y_splits):
        for xi in range(x_splits):
            y0, y1 = y_edges[yi], y_edges[yi + 1]
            x0, x1 = x_edges[xi], x_edges[xi + 1]
            tile = _extract_tile(field, y0, y1, x0, x1, halo)
            tiles.append(tile)
            tile_meta.append((y0, y1, x0, x1))

    result = np.zeros_like(field)
    with ProcessPoolExecutor() as executor:
        outputs = executor.map(_tile_pipeline_worker, [(tile, time_step, halo) for tile in tiles])
        for (y0, y1, x0, x1), processed in zip(tile_meta, outputs):
            result[y0:y1, x0:x1] = processed
    return result


class VedicPDESuite:
    """Reference solvers for PDE validation using Vedic sutra-driven updates."""

    def __init__(self, config: Optional[SolverConfig] = None) -> None:
        self.config = config or SolverConfig()
        self.sutras = get_all_sutras()
        if len(self.sutras) != 29:
            raise ValueError(f"Expected 29 sutras, found {len(self.sutras)}")

    def _field_energy(self, field: np.ndarray) -> float:
        return float(np.linalg.norm(field) / field.size)

    def _sutra_serial(self, field: np.ndarray) -> np.ndarray:
        return _apply_sutra_pipeline_serial_array(field, self.config.time_step)

    def _sutra_concurrent(self, field: np.ndarray) -> np.ndarray:
        return _apply_sutra_pipeline_concurrent_array(
            field, self.config.time_step, self.config.chunk_size
        )

    def _sutra_parallel(self, field: np.ndarray) -> np.ndarray:
        return _apply_sutra_pipeline_parallel_array(
            field, self.config.time_step, self.config.parallel_splits
        )

    def vedic_coefficients(self, field: np.ndarray) -> Dict[str, float]:
        serial_field = self._sutra_serial(field)
        concurrent_field = self._sutra_concurrent(field)
        parallel_field = self._sutra_parallel(field)
        return {
            "serial": self._field_energy(serial_field),
            "concurrent": self._field_energy(concurrent_field),
            "parallel": self._field_energy(parallel_field),
        }

    def laplace_2d(self, boundary: np.ndarray) -> np.ndarray:
        values = boundary.copy().astype(float)
        ny, nx = values.shape
        for _ in range(self.config.max_iterations):
            old = values.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    update = 0.25 * (
                        values[j, i + 1]
                        + values[j, i - 1]
                        + values[j + 1, i]
                        + values[j - 1, i]
                    )
                    values[j, i] = (
                        (1.0 - self.config.relaxation) * values[j, i]
                        + self.config.relaxation * update
                    )
            diff = np.linalg.norm(values - old)
            if diff < self.config.tolerance:
                break
        return values

    def poisson_2d(self, boundary: np.ndarray, source: np.ndarray) -> np.ndarray:
        values = boundary.copy().astype(float)
        ny, nx = values.shape
        h2 = self.config.spatial_step ** 2
        for _ in range(self.config.max_iterations):
            old = values.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    update = 0.25 * (
                        values[j, i + 1]
                        + values[j, i - 1]
                        + values[j + 1, i]
                        + values[j - 1, i]
                        - h2 * source[j, i]
                    )
                    values[j, i] = (
                        (1.0 - self.config.relaxation) * values[j, i]
                        + self.config.relaxation * update
                    )
            diff = np.linalg.norm(values - old)
            if diff < self.config.tolerance:
                break
        return values

    def heat_2d(self, initial: np.ndarray, steps: int = 100) -> np.ndarray:
        field = initial.copy().astype(float)
        ny, nx = field.shape
        alpha = 0.1
        dt = self.config.time_step
        dx = self.config.spatial_step
        coeff = alpha * dt / (dx * dx)
        for _ in range(steps):
            new_field = field.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    new_field[j, i] = field[j, i] + coeff * (
                        field[j + 1, i]
                        + field[j - 1, i]
                        + field[j, i + 1]
                        + field[j, i - 1]
                        - 4.0 * field[j, i]
                    )
            field = new_field
        return field

    def wave_2d(self, initial: np.ndarray, velocity: np.ndarray, steps: int = 100) -> np.ndarray:
        field = initial.copy().astype(float)
        field_prev = field - self.config.time_step * velocity
        ny, nx = field.shape
        c = 1.0
        dt = self.config.time_step
        dx = self.config.spatial_step
        coeff = (c * dt / dx) ** 2
        for _ in range(steps):
            field_next = field.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    laplacian = (
                        field[j + 1, i]
                        + field[j - 1, i]
                        + field[j, i + 1]
                        + field[j, i - 1]
                        - 4.0 * field[j, i]
                    )
                    field_next[j, i] = 2.0 * field[j, i] - field_prev[j, i] + coeff * laplacian
            field_prev, field = field, field_next
        return field

    def burgers_1d(self, initial: np.ndarray, steps: int = 200) -> np.ndarray:
        field = initial.copy().astype(float)
        nu = 0.01
        dt = self.config.time_step
        dx = self.config.spatial_step
        for _ in range(steps):
            new_field = field.copy()
            for i in range(1, len(field) - 1):
                convection = field[i] * (field[i] - field[i - 1]) / dx
                diffusion = nu * (field[i + 1] - 2.0 * field[i] + field[i - 1]) / (dx * dx)
                new_field[i] = field[i] - dt * convection + dt * diffusion
            field = new_field
        return field

    def validate_against_reference(
        self,
        reference_solver: Callable[..., np.ndarray],
        target_solver: Callable[..., np.ndarray],
        *args: np.ndarray,
    ) -> float:
        reference = reference_solver(*args)
        target = target_solver(*args)
        diff = np.linalg.norm(reference - target)
        return float(diff)

    def vq_residual(self, field: np.ndarray) -> float:
        coeffs = self.vedic_coefficients(field)
        serial = coeffs["serial"]
        concurrent = coeffs["concurrent"]
        parallel = coeffs["parallel"]
        residual = np.linalg.norm(field) * (serial + concurrent + parallel) / 3.0
        return float(residual)
