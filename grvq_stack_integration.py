from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import List

import numpy as np
import websockets

from vedic_solutions import VedicPDESuite, SolverConfig


@dataclass
class GRVQStackConfig:
    radial_resolution: int = 16
    theta_resolution: int = 24
    phi_resolution: int = 32
    r_min: float = 0.25
    r_max: float = 4.5
    stream_interval: float = 0.08
    intensity_floor: float = 0.1
    intensity_ceiling: float = 0.95
    pde_shape: tuple[int, int] = (48, 48)


class GRVQShellGenerator:
    def __init__(self, config: GRVQStackConfig) -> None:
        self.config = config
        self.pde_suite = VedicPDESuite(
            SolverConfig(grid_shape=config.pde_shape, time_step=0.01, spatial_step=0.04)
        )
        self.chebyshev = self._chebyshev_grid()

    def _chebyshev_grid(self) -> np.ndarray:
        n = self.config.radial_resolution
        angles = np.linspace(0.0, math.pi, n)
        mid = 0.5 * (self.config.r_max + self.config.r_min)
        half = 0.5 * (self.config.r_max - self.config.r_min)
        return mid + half * np.cos(angles)

    def _r4_suppression(self, r: float) -> float:
        r2 = r * r
        return 1.0 / (1.0 + r2 * r2 / (1.0 + r2))

    def _boundary_field(self, step: int) -> np.ndarray:
        ny, nx = self.config.pde_shape
        field = np.zeros((ny, nx), dtype=float)
        phase = 0.05 * step
        top = np.sin(np.linspace(0.0, 2.0 * math.pi, nx) + phase)
        bottom = np.cos(np.linspace(0.0, 2.0 * math.pi, nx) - phase)
        left = np.sin(np.linspace(0.0, math.pi, ny) - phase)
        right = np.cos(np.linspace(0.0, math.pi, ny) + phase)
        field[0, :] = top
        field[-1, :] = bottom
        field[:, 0] = left
        field[:, -1] = right
        return field

    def _source_field(self, step: int) -> np.ndarray:
        ny, nx = self.config.pde_shape
        y = np.linspace(-1.0, 1.0, ny)
        x = np.linspace(-1.0, 1.0, nx)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx * xx + yy * yy)
        phase = 0.1 * step
        return np.exp(-4.0 * radius * radius) * np.cos(2.0 * math.pi * radius + phase)

    def _compose_field(self, step: int) -> np.ndarray:
        boundary = self._boundary_field(step)
        source = self._source_field(step)
        laplace = self.pde_suite.laplace_2d(boundary)
        poisson = self.pde_suite.poisson_2d(boundary, source)
        heat = self.pde_suite.heat_2d(boundary, steps=40)
        velocity = np.zeros_like(boundary)
        wave = self.pde_suite.wave_2d(boundary, velocity, steps=30)
        composite = 0.25 * (laplace + poisson + heat + wave)
        mid_row = composite.shape[0] // 2
        burgers = self.pde_suite.burgers_1d(composite[mid_row], steps=80)
        composite[mid_row] += 0.15 * burgers
        return composite

    def _shell_energy(
        self,
        r: float,
        step: int,
        coefficients: dict[str, float],
        residual: float,
    ) -> float:
        coefficient_avg = (coefficients["serial"] + coefficients["concurrent"] + coefficients["parallel"]) / 3.0
        phase = 1.0 + 0.25 * math.sin(0.2 * step + r)
        damping = 1.0 / (1.0 + residual)
        return float(self._r4_suppression(r) * phase * damping * math.tanh(coefficient_avg))

    def generate_intensity(self, step: int) -> List[float]:
        field = self._compose_field(step)
        coefficients = self.pde_suite.vedic_coefficients(field)
        residual = self.pde_suite.vq_residual(field)
        intensity = []
        for idx, r in enumerate(self.chebyshev):
            shell_value = self._shell_energy(r, step, coefficients, residual)
            phase = 0.6 + 0.4 * math.sin(0.3 * step + idx)
            value = shell_value * phase
            intensity.append(value)
        normalized = np.interp(
            intensity,
            (min(intensity), max(intensity)),
            (self.config.intensity_floor, self.config.intensity_ceiling),
        )
        return normalized.tolist()


class GRVQStackStreamer:
    def __init__(self, config: GRVQStackConfig) -> None:
        self.config = config
        self.generator = GRVQShellGenerator(config)

    async def _handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        step = 0
        while True:
            payload = {
                "intensity": self.generator.generate_intensity(step),
                "step": step,
            }
            await websocket.send(json.dumps(payload))
            step += 1
            await asyncio.sleep(self.config.stream_interval)

    async def serve(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        async with websockets.serve(self._handler, host, port):
            await asyncio.Future()


def main() -> None:
    config = GRVQStackConfig()
    streamer = GRVQStackStreamer(config)
    asyncio.run(streamer.serve())


if __name__ == "__main__":
    main()
