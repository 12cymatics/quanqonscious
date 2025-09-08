from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List

from .core_engine import run_full_engine


class ExecutionMode(str, Enum):
    """Execution strategy for applying all 29 sutras."""
    SERIAL = "serial"
    THREADS = "threads"
    PROCESSES = "processes"


@dataclass
class SutraExecutor:
    """Run the full GRVQ Vedic engine across multiple inputs.

    Parameters
    ----------
    mode:
        Execution strategy. ``serial`` executes inputs one after another.
        ``threads`` uses a :class:`~concurrent.futures.ThreadPoolExecutor`.
        ``processes`` uses a :class:`~concurrent.futures.ProcessPoolExecutor`.
    max_workers:
        Optional override for the number of workers in threaded or process
        modes.  When ``None`` the executors pick a sensible default based on
        the host environment.
    """

    mode: ExecutionMode = ExecutionMode.SERIAL
    max_workers: int | None = None

    def _run_single(self, params: np.ndarray) -> np.ndarray:
        return run_full_engine(params)

    @staticmethod
    def _run_static(params: np.ndarray) -> np.ndarray:
        """Helper for process-based execution.

        ``ProcessPoolExecutor`` needs a top-level function in order to pickle
        the callable.  This method simply forwards to
        :func:`run_full_engine`.
        """

        return run_full_engine(params)

    def execute(self, inputs: Iterable[np.ndarray]) -> List[np.ndarray]:
        """Apply all sutras to each array in ``inputs``.

        Parameters
        ----------
        inputs:
            Iterable of NumPy arrays that represent the initial parameter
            vectors for the GRVQ-TTGCR engine.

        Returns
        -------
        list of numpy.ndarray
            One refined array for each element of ``inputs`` in the original
            order.
        """

        data = list(inputs)
        if self.mode is ExecutionMode.SERIAL:
            return [self._run_single(arr) for arr in data]

        if self.mode is ExecutionMode.THREADS:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._run_single, arr) for arr in data]
                return [f.result() for f in futures]

        if self.mode is ExecutionMode.PROCESSES:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(SutraExecutor._run_static, arr) for arr in data]
                return [f.result() for f in futures]

        raise ValueError(f"Unsupported execution mode: {self.mode}")
