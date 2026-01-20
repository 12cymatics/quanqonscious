from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from grvq_toroidal_hypercube import GRVQToroidalHypercube, SutraExecutionPlan
from primarysutra import SutraContext, SutraMode


@dataclass
class HybridSimulationConfig:
    """Configuration for hybrid quantum-classical toroidal hypercube runs."""
    n_points: int = 50
    m_mode: int = 3
    n_mode: int = 5
    angle_xw: float = 0.4
    execution_mode: str = "concurrent"
    max_workers: int = 8
    sutra_mode: SutraMode = SutraMode.CLASSICAL
    sutra_base: float = 10.0


class HybridGRVQToroidalSimulator:
    """
    Orchestrates a hybrid quantum-classical simulation using the 29 Vedic sutras
    with selectable serial, concurrent, or parallel execution.
    """

    def __init__(self, config: HybridSimulationConfig,
                 R_major: float = 0.6,
                 R_minor: float = 0.3,
                 tesseract_scale: float = 0.35,
                 R0: float = 1.0) -> None:
        self.config = config
        self.sutra_context = SutraContext(
            mode=config.sutra_mode,
            base=config.sutra_base,
            parallel=config.execution_mode != "serial",
        )
        self.sutra_plan = SutraExecutionPlan(
            context=self.sutra_context,
            execution_mode=config.execution_mode,
            max_workers=config.max_workers,
        )
        self.system = GRVQToroidalHypercube(
            R_major=R_major,
            R_minor=R_minor,
            tesseract_scale=tesseract_scale,
            R0=R0,
            sutra_plan=self.sutra_plan,
        )

    def _run_with_mode(self, execution_mode: str) -> Dict[str, Any]:
        local_plan = SutraExecutionPlan(
            context=self.sutra_context,
            execution_mode=execution_mode,
            max_workers=self.config.max_workers,
        )
        result = self.system.compute_full_system(
            n_points=self.config.n_points,
            m=self.config.m_mode,
            n=self.config.n_mode,
            angle_xw=self.config.angle_xw,
            sutra_plan=local_plan,
        )
        result["sutra_execution_mode"] = execution_mode
        result["sutra_mode"] = self.sutra_context.mode.name
        result["sutra_base"] = self.sutra_context.base
        result["field_energy"] = float(np.linalg.norm(result["transformed_field"]))
        result["field_mean"] = float(np.mean(result["transformed_field"]))
        result["sutra_plan"] = local_plan
        return result

    def run_serial(self) -> Dict[str, Any]:
        """Run with serial sutra execution."""
        return self._run_with_mode("serial")

    def run_concurrent(self) -> Dict[str, Any]:
        """Run with concurrent (threaded) sutra execution."""
        return self._run_with_mode("concurrent")

    def run_parallel(self) -> Dict[str, Any]:
        """Run with parallel (process) sutra execution."""
        return self._run_with_mode("parallel")

    def run(self) -> Dict[str, Any]:
        """Run using the configured execution mode."""
        return self._run_with_mode(self.config.execution_mode)
