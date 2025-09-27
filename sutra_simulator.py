"""High level orchestration utilities for the 29 Vedic sutras.

This module exposes a :class:`HybridQuantumClassicalSimulator` that can execute
the entire sutra corpus serially, concurrently (multi-threaded) or in parallel
across processes.  It wraps :class:`sutra_repository.SutraRepository` and keeps
track of execution timings, intermediate artefacts and aggregation logic so the
caller can fuse the classical and quantum contributions in a single place.

The implementation is intentionally free of pseudo code.  Every helper is fully
implemented and can be used programmatically or from higher-level CLIs (such as
``sutra_orchestrator.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from sutra_repository import SutraRepository, SutraContext, SutraMode


@dataclass
class SutraExecution:
    """Container describing the outcome of a single sutra invocation."""

    name: str
    output: Any
    elapsed: float


@dataclass
class SimulationReport:
    """Structured summary returned after running a suite of sutras."""

    mode: SutraMode
    initial_value: Any
    executions: List[SutraExecution] = field(default_factory=list)
    aggregate: Any = None
    wall_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary for logging or storage."""

        return {
            "mode": self.mode.name,
            "initial_value": self.initial_value,
            "aggregate": self.aggregate,
            "wall_time": self.wall_time,
            "executions": [
                {"name": exec.name, "elapsed": exec.elapsed, "output": exec.output}
                for exec in self.executions
            ],
        }


def _build_context(template: SutraContext, *, mode: Optional[SutraMode] = None) -> SutraContext:
    """Clone the provided context with optional mode override.

    ``SutraContext`` instances carry GPU handles and other non-serialisable
    attributes.  :func:`dataclasses.replace` is used to avoid mutating the
    original reference while preventing inadvertent sharing of execution state.
    """

    context = replace(template)
    if mode is not None:
        context.mode = mode
    # Disable nested parallelism inside helpers when higher-level scheduling is
    # applied.  This avoids recursive thread spawning.
    context.parallel = False
    return context


def _execute_sutra(name: str, value: Any, context: SutraContext) -> SutraExecution:
    """Execute a sutra inside the current process."""

    repo = SutraRepository(context)
    start = perf_counter()
    result = repo.call_sutra(name, value, ctx=context)
    elapsed = perf_counter() - start
    return SutraExecution(name=name, output=result, elapsed=elapsed)


def _execute_sutra_process(payload: Dict[str, Any]) -> SutraExecution:
    """Execute a sutra in a worker process.

    ``SutraContext`` may contain objects that are not picklable (such as CUDA
    handles).  For multi-processing we rebuild a lightweight context using only
    primitive fields from the payload.
    """

    name = payload["name"]
    value = payload["value"]
    context_kwargs = payload["context_kwargs"]
    context = SutraContext(**context_kwargs)
    return _execute_sutra(name, value, context)


class HybridQuantumClassicalSimulator:
    """Coordinator for running all 29 Vedic sutras in different execution styles."""

    def __init__(
        self,
        context: Optional[SutraContext] = None,
        *,
        max_workers: Optional[int] = None,
        aggregation: Optional[Callable[[Any, SutraExecution], Any]] = None,
    ) -> None:
        self._base_context = context or SutraContext()
        self._repository = SutraRepository(self._base_context)
        self._sutra_names = self._repository.list_sutras()
        self._max_workers = max_workers
        self._aggregation = aggregation or self._default_aggregation

    @property
    def sutra_names(self) -> Iterable[str]:
        return tuple(self._sutra_names)

    def _default_aggregation(self, previous: Any, execution: SutraExecution) -> Any:
        """Default aggregation simply forwards the most recent output."""

        return execution.output

    def run_serial(
        self,
        value: Any,
        *,
        mode: Optional[SutraMode] = None,
    ) -> SimulationReport:
        """Run every sutra sequentially, feeding the output into the next sutra."""

        context = _build_context(self._base_context, mode=mode)
        repo = SutraRepository(context)
        aggregate_value: Any = value
        report = SimulationReport(mode=context.mode, initial_value=value)
        wall_start = perf_counter()

        for name in self._sutra_names:
            start = perf_counter()
            aggregate_value = repo.call_sutra(name, aggregate_value, ctx=context)
            elapsed = perf_counter() - start
            execution = SutraExecution(name=name, output=aggregate_value, elapsed=elapsed)
            report.executions.append(execution)
            report.aggregate = self._aggregation(report.aggregate, execution)

        report.wall_time = perf_counter() - wall_start
        if report.aggregate is None:
            report.aggregate = aggregate_value
        return report

    def run_concurrent(
        self,
        value: Any,
        *,
        mode: Optional[SutraMode] = None,
    ) -> SimulationReport:
        """Dispatch every sutra via a thread pool using an identical input value."""

        context = _build_context(self._base_context, mode=mode)
        report = SimulationReport(mode=context.mode, initial_value=value)
        wall_start = perf_counter()

        def submit(name: str) -> SutraExecution:
            local_context = _build_context(context, mode=context.mode)
            return _execute_sutra(name, value, local_context)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(submit, name): name for name in self._sutra_names}
            for future in as_completed(futures):
                report.executions.append(future.result())

        report.executions.sort(key=lambda exec: exec.name)
        for execution in report.executions:
            report.aggregate = self._aggregation(report.aggregate, execution)
        report.wall_time = perf_counter() - wall_start
        return report

    def run_parallel(
        self,
        value: Any,
        *,
        mode: Optional[SutraMode] = None,
    ) -> SimulationReport:
        """Run sutras in separate processes for strict isolation."""

        context = _build_context(self._base_context, mode=mode)
        report = SimulationReport(mode=context.mode, initial_value=value)
        wall_start = perf_counter()

        context_kwargs = {
            "mode": context.mode,
            "quantum_backend": None,
            "precision": context.precision,
            "base": context.base,
            "epsilon": context.epsilon,
            "max_iterations": context.max_iterations,
            "use_gpu": False,
            "device": None,
            "record_performance": context.record_performance,
            "visualization": context.visualization,
            "parallel": False,
        }

        payloads = [
            {"name": name, "value": value, "context_kwargs": context_kwargs}
            for name in self._sutra_names
        ]

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(_execute_sutra_process, payload): payload["name"] for payload in payloads}
            for future in as_completed(futures):
                report.executions.append(future.result())

        report.executions.sort(key=lambda exec: exec.name)
        for execution in report.executions:
            report.aggregate = self._aggregation(report.aggregate, execution)
        report.wall_time = perf_counter() - wall_start
        return report


__all__ = [
    "HybridQuantumClassicalSimulator",
    "SimulationReport",
    "SutraExecution",
]

