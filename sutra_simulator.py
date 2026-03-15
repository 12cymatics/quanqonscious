"""High level orchestration utilities for the 29 Vedic sutras.

This module exposes a :class:`HybridQuantumClassicalSimulator` that can execute
the entire sutra corpus serially, concurrently (multi-threaded) or in parallel
across processes. It wraps :class:`sutra_repository.SutraRepository` and keeps
track of execution timings, intermediate artefacts and aggregation logic so the
caller can fuse the classical and quantum contributions in a single place.

The implementation is intentionally free of pseudo code. Every helper is fully
implemented and can be used programmatically or from higher-level CLIs (such as
``sutra_orchestrator.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import inspect
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from sutra_repository import SutraRepository, SutraContext, SutraMode

REPORT_SCHEMA_VERSION = "hsqcp.report.v1"
DOCTRINE_FIDELITY_TAG = "exact-symbolic-core"

ArgumentResolver = Callable[[str, Any, SutraContext, Any], Tuple[Tuple[Any, ...], Dict[str, Any]]]


def _prepare_args_for_sutra(func: Callable[..., Any], value: Any) -> List[Any]:
    """Generate default positional arguments for a sutra function."""

    sig = inspect.signature(func)
    args: List[Any] = []
    for name, param in sig.parameters.items():
        if name in {"self", "ctx"}:
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Parameter.empty:
                if any(
                    key in name
                    for key in (
                        "coeff",
                        "angles",
                        "values",
                        "parts",
                        "list",
                        "vector",
                    )
                ):
                    args.append([value, value])
                else:
                    args.append(value)
    return args


@dataclass
class SutraExecution:
    """Container describing the outcome of a single sutra invocation."""

    name: str
    output: Any
    elapsed_ns: int


@dataclass
class SimulationReport:
    """Structured summary returned after running a suite of sutras."""

    mode: SutraMode
    initial_value: Any
    executions: List[SutraExecution] = field(default_factory=list)
    aggregate: Any = None
    wall_time_ns: int = 0
    schema_version: str = REPORT_SCHEMA_VERSION
    doctrine_fidelity: str = DOCTRINE_FIDELITY_TAG
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary for logging or storage."""

        return {
            "schema_version": self.schema_version,
            "doctrine_fidelity": self.doctrine_fidelity,
            "generated_at_utc": self.generated_at_utc,
            "mode": self.mode.name,
            "initial_value": self.initial_value,
            "aggregate": self.aggregate,
            "wall_time_ns": self.wall_time_ns,
            "executions": [
                {"name": exec.name, "elapsed_ns": exec.elapsed_ns, "output": exec.output}
                for exec in self.executions
            ],
        }


def _build_context(template: SutraContext, *, mode: Optional[SutraMode] = None) -> SutraContext:
    """Clone the provided context with optional mode override.

    ``SutraContext`` instances carry GPU handles and other non-serialisable
    attributes. :func:`dataclasses.replace` is used to avoid mutating the
    original reference while preventing inadvertent sharing of execution state.
    """

    context = replace(template)
    if mode is not None:
        context.mode = mode
    # Disable nested parallelism inside helpers when higher-level scheduling is
    # applied. This avoids recursive thread spawning.
    context.parallel = False
    return context


def _execute_sutra(
    name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    context: SutraContext,
) -> SutraExecution:
    """Execute a sutra inside the current process."""

    repo = SutraRepository(context)
    start = perf_counter_ns()
    result = repo.call_sutra(name, *args, ctx=context, **kwargs)
    elapsed_ns = perf_counter_ns() - start
    return SutraExecution(name=name, output=result, elapsed_ns=elapsed_ns)


def _execute_sutra_process(payload: Dict[str, Any]) -> SutraExecution:
    """Execute a sutra in a worker process.

    ``SutraContext`` may contain objects that are not picklable (such as CUDA
    handles). For multi-processing we rebuild a lightweight context using only
    primitive fields from the payload.
    """

    name = payload["name"]
    args = payload["args"]
    kwargs = payload["kwargs"]
    context_kwargs = payload["context_kwargs"]
    context = SutraContext(**context_kwargs)
    return _execute_sutra(name, args, kwargs, context)


class HybridQuantumClassicalSimulator:
    """Coordinator for running all 29 Vedic sutras in different execution styles."""

    def __init__(
        self,
        context: Optional[SutraContext] = None,
        *,
        max_workers: Optional[int] = None,
        aggregation: Optional[Callable[[Any, SutraExecution], Any]] = None,
        sutra_filter: Optional[Callable[[str], bool]] = None,
        argument_resolver: Optional[ArgumentResolver] = None,
    ) -> None:
        self._base_context = context or SutraContext()
        self._repository = SutraRepository(self._base_context)
        names = self._repository.list_sutras()
        if sutra_filter is not None:
            names = [name for name in names if sutra_filter(name)]
        self._sutra_names = names
        self._max_workers = max_workers
        self._aggregation = aggregation or self._default_aggregation
        self._argument_resolver = argument_resolver or self._default_argument_resolver

    @property
    def sutra_names(self) -> Iterable[str]:
        return tuple(self._sutra_names)

    def _default_aggregation(self, previous: Any, execution: SutraExecution) -> Any:
        """Default aggregation simply forwards the most recent output."""

        return execution.output

    def _default_argument_resolver(
        self,
        name: str,
        current_value: Any,
        context: SutraContext,
        initial_value: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return (current_value,), {}

    def _resolve_arguments(
        self,
        name: str,
        current_value: Any,
        context: SutraContext,
        initial_value: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        args, kwargs = self._argument_resolver(name, current_value, context, initial_value)
        return tuple(args), dict(kwargs)

    def run_serial(
        self,
        value: Any,
        *,
        mode: Optional[SutraMode] = None,
    ) -> SimulationReport:
        """Run every sutra sequentially, feeding the output into the next sutra."""

        context = _build_context(self._base_context, mode=mode)
        repo = SutraRepository(context)
        initial_value: Any = value
        aggregate_value: Any = value
        report = SimulationReport(mode=context.mode, initial_value=value)
        wall_start = perf_counter_ns()

        for name in self._sutra_names:
            start = perf_counter_ns()
            args, call_kwargs = self._resolve_arguments(
                name, aggregate_value, context, initial_value
            )
            if not args and not call_kwargs:
                args = tuple(_prepare_args_for_sutra(repo._methods[name], aggregate_value))
            aggregate_value = repo.call_sutra(
                name,
                *args,
                ctx=context,
                **call_kwargs,
            )
            elapsed_ns = perf_counter_ns() - start
            execution = SutraExecution(name=name, output=aggregate_value, elapsed_ns=elapsed_ns)
            report.executions.append(execution)
            report.aggregate = self._aggregation(report.aggregate, execution)

        report.wall_time_ns = perf_counter_ns() - wall_start
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
        wall_start = perf_counter_ns()

        def submit(name: str) -> SutraExecution:
            local_context = _build_context(context, mode=context.mode)
            args, call_kwargs = self._resolve_arguments(
                name, value, local_context, value
            )
            if not args and not call_kwargs:
                args = tuple(_prepare_args_for_sutra(self._repository._methods[name], value))
            return _execute_sutra(name, args, call_kwargs, local_context)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(submit, name): name for name in self._sutra_names}
            for future in as_completed(futures):
                report.executions.append(future.result())

        report.executions.sort(key=lambda exec: exec.name)
        for execution in report.executions:
            report.aggregate = self._aggregation(report.aggregate, execution)
        report.wall_time_ns = perf_counter_ns() - wall_start
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
        wall_start = perf_counter_ns()

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

        payloads = []
        for name in self._sutra_names:
            args, call_kwargs = self._resolve_arguments(name, value, context, value)
            if not args and not call_kwargs:
                args = tuple(_prepare_args_for_sutra(self._repository._methods[name], value))
            payloads.append(
                {
                    "name": name,
                    "args": args,
                    "kwargs": call_kwargs,
                    "context_kwargs": context_kwargs,
                }
            )

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(_execute_sutra_process, payload): payload["name"] for payload in payloads}
            for future in as_completed(futures):
                report.executions.append(future.result())

        report.executions.sort(key=lambda exec: exec.name)
        for execution in report.executions:
            report.aggregate = self._aggregation(report.aggregate, execution)
        report.wall_time_ns = perf_counter_ns() - wall_start
        return report


__all__ = [
    "DOCTRINE_FIDELITY_TAG",
    "REPORT_SCHEMA_VERSION",
    "HybridQuantumClassicalSimulator",
    "SimulationReport",
    "SutraExecution",
]
