"""Hybrid Vedic Sutra platform entrypoint.

This module orchestrates the 29 sutras in serial, concurrent, and parallel
execution styles so the same input can be stress-tested across scheduling
strategies. The output is designed to be directly consumable by a
hybrid quantum-classical workflow and can be serialized as JSON for
integration with external services.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sutra_simulator import HybridQuantumClassicalSimulator, SimulationReport
from sutra_repository import SutraContext, SutraMode


@dataclass
class ModeRunConfig:
    """Configuration for a single simulator execution mode."""

    label: str
    mode: SutraMode


@dataclass
class HybridSimulationBundle:
    """Aggregated results across serial, concurrent, and parallel runs."""

    initial_value: Any
    sutra_names: List[str]
    serial: SimulationReport
    concurrent: SimulationReport
    parallel: SimulationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_value": self.initial_value,
            "sutra_names": list(self.sutra_names),
            "serial": self.serial.to_dict(),
            "concurrent": self.concurrent.to_dict(),
            "parallel": self.parallel.to_dict(),
        }


def _parse_mode(value: str) -> SutraMode:
    try:
        return SutraMode[value.upper()]
    except KeyError as exc:
        options = ", ".join(m.name.lower() for m in SutraMode)
        raise ValueError(f"Unknown mode '{value}'. Options: {options}") from exc


def _build_context(mode: SutraMode, *, precision: int, max_iterations: int) -> SutraContext:
    return SutraContext(
        mode=mode,
        precision=precision,
        max_iterations=max_iterations,
        parallel=True,
    )


def _apply_filter(names: Iterable[str], *, include: Optional[str]) -> List[str]:
    if include is None:
        return list(names)
    return [name for name in names if include.lower() in name.lower()]


def run_hybrid_bundle(
    value: Any,
    *,
    mode: SutraMode,
    precision: int,
    max_iterations: int,
    include: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> HybridSimulationBundle:
    """Run serial, concurrent, and parallel suites for the full sutra set."""

    context = _build_context(mode, precision=precision, max_iterations=max_iterations)
    simulator = HybridQuantumClassicalSimulator(context=context, max_workers=max_workers)
    if include:
        simulator = HybridQuantumClassicalSimulator(
            context=context,
            max_workers=max_workers,
            sutra_filter=lambda name: name in _apply_filter(simulator.sutra_names, include=include),
        )

    sutra_names = list(simulator.sutra_names)
    serial_report = simulator.run_serial(value, mode=mode)
    concurrent_report = simulator.run_concurrent(value, mode=mode)
    parallel_report = simulator.run_parallel(value, mode=mode)
    return HybridSimulationBundle(
        initial_value=value,
        sutra_names=sutra_names,
        serial=serial_report,
        concurrent=concurrent_report,
        parallel=parallel_report,
    )


def _summarize_report(report: SimulationReport) -> Dict[str, Any]:
    return {
        "mode": report.mode.name,
        "aggregate": report.aggregate,
        "wall_time": report.wall_time,
        "sutra_count": len(report.executions),
    }


def print_summary(bundle: HybridSimulationBundle) -> None:
    summaries = {
        "serial": _summarize_report(bundle.serial),
        "concurrent": _summarize_report(bundle.concurrent),
        "parallel": _summarize_report(bundle.parallel),
    }
    print(json.dumps(summaries, indent=2))


def write_report(bundle: HybridSimulationBundle, path: Path) -> None:
    payload = bundle.to_dict()
    path.write_text(json.dumps(payload, indent=2))


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 29 Vedic sutras in serial, concurrent, and parallel modes "
            "for hybrid quantum-classical simulation workflows."
        )
    )
    parser.add_argument("value", type=float, help="Input value to feed into sutras")
    parser.add_argument(
        "--mode",
        default="hybrid",
        help="Execution mode: classical, quantum, hybrid, maya_illusion, sulba",
    )
    parser.add_argument("--precision", type=int, default=64, help="Numeric precision")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=128,
        help="Max iterations for recursive sutras",
    )
    parser.add_argument(
        "--include",
        help="Only run sutras containing this substring in their name",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override the worker count for concurrent/parallel runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the full JSON report",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    mode = _parse_mode(args.mode)
    bundle = run_hybrid_bundle(
        args.value,
        mode=mode,
        precision=args.precision,
        max_iterations=args.max_iterations,
        include=args.include,
        max_workers=args.max_workers,
    )
    print_summary(bundle)

    if args.output:
        write_report(bundle, args.output)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
