"""Hybrid Vedic Sutra platform entrypoint.

This module orchestrates the 29 sutras in serial, concurrent, and parallel
execution styles so the same input can be stress-tested across scheduling
strategies. The output is designed to be directly consumable by a
hybrid quantum-classical workflow and can be serialized as JSON for
integration with external services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from sutra_simulator import (
    DOCTRINE_FIDELITY_TAG,
    REPORT_SCHEMA_VERSION,
    HybridQuantumClassicalSimulator,
    SimulationReport,
)
from sutra_repository import SutraContext, SutraMode, SutraRepository

BUNDLE_SCHEMA_VERSION = "hsqcp.bundle.v1"
BENCHMARK_PROTOCOL_VERSION = "hsqcp.benchmark.v1"
DEFAULT_BENCHMARK_SEEDS = ("1", "1618/1000", "2", "31415926535/10000000000")


def _resolve_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def runtime_environment_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _resolve_git_commit(),
    }


def parse_exact_rational(raw: str) -> Fraction:
    value = raw.strip()
    if not value:
        raise ValueError("Empty rational value is not allowed")
    return Fraction(value)


@dataclass(frozen=True)
class RuntimeScalarConfig:
    """Explicit scalar governance configuration for runtime execution."""

    grvq_beta_scale: Fraction = Fraction(1, 1)
    grvq_gamma_scale: Fraction = Fraction(1, 1)
    engine_rebuild_interval: int = 64
    engine_drift_threshold: Fraction = Fraction(1, 1_000_000)

    def __post_init__(self) -> None:
        if self.grvq_beta_scale <= 0:
            raise ValueError("grvq_beta_scale must be > 0")
        if self.grvq_gamma_scale <= 0:
            raise ValueError("grvq_gamma_scale must be > 0")
        if self.engine_rebuild_interval <= 0:
            raise ValueError("engine_rebuild_interval must be > 0")
        if self.engine_drift_threshold <= 0:
            raise ValueError("engine_drift_threshold must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grvq_beta_scale": str(self.grvq_beta_scale),
            "grvq_gamma_scale": str(self.grvq_gamma_scale),
            "engine_rebuild_interval": self.engine_rebuild_interval,
            "engine_drift_threshold": str(self.engine_drift_threshold),
        }


@dataclass
class HybridSimulationBundle:
    """Aggregated results across serial, concurrent, and parallel runs."""

    initial_value: Any
    sutra_names: List[str]
    serial: SimulationReport
    concurrent: SimulationReport
    parallel: SimulationReport
    runtime_scalars: RuntimeScalarConfig

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "doctrine_fidelity": DOCTRINE_FIDELITY_TAG,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "initial_value": self.initial_value,
            "sutra_names": list(self.sutra_names),
            "runtime_scalars": self.runtime_scalars.to_dict(),
            "runtime_environment": runtime_environment_metadata(),
            "serial": self.serial.to_dict(),
            "concurrent": self.concurrent.to_dict(),
            "parallel": self.parallel.to_dict(),
        }
        validate_bundle_payload(payload)
        return payload


@dataclass
class PersistedBundle:
    bundle_id: str
    path: Path


def validate_bundle_payload(payload: Dict[str, Any]) -> None:
    """Validate top-level bundle schema invariants for stable artifacts."""

    required = (
        "schema_version",
        "report_schema_version",
        "benchmark_protocol_version",
        "doctrine_fidelity",
        "generated_at_utc",
        "initial_value",
        "sutra_names",
        "runtime_scalars",
        "runtime_environment",
        "serial",
        "concurrent",
        "parallel",
    )
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required bundle field: {key}")

    if payload["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected bundle schema version: {payload['schema_version']}"
        )
    if payload["report_schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unexpected report schema version: {payload['report_schema_version']}"
        )
    if payload["benchmark_protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError(
            "Unexpected benchmark protocol version: "
            f"{payload['benchmark_protocol_version']}"
        )
    if payload["doctrine_fidelity"] != DOCTRINE_FIDELITY_TAG:
        raise ValueError(
            f"Unexpected doctrine fidelity tag: {payload['doctrine_fidelity']}"
        )
    if len(payload["sutra_names"]) == 0:
        raise ValueError("Bundle must contain at least one sutra")


def _parse_mode(value: str) -> SutraMode:
    try:
        return SutraMode[value.upper()]
    except KeyError as exc:
        options = ", ".join(m.name.lower() for m in SutraMode)
        raise ValueError(f"Unknown mode '{value}'. Options: {options}") from exc


def _build_context(
    mode: SutraMode,
    *,
    precision: int,
    max_iterations: int,
    runtime_scalars: RuntimeScalarConfig,
) -> SutraContext:
    context = SutraContext(
        mode=mode,
        precision=precision,
        max_iterations=max_iterations,
        parallel=True,
    )
    setattr(context, "grvq_beta_scale", runtime_scalars.grvq_beta_scale)
    setattr(context, "grvq_gamma_scale", runtime_scalars.grvq_gamma_scale)
    setattr(context, "engine_rebuild_interval", runtime_scalars.engine_rebuild_interval)
    setattr(context, "engine_drift_threshold", runtime_scalars.engine_drift_threshold)
    return context


def _apply_filter(names: Iterable[str], *, include: Optional[str]) -> List[str]:
    if include is None:
        return list(names)
    lowered = include.lower()
    return [name for name in names if lowered in name.lower()]


def compute_bundle_id(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_hybrid_bundle(
    value: Any,
    *,
    mode: SutraMode,
    precision: int,
    max_iterations: int,
    runtime_scalars: Optional[RuntimeScalarConfig] = None,
    include: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> HybridSimulationBundle:
    """Run serial, concurrent, and parallel suites for the full sutra set."""

    scalar_config = runtime_scalars or RuntimeScalarConfig()
    context = _build_context(
        mode,
        precision=precision,
        max_iterations=max_iterations,
        runtime_scalars=scalar_config,
    )
    name_filter = None
    if include:
        filtered_names: Set[str] = set(
            _apply_filter(SutraRepository(context).list_sutras(), include=include)
        )
        name_filter = lambda name: name in filtered_names

    simulator = HybridQuantumClassicalSimulator(
        context=context,
        max_workers=max_workers,
        sutra_filter=name_filter,
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
        runtime_scalars=scalar_config,
    )


def run_benchmark_suite(
    *,
    mode: SutraMode,
    precision: int,
    max_iterations: int,
    seeds: Iterable[Fraction] = tuple(parse_exact_rational(x) for x in DEFAULT_BENCHMARK_SEEDS),
    runtime_scalars: Optional[RuntimeScalarConfig] = None,
    include: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> List[HybridSimulationBundle]:
    bundles: List[HybridSimulationBundle] = []
    for seed in seeds:
        bundles.append(
            run_hybrid_bundle(
                seed,
                mode=mode,
                precision=precision,
                max_iterations=max_iterations,
                runtime_scalars=runtime_scalars,
                include=include,
                max_workers=max_workers,
            )
        )
    return bundles


def persist_signature_bundle(
    bundle: HybridSimulationBundle,
    vault_dir: Path,
    *,
    run_label: str,
) -> PersistedBundle:
    """Persist a validated bundle into a signature-vault directory."""

    payload = bundle.to_dict()
    bundle_id = compute_bundle_id(payload)

    vault_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = vault_dir / f"{bundle_id}.json"
    bundle_path.write_text(json.dumps(payload, indent=2))

    index_entry = {
        "bundle_id": bundle_id,
        "run_label": run_label,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": payload["generated_at_utc"],
        "path": str(bundle_path.name),
        "sutra_count": len(payload["sutra_names"]),
        "initial_value": payload["initial_value"],
    }
    index_path = vault_dir / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_entry) + "\n")

    return PersistedBundle(bundle_id=bundle_id, path=bundle_path)


def load_bundle_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    validate_bundle_payload(payload)
    return payload


def verify_bundle_file(path: Path) -> str:
    payload = load_bundle_payload(path)
    expected = compute_bundle_id(payload)
    if path.stem != expected:
        raise ValueError(
            f"Bundle filename hash mismatch: file stem '{path.stem}' != computed '{expected}'"
        )
    return expected


def audit_signature_vault(vault_dir: Path) -> Dict[str, Any]:
    index_path = vault_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Signature vault index not found: {index_path}")

    lines = [line for line in index_path.read_text().splitlines() if line.strip()]
    checked = 0
    for line in lines:
        entry = json.loads(line)
        bundle_path = vault_dir / entry["path"]
        bundle_id = verify_bundle_file(bundle_path)
        if entry["bundle_id"] != bundle_id:
            raise ValueError(
                f"Index bundle_id mismatch for {bundle_path.name}: "
                f"index={entry['bundle_id']} computed={bundle_id}"
            )
        checked += 1

    return {
        "vault_dir": str(vault_dir),
        "indexed_bundles": len(lines),
        "verified_bundles": checked,
    }


def _summarize_report(report: SimulationReport) -> Dict[str, Any]:
    return {
        "mode": report.mode.name,
        "aggregate": report.aggregate,
        "wall_time_ns": report.wall_time_ns,
        "sutra_count": len(report.executions),
        "schema_version": report.schema_version,
        "doctrine_fidelity": report.doctrine_fidelity,
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


def _parse_benchmark_seeds(raw: str) -> List[Fraction]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--benchmark-seeds must contain at least one rational seed")
    return [parse_exact_rational(item) for item in values]


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 29 Vedic sutras in serial, concurrent, and parallel modes "
            "for hybrid quantum-classical simulation workflows."
        )
    )
    parser.add_argument("value", type=str, help="Input rational value to feed into sutras (e.g. 1618/1000)")
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
    parser.add_argument(
        "--vault-dir",
        type=Path,
        help="Optional signature-vault directory for indexed bundle persistence",
    )
    parser.add_argument(
        "--run-label",
        default="manual",
        help="Run label recorded in signature-vault index entries",
    )
    parser.add_argument(
        "--benchmark-seeds",
        help="Comma-separated seed list to execute benchmark suite, e.g. 1.0,1.618,2.0",
    )
    parser.add_argument("--grvq-beta-scale", type=str, default="1")
    parser.add_argument("--grvq-gamma-scale", type=str, default="1")
    parser.add_argument("--engine-rebuild-interval", type=int, default=64)
    parser.add_argument("--engine-drift-threshold", type=str, default="1/1000000")
    parser.add_argument(
        "--audit-vault",
        type=Path,
        help="Audit an existing signature vault and verify bundle/index integrity",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    if args.audit_vault:
        audit = audit_signature_vault(args.audit_vault)
        print(json.dumps(audit, indent=2))
        return

    mode = _parse_mode(args.mode)
    runtime_scalars = RuntimeScalarConfig(
        grvq_beta_scale=parse_exact_rational(args.grvq_beta_scale),
        grvq_gamma_scale=parse_exact_rational(args.grvq_gamma_scale),
        engine_rebuild_interval=args.engine_rebuild_interval,
        engine_drift_threshold=parse_exact_rational(args.engine_drift_threshold),
    )

    if args.benchmark_seeds:
        seeds = _parse_benchmark_seeds(args.benchmark_seeds)
        bundles = run_benchmark_suite(
            mode=mode,
            precision=args.precision,
            max_iterations=args.max_iterations,
            seeds=seeds,
            runtime_scalars=runtime_scalars,
            include=args.include,
            max_workers=args.max_workers,
        )
        for bundle in bundles:
            print_summary(bundle)
            if args.vault_dir:
                persisted = persist_signature_bundle(
                    bundle,
                    args.vault_dir,
                    run_label=args.run_label,
                )
                print(f"Persisted bundle {persisted.bundle_id} -> {persisted.path}")
        return

    bundle = run_hybrid_bundle(
        parse_exact_rational(args.value),
        mode=mode,
        precision=args.precision,
        max_iterations=args.max_iterations,
        runtime_scalars=runtime_scalars,
        include=args.include,
        max_workers=args.max_workers,
    )
    print_summary(bundle)

    if args.output:
        write_report(bundle, args.output)
        print(f"Report written to {args.output}")

    if args.vault_dir:
        persisted = persist_signature_bundle(
            bundle,
            args.vault_dir,
            run_label=args.run_label,
        )
        print(f"Persisted bundle {persisted.bundle_id} -> {persisted.path}")


if __name__ == "__main__":
    main()
