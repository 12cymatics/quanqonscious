"""Static contract validator for HSQCP runtime/schema invariants.

This validator avoids runtime dependency loading and verifies that core files keep
required schema/version/field invariants synchronized.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _require(text: str, needle: str, source: str) -> None:
    if needle not in text:
        raise AssertionError(f"Missing required token in {source}: {needle}")


def validate_platform_contract() -> None:
    text = _read("hybrid_sutra_platform.py")
    required = [
        'BUNDLE_SCHEMA_VERSION = "hsqcp.bundle.v1"',
        'BENCHMARK_PROTOCOL_VERSION = "hsqcp.benchmark.v1"',
        '"runtime_scalars"',
        '"runtime_environment"',
        'def parse_exact_rational(raw: str) -> Fraction:',
        'def validate_bundle_payload(payload: Dict[str, Any]) -> None:',
        'def compute_bundle_id(payload: Dict[str, Any]) -> str:',
        'def persist_signature_bundle(',
        'def audit_signature_vault(vault_dir: Path) -> Dict[str, Any]:',
        'def run_benchmark_suite(',
        'def compute_sutra_inventory_hash(sutra_names: Sequence[str]) -> str:',
        'def build_benchmark_matrix(bundles: Sequence[HybridSimulationBundle]) -> Dict[str, Any]:',
        'def write_reproducibility_manifest(',
        'def validate_bundle_semantics(payload: Dict[str, Any]) -> None:',
        'def validate_benchmark_bundle_set(bundles: Sequence[HybridSimulationBundle]) -> None:',
    ]
    for token in required:
        _require(text, token, "hybrid_sutra_platform.py")


def validate_simulator_contract() -> None:
    text = _read("sutra_simulator.py")
    required = [
        'REPORT_SCHEMA_VERSION = "hsqcp.report.v1"',
        'DOCTRINE_FIDELITY_TAG = "exact-symbolic-core"',
        'wall_time_ns: int = 0',
        'elapsed_ns: int',
        '"wall_time_ns"',
        '"elapsed_ns"',
    ]
    for token in required:
        _require(text, token, "sutra_simulator.py")


def validate_docs_contract() -> None:
    schema = _read("docs/hsqcp_report_schema.md")
    protocol = _read("docs/hsqcp_benchmark_protocol.md")

    schema_required = [
        "hsqcp.bundle.v1",
        "hsqcp.report.v1",
        "wall_time_ns",
        "elapsed_ns",
        "runtime_environment",
        "runtime_scalars",
        "Semantic Validation Rules",
    ]
    for token in schema_required:
        _require(schema, token, "docs/hsqcp_report_schema.md")

    protocol_required = [
        "hsqcp.benchmark.v1",
        "serial mode",
        "concurrent mode",
        "parallel mode",
        "Exact Rationals",
        "Benchmark Matrix Artifact",
        "Reproducibility Manifest",
        "Benchmark Set Consistency",
    ]
    for token in protocol_required:
        _require(protocol, token, "docs/hsqcp_benchmark_protocol.md")


def main() -> None:
    validate_platform_contract()
    validate_simulator_contract()
    validate_docs_contract()
    print("HSQCP static contract validation passed")


if __name__ == "__main__":
    main()
