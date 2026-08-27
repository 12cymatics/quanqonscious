# HSQCP Report and Bundle Schema (v1)

## Purpose
This document defines the stable machine-readable artifact contract emitted by
`hybrid_sutra_platform.py` and `sutra_simulator.py` for the deterministic
29-sutra hybrid quantum-classical runtime.

## Doctrine Fidelity Classification
- `exact-symbolic-core`: canonical doctrine-level execution and report payloads.
- `hybrid-bridge`: numerical or bridge modules that remain integrated but are
  not classified as exact-symbolic core.
- `demo-adapter`: UI/reporting wrappers that do not define runtime semantics.

The current simulator and bundle payloads emit `exact-symbolic-core` as the
required doctrine fidelity marker.

## Bundle Schema
Top-level version: `hsqcp.bundle.v1`

Required fields:
- `schema_version` (string)
- `report_schema_version` (string)
- `benchmark_protocol_version` (string)
- `doctrine_fidelity` (string)
- `generated_at_utc` (RFC3339 timestamp string)
- `initial_value` (exact rational or structured value)
- `sutra_names` (array of strings)
- `runtime_scalars` (object containing scalar governance controls)
- `runtime_environment` (object with python/platform/commit provenance)
- `serial` (SimulationReport)
- `concurrent` (SimulationReport)
- `parallel` (SimulationReport)

## SimulationReport Schema
Report version: `hsqcp.report.v1`

Required fields:
- `schema_version` (string)
- `doctrine_fidelity` (string)
- `generated_at_utc` (RFC3339 timestamp string)
- `mode` (string; one of `CLASSICAL|QUANTUM|HYBRID|...` enum names)
- `initial_value` (exact rational or structured value)
- `aggregate` (mode-specific aggregate output)
- `wall_time_ns` (integer nanoseconds)
- `executions` (array of execution records)

Execution record fields:
- `name` (sutra function name)
- `elapsed_ns` (integer nanoseconds)
- `output` (sutra output payload)

## Compatibility Policy
- **Minor, backward-compatible extensions:** add optional fields only.
- **Breaking changes:** bump `hsqcp.bundle.vN` and `hsqcp.report.vN` in lockstep.
- **Validation gate:** any bundle writer must run schema invariant checks before
  persisting artifacts.


## Signature Vault Index Contract
When persisted with the platform vault writer, each bundle emits an index.jsonl entry (a generated artifact, not a tracked file) containing:
- `bundle_id` (sha256 of canonicalized payload),
- `run_label`,
- `benchmark_protocol_version`,
- `report_schema_version`,
- `generated_at_utc`,
- `path` (bundle filename),
- `sutra_count`,
- `initial_value`,
- `sutra_inventory_hash` (sha256 over bundle sutra inventory).


## Vault Audit
Use `hybrid_sutra_platform.py --audit-vault <path>` to validate that each indexed
bundle exists, passes schema validation, and matches its canonical payload hash.


## Reproducibility Manifest Fields
When generated, manifest payloads include:
- `sutra_inventory_hash` (sha256 over sorted sutra names),
- `runtime_scalars`,
- `runtime_environment`,
- `seeds`,
- `seed_count`,
- version and doctrine markers aligned with bundle/report schemas.


## Semantic Validation Rules
Bundle validation enforces that each serial/concurrent/parallel report uses the
report schema marker, doctrine fidelity marker, execution count equal to sutra
inventory length, and execution-name set equality against `sutra_names`.
