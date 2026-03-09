# HSQCP Benchmark Protocol (v1)

Protocol ID: `hsqcp.benchmark.v1`

## Objective
Provide a deterministic, reproducible benchmark procedure for comparing serial,
concurrent, and parallel execution over the same sutra inventory and input
payloads.

## Canonical Procedure
1. Select a fixed input seed set (scalars or structured payloads).
2. For each seed, run all sutras in:
   - serial mode,
   - concurrent mode,
   - parallel mode.
3. Capture full bundle output with schema versions intact.
4. Compute per-mode summary statistics:
   - wall time,
   - execution count,
   - aggregate output consistency checks.
5. Persist bundle artifacts to a signature vault index.
6. Record runtime scalar governance values with each bundle (`grvq_beta_scale`, `grvq_gamma_scale`,
   `engine_rebuild_interval`, `engine_drift_threshold`).
7. Record runtime environment provenance (python version, platform, commit hash).

## Required Invariants
- Same sutra inventory for all three modes per run.
- Stable schema markers (`hsqcp.bundle.v1`, `hsqcp.report.v1`).
- Explicit doctrine fidelity tag per report.
- No placeholder or pseudo execution paths.

## Recommended Baseline Seed Set (Exact Rationals)
- 1
- 1618/1000
- 2
- 31415926535/10000000000

## Storage Guidance
Persist artifacts as immutable JSON payloads keyed by:
- benchmark protocol version,
- runtime commit hash,
- input seed,
- execution timestamp.

This enables longitudinal regression checks and partner-facing audit trails.


## CLI Execution Pattern
Use `hybrid_sutra_platform.py` with `--benchmark-seeds` and optional `--vault-dir`
to execute and persist a complete tri-modal benchmark suite with explicit scalar
settings and indexed artifacts.
