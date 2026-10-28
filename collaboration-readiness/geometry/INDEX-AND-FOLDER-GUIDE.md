# Equation index and readable project structure

## Implemented equation linkage

`geometry-index.template.json` is the unassigned template for the equation index read by `validate_readiness.py`. Its exact top-level fields are `schema_version`, `kind`, `task_id` and `equations`. The template has `kind: template`, a null task ID and an empty equation list. It cannot pass a production acceptance check.

A project index has `schema_version: 1`, `kind: project_record`, the signed contract's task ID and one entry for every signed equation. Each entry contains exactly `id`, `statement_path`, `implementation_paths`, `requirement_ids` and `check_ids`. These values must match the signed equation inventory. The index file and its digest must also be included in the source manifest and referenced by the signed contract. Missing, stale, extra or mismatched links are rejected.

`FOUNDATION_REGISTER.template.yaml` is JSON-compatible YAML with the same four top-level fields. Its project `equations` value carries the complete signed equation objects, including every requirement and category applicability decision. An equation requirement names its actual implementation paths and the bound executable assertions that assess it.

The source statements remain canonical. The index provides validated links rather than copied formulas or invented repository IDs. Passing those links requires the referenced files and checks; production mathematical acceptance additionally requires the actual full specification and sufficient domain checks.

## Broader retrieval graph design retained

The earlier template also described a future typed retrieval graph. That design remains available below; no executable production retrieval implementation is removed or claimed.

| Object | Fields and relationships to retain |
|---|---|
| Node | Stable ID, kind, canonical path, symbol, source revision, content SHA-256, aliases, geometry contract ID, and definition/formula/operator/kernel/test/evidence IDs. |
| Edge | Source ID, relation, destination ID, source revision and verification-evidence ID. |
| Geometry contract | Stable ID, mathematical definition, space kind, domain, metric, dimension, coordinate semantics, axis sizes, periodic axes, equivalence relation, neighbor multiplicity, orientation and operator-sign conventions, coefficient domain, storage precision, layout map/inverse and invariant-test IDs. |
| Index metadata | Repository identity, source revision, generation time, external index digest and retrieval configuration. |

The proposed node kinds are `definition`, `formula`, `operator`, `kernel`, `test`, `evidence` and `decision`. Proposed relations are `defines`, `implements`, `tests`, `supports`, `depends_on` and `supersedes`. A production graph needs explicit edge directions and endpoint-kind rules, unique stable IDs, validated symbols and paths, authenticated source scope, update/tombstone handling and reverse-dependency invalidation. A matching hash establishes bytes, not the correctness of a claim.

Coordinates require declared meanings, domains, units and whether facets are independent. Embedding positions are not exact mathematical coordinates. An unresolved metric is not an implicit Euclidean metric. Keep the index digest outside the bytes it hashes or define an explicit canonical hash scope.

No broader graph loader, populated retrieval corpus, search benchmark or device execution is supplied by the equation-index validator. Those remain explicit production work requiring their actual project inputs.

## Human-readable organization

Use the existing project's conventions. A conceptual map may include:

- `docs/foundations/` for source provenance, exact definitions and operator conventions.
- `docs/decisions/` for actual decisions, rejected alternatives and unresolved questions.
- `docs/architecture/` for dependency boundaries and execution models.
- `src/<domain>/` for the implementation and its required adapters.
- `tests/<domain>/` for exact, integration and behavioral checks.
- `benchmarks/` for reproducible workloads and configurations.
- `evidence/<revision-or-run-id>/` for immutable observed results.
- `metadata/` for consumed cross-reference records and their schemas.

Map existing files before moving anything. Preserve imports, build paths, URLs and instruction discovery. One formula should have one canonical source; crosscutting relationships belong in explicit links. A folder layout alone does not install an index or improve retrieval.
