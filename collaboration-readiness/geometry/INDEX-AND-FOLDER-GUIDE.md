# Registry and readable folder guide

`geometry-index.template.json` is an empty design template, not a JSON Schema, live index or validation claim. `record_shapes` describes fields; its null examples are not nodes. Do not count them as populated evidence. No current repository paths or IDs are fabricated.

On adoption define strict schemas: unique nonempty stable IDs, supported node kinds (`definition`, `formula`, `operator`, `kernel`, `test`, `evidence`, `decision`), allowed edge types (`defines`, `implements`, `tests`, `supports`, `depends_on`, `supersedes`), and endpoint-kind rules. Define each relation's direction explicitly. Validate paths under the authorized repository root, symbol existence, source revisions, content SHA256, resolved references, required semantics and stale evidence. Stable ID identity survives file renames; revisions/hashes do not. Document rename/tombstone handling and reverse dependency invalidation. Evidence may support a claim without proving it. Avoid circular reasoning in evidence graphs even when software dependency cycles are legitimate.

`index_sha256` needs a declared hash scope excluding its own field, or keep the digest in a separate receipt; a naive self-hash is not meaningful. `coordinate_semantics` must state what each coordinate means, whether exact or approximate, allowed values, unit/scale, independent vs constrained facets, and how it maps to actual source entities. Do not use positions in an embedding as exact mathematical coordinates. A `null` metric means unresolved, not implicit Euclidean distance.

Proposed human-readable map for a new project, not a migration instruction or assertion about current paths:

- `docs/foundations/`: source provenance, exact definitions, operator conventions.
- `docs/decisions/`: approved choices, rejected alternatives and open questions.
- `docs/architecture/`: dependency boundaries and execution model.
- `src/<domain>/`: existing language/package conventions; separate domain logic from presentation and hardware adapters when appropriate.
- `tests/<domain>/`: exact, numerical, integration and behavioral test groups.
- `benchmarks/`: reproducible workloads and machine-readable configurations.
- `evidence/<revision-or-run-id>/`: immutable reports and manifests, subject to storage policy.
- `metadata/`: validated cross-reference index and schemas.

Prefer mapping the existing repository into this conceptual model over moving files. Preserve public imports, build scripts, URL references and instruction discovery. Crosscutting relationships belong in a graph or registry; do not duplicate a formula into several folders. Keep source text canonical and make index entries point to it. Deployment requires an actual loader, update process and validation integration; this document installs none.
