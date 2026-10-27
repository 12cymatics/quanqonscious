# Additive migration

Keep `validate_contracts.py --self-test` as the draft-consistency interface. Edition 3 deliberately rejects completion claims in that interface and strengthens lexical ownership checks; use the separate readiness record for assessed completion. Add `validate_readiness.py`, its tests, and `readiness.template.json`; do not silently reinterpret existing project-memory files.

Template consistency:
`python validate_readiness.py --record readiness.template.json --evidence-root . --ownership-root /absolute/project-root`

Assessed record consistency/readiness:
`python validate_readiness.py --record readiness.project.json --evidence-root ./evidence-root --ownership-root /absolute/project-root --readiness`

Set `kind` to `project_record`; select deliverable kind; retain unknown values until observed. Bind `subject_revision` to the actual artifact/source identifier (for non-Git documents, a stable content digest identifier is suitable). Assess all nine gates. Applicable successful or failed checks reference local evidence objects `{path, sha256, subject_revision}`. Paths are relative to the explicit evidence root; unknown/missing/stale evidence cannot pass. For N/A use `applicable:false`, `outcome:not_applicable`, and `applicability_review:{reason,reviewer}`. Profile-mandatory gates cannot be excluded. A deployment without conventional CI should use its remote-revision/platform-build evidence for `remote_revision_and_ci` and clearly state no repository CI is configured; the gate name is legacy, not a mandate to fabricate CI.

`task_state:complete` invokes readiness rules even without `--readiness`. Unknown applicability is acceptable only in unfinished/template records. Concurrent operations must be reconciled before readiness. Completion is scope-specific and is not publication permission.

Run `python -m unittest -v` in this directory. Tests use synthetic evidence expressly to demonstrate shape/integrity behavior, not project scientific validity. All Python source is comment-free as requested. Add project-specific attestation producers only after inspecting actual commands; the checker must never run commands read from untrusted records.

Known limits: no JSON Schema file yet; strictness is the Python validator. It allows extra descriptive keys but ignores them. Reviewer names and subject revisions are declarations, not authenticated identities; file hashes confirm identity, not relevance. Filesystem ownership normalization follows the current host; check case aliases on case-insensitive target hosts. Snapshot evidence into an immutable build artifact to avoid concurrent mutation between check and use. Gate evidence may legitimately share one integrated test report, so reuse is not rejected automatically.

Foundation traceability is mandatory for every profile: for infrastructure/UI work record scope and inherited foundation constraints, not a fabricated geometric derivation. Declare `domain` (general/cymatica/eternal/sites/solver) and boolean `mathematical_claims`; Cymatica, solver and proof work require mathematical integrity. These declarations must be checked by the integrator against actual scope. Readiness rejects cancelled, blocked and not-started records.

`--ownership-root` is required on the CLI and must identify the actual source checkout, independently of the evidence/log root. Ownership aliases and symlink containment are resolved there, not inside the log directory. The Python function accepts `ownership_root=None` only for existing callers whose source and evidence roots are intentionally identical; pass it explicitly otherwise.
