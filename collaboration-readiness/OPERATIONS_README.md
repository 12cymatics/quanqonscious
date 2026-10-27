# Agent, memory and workflow contracts

## Files and adoption
- AGENTS.md: shared operating instructions for agents and the integrator.
- PROJECT_MEMORY.md: project-visible status and evidence discipline.
- policy.yaml: declarative policy, explicitly draft and not automatically enforced.
- project-memory.template.yaml: empty status record; null means not established.
- validate_contracts.py: Python standard-library consistency validator.
- VALIDATION.json: the observed result of validating the supplied templates and rejecting 16 deliberately invalid cases.

Place the shared mathematical documents beside AGENTS.md in the final bundle: MATHEMATICAL_FOUNDATION.md and INTEGRITY_RULES.md. The integrator must check these links and merge with existing project instructions without overwriting them blindly.

policy.yaml and project-memory.template.yaml intentionally use the JSON-compatible subset of YAML 1.2. This makes them valid YAML while allowing Python's standard-library JSON parser to read them without installing a parser package. Retain JSON syntax when editing these files, or change the loader deliberately and test that change. A file extension does not imply GitHub Actions, an agent runner or a website will automatically consume it.

Run from this folder:

    python3 validate_contracts.py --self-test

A pass means only that the checked contract relationships and fields are consistent. It does not verify mathematics, provenance, historical accuracy, physical calibration, code correctness, permissions, deployment or actual agent compliance. The validator does not activate CI, inspect remote state or run project commands. The draft checker rejects lexical duplicate and parent/child ownership paths, traversal and globs; it does not resolve symlinks. The readiness checker resolves ownership against the explicitly supplied --ownership-root and evidence against --evidence-root; set the former to the actual source repository. Neither is a filesystem lock, and target-host case aliases require explicit review. It does not reject every unspecified extra field or implement a complete YAML parser. A separate foundation register written in native YAML is not read or validated by this script; validate that file separately with an appropriate YAML parser and its own domain checks.

## Project record schema
Keep the listed top-level fields from project-memory.template.yaml. Replace status with project_record when beginning a real record. For this draft-format validator, set task_state to not_started, in_progress, blocked or cancelled. Assessed completion uses readiness.template.json migrated into a project record; see MIGRATION.md. Unknown values remain null; dirty_state is unknown, clean or dirty. Source and build identifiers refer to actual inspected states, not proposed names. Candidate SHA-256, when known, is 64 lowercase hexadecimal characters.

Populate the collections with these minimum fields:

| Collection | Minimum entry fields | Meaning |
|---|---|---|
| inputs | id, location, version, sha256, units, provenance, limitations | Exact supplied inputs; unknown fields remain null. Do not claim a hash validates scientific accuracy. |
| foundation_records | id, contract_reference, derivation_reference, status, unresolved_requirements | Traceability to the companion mathematical contracts. |
| claims | text, kind, source, timestamp, time_basis; reasoning for inferred claims | kind is observed, reported, inferred or proposed. time_basis is event, observation, document_metadata or unknown. |
| ownership.assignments | owner, outcome, write_paths | File ownership and bounded worker result; assignments require a named integrator. |
| changes | description, source_revision, files, evidence | Changes actually made, not intentions. |
| checks | id, scope, outcome; evidence for passed/failed; reason for blocked/not_run | outcome is passed, failed, blocked or not_run. Include command, environment and source revision when applicable. |
| active_operations | id, target, state, last_observed_at, evidence, reconciliation_action | Track running or uncertain operations separately from completed checks. |
| blockers | description, affected_scope, required_action, evidence | Specific obstacles, not generic failure labels. |
| approvals | action, target, scope, source, status, limitations | An evidence record, not a mechanism for granting permission. Never store secrets. |

Timestamps, when known, use ISO 8601 with a timezone. In the separate readiness format, a task marked complete requires a completion criterion, decisions for every gate, revision-bound evidence, and no unresolved operations or blockers. For a gate genuinely outside scope, document applicability and review in the evidence record rather than pretending a test ran. Do not label an untested applicable capability complete.

The validator checks selected safety-critical fields and relationships, not every recommended subfield in this table. Human review of provenance, ownership boundaries, applicability and actual completion is still required.

## Connecting real workflows later
1. Identify the actual repository and read its existing build/test/deployment workflows.
2. Map each applicable policy gate to a real command or a documented manual verification. Record command provenance, expected outputs and the target environment.
3. Preserve existing aggregate checks and add regressions for changed behaviour.
4. Add CI only in the correct repository and only within the user's authorized scope. Validate the YAML and run it on an identified source revision.
5. Verify failures are surfaced and that blocked or unrun checks cannot masquerade as successful release gates.
6. Test the exact final archive after extraction, then report remote revision and CI status separately from deployment.

Do not flip the draft activation flags merely to satisfy a validator. This validator intentionally covers the supplied draft contract. A project-specific activated contract needs an explicit schema/version change and corresponding validation work, not a cosmetic assertion that enforcement exists.


Edition 3 adds validate_readiness.py and readiness.template.json. See MIGRATION.md. The older draft checker now refuses task_state=complete; it cannot be used as a completion receipt. Readiness verifies declared gate/evidence consistency, not the truth of scientific claims or authority to act.

Edition 5 adds DOMAIN-EVIDENCE.md for semantic review of each gate. The checker still does not inspect scientific evidence content or certify performance. No schema or acceptance criterion is silently expanded by this documentation.
