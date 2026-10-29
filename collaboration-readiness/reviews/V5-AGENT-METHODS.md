# Fresh v4 instruction-method review

Historical review retained with its original attribution and observations. Its prescriptions are superseded by the current AGENTS.md and INTEGRITY_RULES.md wherever they permit approximations, transformed or replacement formulations, merely declared acceptance, or separate explanatory deliverables. These historical results are not results of the repaired validator.

Reviewed 2026-10-23 UTC. Target: `geometric-review/Refined-Project-Instructions`. Review only; bundle unchanged. Read the previous AGENT-METHODS-AUDIT before investigating; do not treat its v2 reproductions as new v4 defects. No configured agent trial runner was invoked, no account settings changed, and no scientific/product performance was measured.

## Useful results first

Keep v4's compact entrypoint, explicit overlay triggers, scientific-fidelity vetoes, provenance/invalidation rules, single integrator and exact-artifact checks. These already resolve most earlier recommendations. The useful v5 delta is **testable cross-host discovery, worker freshness on arrival, and an evaluation protocol that measures consistency rather than best-of-many success**. Do not expand the universal prompt into a platform manual.

### 1. P1: Portable content is not portable discovery

ACTIVATION.md correctly warns that host behavior differs but leaves consequential cases unnamed. OpenAI's current official page specifies one selected filename per project directory, root-to-working-directory discovery, a combined default 32 KiB budget, and discovery once per run. Splitting files does not automatically make a root-launched session discover arbitrary descendants. [Official OpenAI AGENTS guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

Current Claude documentation specifies direct AGENTS support from v2.1.277. A parent/project CLAUDE.local.md can suppress default AGENTS loading; AGENTS.override.md is not loaded directly. Markdown import expansion and ordinary prose links are different. Imports do not reduce context usage. [Official Claude memory documentation](https://code.claude.com/docs/en/memory).

**Exact addition to ACTIVATION.md after platform paragraphs:**

> Maintain one canonical project contract, with a host-specific adapter only where needed. A file-name convention, Markdown link, import directive and explicit file read are different mechanisms. Do not assume Codex overrides or import syntax work on another host. Record ancestor/local files that can suppress the intended entrypoint. Keep critical scientific constraints in the canonical entrypoint rather than solely in a host-specific override. After instructions change, use the host's documented refresh/new-session mechanism and repeat the applicable discovery fixture; a worker saying “loaded” is insufficient evidence.

**Exact addition to adoption step 6:**

> Include same-directory nonempty, empty and whitespace-only overrides; a competing ancestor/local instruction file; a chain crossing the configured document budget; and working-directory changes. Separate documented expected behavior from observed behavior and record host/version/configuration. Test the nested scientific restriction by an observable harmless decision, not only by reciting instruction names.

Fresh community lead: [Codex issue #54755](https://github.com/openai/codex/issues/54755), opened October 17 with an October 22 corroborating report, claims an empty project override suppresses the base file on CLI 0.163.0, unlike global fallback. This conflicts with an intuitive reading of official “skips empty files.” **Lower-confidence version-specific report, not a verified property of this environment.** Do not import its diagnostic or workaround blindly. Its value is the added negative fixture. The old #36371 budget issue is not a new finding.

### 2. P1: Delegation must carry current constraints, not assume inheritance

OPERATING_PROCEDURES already assigns write ownership but does not require workers to verify the current instruction/foundation generation on arrival. Official Claude subagent documentation distinguishes forks from fresh contexts; some subagent types omit project instructions. It explicitly recommends restating constraints that must reach a delegated worker. [Official subagent documentation](https://code.claude.com/docs/en/sub-agents).

**Exact addition to Parallel ownership:**

> Each delegation carries the active goal, critical no-substitution constraints, relevant foundation/overlay IDs, approved instruction revision or hashes, input revision, ownership, prohibited writes, acceptance criteria and result location. Before consequential work, the worker reads the applicable project-visible files and reports any mismatch with its supplied generation; pause affected work for reconciliation rather than accepting whichever text is newest. Do not assume parent conversation, memory, skills or approvals are inherited. The integrator checks returned base revision, changed files, interface changes and evidence before accepting the result. Assign one writer to shared status/memory indexes; workers return proposed updates instead of racing to overwrite them.

New lower-confidence lead: [Claude issue #88886](https://github.com/anthropics/claude-code/issues/88886) reports stale inherited instruction/memory snapshots in version 2.1.237. This was not reproduced here and should not be generalized to current versions. The discussion's “disk wins” workaround is **not adopted**: a newer disk file could be unreviewed or compromised. Compare against the approved generation and actual user scope instead.

Official team guidance supports disjoint work and notes coordination/token overhead; it does not justify a fixed agent count or guaranteed acceleration. [Official agent teams documentation](https://code.claude.com/docs/en/agent-teams).

### 3. P1: Memory prose needs an easy-to-use claim example

PROJECT_MEMORY.md's requirements are strong; `project-memory.template.yaml` has only an empty claims list. The existing text accurately warns that invalidation fields are documentation requirements unless validated. Do not claim this is a newly reproduced schema vulnerability. Make compliance practical without pretending the example enforces itself.

**Exact addition to PROJECT_MEMORY.md:**

> For a material claim use: claim_id; statement; epistemic_status (observed/reported/inferred/proposed); source_reference and source_revision/hash where available; event_time; observed_at; applies_to_revision; observed_environment; supersedes; revalidate_when. Unknown values remain null. For reported claims identify who reported them; for inferences link premises and reasoning. An evidence file's modified time is not the observation time. A newer claim does not supersede an older one solely because it is newer: record the reason and approving scope. Workers propose memory changes to the designated status owner. Keep revocations and invalidations linked to affected claims, not only appended as prose elsewhere.

This is a project design recommendation, not a vendor memory guarantee. Add a populated **example separate from the blank template**, and document which fields the actual validator checks. Do not fill unknown real-world facts merely to satisfy an example.

### 4. P1: Evaluate v4 versus v5 and consistency, not only v2 versus candidate

BEHAVIOURAL_EVALUATION currently hardcodes v2 as comparator. That cannot isolate the effect of this revision. Keep v2 as optional historical reference, but use v4 as the immediate baseline.

**Exact replacement of opening comparison sentence:**

> Compare the immediately preceding accepted instruction bundle and this candidate on identical repository snapshots, tasks, tools, model/configuration, initial memory state and budgets. Record both bundle hashes and revisions; optionally include older historical baselines separately. Report all trials, including interrupted, blocked and failed trials, with reasons.

**Exact addition to scoring:**

> Predeclare acceptance criteria and evaluators before trials. Separate development fixtures from held-out regression tasks. Report first-attempt success and all-trials consistency, not only whether one of several attempts succeeded. Preserve veto failures per trial. Inspect disagreements between graders and outcomes; do not silently discard inconvenient cases. Measure elapsed time and cost only for comparable, fidelity-preserving work, including coordination and verification overhead. A grader's access to the candidate's own rationale must not replace independent artifact/state checks.

This extends the official distinction between pass-at-k and all-k consistency, outcome versus claimed success, and repeated trials. [Anthropic agent-evaluation guidance](https://www.anthropic.com/research/demystifying-evals-for-ai-agents).

## Small proposed fixture specification (not executed)

Use a disposable repository with no secrets, no external tools and no deployment credentials. Fixture input includes root AGENTS, nested AGENTS, optional override/local adapter, and a tiny exact rational expression with a preservation constraint. Request a proposed optimization and structured outcome. The grader checks whether the agent preserves the rational expression, refuses an unapproved float/downsample substitution, selects the relevant overlay, and reports loader uncertainty correctly. For loading-only diagnostics, generate random markers solely inside public fixture guidance, never in the question; record observed retrieval separately from behavioral adherence. Missing markers are not definitive proof of non-delivery. Never request hidden prompts or inspect private session stores.

Cases: root/nested launch; empty override; whitespace override; competing local adapter; oversized chain; change approved foundation generation between parent start and worker dispatch; compaction/resume with revoked stale approval; concurrent worker proposals to a single-owner memory index. Freeze expectations per documented host/version, preserve unknown outcomes, then run only when a real authorized runner is configured.

## Project-goal consequences

- Cymatica/Faraday: worker brief must carry exact equation/domain/boundary/precision identity, not a generic “make faster” task.
- Vedic/GRVQ: current approved definition and proof-status IDs travel with every mathematical delegation; no worker may invent missing provenance.
- ETERNAL: runtime motion/audio judgments remain separate from build/import success and authored tuning remains distinct from physical validation.
- Geometry-first: preserve topology/operator/source IDs across storage or indexing changes; do not equate a file layout with model truth.
- Monetisation: evaluate measured customer/comparator evidence, not confident prose or instruction compliance alone.

Outcome: actionable research review completed. Adoption, discovery trials, behavioral trials and project runtime validation remain **not run**, not passed or blocked by an access denial.

## Proposed discovery receipt (JSON-compatible YAML)

This is a documentation template, not proof of activation or a proposed addition to an existing validator without corresponding schema/migration work. Each evidence-status field records `observed`, `reported`, or `unknown`; a reported vendor behavior is not an observed local behavior. Record only authorized project-visible settings; never enumerate private prompt text.

```json
{
  "schema_version": 1,
  "receipt_state": "template",
  "recorded_at": null,
  "runtime": {"name": null, "version": null, "evidence_status": "unknown", "evidence": null},
  "project_root": null,
  "working_directory": null,
  "source_revision": null,
  "dirty_state": "unknown",
  "approved_contract": {"revision": null, "sha256": null, "approval_reference": null},
  "loader": {
    "documentation_url": null,
    "documentation_checked_at": null,
    "reported_limit_bytes": null,
    "observed_configuration_limit_bytes": null,
    "configuration_evidence_status": "unknown",
    "mode": null
  },
  "instruction_files": [],
  "manually_read_overlays": [],
  "unresolved_conflicts": [],
  "fixture_results": [],
  "activation_observation": "unknown",
  "revalidate_when": ["runtime changes", "launch directory changes", "instruction files change", "relevant settings change"],
  "supersedes_receipt": null
}
```

Instruction-file entry shape:

```json
{
  "path": null,
  "sha256": null,
  "scope": null,
  "existence": "unknown",
  "discovery_expectation": null,
  "expectation_source": null,
  "host_loading": "unknown",
  "loading_evidence_status": "unknown",
  "loading_evidence": null,
  "manually_read": false,
  "observed_at": null
}
```

Fixture-result entry shape:

```json
{
  "fixture_id": null,
  "host_version": null,
  "fixture_input_sha256": null,
  "expected_behavior_source": null,
  "outcome": "not_run",
  "observed_behavior": null,
  "evidence_reference": null,
  "evidence_sha256": null,
  "run_at": null,
  "limitations": []
}
```

An observed file and even an observed marker do not prove compliance. Keep `activation_observation` unknown until an actual supported-host procedure yields evidence; scope any positive statement to those tested cases and that exact runtime/configuration. New receipts supersede old receipts explicitly; never retrofit a previous host observation onto a new version.
