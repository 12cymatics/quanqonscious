# Discover, adopt and verify instructions

Research snapshot: 23 October 2026. This is a procedure, not evidence of activation.

## Discovery receipt
Record the runtime and version, project root, working directory, project-visible instruction files and scopes, relevant overlay, files read manually, reported loader limits and unresolved conflicts. If discovery cannot be inspected, mark it unknown. Do not reveal private system prompts, unrelated personal memory or credentials. A local file's existence is not proof that an assistant loaded it.

Codex documents a global and project-root-to-working-directory discovery chain, with override behaviour and a default combined project-document limit. Check the actual installed configuration rather than assuming every AGENTS.md, companion file or nested directory is active. Source: https://learn.chatgpt.com/docs/agent-configuration/agents-md

Claude Code has its own version- and setting-dependent loading behaviour. Current documentation describes direct AGENTS.md support under particular conditions; other configurations use CLAUDE.md imports. Inspect the version and existing configuration before choosing an adapter, and avoid duplicate imports. Source: https://code.claude.com/docs/en/memory

For other assistants and Sites, use their supported project-instruction mechanism and record what was actually applied. A repository YAML file does not change account-level instructions, permissions or service settings.

## Adoption sequence
1. Compare this bundle with existing instructions. Preserve project-specific constraints and resolve contradictions explicitly.
2. Keep the short AGENTS.md at the intended entrypoint. Link task-specific procedures instead of duplicating them everywhere.
3. Fill a project record with actual source, environment, inputs, ownership and completion criteria. Leave unknown values unknown.
4. Choose gate applicability from the real deliverable and project. A failed relevant gate cannot be reclassified as irrelevant to obtain a pass.
5. Bind reviewed commands to the actual project. Neither validator executes commands from the record.
6. Test a harmless fixture in a fresh root session and a nested-directory session. Check that the intended overlay and prohibitions affect behaviour, not merely that the agent says it read them.
7. Compare representative task outcomes against the previous instructions before claiming a speed or reliability improvement.

## Reusable skills
Candidate procedures are strict model change, exact-artifact release, interrupted-operation reconciliation and live-DAW control. Create a skill only after its inputs, environment, positive/negative triggers and acceptance tests are known. Use concise metadata for selection and load detailed steps on demand. Do not install a collection of untested skills solely to increase apparent coverage. Source: https://learn.chatgpt.com/docs/build-skills

## Invalidation
Recheck discovery after changing the runtime, launch directory, override files or relevant settings. Recheck project facts after source changes and hardware facts after reconnection. Retain the prior receipt as history; do not silently treat it as current.

## Edition 5 host-specific checks

Use discovery-receipt.template.yaml to distinguish documented expectations, inspected files, manual reads and observed host behaviour. It is a documentation template, not a validator or proof of activation.

Codex selects at most one project instruction filename per directory and documents a default combined 32 KiB budget. Its discovery chain follows the launch working directory; do not assume a root-launched task automatically loads every descendant's restrictions. Verify the actual settings. Source: https://learn.chatgpt.com/docs/agent-configuration/agents-md

Claude's current documentation dates native AGENTS support to v2.1.277 and describes settings and ancestor/local files that affect loading. CLAUDE.local.md can suppress default AGENTS loading; AGENTS.override.md is not loaded directly. Do not transfer Codex override conventions or import syntax blindly. Source: https://code.claude.com/docs/en/memory

Maintain one canonical contract and only the necessary host adapter. A prose link, import and manual read are different mechanisms. Keep critical scientific constraints in the canonical entrypoint. After changes, use the documented refresh or new-session mechanism and repeat applicable fixtures. Compare the approved generation, not whichever file happens to be newest.

Test nonempty, empty and whitespace-only overrides; competing ancestor/local files; an over-budget chain; nested launch directories; and a worker dispatched after instructions change. Observe a harmless fidelity-preserving decision as well as loader diagnostics. Community reports of empty-override and stale-worker behaviour motivate fixtures but have not been reproduced here. See reviews/V5-AGENT-METHODS.md for sources, versions and limitations.
