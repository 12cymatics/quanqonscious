# Critical validation audit — 23 October 2026

Historical review retained with its original attribution and observations. Its prescriptions are superseded by the current AGENTS.md and INTEGRITY_RULES.md wherever they permit approximations, transformed or replacement formulations, merely declared acceptance, or separate explanatory deliverables. These historical results are not results of the repaired validator.

## Verdict

The v2 prose preserves the right project intentions, but its executable validator is only a draft-shape checker. It does not establish completion, mathematical correctness, runnable packaging, or workflow enforcement. Keep ancient mathematics, geometry and supplied higher constructs as the project's traceable mathematical starting point; do not turn historical attribution into empirical proof or force an invented ancient derivation for infrastructure work.

### P0: false completion

`validate_contracts.py` accepts a complete record with one unrelated passed check whose evidence is `not-a-real-file`. It checks all *listed* checks, not all applicable required gates. The foundation register is unvalidated and not linked to completion. Minimal correction: independent readiness record, one decision for every gate, explicit applicability, mandatory profile gates, reviewed exclusions, revision-bound local evidence with existing-file and SHA-256 checks. A matching hash proves content identity, not that a test ran or that its result is honest. Independent review still assesses evidence relevance and validity.

### P1: adoption paradox and generic overconstraint

The current validator rejects installed/configured flags and non-draft status. This is appropriate for a draft-consistency command, not an activation or release validator. Preserve that old CLI and add a separately named readiness checker. Software requires real runtime evidence; a local mathematical proof document does not require a browser or deployment. Unknown applicability is not N/A. N/A decisions need named reviewer and reason, and cannot suppress profile-mandatory gates. Project-specific criteria must bind thresholds and acceptance tests before implementation; do not let a self-authored statement replace measurement.

### P1: ownership is literal, not structural

Original code checks identical strings only. `src` plus `src/module.py`, or aliased spellings of a path, pass. Normalize literal relative paths, reject traversal/globs, resolve symlink escapes, reject ancestor/descendant overlap. This is conservative scheduling hygiene, not a filesystem sandbox or OS lock. Live application control still needs a single owner. Windows/macOS case-insensitive aliases require host-aware treatment; Linux checks do not prove their absence.

### P1: CI wording must describe actual enforcement

A `.yml.template` is inactive. A manually dispatched workflow is not a required merge check. A successful skipped job can look green, and a required workflow skipped through filtering can remain pending; required checks must refer to the relevant commit. Adopt PR triggers/rulesets only when authorized and verified, including merge-group triggers if a merge queue is used. Never make contract-consistency success look like physics validation. See [GitHub status checks](https://docs.github.com/en/pull-requests/reference/status-checks) and [required check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).

Pin third-party actions to verified full commit SHAs and retain a reviewed update process, as recommended by [GitHub security guidance](https://docs.github.com/en/actions/reference/security/secure-use). Verified official v7.0.0 release mappings on the audit date:
- [checkout v7.0.0](https://github.com/actions/checkout/releases/tag/v7.0.0): `9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0`.
- [setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0): `5fda3b95a4ea91299a34e894583c3862153e4b97`.

The Python parser accepts JSON-subset YAML, not general YAML. Name this explicitly, or use `.json`; do not claim general YAML/schema validation. A separate native-YAML foundation register requires its own schema check and references from the assessed evidence.

## Project-specific acceptance evidence

### Cymatica/Faraday and exact kernels

Separate exact symbolic/kernel correctness, discretization verification, physical-model validation, and measured product performance. An exact arithmetic kernel does not make the fluid continuum solution exact. Supply exact rationals/algebraic objects at input; conversion after a binary float was created does not recover an intended decimal value. [SymPy best practices](https://docs.sympy.org/latest/explanation/best-practices.html) supports keeping exact symbolic numbers separate from deliberate numerical evaluation.

Proposed test bindings: exact round-trip serialization fixtures; forbidden float injection at each exact-kernel boundary; independent certificate verifier; dimensional and boundary-condition checks; manufactured or analytic solution where applicable; spatial and temporal refinement with reported observed order; iterative error distinguished from discretization error; correctly accounted forcing/dissipation; held-out experimental comparison with uncertainty. NASA distinguishes verification from agreement with observations in its [V&V tutorial](https://www.grc.nasa.gov/www/wind/valid/tutorial/tutorial.html), [grid convergence discussion](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html), and [validation assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/valassess.html). These are methodological references, not certification of this solver. Not every model exhibits smooth asymptotic convergence in every regime; record invalid assumptions rather than invent a pass.

### Browser and GPU claims

Record OS, browser build/channel, headless/headed mode, solver backend, adapter information, pipeline dispatch/readback evidence and tested feature limits. A GPU-rendered canvas does not prove GPU computation. Reject fallback adapters when physical GPU execution is mandatory; handle null adapter and device loss explicitly, without switching solvers behind the user's back. [WebGPU specification](https://www.w3.org/TR/webgpu/) documents fallback adapter detection and device-loss semantics. Proposed cases: no adapter, fallback-only adapter, device lost mid-step, precision incompatibility, paused worker, resume without simulation-time jump, actual compute output readback.

Playwright WebKit is not the installed Safari browser; record that distinction rather than claiming a Mac Safari pass from Linux WebKit. [Playwright browsers](https://playwright.dev/docs/browsers) explains its browser builds. Use real target-device validation before claiming target-device support, preserving unavailable hardware as blocked or untested.

### ETERNAL Godot build/runtime/audio

Editor import, script parse, export success, headless simulation tests, rendered interaction, and audible output are separate checks. Headless mode uses a Dummy audio driver; it cannot establish listening quality. [Godot command-line reference](https://github.com/godotengine/godot-docs/blob/master/tutorials/editor/command_line_tutorial.rst) documents this. The [official Web export documentation](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_web.html) also distinguishes audio modes, thread/header requirements, and gesture-triggered audio.

Bind export engine/template version; keep the full generated export file set. Test ordinary user input, pointer capture, camera/movement, feet sliding, airborne transitions, timestep variation, impact onset, resting-rubble silence, and maximum voice accumulation. Record waveform/onset/level measurements and an audition review, not merely event counts. Test required effects in the actual Web export: a Sample-mode sound is not evidence that Stream-mode effects work. Do not silently switch audio mode or thread requirements to obtain a pass.

### Sites

Proposed policy distinction: platform-native successful deployment acknowledgement and returned URL establish deployment status; separate behavior checks establish the app's features. Preserve deployment/version identity, audience and source revision. Browser QA is required for a claimed interaction result, not as ritual proof of every hosting acknowledgement. Verify intended data binding, save/reload persistence, accessible controls, responsive layout and error states when relevant. Do not call a source upload a deployment or a screenshot proof of persistence. No public Sites-specific reference was established in this audit, so these are proposed operational controls, not claims about a public platform contract.

### Solver monetisation

Before claiming advantage, freeze objective/constraints/precision, dataset and holdout split, baseline version/configuration, tuning budget, hardware, timing boundaries, compilation/preparation/verification costs, failure counts, and repeated-run spread. Record certificate verification costs separately from solver time. Buyer value and willingness to pay require their own evidence; benchmark speed is not revenue. [SPEC's run/reporting rules](https://www.spec.org/cpu2026/docs/runrules.html) exemplify reproducibility disclosure but do not govern this custom solver unless SPEC benchmarks are used. Do not imply SPEC certification or universal advantage. No price or legal recommendation is made.

## Community advice, separately weighted

[actionlint](https://github.com/rhysd/actionlint) is a maintained community static checker useful for workflow expression/type and shell issues. Recommendation: use a pinned/reviewed version during adoption, alongside actual workflow execution; it is not proof of required-check enforcement or scientific correctness. Confidence: useful tool capability from its maintainers, lower authority than GitHub platform semantics.

[Godot issue #100102](https://github.com/godotengine/godot/issues/100102) reports missing Web audio effects. Treat this as a regression scenario and practitioner warning, not universal proof that all current exports fail. The official Web export audio documentation is the authority for selecting relevant checks. No anonymous forum workaround or fallback setting is adopted without reproducing it against the pinned project version.

## Candidate implementation and limitations

`validate_readiness.py` and `readiness.template.json` implement a separate schema-v2 readiness attestation. `test_readiness.py` contains 25 passing tests including negative missing-gate/evidence/hash/revision/path/N-A/operation cases and positive proof-document exclusions. `test-results.txt` records the run. No external settings, project files, or deployments were modified.

This checker does not execute arbitrary record commands, verify truth of logs, authenticate reviewers, prove authorization, check numerical thresholds, or determine semantic applicability. Evidence/root and record must be selected by trusted orchestration. Do not validate untrusted paths against a privileged unrestricted root. For adoption add project-specific runner-generated evidence and reviewer acceptance, not a generic fake-scientific checker. A readiness attestation records evidence consistency; runtime permissions and explicit action approvals remain independent.

Final review correction: ownership path resolution now uses separately required CLI `--ownership-root`, not the evidence directory. Two distinct-root regression tests reject source-tree symlink escape and overlapping aliases.
