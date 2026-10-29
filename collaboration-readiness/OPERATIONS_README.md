# Agent, memory and execution contracts

## Executable boundaries

`validate_contracts.py` validates the unassigned policy and project-memory format. It rejects completion in that format and enforces categorical prohibitions on substitution, approximation, simplification, truncation, reduced models, transformed formulations, replacement formulas and disabling required functions. Its output describes draft consistency.

`validate_readiness.py` checks a schema-3 project record against a signed schema-2 execution contract, verifies the actual source manifest, parses the evidence, runs the contract's checks and compares observed results with submitted evidence. It rejects legacy records, unassigned templates, empty or failed evidence, invented revisions, changed scope, missing required checks and unresolved operations. `existing_aggregate_checks` is required for every profile.

`validation_identity.py` verifies Ed25519 signatures, source manifests and ownership. Public-key identity and the expected contract digest enter through the trusted operator's invocation; neither is read from the submitting record. The operator establishes the request, root, generation, ownership and exact checks from the user's actual instructions. A signature authenticates those bytes to the pinned key; an agent-generated key or signed approval field does not create user authorization.

`project_handoff.py` implements controlled write admission, handoff verification and serialized application. The integrator owns registry updates and applies accepted worker results. Workers cannot grant themselves ownership, transfer assignments or mark their own changes integrated. Direct writes by an unrestricted external process remain outside this program's control. Its successful states distinguish `admitted`, `handoff_validated` and `applied_requires_acceptance`; none grants publication or scientific completion.

`verify_package.py` verifies the delivered package's exact file inventory and integrity manifest. Its default root is the package; `--root` and `--manifest` select another actual package. The manifest excludes its own file to avoid a circular self-hash. Historical evidence and pre-repair hashes remain distinct from current verification. The checker never regenerates a manifest to conceal differences.

## Formats and source identity

`policy.yaml`, `project-memory.template.yaml` and `FOUNDATION_REGISTER.template.yaml` use JSON-compatible YAML. The executable loaders use JSON parsing; native YAML syntax is not accepted by these interfaces. A file extension does not activate CI, an agent runner or a release process.

The signed execution contract binds the task ID, actual user request and its digest, contract generation, completion criterion, domain, mathematical-claim status, deliverable kind, absolute source root, source manifest, ownership, equation requirements, check commands and all nine gate decisions. The submitting readiness record must match these decisions exactly. A different `domain`, `mathematical_claims` or `deliverable_kind` cannot reduce the signed scope.

The source identity is `sha256:` followed by the canonical manifest digest. Source verification checks the actual bytes of every included file, accounts for scoped dirty and untracked files, and compares the declared base HEAD with the repository state when applicable. Check entrypoints and executable digests are also bound. A revision label alone cannot establish any of these facts.

Ownership paths are literal paths relative to the actual source root. The identity layer checks parent/child overlap, aliases and existing filesystem identity, including planned descendant paths. Its conservative case and Unicode spelling policy may reject distinct spellings on a case-sensitive volume; such a rejection is a portability policy, not a claim that those paths are physically identical. Unsupported symlink and hard-link inputs remain errors.

## Mathematical linkage

When the mathematical integrity gate applies, the contract references actual foundation and geometry records by path and digest. Unassigned templates do not satisfy this requirement. The foundation record must carry the same complete equation inventory as the signed contract. Every equation includes its statement, implementation paths, checks and individually identified requirements for its equations, terms, couplings, modes, domains, initial conditions, boundary conditions and arithmetic, as specified by the user.

Each requirement names the bound implementation files and the executable assertions that check it. The geometry record links these equations, requirement IDs, implementation paths and checks. The validator reads both records and rejects missing, stale, inconsistent or unbound entries. The full equation inventory comes from the supplied specification; shortening that inventory cannot be authorized by a worker's record.

Each equation's signed `requirement_applicability` contains all eight requirement kinds. An applicable kind needs at least one requirement. An inapplicable kind needs an explicit reason and no requirements of that kind. Equation, domain and arithmetic requirements cannot be excluded. These decisions describe the supplied equation; they cannot excuse an omitted required term or condition.

The actual domain checks must exercise the production implementation and evaluate its intended outputs in the required arithmetic. A static mapping, code coverage, a signature, a successful build or the package's finite examples does not establish a derivation, the completeness of an independently supplied specification, or physical validity. Required missing equations, implementations or domain checks remain specific incomplete calculations.

## Execution and evidence

Only commands bound in the trusted contract are eligible for readiness execution. The first argument after the executable must be the literal bound source entrypoint; an unused filename appended to an inline command cannot satisfy entrypoint binding. The caller must already be authorized to execute them in the selected environment. The checker runs the actual entrypoints and requires the exact declared assertion set. An exit code of zero with a failed, missing or malformed assertion cannot pass. Check results identify their check and its structured outcome; gate reports also bind the source and contract identities. Each run retains raw stdout, stderr and execution metadata, including failed execution. The result identifies checked equation and requirement IDs and its execution-evidence locations.

The readiness operation re-executes the bound checks and compares current outcomes with submitted reports. Empty files, opaque text, arbitrary success JSON, mixed failure/success evidence and matching fabricated revision strings cannot replace execution. Failed or unrun checks cannot be marked inapplicable by a reason string or a reviewer name. Applicable gate exclusions must already be allowed by the independently established contract, and mandatory profile gates cannot be excluded.

The GitHub workflow runs these package verification commands with this directory as its working directory. All testing for this repair is performed in GitHub:

```sh
python3 -B verify_package.py
python3 -B validate_contracts.py --self-test
python3 -B -m unittest discover -v
python3 -B validate_readiness.py --record readiness.template.json --evidence-root .
python3 -B geometry/verify_structures.py
```

The template command must report `readiness_attestation: not_evaluated`. Adding `--readiness` must fail because a template cannot establish readiness. `MIGRATION.md` gives the project execution interface. The template workflow runs package checks and a template rejection test; activation in another repository still requires that repository's real paths and existing checks.

The draft project-memory record remains useful for sourced claims, pending operations and blockers. Its approval references are evidence of an actual instruction, never a mechanism that grants new authority. Publication, deployment, installation and mathematical acceptance require their own applicable project checks and the user's existing scope.
