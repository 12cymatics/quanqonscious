# Migration to source-bound execution

The draft policy and memory interface remains `python3 -B validate_contracts.py --self-test`. It cannot accept `task_state: complete`. Readiness uses a schema-3 project record and a signed schema-2 execution contract. Legacy readiness records are rejected; their old passing outputs remain historical evidence.

## Template validation

```sh
python3 -B validate_readiness.py --record readiness.template.json --evidence-root .
```

This checks an unassigned template and returns `readiness_attestation: not_evaluated`. The same invocation with `--readiness` must exit unsuccessfully. Null source identity, null contract identity and unassigned ownership cannot establish admission or completion.

## Establish the execution contract

The operator records the actual user request, source root, complete source manifest, contract generation, ownership, full equation requirements, allowed commands, exact assertions and gate decisions. Preserve every existing required aggregate check. Bind dirty and untracked inputs within scope; a Git commit label alone is insufficient.

For mathematical acceptance, include actual `foundation_register` and `geometry_index` path/digest references in the signed contract. Both records must be included in the source manifest and have `kind: project_record`, the same task ID and complete equation linkage. The provided templates are deliberately unassigned. The foundation `equations` value must equal the signed equation inventory. The geometry record maps each equation's statement path, implementation paths, requirement IDs and check IDs.

Every equation requirement has an ID, kind, statement, implementation paths and `check_assertions` references. Each check/assertion reference must identify a declared executable check and one of its exact required assertions. Each equation also supplies `requirement_applicability` with exactly these kinds: `equation`, `term`, `coupling`, `mode`, `domain`, `initial_condition`, `boundary_condition`, `arithmetic`. Each kind declares `applicable` and `reason`. Applicable kinds require corresponding requirements and a null reason; an inapplicable kind requires an explicit signed reason and no requirements of that kind. Equation, domain and arithmetic cannot be excluded. Enumerate every other requirement from the actual supplied specification.

The approved contract payload is signed with the already established operator identity. Supply the public-key digest and contract payload digest independently of the submitting record. Do not create a new key, invent a scope approval or accept worker-supplied pins merely to make an existing record pass. A change of source or scope creates a new contract identity; retain earlier evidence under its original identity.

## Produce observed results

Use actual absolute paths and independently established digest values in this interface:

```sh
python3 -B validate_readiness.py \
  --run-checks \
  --contract /absolute/control/signed-contract.json \
  --trusted-public-key /absolute/control/operator-public-key.pem \
  --trusted-key-sha256 PUBLIC_KEY_SHA256 \
  --contract-sha256 CONTRACT_PAYLOAD_SHA256 \
  --source-root /absolute/project \
  --evidence-root /absolute/results \
  --output-record /absolute/results/readiness.json
```

The command verifies the contract and source, executes the bound checks and writes results outside the source root. A unique execution directory retains each command's raw stdout, stderr and execution metadata even when the command fails. The uppercase arguments above describe required real digest inputs; they are not usable signatures or supplied project values. A failed command or assertion stays failed. Never replace it with a narrower command, fixed output or manually declared success.

## Validate a submitted record against a fresh run

```sh
python3 -B validate_readiness.py \
  --record /absolute/results/readiness.json \
  --evidence-root /absolute/results \
  --source-root /absolute/project \
  --contract /absolute/control/signed-contract.json \
  --trusted-public-key /absolute/control/operator-public-key.pem \
  --trusted-key-sha256 PUBLIC_KEY_SHA256 \
  --contract-sha256 CONTRACT_PAYLOAD_SHA256 \
  --readiness \
  --execute-checks
```

A submitted project record requires both `--readiness` and `--execute-checks`; setting `task_state: complete` cannot bypass either flag. The checker validates evidence content, scope, ownership and source identity before comparing it with freshly executed results. Evidence from an older source, omitted checks, changed applicability, arbitrary reviewer names, empty files and explicit failure reports are rejected. `--ownership-root`, if supplied, must identify the same source root bound by the contract.

## Controlled project changes

`project_handoff.py` admits requested relative paths against the externally pinned contract and coordination registry. Its required identity arguments are `--contract`, `--trusted-public-key`, `--trusted-key-sha256`, `--contract-sha256`, `--source-root`, `--registry`, `--registry-sha256`, `--registry-generation`, `--worker` and `--assignment`. Repeat `--requested-path` for each intended path. This initial operation is read-only.

Add `--handoff` and `--artifact-root` to verify a worker's candidate changes against those assignments and the actual source delta. Add `--apply --journal-root` only for the integrator's already authorized application. Successful application produces `applied_requires_acceptance`; execute the final candidate's required acceptance checks before declaring the project complete.

The readiness output identifies executed checks, equation IDs, checked requirement IDs and execution-evidence locations. Its validity depends on the established operator pins and actual domain checks. This package supplies no Cymatica, GRVQ, MSTVQ, TGCR or ETERNAL production kernel, calibrated physical data or completed mathematical derivation. Those inputs must be supplied and executed before a claim about those projects can pass their required checks.
