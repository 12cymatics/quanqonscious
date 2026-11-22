# Verified Agent Gate 3

## Browser deployment

You can run this gate on GitHub's hosted Ubuntu machines without owning or installing Linux. The [hosted runner](hosted/README.md) implements the model's tool loop with one execution function, `run_verified_python`, connected directly to the signed gate. The [browser setup page](https://dancrab-verified-gate-setup.danxii.chatgpt.site) generates your authority key locally and links to the GitHub configuration and task launcher.

The repository distribution contains source. Its hosted bootstrap compiles the Lean kernel and native launcher before dispatch. The separately distributed original v3 ZIP also contains the earlier compiled Linux binaries and their build evidence. Commands below that invoke binaries require the build to have completed first when working from the source checkout.

## Execution component

Verified Agent Gate 3 executes an authenticated Python job inside an enforced Linux process boundary. The controller verifies the signed instruction, assignment, executable snapshot, parent record, runtime identity, expiry, and replay state before the job starts. Every job receives the complete authoritative instruction bytes and its authenticated assignment metadata.

The implementation includes the Python controller, the executed Lean serialization/verifier kernel, the native sandbox launcher and worker bootstrap, and adversarial integration checks. Its protected execution interface is `gate3.py dispatch`; the hosted controller routes every model-requested execution through this interface.

## Authoritative source

The supplied original file is retained at [`source/verified_delegation_instruction.txt`](source/verified_delegation_instruction.txt), without rewriting its contents:

| Property | Value |
| --- | --- |
| Version | 3 |
| Exact byte length | 3494 |
| SHA-256 | `b801c62880a6862d716a6107909494e58cf9a6c0c58921eff89296bf293e5fd2` |
| Original first line | `Build this exactly` |

The original downloadable package separately retains the user's historical continuation authorization. That private conversation record is omitted from the public source distribution. Automated tests use the explicitly synthetic `tests/authorization.txt` fixture. The hosted deployment records and signs the actual authenticated manual workflow request for each invocation; test authorization never grants authority for production work.

## What dispatch does

1. Loads the independently enrolled authority identity and verifies the signed controller configuration.
2. Checks the exact original source, authorization record, runtime fingerprints, and signed audit history.
3. Verifies the Ed25519 envelope and invokes the compiled Lean kernel to validate its canonical bytes and trusted instruction, policy, digest, and parent fields.
4. Checks the assignment, unique identifier, expiry, registered parent, immutable input manifest, task metadata, and replay state.
5. Durably records that the identifier is consumed before starting a worker.
6. Launches the copied job read-only in Linux namespaces with mandatory seccomp restrictions. Inside that boundary, the bootstrap verifies the file inventory and every input hash again before executing the selected Python source.
7. Records the process result and hashes of stdout, stderr, and output files in a signed receipt and signed audit chain. Subsequent audit checks rehash the retained artifacts.

Workers cannot start another executable, fork, create a thread, open a network socket, modify the copied job, or access the controller's signing and audit state through the provided mounts. A follow-on job requires another controller-issued, independently verified envelope.

## Requirements

- Linux x86-64 for the supplied binaries, with glibc 2.38 or later for the prebuilt native launcher, unprivileged user/PID/IPC/UTS namespaces, seccomp filters, `close_range`, and bubblewrap at `/usr/bin/bwrap`.
- A controller Python 3.10 or later with `cryptography==50.0.0`.
- System Python 3.10 or later at `/usr/bin/python3`, with its standard library under `/usr/lib`. Workers start with `-I -S`; user site-packages are disabled.
- An ordinary unprivileged operator account. The native launcher rejects root and set-ID execution.
- A private deployment directory outside `/usr`, `/lib`, and `/lib64` and outside the project submitted as a job.

The system must support the mandatory isolation features and the bubblewrap options used in `native/launcher.c`, including `--disable-userns`, `--assert-userns-disabled`, and descriptor-based bind mounts. This build was exercised with bubblewrap 0.12.0. Setup or enforcement failure stops dispatch; there is no unrestricted execution mode. Network calls are denied by seccomp. This implementation does not require a network namespace, which the build environment could not create.

The prebuilt Lean kernel does not require an installed Lean toolchain. Rebuilding it requires the pinned Lean 4.36.0 toolchain. On an otherwise compatible Linux system with older glibc, rebuild the native launcher and library with `make -C native clean all` before enrollment; this requires a C compiler and Linux development headers. Mandatory namespace and seccomp support still apply.

## First deployment

Run these Bash commands from the extracted `verified_agent_gate_v3` directory. The task-specific variables resolve to actual directories on your machine. The deployment directory must be new, because initialization does not replace existing trust state.

```bash
GATE_PACKAGE="$(pwd -P)"
GATE_DEPLOY="$(dirname "$GATE_PACKAGE")/verified-agent-gate-deployment"
umask 077
mkdir -m 700 "$GATE_DEPLOY"
python3 -m venv "$GATE_DEPLOY/controller-venv"
GATE_PYTHON="$GATE_DEPLOY/controller-venv/bin/python"
"$GATE_PYTHON" -m pip install 'cryptography==50.0.0'
GATE_STATE="$GATE_DEPLOY/state"
```

If your Linux installation does not provide bubblewrap or Python's `venv` module, install those system dependencies through your normal system administration process before continuing.

### 1. Check the executable package

The following commands exercise the actual compiled kernel, native launcher, and complete controller/worker dispatch path:

```bash
"$GATE_PACKAGE/lean/gate-kernel" version
"$GATE_PYTHON" "$GATE_PACKAGE/lean/test_kernel.py"
"$GATE_PYTHON" "$GATE_PACKAGE/native/test_native.py"
"$GATE_PYTHON" -m unittest discover -s "$GATE_PACKAGE/tests" -v
```

Do any source rebuild before enrollment. Runtime fingerprints are recorded during initialization, and a later executable or bootstrap change causes existing state to reject the changed runtime.

### 2. Generate and independently enroll an authority key

```bash
"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" keygen \
  --private "$GATE_DEPLOY/authority.key" \
  --public "$GATE_DEPLOY/authority.pub" \
  > "$GATE_DEPLOY/key-enrollment.json"
cat "$GATE_DEPLOY/key-enrollment.json"
```

The output contains the SHA-256 fingerprint of the raw Ed25519 public key. Record that fingerprint through your trusted operator process, separately from an incoming job or policy bundle. For a key supplied by someone else, obtain the expected fingerprint through an already trusted channel. A public key and fingerprint received together in an untrusted archive do not establish the signer's identity.

Enter the fingerprint from your trusted record:

```bash
read -r -p 'Independently recorded authority fingerprint: ' GATE_FINGERPRINT
```

### 3. Record authorization and initialize

The original specification requires the original instruction to be repeated and continuation to be explicitly authorized. For this new deployment, record the operator's actual authorization and its scope:

```bash
printf '%s\n' 'Build this exactly'
read -r -p 'Type AUTHORIZE to permit this deployment and its intended work: ' GATE_DECISION
if [ "$GATE_DECISION" != 'AUTHORIZE' ]; then
  printf '%s\n' 'Deployment was not authorized.'
  exit 1
fi
read -r -p 'Record the authorized scope of work: ' GATE_AUTHORIZATION
if [ -z "$GATE_AUTHORIZATION" ]; then
  printf '%s\n' 'An authorization scope is required.'
  exit 1
fi
printf '%s\n' "$GATE_AUTHORIZATION" > "$GATE_DEPLOY/authorization.txt"

"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" init \
  --state "$GATE_STATE" \
  --spec "$GATE_PACKAGE/source/verified_delegation_instruction.txt" \
  --authority-key "$GATE_DEPLOY/authority.key" \
  --expected-fingerprint "$GATE_FINGERPRINT" \
  --authorization "$GATE_DEPLOY/authorization.txt"
```

Initialization binds the full 3494-byte source and authorization record to the enrolled key. It creates private controller state and a separate audit signing key. Protect the original private key and the state directory as controller secrets; neither belongs in a worker project or distributable archive.

## Run an actual Python job

Choose the project and operation you are authorizing. Its source directory must be disjoint from controller state, and the entrypoint must be a relative `.py` path in that project. The controller rejects symlinks, special files, unsafe paths, and names reserved for its instruction and handoff metadata.

```bash
read -r -p 'Absolute path of the Python project to execute: ' GATE_JOB
read -r -p 'Relative Python entrypoint within that project: ' GATE_ENTRYPOINT
read -r -p 'Authorized assigned operation: ' GATE_OPERATION

"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" create \
  --state "$GATE_STATE" \
  --job "$GATE_JOB" \
  --entrypoint "$GATE_ENTRYPOINT" \
  --operation "$GATE_OPERATION" \
  --timeout-seconds 30 \
  --memory-mib 512 \
  --file-mib 16 \
  --cpu-seconds 30 \
  --ttl-seconds 86400 \
  > "$GATE_DEPLOY/issued.json"

GATE_ENVELOPE="$("$GATE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["envelope"])' "$GATE_DEPLOY/issued.json")"

"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" verify \
  --state "$GATE_STATE" --envelope "$GATE_ENVELOPE"

"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" dispatch \
  --state "$GATE_STATE" --envelope "$GATE_ENVELOPE" \
  > "$GATE_DEPLOY/receipt.json"

"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" audit --state "$GATE_STATE"
```

If the entrypoint takes arguments, append `--` followed by those arguments to `create`; they become part of the signed task. `create` copies and signs the actual project files. Editing the original project afterward does not change that issued snapshot. Create a new envelope for changed code or a retry; an envelope is consumed once execution begins, even if the process fails or the controller is interrupted.

Run `verify` before dispatch. Verification of an already consumed envelope rejects its replay. Use `audit` to verify completed history and retained results.

### Worker interface

The selected entrypoint executes with its working directory at `/output`. Its project is read-only at `/job`. Write persistent results to `/output`; temporary files may use `/tmp`.

| Interface | Value delivered to the Python entrypoint |
| --- | --- |
| `GATE_INSTRUCTION_BYTES` | The exact complete authoritative file as `bytes` |
| `GATE_INSTRUCTION_SHA256` | SHA-256 of those exact bytes |
| `GATE_ASSIGNED_OPERATION` | Authenticated operation text, separate from the original instruction |
| `GATE_HANDOFF_ID` | Unique authenticated handoff identifier |
| `GATE_PARENT_SHA256` | Authenticated parent-envelope digest, or `None` for a root job |
| `sys.argv` | Entrypoint path followed by signed arguments |
| `/job/instruction.bin` | Exact authoritative source file |
| `/job/task.json` | Authenticated entrypoint, arguments, assignment, handoff ID, and parent |
| `/job/envelope.json` | Delivered signed envelope |
| `/job/manifest.json` | Exact executable/input inventory |

The globals above are supplied to the entrypoint by the trusted bootstrap. Environment variables additionally identify `/job`, `/output`, the instruction, envelope, and manifest paths. Pure Python project dependencies can be included in the signed project tree. This worker profile deliberately prohibits subprocesses, threads, and network access; projects needing those facilities require an explicitly reviewed execution profile.

### Results and receipt meaning

The JSON receipt identifies the retained output directory. Retrieve its path with:

```bash
GATE_OUTPUT="$("$GATE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["payload"]["data"]["output_directory"])' "$GATE_DEPLOY/receipt.json")"
printf '%s\n' "$GATE_OUTPUT"
```

Controller state retains `results/<handoff-id>/output/`, `stdout.bin`, `stderr.bin`, and `receipts/<handoff-id>.json`. `audit` validates the signed event chain, its current anchor, each retained receipt, logs, and output inventories. Changed or missing retained artifacts are rejected. Output symlinks and special files are recorded as unsafe and cannot qualify a job as a successful parent.

An exit code of zero means the process completed successfully under this execution policy. It does not prove that the assigned mathematical, scientific, or software objective was accomplished. Assess the actual requested outputs separately.

### Follow-on jobs

Workers cannot launch their own child process. The trusted controller issues a separate signed job whose parent is a successfully completed, audited envelope. For a follow-on operation on the selected project:

```bash
GATE_PARENT_SHA256="$("$GATE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["envelope_sha256"])' "$GATE_DEPLOY/issued.json")"
read -r -p 'Authorized follow-on operation: ' GATE_CHILD_OPERATION

"$GATE_PYTHON" "$GATE_PACKAGE/gate3.py" create \
  --state "$GATE_STATE" \
  --job "$GATE_JOB" \
  --entrypoint "$GATE_ENTRYPOINT" \
  --operation "$GATE_CHILD_OPERATION" \
  --parent "$GATE_PARENT_SHA256" \
  > "$GATE_DEPLOY/child-issued.json"
```

Extract the new envelope path from `child-issued.json`, then verify and dispatch it as above. The controller resolves the parent through its authenticated registry and checks the parent's retained outputs; the child cannot choose an unauthenticated parent record by merely presenting a digest.

## Proofs and rebuilds

[`lean/Main.lean`](lean/Main.lean) contains the functions executed by the compiled kernel and mechanically checked theorems for accepted-byte preservation, trusted-field and parent binding, canonical serialization, and rejection conditions. [`lean/PROTOCOL.md`](lean/PROTOCOL.md) defines the wire format and precise proof scope. The hosted build rechecks the proofs, compiles the kernel, and records its toolchain identity and artifact hashes in build provenance. The original downloadable package also retains its earlier compiler output and build evidence.

To rebuild with the pinned toolchain available:

```bash
bash "$GATE_PACKAGE/lean/build.sh"
make -C "$GATE_PACKAGE/native"
```

`lean/build.sh` also accepts the task-specific `LEAN_TOOLCHAIN_ROOT` variable pointing to a Lean 4.36.0 installation. Rerun the executable checks after rebuilding and initialize fresh trust state for the changed runtime.

## Deployment scope

This implementation enforces jobs started through its trusted local controller. Host operators must route the processes they intend to control through that controller and protect its state from other host processes. The package does not modify ChatGPT's tool routing, GitHub delegation infrastructure, or unrelated host execution interfaces.

The Lean proofs cover the executed handoff kernel. Python orchestration, Ed25519/SHA-256 implementations, the native compiler, operating system, and sandbox are explicit trusted implementation boundaries, as permitted by clauses 7 and 10 of the supplied contract. The package implements the delegation system; it contains no Faraday solver or implementation of the 29 Vedic methods.

Resource limits include wall time, CPU time, process address space, open descriptors, and individual file sizes. They do not provide an aggregate host-output disk quota. Signed audit records detect changes relative to the retained current anchor; complete rollback or replacement by a hostile host possessing the signing keys requires independently protected storage or external checkpoints to detect. See [`SECURITY_MODEL.md`](SECURITY_MODEL.md) for the operational trust model.
