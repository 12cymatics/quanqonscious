# Verified Agent Gate 3: security and execution model

## Protected operation

The protected operation is the transition from a controller-authenticated handoff to execution of one copied Python job. `gate3.py dispatch` is the authorization interface. The controller consumes a unique handoff before starting its worker, and the worker runs inside mandatory kernel-enforced restrictions.

The threat model treats the submitted project, worker code, worker outputs, and unverified envelopes as untrusted. It trusts the operator, enrolled key identity, controller installation and state, operating system, installed runtime, and cryptographic implementation. An arbitrary process with access to the same host account and controller secrets is outside that boundary; filesystem mode bits do not isolate two hostile processes sharing a UID.

## Trust enrollment and authorization

`init` requires a complete source file, a signing key, an independently established public-key fingerprint, and a nonempty explicit authorization record. The fingerprint is SHA-256 over the raw 32-byte Ed25519 public key. The operator must obtain or record it through a trusted process. Calculating a fingerprint from an attacker-supplied key and trusting that value provides no independent origin authentication.

Initialization rejects existing state. It records the exact source bytes and length, source and authorization hashes, authority identity, audit identity, and selected runtime hashes in a signed configuration. Source and authorization records are checked on subsequent controller operations. The current source is exactly 3494 bytes, SHA-256 `b801c62880a6862d716a6107909494e58cf9a6c0c58921eff89296bf293e5fd2`.

`source/continuation-authorization.txt` preserves the user's authorization of the package work. The operator of a new deployment must separately record the authority and scope for that deployment. A nonempty text file is a recorded human assertion; the program does not infer legal authority or semantic consent from arbitrary prose.

State and signing keys must be kept outside worker source trees and system runtime mounts. Initialization rejects protected state or authority-key placement under `/usr`, `/lib`, or `/lib64`. Job creation rejects overlap between the source project and controller state. Filesystem isolation prevents the managed worker from accessing the controller's private state; it does not remove the operator's access.

## Authenticated handoff and executable identity

Each handoff contains eight fields in the executed Lean kernel's `VGK3` wire format:

| Field | Binding |
| --- | --- |
| Authoritative instruction | Exact complete source bytes |
| Policy digest | Exact enrolled manifest identity |
| Source digest | SHA-256 of the complete instruction |
| Assigned operation | Separate nonempty UTF-8 allocation of work |
| Parent digest | Empty for a root job, otherwise the registered envelope digest |
| Handoff ID | Unique 32-byte identifier |
| Signing-key ID | Enrolled authority identity |
| Execution specification | Canonical JSON binding files, entrypoint, arguments, limits, runtime, authorization, and validity interval |

The signature scope is the domain separator `VERIFIED-AGENT-GATE-V3` followed by NUL and the complete serialized payload. Every wire field has an unsigned 64-bit big-endian byte length. The parser rejects malformed framing, extra bytes, and noncanonical encodings. The exact definition is in `lean/PROTOCOL.md`.

The controller checks Ed25519 authenticity, invokes the compiled Lean verifier, and checks JSON semantics, source length and hash, authorization, registered issuance, runtime fingerprints, expiry, and replay state. It resolves a parent through the signed registry, verifies its stored envelope, and requires one successful completed receipt with retained artifacts still matching their recorded identities. A digest supplied by a child is a lookup value, not an independently trusted parent record.

`create` copies project files into a private snapshot and binds each relative path, size, and SHA-256 to the signed execution specification. It also binds the delivered instruction, task metadata, source manifest, and authorization record. The manager requires the exact file set and exact delivered envelope. The bootstrap repeats file-set, regular-file, size, and digest checks inside the read-only job mount before the entrypoint runs.

The task metadata is compared with verified fields before dispatch. The worker receives the full instruction bytes, operation, handoff ID, and parent digest directly through the bootstrap and through read-only metadata files. Carrying an assignment never substitutes its text for the original instruction.

## Mandatory process boundary

The native launcher runs as an ordinary user and refuses root or set-ID execution. It requires bubblewrap user, PID, IPC, and UTS namespace creation, disables further user namespaces, drops capabilities, sets no-new-privileges, and fails closed if setup cannot complete.

| Worker-visible path | Access and purpose |
| --- | --- |
| `/job` | Read-only signed snapshot, including complete instruction and handoff metadata |
| `/gate` | Read-only trusted bootstrap and seccomp library |
| `/output` | Writable persistent output directory for this job only |
| `/tmp` | Ephemeral tmpfs, 64 MiB |
| `/dev/shm` | Ephemeral tmpfs, 64 MiB |
| `/usr/lib`, `/lib`, `/lib64` | Read-only installed runtime libraries |
| `/usr/bin/python3` | Read-only system Python executable |
| `/bin` | Symlink to the minimal `/usr/bin` view |
| `/proc` | Read-only procfs within the worker PID namespace |
| `/dev` | Bubblewrap's basic virtual devices; no host GPU or arbitrary host devices |

The environment is cleared and rebuilt with fixed runtime and job-location variables. Stdin is disconnected. Descriptors beyond stdin/stdout/stderr are closed before user code starts. Stdout and stderr are controlled result files, subject to the worker's per-file limit. Host home directories, project source outside the copied snapshot, signing keys, audit logs, and controller state are not mounted.

The first seccomp filter is installed before the interpreter executes. It denies socket creation and network operations, io_uring, ptrace and process-memory access, filesystem-handle access, mount and namespace manipulation, keyring operations, and other selected privileged interfaces. The host's restrictions prevented network namespace creation during this build; the mandatory socket/syscall denial supplies the network-access restriction.

System Python starts with `-I -S`. The trusted bootstrap installs a second filter before invoking the job. It additionally denies `execve`, `execveat`, `fork`, `vfork`, `clone`, and `clone3`. Thus a worker cannot start an independent executable, process, or thread and cannot manufacture an unchecked child agent. Further authorized work is issued through the controller with a new signed envelope.

These filters constrain operating-system operations. They do not prove that arbitrary Python expressions implement an English instruction. A Python program can still perform computations within its own process and permitted resources; semantic correctness must be established from its implementation and requested outputs.

## Lifetime and resource controls

The launcher requires the actual controller PID, checks parent identity, and uses a parent-death signal. The native supervisor and bubblewrap maintain a death-signal chain to the worker namespace. An explicit expected parent closes the startup race in which the controller exits before the launcher begins initialization.

The supervisor uses a monotonic wall timer. On expiry it kills the sandbox process group and returns `124`. Controller death, supervisor death, and timeout are exercised with actual running processes in `native/test_native.py`. The tests confirm that the native runner, bubblewrap processes, and worker cease executing.

Signed configurable limits are:

| Setting | Default | Accepted range | Meaning |
| --- | --- | --- | --- |
| `timeout_seconds` | 30 | 1–3600 | Supervisor wall time |
| `cpu_seconds` | 30 | 1–3600 | CPU rlimit inherited by the worker |
| `memory_mib` | 512 | 64–8192 | Process virtual-address-space rlimit |
| `file_mib` | 16 | 1–1024 | Maximum size of an individual output/log file |

The launcher also limits open descriptors and process count; the worker's stricter no-fork/no-clone policy applies after bootstrap. Address-space limits are not total cgroup memory accounting. Per-file limits are not a total output-directory quota. A deployment requiring a hard aggregate storage limit must provide a quota-limited output filesystem through its trusted host administration. This package does not claim such a quota.

## Replay and parent acceptance

A controller lock serializes changes to the registry and audit chain. Before spawning the worker, dispatch appends and fsyncs a signed `started` record. That record consumes the handoff identifier. Failure, timeout, or controller interruption does not make the identifier reusable. A new issuance is required for a retry.

A parent must have an authenticated issuance, a valid stored envelope, exactly one completed successful dispatch, safe output types, and unchanged retained artifacts. A child cannot treat a still-running, failed, unauthenticated, or altered parent as complete. A process exit code of zero establishes process success under the recorded policy; it is not a correctness proof for the job's domain objective.

## Receipts, audit history, and retained artifacts

The controller uses a distinct audit signing key. Audit events have signed payloads, monotonic sequence numbers, and the previous record's digest. Each append is flushed and fsynced; a signed anchor records the current count and head. This detects edited, removed, reordered, duplicated, and truncated records relative to the retained current anchor. A mismatched anchor or interrupted audit commit stops normal operation for explicit operator review.

Completed receipts bind the handoff, source, parent, process exit code, timing, logs, and output-file identities. Retained receipt files must exactly match their signed audit records. `audit` rehashes stdout, stderr, and the output inventory; missing, modified, newly added, or retyped retained outputs are detected. Symlinks are inspected without following them, and symlinks/special files are recorded as unsafe. Such a result cannot qualify as a successful parent.

Audit files are protected from managed workers through the process boundary. They are not hardware append-only storage. A hostile host that can replace the entire state, keys, and anchor can restore an older consistent signed state or create a new authority. Detecting that threat requires an independently retained checkpoint, external trusted log, or storage controls administered outside that host. No such external service is claimed by this package.

An interrupted job remains consumed. `audit` reports started handoffs without a completed receipt as unfinished. The operator should inspect their retained state and issue a fresh authorized task if appropriate; the package does not silently erase history or retry the old handoff.

## Mechanically checked proof boundary

The controller calls the executable compiled from `lean/Main.lean`; it does not substitute a separate Python implementation for the proved acceptance functions. Lean 4.36.0 mechanically checked the retained source. The repository includes compiler output, native build hashes, and toolchain provenance.

The checked theorems establish properties of successful executions of the modeled encoder/verifier: exact instruction preservation, binding to trusted policy/source/parent values, canonical reserialization, accepted-field shapes, relevant rejection conditions, and handoff inheritance. Successful encoding is checked by the runtime roundtrip; its theorem is a soundness result for accepted encodings, not unconditional totality for arbitrarily large input.

The trusted implementation boundary includes Lean's kernel, compiler/runtime, C toolchain, standard I/O, Python orchestration, the cryptography library and hash implementation, key provenance, the registry, bubblewrap, the native filters, and the operating system. The Lean theorems do not prove Ed25519, Python's JSON implementation, the C launcher, or the semantic fulfillment of an assignment. This explicit separation follows clauses 7 and 10 of the supplied specification.

## Host integration and functional scope

Managed workers have no unrestricted execution interface within this runner. Other programs, ChatGPT tools, GitHub processes, or agents outside it remain controlled by their own host/orchestrator. This archive does not change those systems. A host deployment that intends to mediate all agent actions must make this controller the only authorized execution path and deny alternatives using host-owned controls.

The delivered functionality is the authenticated Lean-plus-cryptographic handoff and isolated Python execution system. It does not contain the Faraday solver or the 29 Vedic methods. Cryptographic authenticity and byte preservation do not establish scientific validity, mathematical correctness, or compliance with every sentence of an operation description.

For reproducibility, `lean/test_kernel.py` drives the compiled wire kernel, `native/test_native.py` drives real isolated processes, and `tests/test_acceptance.py` exercises signed controller issuance, real dispatch, adversarial handoffs, replay, parent binding, audit changes, and retained-artifact changes. These checks support the described implementation; they are not a substitute for the implementation or a proof of unmodeled properties.
