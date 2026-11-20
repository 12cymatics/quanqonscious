# Byte-preserving execution gate v3

This component authenticates a handoff and its full instruction bytes before it executes the signed command. It preserves the uploaded files in `originals/`, binds command files to their actual bytes, records signed acceptance and outcome receipts, rejects replay, and calls a compiled Lean acceptance predicate on every dispatch.

It does not implement the Faraday solver. A successful command or receipt is not evidence that the simulator implements the 29 sutras.

## Exact sources

The original 18-byte instruction, the original signed policy/key/signature/digest, the complete 3,494-byte delegation contract, and the original `verify.py` are retained unchanged. The historical `verify.py` is evidence; the new receiver does not use its removable Python assertions. The v3 receiver uses explicit checks and exceptions.

Every authority and handoff contains the full original instruction and full contract bytes. Additional source files supplied at authority creation are also included verbatim. Supply the complete simulator task and its authoritative specifications as additional sources when using this component for simulator work: the three-word instruction alone does not define that mathematics.

| Source | SHA-256 |
| --- | --- |
| `instruction.bin` | `fdb95ea54d90a1875752668e55c5857618f4e0275a327dbb4d03181a8256a0c7` |
| `verified_delegation_instruction.txt` | `b801c62880a6862d716a6107909494e58cf9a6c0c58921eff89296bf293e5fd2` |
| `policy.canonical.json` | `bdcd68b207acca59f428e0a2ef3053bea235bdca64c6a61ce9f8894b60d82a85` |

The original policy signature is checked against its retained public key. New execution envelopes use a newly generated execution key. They are execution records, not new signatures by the user. Pin that execution public key independently at the receiver. Keep its private key outside the authority directory, with access restricted to the signing process. The local helper requires permissions excluding group and other users.

## Build

The source requires Linux, Python 3.12 and `cryptography==50.0.0`. `lean-toolchain` pins Lean 4.34.1. The Lean model uses only that toolchain's core/Std libraries; it has no Mathlib dependency.

```sh
python3 -m pip install -r requirements.txt
lake build
python3 -m unittest -v test_gate.py
```

The resulting executable is `.lake/build/bin/bytegate`. The authority copies and signs the digest of this actual executable, the receiver source, and the snapshot helper. Changing these after authority creation is rejected. Rebuilding intentionally requires a new authority, with the new pinned public key conveyed to receivers.

## Execute a bound standalone program

```sh
python3 gate.py init --authority trust --private execution.key --checker .lake/build/bin/bytegate --source simulator-task.txt=simulator-task.txt
python3 gate.py create --authority trust --pin PUBLIC_KEY_SHA256 --private execution.key --operation-file operation.txt --output handoff.json --file-arg 4 -- /absolute/python3 -I -S -B /absolute/worker.py
python3 gate.py dispatch --authority trust --pin PUBLIC_KEY_SHA256 --private execution.key --envelope handoff.json --digest HANDOFF_SHA256 --journal audit --head 0000000000000000000000000000000000000000000000000000000000000000
```

`init` prints the actual public-key digest. `create` prints the SHA-256 of the exact canonical handoff file. Pass those actual values; the uppercase strings above are parameter names. The command executable is always bound. Each `--file-arg` is a zero-based command argument index. In the example the Python script is argument 4.

The executable and selected command-file arguments are read once, checked against their signed lengths and hashes, placed in Linux memory files, sealed against changes, and executed through the sealed file descriptors. This prevents replacing a pathname after verification from changing the bound code actually executed. The regression test replaces the script pathname after verification and establishes that the original sealed bytes execute.

Relative imports need the source-snapshot route below, because a sealed script has a `/proc/self/fd/...` pathname. The gate does not silently reopen the original script path to make imports work.

## Source-tree snapshots

`snapshot.create` copies a selected inventory into a new directory. It preserves relative filenames, records each file's bytes and executable flag, uses read-only input files/directories, and makes only declared output directories writable. The exact inventory, hashes, working directory, environment, and isolation mode are signed in the handoff. The receiver checks all protected source bytes and their inventory before execution and again before recording a successful result. Output directories cannot overlap immutable inputs.

```python
from pathlib import Path
import gate
import snapshot
manifest = snapshot.create(
    "/absolute/repository",
    ["dns/dyadic128.js", "dns/check-dyadic128.mjs"],
    "/absolute/snapshot",
    writable=["output"]
)
pin = gate.init_authority(
    "/absolute/trust",
    "/absolute/execution.key",
    ".lake/build/bin/bytegate",
    extras=[("simulator-task.txt", "/absolute/simulator-task.txt")]
)
authority = gate.Authority("/absolute/trust", pin)
raw = gate.create_envelope(
    authority,
    "/absolute/execution.key",
    Path("/absolute/operation.txt").read_bytes(),
    ["/absolute/node", "/absolute/snapshot/dns/check-dyadic128.mjs"],
    snapshot=manifest,
    snapshot_mode="trusted-host-readonly",
    timeout_ms=600000
)
result = gate.dispatch(
    authority,
    "/absolute/execution.key",
    raw,
    gate.sha(raw),
    None,
    "/absolute/audit",
    gate.GENESIS
)
```

Include every actual required project source and input file in the selected inventory. The two-item example is illustrative; it does not assert that those two files are sufficient for a specific simulator suite.

There are two explicit signed snapshot modes:

| Mode | Enforcement and boundary |
| --- | --- |
| `landlock` | The default. Requires a pinned Landrun executable, read/execute runtime roots, and kernel Landlock support. The command receives write permission only to specified output paths. Failure to establish the sandbox rejects execution. |
| `trusted-host-readonly` | Requires an explicit choice. Uses a read-only copied source tree, signed hashes, and validation before/after execution. The interpreter, loader, imported runtime libraries, OS, and absence of concurrent same-authority tampering are trusted. It is not kernel isolation, and a same-user process able to change modes or replace directory entries can defeat the filesystem protection. |

No automatic fallback exists between these modes. In this environment Landrun reports Landlock ABI v0 and rejects strict execution. That failure is retained in the verification records. The trusted-host mode makes useful source-tree execution possible under its stated assumptions; it does not fulfill a requirement for strict host isolation.

For strict mode, provide `landrun=/absolute/landrun` to `init_authority` and `runtime_roots=[...]` to `create_envelope`. Runtime roots must not encompass the source snapshot, authority, or private key. The CLI equivalents are `--landrun`, `--runtime-root`, `--snapshot-manifest` and `--snapshot-mode`.

## Parent handoffs, replay and receipts

A child includes its parent's exact envelope digest. Receivers must supply the expected parent independently through `--parent`. The parent must already have an authenticated acceptance in the journal. Every handoff has its own random 256-bit identifier and signature; an accepted identifier cannot execute again, including after a command failure or interrupted run.

The journal is opened in append mode and serialized with a file lock. Each record is signed and includes its sequence, the preceding record's exact digest, the handoff digest, source digest, code descriptors, parent, and actual outcome. Completion receipts also bind the stdout/stderr digests and process exit status. An acceptance is committed before the process starts; a crash leaves an unfinished acceptance that cannot be replayed.

Pass the last **externally retained** signed journal-head digest for each next dispatch. `GENESIS` is only for an empty journal. A correct previous prefix alone cannot reveal that a journal's tail was deleted; the independently retained checkpoint is what makes truncation detectable. Do not recover an allegedly trusted checkpoint by reading an untrusted journal tail. The dispatch return object includes the new head to retain. Both expected handoff digest and expected journal head are mandatory.

```sh
python3 gate.py audit --authority trust --pin PUBLIC_KEY_SHA256 --journal audit --head LAST_RETAINED_HEAD
```

`verify` checks a handoff's structure, authority, signature, bytes and pinned identity. It does not consume it or authorize process execution; `dispatch` also checks parent acceptance, replay state, code files and the compiled Lean predicate.

## Canonical encoding

Signed objects use JSON encoded as ASCII with lexicographically sorted keys, no separators other than `,` and `:`, `ensure_ascii=True`, and no non-finite values. The entire signed envelope ends with exactly one LF. Embedded original files use canonical RFC 4648 base64, a byte length, and SHA-256; they are never Unicode-normalized or newline-converted. Duplicate keys, extra or missing fields, noncanonical encodings and noncanonical base64 are rejected. The signature covers the canonical `payload` object. An envelope/receipt digest covers its complete serialized file bytes, including its terminating LF.

## Formal status and scope

Lean 4.34.1 successfully compiled 14 theorems and the executable acceptance predicate. `Gate.lean` contains the solutions; `ComparatorChallenges/Gate.lean` contains independently readable statements; its configuration permits exactly `propext`, `Quot.sound`, and `Classical.choice`. The compiled solutions' axiom audit reports only `propext` and, for the history invariant, `Quot.sound`. No solution uses `sorry`, `native_decide` or a custom axiom. The required challenge placeholders are not runtime code.

The pinned Comparator, lean4export and Landrun were built and the actual Comparator command was attempted. Inspection established that the pinned Comparator passes `--best-effort` to Landrun. Its unmodified invocation printed a successful comparison even though this kernel provides no Landlock; that result does not meet the required sandbox assumptions and is retained as non-qualifying evidence. A strict invocation, through a recorded wrapper that removes only `--best-effort`, rejects the missing protection. Under the repository's CODEX_OPENAI_MATH.md terminology these results are **not Comparator-verified**. Compilation and axiom-audit evidence remain available; the non-qualifying output is not promoted to a passing formal verification.

The dispatcher executes the compiled `check` predicate before launching an accepted command. The model proves byte equality, length equality, rejection conditions, one-use acceptance, and preservation over a history of accepted handoffs. Python supplies the facts established by SHA-256, Ed25519, canonical parsing and sealed-file checks. The Python/cryptography implementation, JSON-to-Lean transport, Lean compiler/code generator, OS and hardware remain explicit unverified trust boundaries. This is an executable connection to the proved acceptance predicate, not a verified refinement of Python, SHA-256 or Ed25519 into Lean.

The gate controls commands invoked through its receiver. It cannot disable other host execution interfaces or intercept ChatGPT collaboration/GitHub tools. The recorded v2 handoffs that initiated this work remain v2 handoffs; this component does not retroactively claim they used v3. Cryptographic byte preservation does not prove semantic obedience or mathematical correctness. Those require evidence about the actual simulator calculations.

See `verification/` for compilation, axiom, runtime-test and Comparator results, and `docs/scope.md` for the formal statement boundary.
