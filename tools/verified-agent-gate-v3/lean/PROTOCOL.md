# Executed Lean kernel

`gate-kernel` is compiled from `Main.lean` using the official Lean 4.36.0 toolchain. The included Linux x86-64 executable depends on standard glibc libraries and does not require Lean to be installed. `build.sh` rebuilds from source when the pinned toolchain is available, either on PATH or at LEAN_TOOLCHAIN_ROOT.

## Wire payload

The byte payload begins with the five bytes `56 47 4b 33 00` (hex, VGK3 followed by NUL). Exactly eight fields follow. Every field has an unsigned 64-bit big-endian byte length followed by that many uninterpreted bytes. There is no trailing data.

| Position | Field | Required shape |
| --- | --- | --- |
| 0 | Authoritative instruction | Nonempty exact bytes |
| 1 | Policy SHA-256 | 32 raw bytes |
| 2 | Instruction SHA-256 | 32 raw bytes |
| 3 | Assigned operation | Nonempty bytes; UTF-8 validation is performed by the host |
| 4 | Parent envelope SHA-256 | Empty for root, otherwise 32 raw bytes |
| 5 | Unique handoff identifier | 32 raw bytes |
| 6 | Trusted signing-key identifier | 32 raw bytes |
| 7 | Execution specification | Nonempty bytes; canonical JSON and semantics are validated by the host |

## Executable transport

Every input line is lowercase hexadecimal followed by one LF. An empty field is a line containing only LF. Additional lines, uppercase hex, whitespace, odd-length hex, and missing terminal LF are rejected. Output has the same representation. On rejection, the executable returns 1, writes an explanation to stderr, and emits no fields.

- `gate-kernel encode`: stdin contains all eight field lines; stdout contains one serialized payload line.
- `gate-kernel decode`: stdin contains one payload line; stdout contains all eight canonical decoded field lines.
- `gate-kernel verify`: stdin contains five lines in order: payload, trusted instruction bytes, trusted policy digest, trusted instruction digest, expected registered parent digest. On acceptance, stdout contains all eight verified field lines.
- `gate-kernel version`: prints the implementation and compiler version.

The host signs `b"VERIFIED-AGENT-GATE-V3\0" + payload`. Cryptographic signing and verification are separate operations performed by the host. A successful `decode` is not authorization; dispatch requires a successful `verify`, trusted-key signature verification, registry/replay checks, and the host execution policy.

## Mechanically checked connection

The actual CLI calls the `encodeChecked`, `decodeCanonical`, and `verify` definitions proved in Main.lean. It does not reimplement their decisions in a separate Python path.

`verify_sound` proves that a successful result has the required field shapes, has exactly the trusted instruction, policy digest, instruction digest and parent digest, and reserializes to exactly the supplied payload. The specialized theorems establish exact source preservation, policy and parent binding, and rejection of a different source. `child_inherits_exact` applies the executed verifier to a parent instruction and registered parent-envelope digest. `encodeChecked_roundtrip` proves that every successful encoding decodes canonically to exactly its input frame. The encoder performs that roundtrip check at runtime; the theorem is a soundness property of successful encoding, not an unconditional totality theorem for all arbitrarily large inputs.

Lean checked these theorems with only the standard propositional-extensionality axiom `propext`; no sorry, admit, or project-defined axiom is present. The compiler output is retained in proof-check.txt. test_kernel.py executes the native binary against an independent struct-based wire encoder and adversarial payloads, including every possible byte value and exact 64 KiB source preservation.

The trusted implementation boundary includes Lean's kernel, native compiler and runtime, C toolchain, operating system and standard I/O. The cryptography library, hash functions, key provenance, execution-specification semantics, registry and sandbox implementation are not proved by these theorems. In particular, instruction identity is not a proof that the instruction's requested work was completed. The host must establish those additional obligations and must never label signature or byte-preservation success as task completion.
