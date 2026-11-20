# Scope of ByteGate

The trusted model is `Defs.lean`. The executable predicate `check` decides a conjunction of exact byte equality, byte length, signature/digest/encoding/file-check facts, a fresh handoff identifier, the expected parent, and membership of any parent in the accepted parent list.

Fourteen statements in `Gate.lean` establish soundness and completeness of that decision, byte and length preservation, rejection on changed bytes, replay, false signature/digest/encoding/file facts, a mismatched or unknown parent, rejection after consuming an identifier, and source-byte preservation throughout a modeled accepted history.

The executable in `Main.lean` parses concrete byte arrays and evaluates `check`. The production Python dispatcher calls that executable after authenticating the handoff and bound files, and before committing acceptance or launching the signed command.

The model does not prove SHA-256 collision resistance, Ed25519 security, Python parsing or verification, filesystem immutability, sandbox correctness, the JSON parser or compiler, or the correspondence between English requirements and mathematical behavior. Signature and code-check predicates are facts supplied by those external checks. Receipt signing and journal I/O are implemented and adversarially tested in Python; they are not proved by the modeled history invariant.

Lean compilation and an axiom audit completed with the pinned v4.34.1 toolchain. The repository's stronger Comparator requirement remains unsatisfied because the pinned Landrun cannot establish its required sandbox on this host. The upstream Comparator injects `--best-effort` and its invocation printed success without the required isolation. That output is retained as non-qualifying evidence. The strict rerun removes only the weakening flag, records the resulting sandbox failure, and does not claim a passing Comparator result.
