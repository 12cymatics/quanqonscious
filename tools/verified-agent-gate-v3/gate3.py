#!/usr/bin/env python3
"""Authenticated, isolated instruction dispatch. The controller is trusted; jobs are not."""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import subprocess
import sys
import time
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption

ROOT = Path(__file__).resolve().parent
VERSION = "3.0.0"
HANDOFF_DOMAIN = b"VERIFIED-AGENT-GATE-V3\0"
CONFIG_DOMAIN = b"VERIFIED-AGENT-GATE-CONFIG-V3\0"
AUDIT_DOMAIN = b"VERIFIED-AGENT-GATE-AUDIT-V3\0"
ANCHOR_DOMAIN = b"VERIFIED-AGENT-GATE-ANCHOR-V3\0"
SCHEMA = "verified-agent-handoff-v3"
ZERO_HASH = "0" * 64
RESERVED = {"instruction.bin", "task.json", "manifest.json", "envelope.json", "source-manifest.json", "authorization.txt"}
DEFAULT_LIMITS = {"timeout_seconds": 30, "memory_mib": 512, "file_mib": 16, "cpu_seconds": 30}
LIMIT_RANGES = {"timeout_seconds": (1, 3600), "memory_mib": (64, 8192), "file_mib": (1, 1024), "cpu_seconds": (1, 3600)}
SYSTEM_MOUNTS = (Path("/usr"), Path("/lib"), Path("/lib64"))


class Rejected(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(value: str) -> bytes:
    if not isinstance(value, str):
        raise Rejected("base64 field must be text")
    raw = base64.b64decode(value, validate=True)
    if b64(raw) != value:
        raise Rejected("noncanonical base64")
    return raw


def parse(raw: bytes) -> Any:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise Rejected("duplicate JSON key")
            out[key] = value
        return out
    obj = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(Rejected("invalid JSON number")))
    if canonical(obj) != raw:
        raise Rejected("noncanonical JSON")
    return obj


def raw_private(path: Path) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes())


def public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def private_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def signed(domain: bytes, payload: Any, key: Ed25519PrivateKey) -> dict:
    return {"payload": payload, "signature_b64": b64(key.sign(domain + canonical(payload)))}


def check_signed(domain: bytes, record: dict, public: bytes) -> Any:
    if not isinstance(record, dict) or set(record) != {"payload", "signature_b64"}:
        raise Rejected("invalid signed record fields")
    Ed25519PublicKey.from_public_bytes(public).verify(unb64(record["signature_b64"]), domain + canonical(record["payload"]))
    return record["payload"]


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path: Path, raw: bytes, *, replace: bool = False, mode: int = 0o600) -> None:
    if not replace and path.exists():
        raise Rejected(f"refusing to overwrite {path.name}")
    temp = path.parent / (".write-" + secrets.token_hex(16))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise Rejected(f"refusing to overwrite {path.name}")
        os.replace(temp, path)
        fsync_dir(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def path_safe(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise Rejected("invalid relative job path")
    p = PurePosixPath(name)
    if p.is_absolute() or any(x in {"", ".", ".."} for x in name.split("/")):
        raise Rejected("unsafe relative job path")
    return name


def regular(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Rejected("only regular files are accepted")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def file_digest(path: Path) -> tuple[int, str]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise Rejected("artifact is not a regular file")
        value = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                value.update(block)
        if size != info.st_size:
            raise Rejected("artifact changed during verification")
        return size, value.hexdigest()
    finally:
        os.close(fd)


def validate_limits(chosen: dict) -> None:
    if not isinstance(chosen, dict) or set(chosen) != set(LIMIT_RANGES):
        raise Rejected("unknown or missing execution limit")
    for name, (lower, upper) in LIMIT_RANGES.items():
        value = chosen[name]
        if type(value) is not int or not lower <= value <= upper:
            raise Rejected(f"{name} must be an integer from {lower} through {upper}")


def validate_argv(args: list[str]) -> None:
    if not isinstance(args, list) or len(args) > 128 or not all(isinstance(x, str) and "\x00" not in x for x in args):
        raise Rejected("argv must contain at most 128 valid text arguments")
    if sum(len(x.encode("utf-8", "strict")) for x in args) > 65536:
        raise Rejected("argv exceeds the supported 65536 UTF-8 bytes")


def output_inventory(directory: Path) -> tuple[list[dict], list[dict]]:
    files, unsafe = [], []
    queue = [directory]
    while queue:
        current = queue.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                name = path.relative_to(directory).as_posix()
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    queue.append(path)
                elif stat.S_ISREG(info.st_mode):
                    size, digest = file_digest(path)
                    files.append({"path": name, "size": size, "sha256": digest})
                elif stat.S_ISLNK(info.st_mode):
                    unsafe.append({"path": name, "kind": "symlink", "target": os.readlink(path)})
                else:
                    unsafe.append({"path": name, "kind": "special", "mode": stat.S_IFMT(info.st_mode)})
                if len(files) + len(unsafe) + len(queue) > 100000:
                    raise Rejected("output exceeds the supported 100000-file inspection limit")
    files.sort(key=lambda item: item["path"])
    unsafe.sort(key=lambda item: item["path"])
    return files, unsafe


def runtime_inventory() -> dict:
    paths = [ROOT / "gate3.py", ROOT / "lean" / "gate-kernel", ROOT / "native" / "gate-sandbox",
             ROOT / "native" / "bootstrap.py", ROOT / "native" / "libgate_seccomp.so"]
    # The trusted operator installs these system executables. Jobs cannot modify their mounts.
    for executable in ["bwrap", "python3"]:
        candidate = "/usr/bin/" + executable
        if not Path(candidate).exists():
            raise Rejected(f"required executable unavailable: {candidate}")
        paths.append(Path(candidate).resolve())
    result = {}
    for path in paths:
        if not path.is_file():
            raise Rejected(f"required runtime has not been built: {path}")
        result[str(path.resolve())] = sha(regular(path.resolve()))
    return result


def kernel(action: str, fields: list[bytes]) -> list[bytes]:
    result = subprocess.run([str(ROOT / "lean" / "gate-kernel"), action],
                            input=b"".join(x.hex().encode("ascii") + b"\n" for x in fields),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
                            env={"PATH": "/usr/bin:/bin", "LANG": "C"})
    if result.returncode != 0:
        raise Rejected("compiled Lean kernel rejected handoff: " + result.stderr.decode("utf-8", "replace").strip())
    try:
        return [bytes.fromhex(line.decode("ascii")) for line in result.stdout.splitlines()]
    except (ValueError, UnicodeError) as exc:
        raise Rejected("invalid compiled kernel response") from exc


def keygen(private: Path, public: Path) -> str:
    key = Ed25519PrivateKey.generate()
    atomic(private, private_bytes(key))
    atomic(public, public_bytes(key), mode=0o644)
    return sha(public_bytes(key))


def init_state(state: Path, spec: Path, authority_key: Path, expected_fingerprint: str, authorization: Path) -> dict:
    state = state.resolve()
    if state.exists():
        raise Rejected("state already exists; existing trust cannot be silently replaced")
    for protected in (state, authority_key.resolve()):
        if any(protected == base.resolve() or protected.is_relative_to(base.resolve()) for base in SYSTEM_MOUNTS):
            raise Rejected("controller state and original private key must stay outside worker-visible system mounts")
    key = raw_private(authority_key)
    pub = public_bytes(key)
    if sha(pub) != expected_fingerprint:
        raise Rejected("authority key does not match independently enrolled fingerprint")
    source = regular(spec)
    consent = regular(authorization)
    if not source or not consent.strip():
        raise Rejected("complete original instruction and explicit authorization are required")
    if len(source) > 16 * 1024 * 1024:
        raise Rejected("authoritative instruction exceeds the supported 16 MiB")
    if source.split(b"\n", 1)[0] != b"Build this exactly":
        raise Rejected("this deployment requires the original three-word instruction")
    runtimes = runtime_inventory()
    state.mkdir(mode=0o700)
    for name in ["jobs", "results", "receipts"]:
        (state / name).mkdir(mode=0o700)
    audit_key = Ed25519PrivateKey.generate()
    manifest = {"schema": "authoritative-instruction-v3", "source_length": len(source), "source_sha256": sha(source),
                "original_instruction_hex": source.split(b"\n", 1)[0].hex(),
                "authorization_sha256": sha(consent), "authority_key_id": sha(pub),
                "encoding": "VGK3-U64BE-fixed-eight-fields", "version": VERSION}
    manifest_raw = canonical(manifest)
    config = {"schema": "gate-controller-config-v3", "version": VERSION, "authority_key_id": sha(pub),
              "audit_public_b64": b64(public_bytes(audit_key)), "audit_key_id": sha(public_bytes(audit_key)),
              "policy_sha256": sha(manifest_raw), "manifest": manifest, "runtime_sha256": runtimes,
              "created_ns": time.time_ns()}
    atomic(state / "authority.pub", pub, mode=0o400)
    atomic(state / "authority.key", private_bytes(key), mode=0o400)
    atomic(state / "audit.key", private_bytes(audit_key), mode=0o400)
    atomic(state / "instruction.bin", source, mode=0o400)
    atomic(state / "authorization.txt", consent, mode=0o400)
    atomic(state / "source-manifest.json", manifest_raw, mode=0o400)
    atomic(state / "config.json", canonical(signed(CONFIG_DOMAIN, config, key)), mode=0o400)
    atomic(state / "enrolled-fingerprint", expected_fingerprint.encode("ascii"), mode=0o400)
    atomic(state / "audit.jsonl", b"")
    atomic(state / "controller.lock", b"")
    initial = {"schema": "audit-anchor-v3", "count": 0, "head_sha256": ZERO_HASH, "audit_key_id": config["audit_key_id"]}
    atomic(state / "anchor.json", canonical(signed(ANCHOR_DOMAIN, initial, audit_key)))
    gate = Gate(state)
    with gate.locked():
        gate._append("initialized", {"source_sha256": sha(source), "source_length": len(source),
                                      "authorization_sha256": sha(consent), "authority_key_id": sha(pub)})
    return {"state": str(state), "authority_fingerprint": sha(pub), "audit_fingerprint": config["audit_key_id"],
            "source_sha256": sha(source), "source_length": len(source), "version": VERSION}


class Gate:
    def __init__(self, state: str | Path):
        self.state = Path(state).absolute()
        st = self.state.lstat()
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
            raise Rejected("controller state must be an operator-owned real directory with mode 0700")
        self.public = regular(self.state / "authority.pub")
        if sha(self.public) != regular(self.state / "enrolled-fingerprint").decode("ascii"):
            raise Rejected("enrolled authority fingerprint changed")
        self.config = check_signed(CONFIG_DOMAIN, parse(regular(self.state / "config.json")), self.public)
        if self.config["authority_key_id"] != sha(self.public):
            raise Rejected("configuration authority mismatch")
        self.source = regular(self.state / "instruction.bin")
        self.manifest_raw = regular(self.state / "source-manifest.json")
        manifest = parse(self.manifest_raw)
        if manifest != self.config["manifest"] or sha(self.manifest_raw) != self.config["policy_sha256"]:
            raise Rejected("authoritative manifest changed")
        if len(self.source) != manifest["source_length"] or sha(self.source) != manifest["source_sha256"]:
            raise Rejected("authoritative instruction changed")
        if sha(regular(self.state / "authorization.txt")) != manifest["authorization_sha256"]:
            raise Rejected("authorization changed")
        self.audit_public = unb64(self.config["audit_public_b64"])
        if sha(self.audit_public) != self.config["audit_key_id"]:
            raise Rejected("audit key mismatch")
        self._check_runtime()

    def _check_runtime(self) -> None:
        if runtime_inventory() != self.config["runtime_sha256"]:
            raise Rejected("trusted executable runtime changed; operator re-enrollment is required")

    @contextlib.contextmanager
    def locked(self):
        fd = os.open(self.state / "controller.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _records(self) -> list[dict]:
        records = []
        previous = ZERO_HASH
        raw = regular(self.state / "audit.jsonl")
        if raw and not raw.endswith(b"\n"):
            raise Rejected("torn audit record")
        for index, line in enumerate(raw.splitlines()):
            envelope = parse(line)
            event = check_signed(AUDIT_DOMAIN, envelope, self.audit_public)
            if event["schema"] != "audit-event-v3" or event["sequence"] != index + 1 or event["previous_sha256"] != previous:
                raise Rejected("audit chain was edited, removed, reordered, or duplicated")
            previous = sha(line)
            records.append(envelope)
        anchor = check_signed(ANCHOR_DOMAIN, parse(regular(self.state / "anchor.json")), self.audit_public)
        if anchor != {"schema": "audit-anchor-v3", "count": len(records), "head_sha256": previous, "audit_key_id": self.config["audit_key_id"]}:
            raise Rejected("audit checkpoint mismatch; truncation or interrupted commit requires operator review")
        return records

    def _append(self, kind: str, data: dict) -> dict:
        records = self._records()
        previous = sha(canonical(records[-1])) if records else ZERO_HASH
        key = raw_private(self.state / "audit.key")
        if public_bytes(key) != self.audit_public:
            raise Rejected("audit signer changed")
        event = {"schema": "audit-event-v3", "sequence": len(records) + 1, "previous_sha256": previous,
                 "time_ns": time.time_ns(), "event": kind, "data": data}
        record = signed(AUDIT_DOMAIN, event, key)
        raw = canonical(record)
        fd = os.open(self.state / "audit.jsonl", os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(raw + b"\n")
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
        anchor = {"schema": "audit-anchor-v3", "count": len(records) + 1, "head_sha256": sha(raw), "audit_key_id": self.config["audit_key_id"]}
        atomic(self.state / "anchor.json", canonical(signed(ANCHOR_DOMAIN, anchor, key)), replace=True)
        return record

    def audit_verify(self) -> dict:
        with self.locked():
            records = self._records()
            for record in records:
                if record["payload"]["event"] == "completed":
                    self._verify_completed_artifacts(record)
            started = {r["payload"]["data"]["handoff_id"] for r in records if r["payload"]["event"] == "started"}
            finished = {r["payload"]["data"]["handoff_id"] for r in records if r["payload"]["event"] == "completed"}
            return {"records": len(records), "head_sha256": sha(canonical(records[-1])) if records else ZERO_HASH,
                    "audit_fingerprint": self.config["audit_key_id"],
                    "events": [r["payload"]["event"] for r in records], "unfinished_handoffs": sorted(started - finished)}

    def _verify_completed_artifacts(self, record: dict) -> None:
        data = record["payload"]["data"]
        handoff = data["handoff_id"]
        if regular(self.state / "receipts" / (handoff + ".json")) != canonical(record):
            raise Rejected("retained receipt differs from its signed audit record")
        result = self.state / "results" / handoff
        if file_digest(result / "stdout.bin")[1] != data["stdout_sha256"] or file_digest(result / "stderr.bin")[1] != data["stderr_sha256"]:
            raise Rejected("retained stdout or stderr changed")
        files, unsafe = output_inventory(result / "output")
        if files != data["output_files"] or unsafe != data["unsafe_entries"] or bool(unsafe) != data["unsafe_output"]:
            raise Rejected("retained output artifacts changed")

    def _find(self, digest: str, records: list[dict]) -> dict:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Rejected("invalid parent envelope digest")
        for record in records:
            event = record["payload"]
            if event["event"] == "issued" and event["data"]["envelope_sha256"] == digest:
                return event["data"]
        raise Rejected("parent is not in the trusted controller registry")

    def _parent(self, digest: str | None, records: list[dict]) -> bytes:
        if digest is None:
            return b""
        parent = self._find(digest, records)
        stored = regular(self.state / "jobs" / parent["handoff_id"] / "envelope.json")
        if sha(stored) != digest:
            raise Rejected("stored parent envelope changed")
        env = parse(stored)
        payload = unb64(env["payload_b64"])
        Ed25519PublicKey.from_public_bytes(self.public).verify(unb64(env["signature_b64"]), HANDOFF_DOMAIN + payload)
        completed = [r for r in records if r["payload"]["event"] == "completed"
                     and r["payload"]["data"].get("envelope_sha256") == digest
                     and r["payload"]["data"].get("exit_code") == 0
                     and r["payload"]["data"].get("unsafe_output") is False]
        if len(completed) != 1:
            raise Rejected("parent has no successfully completed authenticated dispatch")
        self._verify_completed_artifacts(completed[0])
        return bytes.fromhex(digest)

    def create(self, job_dir: str | Path, entrypoint: str, operation: str, argv: list[str] | None = None,
               parent: str | None = None, limits: dict | None = None, ttl_seconds: int = 86400) -> Path:
        source_dir = Path(job_dir).resolve(strict=True)
        state_dir = self.state.resolve(strict=True)
        if source_dir == state_dir or source_dir.is_relative_to(state_dir) or state_dir.is_relative_to(source_dir):
            raise Rejected("job inputs must not overlap controller state")
        entrypoint = path_safe(entrypoint)
        if not isinstance(operation, str) or not operation.strip():
            raise Rejected("assigned operation must be nonempty UTF-8 text")
        operation_bytes = operation.encode("utf-8", "strict")
        args = [] if argv is None else argv
        validate_argv(args)
        chosen = dict(DEFAULT_LIMITS)
        if limits:
            if not set(limits).issubset(chosen):
                raise Rejected("unknown execution limit")
            chosen.update(limits)
        validate_limits(chosen)
        if type(ttl_seconds) is not int or ttl_seconds < 1:
            raise Rejected("handoff lifetime must be a positive integer")
        if not entrypoint.endswith(".py"):
            raise Rejected("this runtime accepts Python jobs through the trusted no-exec bootstrap")
        with self.locked():
            self._check_runtime()
            records = self._records()
            parent_raw = self._parent(parent, records)
            handoff_id = secrets.token_hex(32)
            task = {"entrypoint": entrypoint, "argv": args, "assigned_operation": operation,
                    "handoff_id": handoff_id, "parent_sha256": parent}
            if len(canonical(task)) > 65536:
                raise Rejected("task metadata exceeds the supported 65536 bytes")
            job = self.state / "jobs" / handoff_id
            code = job / "code"
            job.mkdir(mode=0o700)
            code.mkdir(mode=0o700)
            entries = []
            for path in sorted(source_dir.rglob("*")):
                relative = path.relative_to(source_dir).as_posix()
                path_safe(relative)
                info = path.lstat()
                if path.is_symlink():
                    raise Rejected("symlinks are not permitted in signed job inputs")
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise Rejected("special files are not permitted in signed job inputs")
                if relative.split("/", 1)[0] in RESERVED:
                    raise Rejected("job conflicts with a protected handoff filename")
                data = regular(path)
                destination = code / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic(destination, data, mode=0o400)
                entries.append({"path": relative, "size": len(data), "sha256": sha(data)})
            if entrypoint not in {item["path"] for item in entries}:
                raise Rejected("entrypoint is missing from signed snapshot")
            source_files = {"instruction.bin": self.source, "source-manifest.json": self.manifest_raw,
                            "authorization.txt": regular(self.state / "authorization.txt"),
                            "task.json": canonical(task)}
            for name, data in source_files.items():
                atomic(code / name, data, mode=0o400)
                entries.append({"path": name, "size": len(data), "sha256": sha(data)})
            entries.sort(key=lambda item: item["path"])
            manifest = {"schema": "job-files-v3", "files": entries}
            manifest_raw = canonical(manifest)
            if len(entries) > 100000 or len(manifest_raw) > 16 * 1024 * 1024:
                raise Rejected("signed file manifest exceeds supported file-count or byte limits")
            atomic(code / "manifest.json", manifest_raw, mode=0o400)
            now = time.time_ns()
            execution = {"schema": "isolated-python-execution-v3", "entrypoint": entrypoint, "argv": args,
                         "manifest_sha256": sha(manifest_raw), "files": entries, "limits": chosen,
                         "source_length": len(self.source), "issued_ns": now,
                         "expires_ns": now + ttl_seconds * 1_000_000_000,
                         "authorization_sha256": self.config["manifest"]["authorization_sha256"],
                         "runtime_sha256": sha(canonical(self.config["runtime_sha256"]))}
            fields = [self.source, bytes.fromhex(self.config["policy_sha256"]), bytes.fromhex(sha(self.source)),
                      operation_bytes, parent_raw, bytes.fromhex(handoff_id), bytes.fromhex(sha(self.public)), canonical(execution)]
            encoded = kernel("encode", fields)
            if len(encoded) != 1:
                raise Rejected("compiled kernel did not return exactly one encoded payload")
            payload = encoded[0]
            key = raw_private(self.state / "authority.key")
            if public_bytes(key) != self.public:
                raise Rejected("authority signer changed")
            envelope = {"schema": SCHEMA, "payload_b64": b64(payload), "signature_b64": b64(key.sign(HANDOFF_DOMAIN + payload))}
            envelope_raw = canonical(envelope)
            if len(envelope_raw) > 16 * 1024 * 1024:
                raise Rejected("signed envelope exceeds the supported 16 MiB")
            atomic(job / "envelope.json", envelope_raw, mode=0o400)
            atomic(code / "envelope.json", envelope_raw, mode=0o400)
            self._append("issued", {"handoff_id": handoff_id, "envelope_sha256": sha(envelope_raw), "parent_sha256": parent,
                                     "operation_sha256": sha(operation_bytes), "manifest_sha256": sha(manifest_raw)})
            return job / "envelope.json"

    def _verify(self, envelope: str | Path, records: list[dict], *, require_unspent: bool = True) -> dict:
        raw = regular(Path(envelope))
        env = parse(raw)
        if not isinstance(env, dict) or set(env) != {"schema", "payload_b64", "signature_b64"} or env["schema"] != SCHEMA:
            raise Rejected("invalid handoff envelope")
        payload = unb64(env["payload_b64"])
        Ed25519PublicKey.from_public_bytes(self.public).verify(unb64(env["signature_b64"]), HANDOFF_DOMAIN + payload)
        # The untrusted digest is used only as a lookup key. The expected parent comes from the signed registry.
        envelope_hash = sha(raw)
        issued = self._find(envelope_hash, records)
        expected_parent = self._parent(issued["parent_sha256"], records)
        fields = kernel("verify", [payload, self.source, bytes.fromhex(self.config["policy_sha256"]),
                                   bytes.fromhex(sha(self.source)), expected_parent])
        if len(fields) != 8:
            raise Rejected("compiled verifier returned an incomplete handoff")
        original, policy_hash, source_hash, operation, parent, handoff, key_id, execraw = fields
        if len(handoff) != 32 or handoff.hex() != issued["handoff_id"] or key_id != bytes.fromhex(sha(self.public)):
            raise Rejected("handoff or enrolled signing key mismatch")
        if sha(operation) != issued["operation_sha256"] or not operation.decode("utf-8", "strict").strip():
            raise Rejected("assigned operation changed")
        execution = parse(execraw)
        keys = {"schema", "entrypoint", "argv", "manifest_sha256", "files", "limits", "source_length", "issued_ns", "expires_ns", "authorization_sha256", "runtime_sha256"}
        if not isinstance(execution, dict) or set(execution) != keys or execution["schema"] != "isolated-python-execution-v3":
            raise Rejected("invalid execution specification")
        if execution["source_length"] != len(self.source) or execution["authorization_sha256"] != self.config["manifest"]["authorization_sha256"]:
            raise Rejected("execution instruction or authorization mismatch")
        if execution["runtime_sha256"] != sha(canonical(self.config["runtime_sha256"])):
            raise Rejected("execution runtime differs from enrolled runtime")
        if type(execution["issued_ns"]) is not int or type(execution["expires_ns"]) is not int or not execution["issued_ns"] <= time.time_ns() < execution["expires_ns"]:
            raise Rejected("handoff is not within its signed validity interval")
        validate_limits(execution["limits"])
        validate_argv(execution["argv"])
        if require_unspent and any(r["payload"]["event"] == "started" and r["payload"]["data"].get("handoff_id") == handoff.hex() for r in records):
            raise Rejected("handoff has already been consumed; replay is forbidden")
        code = self.state / "jobs" / handoff.hex() / "code"
        manifest_raw = regular(code / "manifest.json")
        if sha(manifest_raw) != execution["manifest_sha256"] or sha(manifest_raw) != issued["manifest_sha256"]:
            raise Rejected("signed job manifest changed")
        manifest = parse(manifest_raw)
        if manifest != {"schema": "job-files-v3", "files": execution["files"]}:
            raise Rejected("manifest does not match signed job files")
        expected = set()
        for item in execution["files"]:
            if set(item) != {"path", "size", "sha256"}:
                raise Rejected("invalid file manifest entry")
            name = path_safe(item["path"])
            if name in expected:
                raise Rejected("duplicate job path")
            expected.add(name)
            data = regular(code / name)
            if len(data) != item["size"] or sha(data) != item["sha256"]:
                raise Rejected("signed input content changed: " + name)
        actual = set()
        for path in code.rglob("*"):
            if path.is_symlink():
                raise Rejected("symlink appeared in signed snapshot")
            if path.is_file():
                actual.add(path.relative_to(code).as_posix())
            elif not path.is_dir():
                raise Rejected("special file appeared in signed snapshot")
        if actual != expected | {"manifest.json", "envelope.json"} or regular(code / "envelope.json") != raw:
            raise Rejected("signed snapshot file set or delivered envelope changed")
        if regular(code / "instruction.bin") != self.source:
            raise Rejected("delivered authoritative instruction changed")
        if parse(regular(code / "task.json")) != {"entrypoint": execution["entrypoint"], "argv": execution["argv"],
                                               "assigned_operation": operation.decode("utf-8"),
                                               "handoff_id": handoff.hex(), "parent_sha256": parent.hex() if parent else None}:
            raise Rejected("bootstrap task differs from signed execution")
        return {"handoff_id": handoff.hex(), "envelope_sha256": envelope_hash,
                "parent_sha256": parent.hex() if parent else None,
                "operation": operation.decode("utf-8"), "execution": execution, "code": str(code)}

    def verify(self, envelope: str | Path) -> dict:
        with self.locked():
            self._check_runtime()
            return self._verify(envelope, self._records())

    def dispatch(self, envelope: str | Path) -> dict:
        with self.locked():
            self._check_runtime()
            data = self._verify(envelope, self._records())
            handoff = data["handoff_id"]
            limits = data["execution"]["limits"]
            result_dir = self.state / "results" / handoff
            output_dir = result_dir / "output"
            result_dir.mkdir(mode=0o700)
            output_dir.mkdir(mode=0o700)
            self._append("started", {"handoff_id": handoff, "envelope_sha256": data["envelope_sha256"],
                                     "source_sha256": sha(self.source), "parent_sha256": data["parent_sha256"]})
        # The consumed record is durable before any worker can begin. A crash never makes this id reusable.
        command = [str(ROOT / "native" / "gate-sandbox"), "--controller-pid", str(os.getpid()), "--job", data["code"], "--output", str(output_dir),
                   "--timeout", str(limits["timeout_seconds"]), "--memory-mib", str(limits["memory_mib"]),
                   "--file-mib", str(limits["file_mib"]), "--cpu-seconds", str(limits["cpu_seconds"])]
        stdout_path, stderr_path = result_dir / "stdout.bin", result_dir / "stderr.bin"
        started = time.time_ns()
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, check=False,
                                        timeout=limits["timeout_seconds"] + 15)
                returncode = result.returncode
                stdout.flush(); os.fsync(stdout.fileno())
                stderr.flush(); os.fsync(stderr.fileno())
        except Exception as exc:
            with self.locked():
                self._append("controller_failure", {"handoff_id": handoff, "envelope_sha256": data["envelope_sha256"],
                                                     "error_type": type(exc).__name__})
            raise
        output_files, unsafe_entries = output_inventory(output_dir)
        receipt_data = {"handoff_id": handoff, "envelope_sha256": data["envelope_sha256"], "source_sha256": sha(self.source),
                        "source_length": len(self.source), "parent_sha256": data["parent_sha256"], "exit_code": returncode,
                        "started_ns": started, "finished_ns": time.time_ns(), "output_files": output_files,
                        "stdout_sha256": file_digest(stdout_path)[1], "stderr_sha256": file_digest(stderr_path)[1],
                        "unsafe_output": bool(unsafe_entries), "unsafe_entries": unsafe_entries, "output_directory": str(output_dir)}
        with self.locked():
            record = self._append("completed", receipt_data)
            atomic(self.state / "receipts" / (handoff + ".json"), canonical(record), mode=0o400)
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("keygen")
    p.add_argument("--private", type=Path, required=True); p.add_argument("--public", type=Path, required=True)
    p = sub.add_parser("init")
    p.add_argument("--state", type=Path, required=True); p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--authority-key", type=Path, required=True); p.add_argument("--expected-fingerprint", required=True)
    p.add_argument("--authorization", type=Path, required=True)
    for action in ["create", "verify", "dispatch", "audit"]:
        p = sub.add_parser(action); p.add_argument("--state", type=Path, required=True)
        if action == "create":
            p.add_argument("--job", type=Path, required=True); p.add_argument("--entrypoint", required=True)
            p.add_argument("--operation", required=True); p.add_argument("--parent")
            p.add_argument("--ttl-seconds", type=int, default=86400)
            for flag, default in DEFAULT_LIMITS.items():
                p.add_argument("--" + flag.replace("_", "-"), type=int, default=default)
            p.add_argument("argv", nargs=argparse.REMAINDER)
        elif action in {"verify", "dispatch"}:
            p.add_argument("--envelope", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "keygen":
            value = {"authority_fingerprint": keygen(args.private, args.public)}
        elif args.action == "init":
            value = init_state(args.state, args.spec, args.authority_key, args.expected_fingerprint, args.authorization)
        else:
            gate = Gate(args.state)
            if args.action == "create":
                argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
                path = gate.create(args.job, args.entrypoint, args.operation, argv, args.parent,
                                   {k: getattr(args, k) for k in DEFAULT_LIMITS}, args.ttl_seconds)
                value = {"envelope": str(path), "envelope_sha256": sha(regular(path))}
            elif args.action == "verify":
                value = gate.verify(args.envelope)
            elif args.action == "dispatch":
                value = gate.dispatch(args.envelope)
            else:
                value = gate.audit_verify()
        print(canonical(value).decode("ascii"))
        if args.action == "dispatch":
            return 0 if value["payload"]["data"]["exit_code"] == 0 and not value["payload"]["data"]["unsafe_output"] else 1
        return 0
    except Exception as exc:
        print(canonical({"status": "REJECTED", "type": type(exc).__name__, "reason": str(exc)}).decode("ascii"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
