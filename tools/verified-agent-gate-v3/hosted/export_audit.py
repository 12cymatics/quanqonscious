#!/usr/bin/env python3
"""Export and verify a public, signed, read-only record of one hosted batch."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate3 import Gate, CONFIG_DOMAIN, AUDIT_DOMAIN, ANCHOR_DOMAIN, canonical, parse, check_signed, sha

EXPORT_DOMAIN = b"VERIFIED-AGENT-GATE-PUBLIC-EXPORT-V1\0"
MAX_BYTES = 256 * 1024 * 1024
MAX_FILES = 10000
PUBLIC_STATE_FILES = (
    "authority.pub", "config.json", "instruction.bin", "authorization.txt",
    "source-manifest.json", "audit.jsonl", "anchor.json", "enrolled-fingerprint",
)


def regular_bytes(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("Export accepts bounded regular files only")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) != info.st_size or len(raw) > MAX_BYTES:
            raise ValueError("Export file changed or exceeded its limit")
        return raw
    finally:
        os.close(fd)


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or path.as_posix() != name or any(p in {"", ".", ".."} for p in name.split("/")) or "\\" in name or "\0" in name:
        raise ValueError("Unsafe archive path")
    return name


def collect_tree(directory: Path, prefix: str, add, omissions: list[dict]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected an ordinary export directory")
    pending = [directory]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                name = safe_name(prefix + "/" + path.relative_to(directory).as_posix())
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    add(name, regular_bytes(path))
                elif stat.S_ISLNK(info.st_mode):
                    omissions.append({"path": name, "kind": "symlink", "target": os.readlink(path)})
                else:
                    omissions.append({"path": name, "kind": "special", "mode": stat.S_IFMT(info.st_mode)})


def export_public(state: Path, results: Path | None, destination: Path, metadata: dict,
                  sensitive_values: tuple[bytes, ...] = (), build_provenance: Path | None = None) -> dict:
    state = state.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(state):
        raise ValueError("Archive must be new and outside private controller state")
    gate = Gate(state)
    audit_summary = gate.audit_verify()
    private_authority = regular_bytes(state / "authority.key")
    private_audit = regular_bytes(state / "audit.key")
    sensitive_values = sensitive_values + tuple(
        representation
        for key in (private_authority, private_audit)
        for representation in (key, base64.b64encode(key), key.hex().encode("ascii"))
    )
    files: dict[str, bytes] = {}
    omissions: list[dict] = []
    total = 0

    def add(name: str, raw: bytes) -> None:
        nonlocal total
        name = safe_name(name)
        if name in files or len(files) >= MAX_FILES or total + len(raw) > MAX_BYTES:
            raise ValueError("Duplicate archive path or export limit exceeded")
        if any(secret and secret in raw for secret in sensitive_values):
            raise ValueError("A credential value appeared in public export material; publication blocked")
        files[name] = raw
        total += len(raw)

    for name in PUBLIC_STATE_FILES:
        add("controller/" + name, regular_bytes(state / name))
    collect_tree(state / "receipts", "controller/receipts", add, omissions)
    collect_tree(state / "results", "controller/results", add, omissions)
    for job in sorted((state / "jobs").iterdir()):
        if job.is_symlink() or not job.is_dir() or len(job.name) != 64 or any(c not in "0123456789abcdef" for c in job.name):
            raise ValueError("Unexpected controller job directory")
        add("controller/jobs/" + job.name + "/envelope.json", regular_bytes(job / "envelope.json"))
        collect_tree(job / "code", "controller/jobs/" + job.name + "/code", add, omissions)
    if results is not None and results.exists():
        collect_tree(results.resolve(strict=True), "agent-results", add, omissions)
    if build_provenance is not None:
        add("build-provenance.json", regular_bytes(build_provenance))
    add("audit-verification.json", canonical(audit_summary))
    add("run-metadata.json", canonical(metadata))
    add("omitted-unsafe-entries.json", canonical(sorted(omissions, key=lambda x: x["path"])))

    private = Ed25519PrivateKey.from_private_bytes(private_authority)
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    if public != files["controller/authority.pub"]:
        raise ValueError("Public export signer differs from enrolled authority")
    inventory = {
        "schema": "verified-agent-gate-public-export-v1", "authority_fingerprint": sha(public),
        "github": metadata.get("github", {}), "batch_scope": "one-workflow-run",
        "resumable_controller_state": False,
        "files": [{"path": name, "size": len(raw), "sha256": sha(raw)} for name, raw in sorted(files.items())],
    }
    signed_inventory = {"payload": inventory, "signature_b64": base64.b64encode(private.sign(EXPORT_DOMAIN + canonical(inventory))).decode("ascii")}
    manifest = canonical(signed_inventory)
    if any(secret and secret in manifest for secret in sensitive_values):
        raise ValueError("A credential value appeared in the export inventory")
    files["export-manifest.json"] = manifest
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise ValueError("Export staging file already exists")
    try:
        with temporary.open("xb") as output:
            with tarfile.open(fileobj=output, mode="w:gz") as archive:
                for name, raw in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(raw), 0o600, 0
                    info.uid = info.gid = 0
                    archive.addfile(info, io.BytesIO(raw))
            output.flush()
            os.fsync(output.fileno())
        verified = verify_public(temporary, sha(public))
        if not verified["verified"]:
            raise ValueError("Export self-verification failed")
        archive_digest = sha(regular_bytes(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"archive": str(destination), "sha256": archive_digest,
            "authority_fingerprint": sha(public), "files": len(files), "audit": audit_summary}


def verify_public(archive_path: Path, expected_fingerprint: str) -> dict:
    if len(expected_fingerprint) != 64 or any(c not in "0123456789abcdef" for c in expected_fingerprint):
        raise ValueError("An independently trusted authority fingerprint is required")
    files: dict[str, bytes] = {}
    total = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            name = safe_name(member.name)
            if not member.isfile() or name in files or len(files) >= MAX_FILES + 1 or member.size < 0 or total + member.size > MAX_BYTES + 16 * 1024 * 1024:
                raise ValueError("Archive contains an unsafe member or exceeds verification limits")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Archive member is unreadable")
            raw = stream.read(member.size + 1)
            if len(raw) != member.size:
                raise ValueError("Truncated archive member")
            files[name] = raw
            total += len(raw)
    public = files["controller/authority.pub"]
    if len(public) != 32 or sha(public) != expected_fingerprint:
        raise ValueError("Archive authority does not match the independently enrolled fingerprint")
    manifest = parse(files["export-manifest.json"])
    inventory = check_signed(EXPORT_DOMAIN, manifest, public)
    if inventory.get("schema") != "verified-agent-gate-public-export-v1" or inventory.get("authority_fingerprint") != expected_fingerprint:
        raise ValueError("Unexpected export identity")
    expected = {}
    for item in inventory["files"]:
        if set(item) != {"path", "size", "sha256"} or item["path"] in expected:
            raise ValueError("Invalid export inventory")
        expected[safe_name(item["path"])] = item
    if set(files) != set(expected) | {"export-manifest.json"}:
        raise ValueError("Export file set changed")
    for name, item in expected.items():
        if len(files[name]) != item["size"] or sha(files[name]) != item["sha256"]:
            raise ValueError("Export file contents changed")
    config = check_signed(CONFIG_DOMAIN, parse(files["controller/config.json"]), public)
    audit_public = base64.b64decode(config["audit_public_b64"], validate=True)
    if sha(audit_public) != config["audit_key_id"]:
        raise ValueError("Audit identity changed")
    source = files["controller/instruction.bin"]
    if len(source) != config["manifest"]["source_length"] or sha(source) != config["manifest"]["source_sha256"]:
        raise ValueError("Authoritative source changed")
    if sha(files["controller/authorization.txt"]) != config["manifest"]["authorization_sha256"]:
        raise ValueError("Authorization record changed")
    previous, records = "0" * 64, []
    for number, line in enumerate(files["controller/audit.jsonl"].splitlines(), 1):
        envelope = parse(line)
        record = check_signed(AUDIT_DOMAIN, envelope, audit_public)
        if record["schema"] != "audit-event-v3" or record["sequence"] != number or record["previous_sha256"] != previous:
            raise ValueError("Audit chain changed")
        previous = sha(line)
        records.append(envelope)
        if record["event"] == "completed":
            data = record["data"]
            handoff = data["handoff_id"]
            if files["controller/receipts/" + handoff + ".json"] != canonical(envelope):
                raise ValueError("Receipt differs from signed audit record")
            prefix = "controller/results/" + handoff + "/"
            for stream_name in ("stdout", "stderr"):
                if sha(files[prefix + stream_name + ".bin"]) != data[stream_name + "_sha256"]:
                    raise ValueError("Retained process log changed")
            for item in data["output_files"]:
                raw = files[prefix + "output/" + safe_name(item["path"])]
                if len(raw) != item["size"] or sha(raw) != item["sha256"]:
                    raise ValueError("Retained job output changed")
    anchor = check_signed(ANCHOR_DOMAIN, parse(files["controller/anchor.json"]), audit_public)
    if anchor != {"schema": "audit-anchor-v3", "count": len(records), "head_sha256": previous, "audit_key_id": config["audit_key_id"]}:
        raise ValueError("Audit anchor mismatch")
    return {"verified": True, "authority_fingerprint": expected_fingerprint, "records": len(records),
            "github": inventory["github"], "resumable_controller_state": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--expected-fingerprint", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_public(args.archive, args.expected_fingerprint), sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "REJECTED", "type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
