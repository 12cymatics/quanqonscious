"""Verify and restore complete exact states and optimisation candidate logs.

Uses only the Python standard library.  Every archive part, complete archive
and original member is checked against its SHA-256 before restoration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


SCHEMA = "grvq-exact-data-archives/v1"


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def _relative(name, member=False):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    if member:
        if len(path.parts) != 3 or path.parts[0] != "results" or path.parts[1] not in ("states","traces"):
            raise ValueError(f"Unexpected restored path: {name}")
    elif len(path.parts) != 1:
        raise ValueError("Archive parts must be adjacent filenames")
    return Path(*path.parts)


def unpack_exact_data(manifest_path=None,destination=None,models=None,verify_only=False):
    base = Path(__file__).resolve().parent
    manifest_path = Path(manifest_path or base/"results/exact_data/manifest.json").resolve()
    destination = Path(destination or base).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported exact-data manifest")
    known = {model["model"]:model for model in manifest["models"]}
    selected = set(known) if models is None else set(models)
    if selected-set(known):
        raise ValueError(f"Unknown models: {sorted(selected-set(known))}")
    restored_files = restored_bytes = 0
    for model_name in sorted(selected):
        model = known[model_name]
        expected = {file["path"]:file for file in model["files"]}
        if len(expected) != model["file_count"]:
            raise ValueError("Duplicate or missing file manifest entries")
        with tempfile.TemporaryDirectory(prefix="grvq-exact-data-") as temporary:
            archive_path = Path(temporary)/"model.tar.gz"
            with archive_path.open("wb") as archive:
                for part in model["parts"]:
                    path = manifest_path.parent/_relative(part["path"])
                    if path.stat().st_size != part["bytes"] or _hash(path) != part["sha256"]:
                        raise ValueError(f"Archive part failed integrity verification: {part['path']}")
                    with path.open("rb") as source:
                        shutil.copyfileobj(source,archive,1024*1024)
            if archive_path.stat().st_size != model["archive"]["bytes"] or _hash(archive_path) != model["archive"]["sha256"]:
                raise ValueError(f"Archive failed integrity verification: {model_name}")
            observed = set()
            with tarfile.open(archive_path,"r:gz") as archive:
                for member in archive:
                    relative = _relative(member.name,member=True)
                    if not member.isfile() or member.name not in expected or member.name in observed:
                        raise ValueError(f"Unexpected or repeated archive member: {member.name}")
                    record = expected[member.name]
                    if member.size != record["bytes"]:
                        raise ValueError(f"Member length mismatch: {member.name}")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Unreadable archive member: {member.name}")
                    target = destination/relative
                    if not target.resolve().is_relative_to(destination):
                        raise ValueError(f"Restored path escapes destination: {member.name}")
                    digest = hashlib.sha256()
                    written = 0
                    if verify_only:
                        for block in iter(lambda:source.read(1024*1024),b""):
                            digest.update(block); written += len(block)
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True)
                        with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as output:
                            temporary_file = Path(output.name)
                            for block in iter(lambda:source.read(1024*1024),b""):
                                digest.update(block); written += len(block); output.write(block)
                        if written != record["bytes"] or digest.hexdigest() != record["sha256"]:
                            temporary_file.unlink()
                            raise ValueError(f"Restored member failed integrity verification: {member.name}")
                        os.replace(temporary_file,target)
                    if written != record["bytes"] or digest.hexdigest() != record["sha256"]:
                        raise ValueError(f"Member failed integrity verification: {member.name}")
                    observed.add(member.name)
                    restored_files += 1
                    restored_bytes += written
            if observed != set(expected):
                raise ValueError(f"Archive omitted expected members: {model_name}")
    return {"models":len(selected),"files":restored_files,"bytes":restored_bytes,
            "verified":True,"restored":not verify_only,"destination":str(destination)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--destination",type=Path)
    parser.add_argument("--models",nargs="+")
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    print(json.dumps(unpack_exact_data(args.manifest,args.destination,args.models,args.verify_only)))
