"""Trusted Python entrypoint for a single signed, confined job."""
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import sys


def reject(message):
    print(f"gate-bootstrap: {message}", file=sys.stderr, flush=True)
    raise SystemExit(125)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            reject("duplicate JSON key")
        value[key] = item
    return value


def load_json(raw, description):
    try:
        return json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeError, ValueError) as error:
        reject(f"invalid {description}: {error}")


def regular_file(path, maximum=None):
    try:
        info = path.lstat()
    except OSError as error:
        reject(f"required file unavailable: {path}: {error}")
    if not stat.S_ISREG(info.st_mode):
        reject(f"required file is not regular: {path}")
    if maximum is not None and info.st_size > maximum:
        reject(f"required file exceeds its limit: {path}")
    return path.read_bytes()


def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        reject("invalid manifest path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or any(part in {".", ".."} for part in name.split("/")):
        reject("manifest paths must stay inside the job")
    return relative


def validate_manifest():
    manifest = load_json(regular_file(Path("/job/manifest.json"), 16 * 1024 * 1024), "manifest JSON")
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "files"} or manifest["schema"] != "job-files-v3":
        reject("unexpected manifest schema")
    files = manifest["files"]
    if not isinstance(files, list) or len(files) > 100000:
        reject("invalid manifest file list")
    entries = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            reject("invalid manifest entry")
        name = str(safe_relative(entry["path"]))
        if name in entries or name in {"manifest.json", "envelope.json"}:
            reject("duplicate or reserved manifest path")
        if type(entry["size"]) is not int or entry["size"] < 0:
            reject("invalid manifest size")
        expected_hash = entry["sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            reject("invalid manifest digest")
        entries[name] = entry
    required = {"instruction.bin", "task.json", "source-manifest.json", "authorization.txt"}
    if not required.issubset(entries):
        reject("manifest omits required authorization inputs")
    observed = set()
    directories = [Path("/job")]
    while directories:
        directory = directories.pop()
        with os.scandir(directory) as children:
            for child in children:
                info = child.stat(follow_symlinks=False)
                name = Path(child.path).relative_to("/job").as_posix()
                safe_relative(name)
                if stat.S_ISLNK(info.st_mode):
                    reject(f"job symlink prohibited: {name}")
                if stat.S_ISDIR(info.st_mode):
                    directories.append(Path(child.path))
                elif stat.S_ISREG(info.st_mode):
                    observed.add(name)
                else:
                    reject(f"job special file prohibited: {name}")
    if observed != set(entries) | {"manifest.json", "envelope.json"}:
        reject("job file set differs from the authenticated manifest")
    for name, entry in entries.items():
        target = Path("/job") / name
        info = target.stat(follow_symlinks=False)
        if info.st_size != entry["size"]:
            reject(f"job file size mismatch: {name}")
        value = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        if value.hexdigest() != entry["sha256"]:
            reject(f"job file hash mismatch: {name}")


def relative_file(name):
    relative = safe_relative(name)
    if relative.suffix != ".py":
        reject("entrypoint must be a Python source file")
    candidate = Path("/job")
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            reject("entrypoint symlinks are prohibited")
    regular_file(candidate)
    return candidate


def main():
    native = ctypes.CDLL("/gate/libgate_seccomp.so", use_errno=True)
    native.gate_worker_policy_version.argtypes = []
    native.gate_worker_policy_version.restype = ctypes.c_int
    native.gate_install_worker_filter.argtypes = []
    native.gate_install_worker_filter.restype = ctypes.c_int
    if native.gate_worker_policy_version() != 1:
        reject("unsupported native worker policy")
    if native.gate_install_worker_filter() != 0:
        reject(f"mandatory worker filter failed: errno {ctypes.get_errno()}")
    sys.dont_write_bytecode = True
    validate_manifest()
    task_raw = regular_file(Path("/job/task.json"), 65536)
    task = load_json(task_raw, "task JSON")
    if not isinstance(task, dict) or set(task) != {"entrypoint", "argv", "assigned_operation", "handoff_id", "parent_sha256"}:
        reject("unexpected task fields")
    operation = task["assigned_operation"]
    if not isinstance(operation, str) or not operation.strip() or "\0" in operation:
        reject("invalid assigned operation")
    try:
        operation.encode("utf-8", "strict")
    except UnicodeError:
        reject("assigned operation is not valid UTF-8")
    handoff_id = task["handoff_id"]
    parent = task["parent_sha256"]
    for value in [handoff_id] + ([] if parent is None else [parent]):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            reject("invalid task handoff or parent digest")
    arguments = task["argv"]
    if not isinstance(arguments, list) or len(arguments) > 128 or not all(isinstance(item, str) and "\0" not in item for item in arguments):
        reject("invalid job arguments")
    if sum(len(item.encode("utf-8")) for item in arguments) > 65536:
        reject("job arguments exceed their limit")
    instruction = regular_file(Path("/job/instruction.bin"), 16 * 1024 * 1024)
    regular_file(Path("/job/envelope.json"), 16 * 1024 * 1024)
    regular_file(Path("/job/manifest.json"), 16 * 1024 * 1024)
    entrypoint = relative_file(task["entrypoint"])
    sys.path.insert(0, "/job")
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *arguments]
    os.chdir("/output")
    runpy.run_path(str(entrypoint), run_name="__main__", init_globals={
        "GATE_INSTRUCTION_BYTES": instruction,
        "GATE_INSTRUCTION_SHA256": hashlib.sha256(instruction).hexdigest(),
        "GATE_ASSIGNED_OPERATION": operation,
        "GATE_HANDOFF_ID": handoff_id,
        "GATE_PARENT_SHA256": parent,
    })


if __name__ == "__main__":
    main()
