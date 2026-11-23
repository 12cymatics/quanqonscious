"""Package complete experiment states and candidate logs without losing digits.

Each complete model receives a deterministic tar.gz split into parts no larger
than two MiB.  Original files remain in results/states and results/traces.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path


PART_BYTES = 2 * 1024 * 1024
SCHEMA = "grvq-exact-data-archives/v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.replace(temporary, path)


def _run_paths(root, run):
    state = Path(run["state_file"])
    trace = Path(run["complete_trace"])
    if state.is_absolute() or ".." in state.parts or state.parts[0] != "states":
        raise ValueError("An experiment state path is outside results/states")
    if trace.is_absolute() or ".." in trace.parts or len(trace.parts) != 1:
        raise ValueError("An experiment trace path must be a filename")
    for relative in (Path("results")/state, Path("results/traces")/trace):
        path = root/relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"A completed run is missing its regular data file: {relative}")
        yield relative.as_posix(), path


def _completed_models(result):
    configuration = result["configuration"]
    expected = {(family, seed) for family in configuration["families"] for seed in configuration["seeds"]}
    if len(expected) != 21:
        raise ValueError("This release requires exactly twenty-one family/seed runs per model")
    grouped = defaultdict(list)
    for run in result["runs"]:
        grouped[run["model"]].append(run)
    complete, pending = {}, {}
    for model in (entry["name"] for entry in result["models"]):
        runs = grouped[model]
        identities = [(run["family"], run["seed"]) for run in runs]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate run identities for {model}")
        if len(runs) == len(expected) and set(identities) == expected:
            if any(run["objective_evaluations"] != configuration["budget"] for run in runs):
                raise ValueError(f"Model {model} has an unfinished evaluation budget")
            complete[model] = runs
        else:
            pending[model] = len(runs)
    return complete, pending


def _package_model(root, destination, model, runs, budget):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model):
        raise ValueError("A model name cannot be used safely as an archive basename")
    sources = {}
    for run in runs:
        for relative, path in _run_paths(root, run):
            if relative in sources:
                raise ValueError(f"Run data path is shared: {relative}")
            sources[relative] = path
    files = []
    for relative, path in sorted(sources.items()):
        record = {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if relative.startswith("results/traces/"):
            with gzip.open(path, "rb") as stream:
                evaluations = sum(1 for _ in stream)
            if evaluations != budget:
                raise ValueError(f"{relative} contains {evaluations} candidates; expected {budget}")
            record["candidate_evaluations"] = evaluations
        files.append(record)
    with tempfile.TemporaryDirectory(prefix="grvq-model-archive-") as temporary:
        archive = Path(temporary)/(model+".tar.gz")
        with archive.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                    for record in files:
                        info = tarfile.TarInfo(record["path"])
                        info.size = record["bytes"]
                        info.mtime = 0
                        info.mode = 0o644
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with sources[record["path"]].open("rb") as stream:
                            tar.addfile(info, stream)
        archive_digest = file_sha256(archive)
        parts = []
        with archive.open("rb") as stream:
            index = 1
            while True:
                block = stream.read(PART_BYTES)
                if not block:
                    break
                name = f"{model}.tar.gz.part{index:03d}"
                _atomic_bytes(destination/name, block)
                parts.append({"path": name, "bytes": len(block),
                              "sha256": hashlib.sha256(block).hexdigest(),
                              "git_blob_sha1": git_blob_sha(block)})
                index += 1
        record = {"model": model, "run_count": len(runs), "file_count": len(files),
                  "runs": [{"family": run["family"], "seed": run["seed"],
                            "objective_evaluations": run["objective_evaluations"]}
                           for run in sorted(runs,key=lambda r:(r["family"],r["seed"]))],
                  "archive": {"name": model+".tar.gz", "bytes": archive.stat().st_size,
                              "sha256": archive_digest, "format": "deterministic ustar+gzip"},
                  "parts": parts, "files": files}
    for file_record in files:
        if file_sha256(sources[file_record["path"]]) != file_record["sha256"]:
            raise RuntimeError(f"A completed run changed during packaging: {file_record['path']}")
    return record


def pack_exact_data(root=None, models=None):
    root = Path(root or Path(__file__).resolve().parent).resolve()
    results = root/"results/results.json"
    data = json.loads(results.read_text())
    complete, pending = _completed_models(data)
    selected = set(complete) if models is None else set(models)
    if selected-set(complete):
        raise ValueError(f"Models are not complete: {sorted(selected-set(complete))}")
    destination = root/"results/exact_data"
    destination.mkdir(parents=True,exist_ok=True)
    manifest_path = destination/"manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    packaged = {m["model"]:m for m in previous["models"]} if previous and models is not None else {}
    for model in sorted(selected):
        packaged[model] = _package_model(root,destination,model,complete[model],data["configuration"]["budget"])
    manifest = {"schema":SCHEMA, "source_commit":data["source_commit"],
                "part_limit_bytes":PART_BYTES,"compression":"lossless gzip of original bytes",
                "restored_paths":"Relative to the experiments/grvq_ansatz_comparison directory",
                "model_count":len(packaged),"models":[packaged[m] for m in sorted(packaged)],
                "pending_models":pending}
    _atomic_bytes(manifest_path,(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--models",nargs="+")
    args = parser.parse_args()
    manifest = pack_exact_data(args.root,args.models)
    print(json.dumps({"models":manifest["model_count"],
                      "parts":sum(len(m["parts"]) for m in manifest["models"]),
                      "archive_bytes":sum(m["archive"]["bytes"] for m in manifest["models"]),
                      "pending_models":manifest["pending_models"]}))
