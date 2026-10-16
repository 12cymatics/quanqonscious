"""What a run record must say about where it came from.

Every training and evaluation script writes this block beside its numbers.
A figure that cannot be traced to the code, the library versions, the model
weights and the data that produced it cannot be reproduced. This package
has already had to withdraw results for exactly that reason: they were
measured on a corpus it no longer contains, under loss definitions it no
longer implements.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

#: Distributions whose versions decide the numbers a run produces.
DISTRIBUTIONS = ("torch", "transformers", "peft", "datasets", "accelerate",
                 "tokenizers", "safetensors", "numpy", "pydantic")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True,
                          capture_output=True, text=True).stdout


def code_state() -> dict[str, Any]:
    """The commit the code ran from, and whether the tree differed from it.

    Raises when git cannot answer: a record that says nothing about its code
    is the defect this module exists to prevent.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".")
    return {"commit": _git("rev-parse", "HEAD").strip(),
            "tracked_files_modified": bool(status.strip())}


def library_versions() -> dict[str, str]:
    return {d: metadata.version(d) for d in DISTRIBUTIONS}


def environment() -> dict[str, Any]:
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_parallel_info": torch.__config__.parallel_info(),
        "libraries": library_versions(),
        "code": code_state(),
    }
