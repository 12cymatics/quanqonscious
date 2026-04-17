#!/usr/bin/env python3
"""
Strict heavy dependency verifier for the hybrid 29-sutra simulator runtime.
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict, List, Tuple

REQUIRED_MODULES: List[Tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sympy", "sympy"),
    ("matplotlib.pyplot", "matplotlib"),
    ("scipy.linalg", "scipy"),
    ("cirq", "cirq"),
    ("torch", "torch"),
    ("cudaq", "cudaq"),
]


def module_version(top_level_name: str) -> str:
    module = importlib.import_module(top_level_name)
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else "unknown"


def main() -> int:
    failures: List[Tuple[str, str]] = []
    versions: Dict[str, str] = {}
    for import_name, top_level_name in REQUIRED_MODULES:
        try:
            importlib.import_module(import_name)
            versions[import_name] = module_version(top_level_name)
        except Exception as exc:
            failures.append((import_name, str(exc)))

    if failures:
        print("Heavy dependency verification failed:")
        for name, error in failures:
            print(f" - {name}: {error}")
        return 1

    print("Heavy dependency verification passed.")
    for import_name in [item[0] for item in REQUIRED_MODULES]:
        print(f" - {import_name}: {versions[import_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
