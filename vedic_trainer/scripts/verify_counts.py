"""Single authority for every test count this package reports.

Why this exists
---------------
Counts were repeatedly reported wrong because they were *inferred from
terminal output* rather than measured. The specific mechanism: `pytest -q`
prints one character per test and **no summary line**, so a `| tail` of that
output physically cannot contain the number. Counts were then read off by
eyeballing wrapped dots — which produced 145 for a suite of 217.

The structural fix is not "be careful reading pytest output". It is to make
the number come from one place that computes it, and to make every document
that quotes a count checkable against that place.

Usage
-----
    python scripts/verify_counts.py            # print the measured counts
    python scripts/verify_counts.py --check    # also verify README agrees
                                               # exits 1 on any mismatch

`--check` is the gate. Run it before quoting a count anywhere.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# layer name -> pytest paths. This mapping IS the README table's meaning;
# if a layer is added here it must appear in the README, and vice versa.
LAYERS: dict[str, list[str]] = {
    "Kernel (ℚ)": [
        "vedic/kernel/tests/test_simulator_match.py",
        "vedic/kernel/tests/test_conservation_laws.py",
        "vedic/kernel/tests/test_interaction_matrix.py",
    ],
    "Operands": ["vedic/kernel/tests/test_sutra_operands.py"],
    "Composition": ["vedic/kernel/tests/test_composition.py"],
    "Canonical 29": ["vedic/kernel/tests/test_sutras_canonical.py"],
    "Blueprint gates": ["vedic/kernel/tests/test_blueprint_gates.py"],
    "Kernel (torch)": ["vedic/kernel/tests/test_torch_buffers.py"],
    "Data": ["vedic/data/tests"],
    "External sidecar": ["vedic/external/tests"],
    "Script validity": ["vedic/kernel/tests/test_scripts_are_valid.py"],
    "Reported numbers": ["vedic/kernel/tests/test_reported_ablation.py"],
}

_SUMMARY = re.compile(r"(\d+) passed")


def measure(paths: list[str]) -> int:
    """Collected-and-passing count, read from pytest's summary line.

    Deliberately does NOT pass -q: that flag suppresses the summary, which is
    the exact condition that made the number unreadable in the first place.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "--no-header"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/usr/local/bin:/bin"},
    )
    m = _SUMMARY.search(proc.stdout)
    if not m:
        raise RuntimeError(
            f"no pytest summary line for {paths}; output tail:\n"
            + "\n".join(proc.stdout.strip().splitlines()[-5:])
        )
    return int(m.group(1))


def readme_counts() -> dict[str, int]:
    """Counts the README currently claims, parsed from its status table."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = re.search(r"(\d+)\s+(?:tests?|buffer tests?)", cells[2])
        if m:
            out[cells[0]] = int(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the README table matches; exit 1 on mismatch")
    args = ap.parse_args()

    measured = {name: measure(paths) for name, paths in LAYERS.items()}
    total = measure(["vedic/"])

    width = max(len(n) for n in measured)
    for name, n in measured.items():
        print(f"  {name:<{width}}  {n:>4}")
    print(f"  {'TOTAL (vedic/)':<{width}}  {total:>4}")

    if not args.check:
        return 0

    claimed = readme_counts()
    problems: list[str] = []
    for name, n in measured.items():
        if name not in claimed:
            problems.append(f"README has no row for layer {name!r} ({n} tests)")
        elif claimed[name] != n:
            problems.append(
                f"README says {name!r} = {claimed[name]}, measured {n}")
    for name in claimed:
        if name not in measured:
            problems.append(f"README row {name!r} maps to no layer here")

    if problems:
        print("\nMISMATCH:")
        for p in problems:
            print(f"  - {p}")
        print(f"\n{len(problems)} count claims do not match the code.")
        return 1
    print("\nOK — every README count matches the measured suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
