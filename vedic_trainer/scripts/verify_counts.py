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
    "Data": ["vedic/data/tests/test_synthetic_quality.py"],
    "Split integrity": ["vedic/data/tests/test_split_is_disjoint.py"],
    "External sidecar": ["vedic/external/tests"],
    "Script validity": ["vedic/kernel/tests/test_scripts_are_valid.py"],
    "Reported numbers": ["vedic/kernel/tests/test_reported_ablation.py"],
    "Documented paths": ["vedic/kernel/tests/test_documented_paths.py"],
    "Conservation (torch)": ["vedic/kernel/tests/test_conservation_torch.py"],
    "Audit closure": ["vedic/eval/tests/test_audit_closure_degeneracy.py"],
    "Benchmark honesty": ["vedic/eval/tests/test_no_subset_is_quoted_as_a_benchmark.py"],
    "Gates reject": ["vedic/kernel/tests/test_gates_reject.py"],
    "Aux checkpoint": ["vedic/training/tests"],
}

_COLLECTED = re.compile(r"(\d+) tests? collected")
_FAILED = re.compile(r"(\d+) (?:failed|error)")


def measure(paths: list[str]) -> int:
    """Number of tests COLLECTED, not passed.

    It used to be the "N passed" count, which is environment-dependent: three
    tests here are skipped when no Lean toolchain is present, so the same
    README was correct on a machine with Lean and wrong in CI without it. A
    count that changes with the machine cannot be a claim about the suite.

    Collection is environment-independent -- a skipped test is still collected
    -- so this is what the README's numbers mean. Whether they *pass* is a
    separate question, asked by `failures()`.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "--collect-only", "-q",
         "--no-header"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/usr/local/bin:/bin"},
    )
    m = _COLLECTED.search(proc.stdout)
    if not m:
        raise RuntimeError(
            f"no collection summary for {paths}; output tail:\n"
            + "\n".join(proc.stdout.strip().splitlines()[-5:])
        )
    return int(m.group(1))


def failures() -> tuple[int, str]:
    """Run the suite; return (failure count, the summary line).

    Counting collected tests alone would let the README stay "correct" while
    every one of them failed. Deliberately does NOT pass -q: that suppresses
    the summary line, the exact condition this script exists to prevent.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "vedic/", "--no-header"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/usr/local/bin:/bin"},
    )
    tail = [ln for ln in proc.stdout.strip().splitlines()
            if " passed" in ln or " failed" in ln or " error" in ln]
    line = tail[-1] if tail else "(no summary line)"
    m = _FAILED.search(line)
    return (int(m.group(1)) if m else 0), line.strip()


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


def reconcile(measured: dict[str, int], total: int,
              claimed: dict[str, int], n_failed: int = 0,
              summary: str = "") -> list[str]:
    """Judge measured counts against claimed ones. Pure: no I/O, no pytest.

    Split out of `main` so the judgment can be regeneration-tested directly.
    A gate whose decision logic can only be exercised by running the entire
    suite is a gate nobody re-tests -- and the decisions are exactly the part
    that can silently stop working.
    """
    problems: list[str] = []

    if n_failed:
        problems.append(
            f"{n_failed} test(s) do not pass: {summary}. A count of collected "
            f"tests says nothing about whether they work.")

    # The layers must account for every test. Without this, a new test file
    # that no layer names is simply invisible: the README stays "correct"
    # while the suite it describes has grown past it.
    layered = sum(measured.values())
    if layered != total:
        problems.append(
            f"layers sum to {layered} but the suite has {total} tests — "
            f"{total - layered} belong to no layer. Add them to LAYERS (and "
            f"to the README table) rather than leaving them unaccounted.")

    claimed = claimed
    for name, n in measured.items():
        if name not in claimed:
            problems.append(f"README has no row for layer {name!r} ({n} tests)")
        elif claimed[name] != n:
            problems.append(
                f"README says {name!r} = {claimed[name]}, measured {n}")
    for name in claimed:
        if name not in measured:
            problems.append(f"README row {name!r} maps to no layer here")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the README table matches; exit 1 on mismatch")
    args = ap.parse_args()

    measured = {name: measure(paths) for name, paths in LAYERS.items()}
    total = measure(["vedic/"])
    n_failed, summary = failures()

    width = max(len(n) for n in measured)
    for name, n in measured.items():
        print(f"  {name:<{width}}  {n:>4}")
    print(f"  {'TOTAL (vedic/)':<{width}}  {total:>4}   (collected)")
    print(f"\n  suite: {summary}")

    if not args.check:
        return 0

    problems = reconcile(measured, total, readme_counts(),
                         n_failed, summary)

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
