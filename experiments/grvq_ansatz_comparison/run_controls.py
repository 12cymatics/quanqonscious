"""Reproduce all 29 conventional exact coefficient-generation controls.

Run from the extracted repository root:
    python -m experiments.grvq_ansatz_comparison.run_controls

The same consumed compiler pipeline runs unchanged, with all 29 extensions
replaced, and with each individual replacement. The default nine repetitions
shuffle all 31 variants independently within each block. Equality is
structural equality of every retained mathematical output and original-source
fingerprint. Timings, tracing, method counters and replacement IDs are excluded.
The SHA-256 serialization preserves the existing v1 comparison artifacts.
Each retained compiler output contains the last repetition, while the summary
retains every timing, exact-comparison result and block order. These controls
compare arithmetic identities, not different physical models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess


HERE = Path(__file__).resolve().parent
NONSEMANTIC = {"methods", "generation_ms", "classical_comparison_ids", "tracing"}


def mathematical_payload(record):
    return {key: value for key, value in record.items() if key not in NONSEMANTIC}


def digest(payload):
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def compile_variant(node, ids, output):
    command = [node, str(HERE / "compile_polynomials.cjs")]
    if ids:
        command.append("--classical-ids=" + ",".join(map(str, ids)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    record = json.loads(completed.stdout)
    if record.get("classical_comparison_ids") != ids:
        raise ValueError("Compiler returned different conventional replacement IDs")
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/conventional_comparison")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--order-seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to execute the original JavaScript coefficient compiler")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    variants = [([], "vedic_baseline"), (list(range(1, 30)), "classical_all29")]
    variants.extend(([i], f"classical_s{i:02d}") for i in range(1, 30))
    observations = {name: [] for _, name in variants}
    latest = {}
    expected = None
    baseline_all = True
    rng = random.Random(args.order_seed)
    orders = []
    for repeat in range(args.repeats):
        order = list(variants)
        rng.shuffle(order)
        orders.append([name for _, name in order])
        block = []
        for ids, name in order:
            actual = compile_variant(node, ids, output / (name + ".json"))
            latest[name] = actual
            block.append((name, actual, mathematical_payload(actual)))
        baseline = latest["vedic_baseline"]
        if expected is None:
            expected = mathematical_payload(baseline)
        baseline_all = baseline_all and ({m["id"] for m in baseline["methods"] if m["calls"] > 0} == set(range(1, 30)))
        for name, actual, payload in block:
            observations[name].append({"generation_ms": actual["generation_ms"],
                                       "exact_output_equality": payload == expected,
                                       "semantic_sha256": digest(payload)})
    baseline_times = [x["generation_ms"] for x in observations["vedic_baseline"]]
    baseline_median = statistics.median(baseline_times)
    summary = {"schema": "grvq-conventional-identity-comparison/v1",
               "baseline_generation_ms": baseline_median,
               "baseline_semantic_sha256": digest(expected),
               "all_29_baseline_methods_executed": baseline_all,
               "baseline_exact_equality_every_repeat": all(x["exact_output_equality"] for x in observations["vedic_baseline"]),
               "baseline_timing_raw_ms": baseline_times,
               "baseline_timing_min_ms": min(baseline_times), "baseline_timing_max_ms": max(baseline_times),
               "repeats": args.repeats, "randomization_seed": args.order_seed,
               "block_orders": orders, "total_compiler_executions": args.repeats * len(variants),
               "timing_note": "Independent fresh Node processes; source loading, instrumentation and process startup excluded by the in-compiler timer. Variant order is shuffled within each repetition, with one original baseline per block. These modest exact-arithmetic microbenchmarks do not measure quantum speedup or establish general performance superiority.",
               "variants": []}
    for ids, name in variants[1:]:
        actual = latest[name]
        observed = observations[name]
        times = [x["generation_ms"] for x in observed]
        median = statistics.median(times)
        paired_ratios = [time / base for time, base in zip(times, baseline_times)]
        result = {"classical_ids": ids, "exact_output_equality": all(x["exact_output_equality"] for x in observed),
                  "generation_ms": median, "timing_min_ms": min(times), "timing_max_ms": max(times),
                  "timing_raw_ms": times,
                  "relative_median_to_baseline": median / baseline_median,
                  "paired_baseline_ratios": paired_ratios,
                  "median_paired_ratio": statistics.median(paired_ratios),
                  "exact_equality_by_repeat": [x["exact_output_equality"] for x in observed],
                  "entry_counts": {m["implementation"]: m["calls"] for m in actual["methods"]},
                  "semantic_sha256": digest(mathematical_payload(actual)),
                  "semantic_sha256_by_repeat": [x["semantic_sha256"] for x in observed]}
        summary["variants"].append(result)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    passed = sum(item["exact_output_equality"] for item in summary["variants"])
    print(json.dumps({"output": str(output), "exactly_equal": passed,
                      "variants": len(variants)-1, "repeats": args.repeats,
                      "compiler_executions": args.repeats * len(variants),
                      "baseline_semantic_sha256": digest(expected)}))
    if passed != len(variants)-1 or not baseline_all or not summary["baseline_exact_equality_every_repeat"]:
        raise RuntimeError("Conventional controls differ; inspect summary.json and retained compiler outputs")


if __name__ == "__main__":
    main()
