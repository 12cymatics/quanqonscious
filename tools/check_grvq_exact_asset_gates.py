"""Require selected exact-asset gates to reject one defect per disposable copy.

This runs in the GRVQ exact assets GitHub workflow. Neither a collection error
nor an arbitrary nonzero exit is accepted as evidence that a test caught a
wrong calculation. The unchanged tests must pass before and after the defects.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = "tests/test_grvq_exact_assets.py"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    before: str
    after: str
    test: str
    test_file: str = TEST_FILE


MUTATIONS = (
    Mutation(
        "omitted_turyavrtti_factor", "grvq_exact_assets.py",
        '"canonical_GRVQ_scalar", field.value,',
        '"canonical_GRVQ_scalar", field.value / field.components["turyavrtti_modulation"],',
        "test_canonical_scalar_retains_all_factors_at_a_nontrivial_point",
    ),
    Mutation(
        "omitted_final_book_coefficient", "vedic_field.py",
        "power, result = values, values\n        for coefficient in self.result.spectral_coefficients:",
        "power, result = values, values\n        for coefficient in self.result.spectral_coefficients[:-1]:",
        "test_authored_mapping_retains_every_coefficient_and_laplacian_sign",
    ),
    Mutation(
        "reversed_d2_orientation", "grvq_exact_assets.py",
        "flux = q4_complex.d2(plaquette_cochain)",
        "flux = {cell: -value for cell, value in q4_complex.d2(plaquette_cochain).items()}",
        "test_independent_cochains_do_not_inherit_zero_gradient_curl",
    ),
    Mutation(
        "dropped_highest_walsh_mode", "grvq_exact_assets.py",
        "coefficients = wht.walsh_transform_exact(values)",
        "coefficients = wht.walsh_transform_exact(values)\n    coefficients = coefficients[:15] + (Fraction(0),)",
        "test_q4_retains_the_highest_weight_walsh_mode",
    ),
    Mutation(
        "total_norm_used_instead_of_hermitian", "grvq_exact_assets.py",
        "left, right = _k2_coordinates(numerator), _k2_coordinates(denominator)\n    quotient = left / right",
        "left, right = _k2_coordinates(numerator), _k2_coordinates(denominator)\n"
        "    quotient = left * right.sigma_i() * k2_field.K2.from_rational(1 / right.total_norm())",
        "test_k2_division_uses_the_hermitian_norm_in_its_own_field",
    ),
    Mutation(
        "silent_float_to_rational_coercion", "grvq_exact_assets.py",
        "q = vedic_book.rational(value)",
        "q = sp.Rational(value)",
        "test_exact_asset_boundaries_reject_inexact_or_coerced_values",
    ),
    Mutation(
        "decimal_limit_in_symbolic_receipts", "grvq_exact_assets.py",
        'return _ExactReprPrinter({"order": "none"}).doprint(value)',
        "return sp.srepr(value)",
        "test_symbolic_receipts_preserve_large_exact_coefficient_atoms",
    ),
    Mutation(
        "omitted_anti_self_dual_strength", "grvq_q4_hodge.py",
        "source_strength_squared = _norm_squared(source_plus) + _norm_squared(source_minus)",
        "source_strength_squared = _norm_squared(source_plus)",
        "test_anti_self_dual_field_remains_active_in_source_strength",
        "tests/test_grvq_q4_hodge.py",
    ),
    Mutation(
        "omitted_source_minus_normalization", "grvq_q4_hodge.py",
        'source_minus = tuple(value / 6 for value in split["minus_triple"])',
        'source_minus = tuple(value for value in split["minus_triple"])',
        "test_hodge_keeps_both_chiralities_and_source_normalization",
        "tests/test_grvq_exact_asset_continuation.py",
    ),
    Mutation(
        "omitted_final_book_stage", "grvq_book_chain.py",
        "for index, stage in enumerate(prepared, 1):",
        "for index, stage in enumerate(prepared[:-1], 1):",
        "test_full_chain_retains_every_computed_stage_and_sequential_state[laplacian_polynomial]",
        "tests/test_grvq_book_chain.py",
    ),
    Mutation(
        "restarted_book_stage_from_initial_state", "grvq_book_chain.py",
        "record = _execute_stage(stage, current, mapping, index, completed)",
        "record = _execute_stage(stage, initial_values, mapping, index, completed)",
        "test_full_chain_retains_every_computed_stage_and_sequential_state[laplacian_polynomial]",
        "tests/test_grvq_book_chain.py",
    ),
)


def source_manifest():
    """Hash every tracked source byte, including tests and instructions."""
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                            check=True, capture_output=True)
    paths = [part.decode("utf-8") for part in result.stdout.split(b"\0") if part]
    if not paths:
        raise RuntimeError("The checked-out source inventory is empty")
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths if (ROOT / path).is_file()}


def isolated_copy(destination):
    """Copy source without caches, generated evidence, or shared git metadata."""
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", "ci-results", ".lake",
    ))


def run_gate(copy, targets, label, output, expect_red):
    report = output / f"{label}.xml"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(copy), str(copy / "vedic_trainer")))
    environment["PYTHONNOUSERSITE"] = "1"
    command = [sys.executable, "-m", "pytest", *targets, f"--junitxml={report}"]
    process = subprocess.run(command, cwd=copy, env=environment,
                             capture_output=True, text=True)
    (output / f"{label}.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if not report.is_file():
        raise RuntimeError(f"{label}: no JUnit report; this does not establish a working gate")
    cases = list(ET.parse(report).getroot().iter("testcase"))
    failures = [failure for case in cases for failure in case.findall("failure")]
    errors = [error for case in cases for error in case.findall("error")]
    skipped = [skip for case in cases for skip in case.findall("skipped")]
    details = {"label": label, "returncode": process.returncode, "tests": len(cases),
               "failures": len(failures), "errors": len(errors), "skipped": len(skipped),
               "targets": targets, "copied_source": str(copy)}
    if not cases or errors or skipped:
        raise RuntimeError(f"{label}: missing tests, errors, or skipped tests: {details}")
    if expect_red:
        if process.returncode != 1 or not failures:
            raise RuntimeError(f"{label}: deliberate calculation defect was not rejected: {details}")
        assertion_failures = [failure for failure in failures
                              if "AssertionError" in (failure.text or "")
                              or "Failed: DID NOT RAISE" in (failure.text or "")]
        if len(assertion_failures) != len(failures):
            raise RuntimeError(f"{label}: failure was not an assertion about the intended behavior")
    elif process.returncode != 0 or failures:
        raise RuntimeError(f"{label}: unchanged implementation did not pass: {details}")
    print(json.dumps(details), flush=True)
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = source_manifest()
    summary = {"source_before": before, "runs": [], "working_source_unchanged": False}
    targets = [f"{mutation.test_file}::{mutation.test}" for mutation in MUTATIONS]
    try:
        with tempfile.TemporaryDirectory(prefix="grvq-exact-gates-") as temporary:
            directory = Path(temporary)
            initial = directory / "unchanged-before"
            isolated_copy(initial)
            summary["runs"].append(run_gate(initial, targets, "unchanged-before", output, False))
            shutil.rmtree(initial)
            for mutation in MUTATIONS:
                copy = directory / mutation.name
                isolated_copy(copy)
                path = copy / mutation.path
                original = path.read_text(encoding="utf-8")
                if original.count(mutation.before) != 1:
                    raise RuntimeError(f"{mutation.name}: expected exactly one source anchor")
                changed = original.replace(mutation.before, mutation.after, 1)
                path.write_text(changed, encoding="utf-8")
                patch = "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                                    changed.splitlines(keepends=True),
                                                    fromfile=mutation.path, tofile=mutation.path))
                (output / f"{mutation.name}.patch").write_text(patch, encoding="utf-8")
                result = run_gate(copy, [f"{mutation.test_file}::{mutation.test}"], mutation.name, output, True)
                result["mutated_path"] = mutation.path
                result["mutated_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                summary["runs"].append(result)
                shutil.rmtree(copy)
            final = directory / "unchanged-after"
            isolated_copy(final)
            summary["runs"].append(run_gate(final, targets, "unchanged-after", output, False))
    finally:
        after = source_manifest()
        summary["working_source_unchanged"] = before == after
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if before != after:
            raise RuntimeError("The checked-out source changed during defect checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
