"""Run complete exact quantum-GRVQ trajectories and their paired controls.

Reconstruction provenance: the first 240 lines of the earlier launcher were
retained in the conversation; the missing tail and worker-control entry point
were rebuilt after its execution host became unavailable. This is a new source
revision. Its results are obtained only by actual execution.
Mathematical Vedic composition and independent CPU scheduling remain distinct.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "latest_hybrid_results"
MODES = ("SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initial_parameters(seed):
    generator = random.Random(seed)
    return [str(Fraction(generator.randint(-3, 3), 8)) for _ in range(12)]


def qasm(parameters, basis="ZZZZ"):
    import mpmath as mp
    from .ansatz import _measurement_qasm

    parameters = [Fraction(p) for p in parameters]
    if len(parameters) != 12:
        raise ValueError("The recovered circuit requires twelve angles")
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[4];", "creg c[4];"]
    with mp.workdps(70):
        for layer in range(3):
            for q in range(4):
                t = parameters[4 * layer + q]
                angle = 4 * mp.atan(mp.mpf(t.numerator) / t.denominator)
                encoded = mp.nstr(angle, 40, min_fixed=-8, max_fixed=8)
                if "e" in encoded:
                    mantissa, exponent = encoded.split("e")
                    encoded = mantissa + "e" + str(int(exponent))
                lines.append(f"ry({encoded}) q[{q}];")
            lines.extend(f"cx q[{q}],q[{q+1}];" for q in range(3))
    return _measurement_qasm(lines, 4, basis, True)


def export_qasm(case_directory, result):
    from .ansatz import ExactState

    directory = Path(case_directory)
    manifest = {"schema": "grvq-feedback-qasm-exports/v1", "circuits": [],
                "status": "prepared; provider execution is separate",
                "angles": "40-digit decimal output projection of exact Cayley parameters"}
    for step in result["steps"]:
        state_record = json.loads((directory / step["quantum_file"]).read_text())
        state = ExactState.from_fractions(tuple(Fraction(x) for x in state_record["amplitudes"]))
        if state.norm_squared != 1:
            raise ArithmeticError("Exported exact quantum state is not normalized")
        expected_z = tuple(Fraction(x) for x in state_record["probabilities"])
        if state.probabilities() != expected_z:
            raise ArithmeticError("Saved quantum probabilities differ from the state")
        for basis in ("ZZZZ", "XXXX", "YYYY", "XYZX"):
            relative = Path("qasm") / f"step_{step['step']:03d}_{basis}.qasm"
            target = directory / relative
            target.parent.mkdir(exist_ok=True)
            target.write_text(qasm(state_record["parameters"], basis))
            probabilities = state.measurement_probabilities(basis)
            expected = target.with_suffix(".expected.json")
            _write(expected, {"basis": basis, "qubit_order": "letter q is qubit q; bitstrings display q3..q0",
                              "probabilities_exact": {f"{i:04b}": str(p) for i, p in enumerate(probabilities)},
                              "probabilities_numeric_display": {f"{i:04b}": float(p) for i, p in enumerate(probabilities)}})
            manifest["circuits"].append({"step": step["step"], "basis": basis,
                                          "qasm": relative.as_posix(), "qasm_sha256": _sha(target),
                                          "expected": expected.relative_to(directory).as_posix()})
    _write(directory / "qasm_manifest.json", manifest)
    return manifest


def _run_case(configuration, output):
    destination = Path(output) / "cases" / configuration["case_id"]
    request = Path(output) / "requests" / (configuration["case_id"] + ".json")
    _write(request, configuration)
    logs = Path(output) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = logs / (configuration["case_id"] + ".jsonl")
    stderr = logs / (configuration["case_id"] + ".stderr.jsonl")
    with stdout.open("w") as out, stderr.open("w") as err:
        completed = subprocess.run(["node", str(HERE / "hybrid_quantum.cjs"), "--input", str(request),
                                    "--output", str(destination), "--progress"], cwd=ROOT,
                                   stdout=out, stderr=err, check=False)
    result_path = destination / "result.json"
    if completed.returncode or not result_path.exists():
        return {"case_id": configuration["case_id"], "status": "failed", "exit_code": completed.returncode,
                "configuration": configuration, "result_file": str(result_path.relative_to(output)),
                "error_file": str(stderr.relative_to(output))}
    result = json.loads(result_path.read_text())
    if result["status"] != "complete" or len(result["steps"]) != configuration["steps"]:
        raise ArithmeticError("The child reported success without completing its declared trajectory")
    for step in result["steps"]:
        expected = 58 if configuration["composition"] == "SYMMETRIC_CONCURRENT" else 29
        if not step["all29_consumed"] or step["core_method_calls"] != expected or step["distributed_grvq_updates"] != 25:
            raise ArithmeticError("A completed frame lacks required source operations")
    exports = export_qasm(destination, result)
    return {"case_id": configuration["case_id"], "status": "complete", "configuration": configuration,
            "result_file": str(result_path.relative_to(output)), "result_sha256": _sha(result_path),
            "elapsed_seconds": result["elapsed_seconds"], "steps": result["steps"],
            "totals": result["totals"], "exact_arithmetic_statistics": result["exact_arithmetic_statistics"],
            "qasm_circuits": len(exports["circuits"])}


def paired_analysis(cases):
    groups = {}
    for case in cases:
        if case["status"] != "complete":
            continue
        config = case["configuration"]
        key = (config["composition"], config["strength"], config["seed"])
        groups.setdefault(key, {})[config["feedback"]] = case
    pairs = []
    for (mode, strength, seed), group in groups.items():
        if set(group) != {False, True}:
            continue
        off, on = group[False], group[True]
        first_on, first_off = on["steps"][0], off["steps"][0]
        if first_on["input_field_digests"] != first_off["input_field_digests"]:
            raise ArithmeticError("Paired controls do not share the exact initial field")
        if first_on["after_formula_field_digests"] != first_off["after_formula_field_digests"]:
            raise ArithmeticError("Paired controls differ before quantum feedback is applied")
        if any(Fraction(x) for step in off["steps"] for x in step["feedback_deltas"]):
            raise ArithmeticError("The frozen-feedback control applied a nonzero quantum increment")
        stages = []
        for a, b in zip(on["steps"], off["steps"]):
            if a["step"] and a["input_field_digests"] != on["steps"][a["step"] - 1]["next_field_digests"]:
                raise ArithmeticError("The enabled trajectory did not consume its previous feedback output")
            stages.append({"step": a["step"],
                           "input_fields_equal": a["input_field_digests"] == b["input_field_digests"],
                           "source_output_fields_equal": a["after_formula_field_digests"] == b["after_formula_field_digests"],
                           "next_fields_equal": a["next_field_digests"] == b["next_field_digests"],
                           "feedback_on_gap": a["display"]["energy_gap"]["value"],
                           "feedback_off_gap": b["display"]["energy_gap"]["value"],
                           "feedback_on_lambda": [x["value"] for x in a["display"]["lambda_after_feedback"]],
                           "feedback_off_lambda": [x["value"] for x in b["display"]["lambda_after_feedback"]]})
        pairs.append({"composition": mode, "strength": strength, "seed": seed,
                      "feedback_on_case": on["case_id"], "feedback_off_case": off["case_id"],
                      "same_initial_conditions": True, "steps": stages})
    return {"pairs": pairs, "interpretation": "Each energy gap uses that frame's own exact H(lambda); cross-frame raw energy decreases are not fixed-objective VQE gains."}


def run_sweep(output=DEFAULT_OUTPUT, *, modes=MODES, strengths=("1/29", "1/58"), seeds=(7, 19, 43),
              feedback="both", steps=2, sweeps=4, grid_denominator=8, workers=2,
              parallel_backend="serial", branch_workers=2):
    modes, strengths, seeds = tuple(modes), tuple(strengths), tuple(seeds)
    if not modes or not strengths or not seeds:
        raise ValueError("A sweep requires nonempty modes, strengths and seeds")
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Select distinct integer seeds")
    if any(isinstance(x, (float, bool)) or not isinstance(x, (str, int, Fraction)) for x in strengths):
        raise TypeError("Strengths require exact rational strings, integers or Fractions")
    strengths = tuple(str(Fraction(x)) for x in strengths)
    if len(set(strengths)) != len(strengths):
        raise ValueError("Select distinct exact strengths; equivalent fractions define the same case")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("The sweep output already exists; select a new output directory to retain the earlier work")
    for name, count in (("steps", steps), ("sweeps", sweeps), ("grid_denominator", grid_denominator),
                        ("workers", workers), ("branch_workers", branch_workers)):
        if type(count) is not int or count < 1:
            raise ValueError(name + " must be a positive integer")
    if any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("Select distinct supported mathematical modes")
    if feedback not in ("both", "on", "off") or parallel_backend not in ("serial", "workers"):
        raise ValueError("Invalid feedback or scheduling selection")
    feedback_choices = (False, True) if feedback == "both" else (feedback == "on",)
    configurations = []
    for mode in modes:
        for strength in strengths:
            fraction = Fraction(strength)
            if not 0 < fraction <= 1:
                raise ValueError("All methods require nonzero declared strengths")
            for seed in seeds:
                for enabled in feedback_choices:
                    case_id = f"{mode}__s{fraction.numerator}_{fraction.denominator}__seed{seed}__feedback_{'on' if enabled else 'off'}"
                    configurations.append({"case_id": case_id, "composition": mode, "strength": str(fraction),
                                           "seed": seed, "feedback": enabled, "steps": steps, "sweeps": sweeps,
                                           "grid_denominator": grid_denominator, "branch_workers": branch_workers,
                                           "backend": parallel_backend if mode == "PARALLEL" else "serial",
                                           "initial_parameters": initial_parameters(seed)})
    started = time.perf_counter()
    report = {"schema": "grvq-quantum-feedback-sweep/v2", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "runtime": {"python": platform.python_version()},
              "expected_cases": len(configurations),
              "expected_frames": len(configurations) * steps, "cases": [], "total_frames": 0,
              "configuration": {"modes": list(modes), "strengths": list(strengths), "seeds": list(seeds),
                                "feedback": feedback, "steps": steps, "sweeps": sweeps,
                                "grid_denominator": grid_denominator, "workers": workers,
                                "parallel_backend": parallel_backend, "branch_workers": branch_workers,
                                "scheduling": "independent trajectory processes; source SERIES/concurrent dependencies are preserved",
                                "initialization": "random.Random(seed).randint(-3,3)/8 for twelve exact Cayley parameters",
                                "code_sha256": {name: _sha(HERE / name) for name in
                                                ("run_latest_hybrid.py", "hybrid_quantum.cjs", "latest_hybrid_executor.cjs", "exact_dag.cjs")}},
              "totals": {},
              "reconstruction": {"kind": "new launcher source revision with genuine execution",
                                 "previous_launcher_sha256": "75ef2922c46b4dafb721601aaf5cae87d6cd7f00c553d56311ac7cbf2fb27ee2",
                                 "historical_completed_results_sha256": "378d9188fb150932050e3d0538e1b0f6fe7ee756aa34056aa02fa6ed34a06796",
                                 "current_results": "computed by this run, not copied or asserted from the historical run"}}
    _write(output / "results.json", report)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_run_case, c, output): c for c in configurations}
        for future in as_completed(jobs):
            try:
                result = future.result()
            except Exception as error:
                configuration = jobs[future]
                result = {"case_id": configuration["case_id"], "status": "failed",
                          "configuration": configuration,
                          "error": {"name": type(error).__name__, "message": str(error)},
                          "result_file": str(Path("cases") / configuration["case_id"] / "result.json")}
            report["cases"].append(result)
            report["cases"].sort(key=lambda c: c["case_id"])
            report["total_frames"] += len(result.get("steps", ()))
            for key, value in result.get("totals", {}).items():
                report["totals"][key] = report["totals"].get(key, 0) + value
            report["elapsed_seconds"] = time.perf_counter() - started
            _write(output / "results.json", report)
            print(json.dumps({"completed_cases": len(report["cases"]), "expected_cases": len(configurations),
                              "case_id": result["case_id"], "status": result["status"],
                              "total_frames": report["total_frames"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    failures = [case for case in report["cases"] if case["status"] != "complete"]
    report["elapsed_seconds"] = time.perf_counter() - started
    if failures or report["total_frames"] != report["expected_frames"]:
        report["status"] = "failed"
        report["failures"] = [case["case_id"] for case in failures]
        _write(output / "results.json", report)
        raise RuntimeError("The complete sweep failed; every completed and failed case is retained in " + str(output))
    report["paired_feedback_analysis"] = paired_analysis(report["cases"])
    if feedback == "both":
        expected_pairs = len(modes) * len(strengths) * len(seeds)
        if len(report["paired_feedback_analysis"]["pairs"]) != expected_pairs:
            raise ArithmeticError("A declared paired control is missing")
    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report["qasm_circuits"] = sum(c["qasm_circuits"] for c in report["cases"])
    _write(output / "results.json", report)
    return report


def run_workers_control(sweep_input=DEFAULT_OUTPUT, output=HERE / "latest_hybrid_backend_validation" / "workers_case"):
    """Execute both coupled frames using the actual independent worker branches."""
    sweep_input, output = Path(sweep_input).resolve(), Path(output).resolve()
    canonical = sweep_input / "requests" / "PARALLEL__s1_29__seed7__feedback_on.json"
    configuration = json.loads(canonical.read_text())
    configuration["backend"] = "workers"
    configuration["case_id"] = "PARALLEL_workers_control__seed7"
    if configuration["branch_workers"] != 2 or configuration["steps"] != 2:
        raise ValueError("The canonical worker control requires two workers and two complete coupled frames")
    request = output.parent / "workers_control_request.json"
    _write(request, configuration)
    subprocess.run(["node", str(HERE / "hybrid_quantum.cjs"), "--input", str(request),
                    "--output", str(output), "--progress"], cwd=ROOT, check=True)
    result = json.loads((output / "result.json").read_text())
    if result["status"] != "complete" or len(result["steps"]) != 2:
        raise ArithmeticError("The worker control did not finish both coupled frames")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--strengths", default="1/29,1/58")
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--feedback", choices=("both", "on", "off"), default="both")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--grid-denominator", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parallel-backend", choices=("serial", "workers"), default="serial")
    parser.add_argument("--branch-workers", type=int, default=2)
    parser.add_argument("--workers-control", action="store_true")
    parser.add_argument("--sweep-input", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.workers_control:
        result = run_workers_control(args.sweep_input, args.output or HERE / "latest_hybrid_backend_validation" / "workers_case")
        print(json.dumps({"status": result["status"], "frames": len(result["steps"]), "kind": "actual worker control"}))
        return
    split = lambda value: tuple(x.strip() for x in value.split(",") if x.strip())
    result = run_sweep(args.output or DEFAULT_OUTPUT, modes=split(args.modes), strengths=split(args.strengths),
                       seeds=tuple(map(int, split(args.seeds))), feedback=args.feedback, steps=args.steps,
                       sweeps=args.sweeps, grid_denominator=args.grid_denominator, workers=args.workers,
                       parallel_backend=args.parallel_backend, branch_workers=args.branch_workers)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "frames": result["total_frames"],
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
