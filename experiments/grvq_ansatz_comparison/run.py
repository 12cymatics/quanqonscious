"""Run the source-derived ansatz comparison and retain complete experiment data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import scipy.linalg

from .ansatz import (ExactSparseOperator, ExactState, GRVQProductAnsatz,
                     mix_hypercube, polynomial_value, qasm_product_mixer,
                     qasm_source_ry_cz, source_ry_cz_state)
from .arithmetic import run_arithmetic_suite
from .hamiltonians import q4_laplacian, tfim, xxz, torus_graph, schrodinger_1d, self_check
from .spectral import build_spectral_suite


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "e7aca179d2a492769a1f2dfb8e5b4f56e9b58fa0"
F = Fraction
FAMILIES = ("source_ry_cz", "grvq_R2", "grvq_R4", "grvq_R2_cayley",
            "grvq_R4_cayley", "grvq_R4_reverse", "grvq_R4_local")


def _json(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_json, indent=2, allow_nan=False) + "\n")


def sparse_dense(matrix):
    return tuple(tuple((j, value) for j, value in enumerate(row) if value)
                 for row in matrix)


def dense_sparse(rows):
    dimension = len(rows)
    result = np.zeros((dimension, dimension))
    for i, row in enumerate(rows):
        for j, value in row:
            result[i, j] += float(value)
    return result


@dataclass
class Objective:
    name: str
    nqubits: int
    rows: tuple
    mass_rows: tuple
    metadata: dict

    def __post_init__(self):
        self.operator = ExactSparseOperator.from_rows(self.rows)
        self.metric = ExactSparseOperator.from_rows(self.mass_rows)
        if not self.operator.is_symmetric() or not self.metric.is_symmetric():
            raise ValueError("A variational problem requires symmetric exact A and M")
        self.A_float = dense_sparse(self.rows)
        self.M_float = dense_sparse(self.mass_rows)
        self.eigenvalues, self.eigenvectors = scipy.linalg.eigh(self.A_float, self.M_float)
        self.reference_energy = float(self.eigenvalues[0])
        self.eigenspace_tolerance = 1e-10 * max(1., float(np.linalg.norm(self.A_float, 2)))
        self.ground_indices = np.flatnonzero(self.eigenvalues-self.eigenvalues[0]
                                           <= self.eigenspace_tolerance)
        self.generalized = any(row != ((i, F(1)),) for i, row in enumerate(self.mass_rows))
        self.reference_residual = float(np.linalg.norm(
            self.A_float @ self.eigenvectors[:, 0]
            - self.reference_energy * self.M_float @ self.eigenvectors[:, 0]))

    def energy(self, state):
        v = state.numerators
        av = self.operator.apply_numerators(v)
        mv = self.metric.apply_numerators(v)
        a = sum(x*y for x, y in zip(v, av))
        m = sum(x*y for x, y in zip(v, mv))
        if m <= 0:
            raise ArithmeticError("The exact metric norm must be positive")
        return F(a*self.metric.denominator, m*self.operator.denominator)

    def metrics(self, state):
        energy = self.energy(state)
        scale = max(abs(value) for value in state.numerators)
        v = np.array([float(F(value, scale)) for value in state.numerators])
        v /= np.sqrt(v @ self.M_float @ v)
        overlaps = self.eigenvectors[:, self.ground_indices].T @ self.M_float @ v
        residual = self.A_float @ v - float(energy) * self.M_float @ v
        metric_residual = float(residual @ scipy.linalg.solve(self.M_float, residual,
                                                            assume_a="pos"))
        result = {
            "energy_exact": str(energy),
            "energy": float(energy),
            "ground_energy_error": float(energy)-self.reference_energy,
            "ground_subspace_fidelity": float(overlaps @ overlaps),
            "generalized_residual_squared_numeric": metric_residual,
            "maximum_amplitude_numerator_bits": max(abs(x).bit_length() for x in state.numerators),
            "amplitude_denominator_bits": state.denominator.bit_length(),
        }
        if not self.generalized:
            variance = state.variance(self.operator)
            result["energy_variance_exact"] = str(variance)
            result["energy_variance"] = float(variance)
        return result

    def record(self):
        raw = {"A_rows": self.rows, "M_rows": self.mass_rows}
        digest = hashlib.sha256(json.dumps(raw, default=_json, sort_keys=True).encode()).hexdigest()
        return {
            "name": self.name, "nqubits": self.nqubits, "dimension": 1 << self.nqubits,
            "generalized_metric": self.generalized, "metadata": self.metadata,
            "reference_ground_energy": self.reference_energy,
            "reference_method": "scipy.linalg.eigh(A,M), numerical diagnostic of exact rational matrices",
            "reference_residual_l2": self.reference_residual,
            "ground_subspace_dimension": len(self.ground_indices),
            "ground_subspace_tolerance": self.eigenspace_tolerance,
            "exact_matrices_sha256": digest,
        }


def identity_rows(size):
    return tuple(((i, F(1)),) for i in range(size))


def build_objectives(polynomial_path):
    spectral_problems, polynomial_verification = build_spectral_suite(polynomial_path)
    objectives = []

    def append(name, operator):
        objectives.append(Objective(name, operator.nqubits, operator.rows,
                                    identity_rows(operator.dimension), operator.metadata))

    x = tuple(F(i, 3)-F(1, 2) for i in range(4))
    harmonic = tuple(4*a*a + 2*b*b + a*b for a in x for b in x)
    doublewell = tuple(12*(a*a-F(1, 9))**2 + 2*b*b for a in x for b in x)
    append("Q4_harmonic", q4_laplacian(4, harmonic))
    append("Q4_double_well", q4_laplacian(4, doublewell))
    append("torus_4x4", torus_graph((4, 4), harmonic))
    for h in (F(1, 2), F(1), F(3, 2)):
        append("TFIM4_h"+str(h).replace("/", "_"), tfim(4, h=h))
    for anisotropy in (F(1, 2), F(1), F(3, 2)):
        append("XXZ4_delta"+str(anisotropy).replace("/", "_"), xxz(4, Jz=anisotropy))
    append("TFIM6_h1", tfim(6, h=F(1)))
    append("XXZ6_delta1", xxz(6, Jz=F(1)))
    append("TFIM8_h1", tfim(8, h=F(1)))
    append("XXZ8_delta1", xxz(8, Jz=F(1)))
    append("line16_quartic", schrodinger_1d(4, spacing=F(1, 8),
           potential=tuple(4*(F(i, 8)-1)**4 for i in range(16)), kinetic=F(1, 2)))
    for problem in spectral_problems:
        objectives.append(Objective(problem["name"], problem["nqubits"],
                          sparse_dense(problem["A"]), sparse_dense(problem["M"]),
                          problem["metadata"]))
    return objectives, polynomial_verification


def product_family(data, nqubits, shell_scale=F(1)):
    """Explicit polynomial update of the source GRVQ product family.

    All four source cardinal polynomials are used on the full binary-index
    sample grid. The nonconstant carrier is interpolation of 1+x at the source
    nodes. This is a declared experiment ansatz, not the reciprocal scalar in
    grvq_canonical.py. On at least16 distinct points the resulting degree13
    product cannot be the zero vector for finite alpha values.
    """
    size = 1 << nqubits
    basis = tuple(tuple(F(c) for c in row) for row in data["basis"])
    nodes = tuple(F(x) for x in data["nodes"])
    points = tuple(F(i, size-1) for i in range(size))
    shape_table = tuple(tuple(polynomial_value(p, x) for p in basis) for x in points)
    carrier = tuple(sum((1+node)*value for node, value in zip(nodes, row))
                    for row in shape_table)
    radii = tuple(abs(2*x-1) for x in points)
    widths = tuple(shell_scale*w for w in (F(1, 2), F(1), F(3, 2), F(2)))
    return GRVQProductAnsatz(shape_table, radii, carrier, widths)


def parameter_count(family, nqubits):
    if family == "source_ry_cz":
        return 2*nqubits
    return 4 if family in ("grvq_R2", "grvq_R4") else 4+nqubits


def evaluate_family(family, product, nqubits, parameters):
    if family == "source_ry_cz":
        return source_ry_cz_state(nqubits, parameters, layers=2, ring=True)
    power = 2 if "R2" in family else 4
    state = product.evaluate(parameters[:4], power=power)
    if family in ("grvq_R2", "grvq_R4"):
        return state
    order = tuple(reversed(range(nqubits))) if family == "grvq_R4_reverse" else tuple(range(nqubits))
    return mix_hypercube(state, parameters[4:], ordering=order,
                         conditional_signs=family != "grvq_R4_local", control_mode="neighbor")


def optimize(objective, family, product, seed, budget, trace_path):
    count = parameter_count(family, objective.nqubits)
    rng = random.Random(seed)
    parameters = tuple(F(rng.randint(-3, 3), 8) for _ in range(count))
    state = evaluate_family(family, product, objective.nqubits, parameters)
    current = objective.energy(state)
    initial = current
    evaluations = 1
    step = F(1, 2)
    history = [{"evaluation": 1, "energy": float(current), "step": str(step)}]
    started = time.perf_counter()
    with gzip.open(trace_path, "wt") as trace:
        trace.write(json.dumps({"evaluation": 1, "parameters": list(map(str, parameters)),
                                "energy_exact": str(current), "accepted": True})+"\n")
        while evaluations < budget:
            improved = False
            for index in range(count):
                origin = parameters
                best = current
                best_state, best_parameters = state, parameters
                for sign in (1, -1):
                    if evaluations >= budget:
                        break
                    candidate = list(origin)
                    candidate[index] += sign*step
                    candidate = tuple(candidate)
                    candidate_state = evaluate_family(family, product, objective.nqubits, candidate)
                    score = objective.energy(candidate_state)
                    evaluations += 1
                    accepted = score < best
                    trace.write(json.dumps({"evaluation": evaluations, "coordinate": index,
                            "origin": list(map(str, origin)), "delta": str(sign*step),
                            "energy_exact": str(score), "accepted_in_pair": accepted})+"\n")
                    if accepted:
                        best, best_state, best_parameters = score, candidate_state, candidate
                if best < current:
                    improved = True
                    current, state, parameters = best, best_state, best_parameters
                    history.append({"evaluation": evaluations, "energy": float(current),
                                    "step": str(step)})
                if evaluations >= budget:
                    break
            if not improved:
                step /= 2
        history.append({"evaluation": evaluations, "energy": float(current), "step": str(step)})
    elapsed = time.perf_counter()-started
    return {
        "model": objective.name, "family": family, "seed": seed,
        "nqubits": objective.nqubits, "parameter_count": count,
        "objective_evaluations": evaluations, "initial_energy_exact": str(initial),
        "initial_energy": float(initial), "parameters": list(map(str, parameters)),
        "final_step": str(step), "seconds": elapsed, "trace": history,
        "complete_trace": trace_path.name, **objective.metrics(state),
    }, state


def export_run(run, state, product, destination):
    destination.mkdir(parents=True, exist_ok=True)
    parameters = tuple(F(value) for value in run["parameters"])
    family, n = run["family"], run["nqubits"]
    save_json(destination / "state.json", state.to_dict())
    save_json(destination / "run.json", run)
    manifests = []
    for basis in ("Z"*n, "X"*n, "Y"*n, ("XYZ"*((n+2)//3))[:n]):
        if family == "source_ry_cz":
            qasm = qasm_source_ry_cz(n, parameters, layers=2, ring=True, basis=basis)
        elif family in ("grvq_R2", "grvq_R4"):
            qasm = state.to_qasm(basis=basis)
        else:
            order = tuple(reversed(range(n))) if family == "grvq_R4_reverse" else tuple(range(n))
            qasm = qasm_product_mixer(product.evaluate(parameters[:4], power=2 if "R2" in family else 4),
                     parameters[4:], ordering=order, conditional_signs=family != "grvq_R4_local",
                     control_mode="neighbor", basis=basis)
        path = destination / (basis+".qasm")
        path.write_text(qasm)
        probabilities = state.measurement_probabilities(basis)
        expected = {format(i, f"0{n}b"): str(p) for i, p in enumerate(probabilities)}
        save_json(destination / (basis+"_expected.json"), expected)
        manifests.append({"basis": basis, "qasm": str(path.relative_to(destination.parent.parent)),
                          "expected": str((destination/(basis+"_expected.json")).relative_to(destination.parent.parent)),
                          "sha256": hashlib.sha256(qasm.encode()).hexdigest()})
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/grvq_ansatz_comparison/results")
    parser.add_argument("--budget", type=int, default=385)
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--models", default="all")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--arithmetic-cases", type=int, default=64)
    parser.add_argument("--reuse-arithmetic", type=Path)
    args = parser.parse_args(argv)
    if args.budget < 2:
        parser.error("The objective budget must be at least two")
    sys.set_int_max_str_digits(0)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/"traces").mkdir(exist_ok=True)
    (output/"states").mkdir(exist_ok=True)
    begin = time.perf_counter()
    generation = subprocess.run(["node", str(Path(__file__).with_name("compile_polynomials.cjs"))],
                                cwd=ROOT, check=True, capture_output=True, text=True)
    polynomial_path = output/"polynomial_data.json"
    polynomial_path.write_text(generation.stdout)
    data = json.loads(generation.stdout)
    print("All 29 polynomial methods executed in the actual coefficient compiler.", flush=True)
    objectives, verification = build_objectives(polynomial_path)
    if args.models != "all":
        names = args.models.split(",")
        known = {objective.name for objective in objectives}
        if set(names)-known:
            parser.error(f"Unknown models: {set(names)-known}")
        objectives = [objective for objective in objectives if objective.name in names]
    seeds = tuple(int(value) for value in args.seeds.split(","))
    families = tuple(args.families.split(","))
    if set(families)-set(FAMILIES):
        parser.error("Unknown ansatz family")
    checks = self_check()
    if args.reuse_arithmetic:
        arithmetic = json.loads(args.reuse_arithmetic.read_text())
    else:
        arithmetic = run_arithmetic_suite(seed=20261122, cases_per_sutra=args.arithmetic_cases)
    save_json(output/"arithmetic.json", arithmetic)
    save_json(output/"polynomial_verification.json", verification)
    report = {
        "schema": "grvq-ansatz-comparison/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "configuration": {
            "budget": args.budget, "seeds": seeds, "families": families,
            "optimizer": "exact rational coordinate pattern search; initial step1/2, halve after a nonimproving sweep",
            "state_arithmetic": "arbitrary precision integers and Fraction; projective exact normalization",
            "shape_family": "all four exact source cardinal polynomials on x_i=i/(2^n-1)",
            "carrier": "source cardinal interpolant of 1+x (nonconstant)",
            "radius": "abs(2*x_i-1)", "shell_widths": ["1/2", "1", "3/2", "2"],
            "cayley_refinement": "t_d times (-1)^bit((d+1) modulo n), ordered by qubit; noncommuting neighboring controls",
            "source_baseline": "source H/Ry/CZ structure; two Ry/CZ ring layers, rational Cayley gate parameters",
            "diagnostics": "Eigensolver, plotted decimals and QASM physical angles are numerical boundary outputs; optimizer comparisons stay rational.",
        },
        "models": [objective.record() for objective in objectives],
        "polynomial_method_trace": data["methods"],
        "polynomial_verification": verification,
        "hamiltonian_checks": checks,
        "arithmetic_total_cases": arithmetic["total_cases"],
        "runs": [], "exports": [],
    }
    for objective in objectives:
        save_json(output/(objective.name+"_matrices.json"), {
            "A_rows": objective.rows, "M_rows": objective.mass_rows, **objective.record()})
        product = product_family(data, objective.nqubits)
        for family in families:
            for seed in seeds:
                label = f"{objective.name}__{family}__{seed}"
                run, state = optimize(objective, family, product, seed, args.budget,
                                      output/"traces"/(label+".jsonl.gz"))
                save_json(output/"states"/(label+".json"), state.to_dict())
                run["state_file"] = "states/"+label+".json"
                report["runs"].append(run)
            recent = [r for r in report["runs"] if r["model"] == objective.name and r["family"] == family]
            print(f"{objective.name:24} {family:22} median_error={np.median([r['ground_energy_error'] for r in recent]):.9g}", flush=True)
        save_json(output/"results.json", report)
    for name in ("Q4_harmonic", "TFIM4_h1", "XXZ4_delta1", "vedic_spectral_2d", "TFIM6_h1"):
        candidates = [run for run in report["runs"] if run["model"] == name]
        if not candidates:
            continue
        winner = min(candidates, key=lambda run: F(run["energy_exact"]))
        raw = json.loads((output/winner["state_file"]).read_text())
        state = ExactState(tuple(int(value, 16) for value in raw["numerators_hex"]),
                           int(raw["denominator_hex"], 16))
        product = product_family(data, winner["nqubits"])
        exported = export_run(winner, state, product, output/"exports"/name)
        report["exports"].append({"model": name, "family": winner["family"], "files": exported})
    report["elapsed_seconds"] = time.perf_counter()-begin
    report["total_objective_evaluations"] = sum(run["objective_evaluations"] for run in report["runs"])
    save_json(output/"results.json", report)
    print(json.dumps({"runs": len(report["runs"]), "objective_evaluations": report["total_objective_evaluations"],
                      "seconds": report["elapsed_seconds"], "results": str(output/"results.json")}))


if __name__ == "__main__":
    main()
