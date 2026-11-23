"""Audit every retained qBraid response against the immutable exact sweep.

This program submits no jobs and reruns no field or quantum optimization.
Provider JSON numbers are interpreted as their exact binary64 rationals. No
probability, state, coefficient, or feedback value is normalized or clamped.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.set_int_max_str_digits(0)
BASE = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f2f3bc798294e1fe56b3dacbecb0f494a6d6d8bd"
BASES = ("ZZZZ", "XXXX", "YYYY", "XYZX")
BITSTRINGS = tuple(f"{i:04b}" for i in range(16))
PROBABILITY_TOLERANCE = F(1, 10**12)
MOMENT_TOLERANCE = 16 * PROBABILITY_TOLERANCE


class VerificationError(ArithmeticError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def parse_json(text):
    def reject_constant(value):
        raise VerificationError("Nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained_path(directory, relative):
    require(isinstance(relative, str), "A manifest file path is not a string")
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory), "A manifest path leaves the results directory")
    return path


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(record):
    return record["case_id"], record["step"], record["basis"]


def label(key):
    return f"{key[0]} / step {key[1]} / {key[2]}"


def exact_binary64(value):
    require(type(value) in (int, float), "A provider probability is not a JSON number")
    converted = float(value)
    require(math.isfinite(converted), "A provider probability is not finite")
    return F.from_float(converted)


def correlation(probabilities, mask):
    return sum(((-1) ** ((index & mask).bit_count())) * probabilities[key]
               for index, key in enumerate(BITSTRINGS))


def exact_error(observed, expected):
    delta = observed - expected
    return {"observed_rational": str(observed), "expected_rational": str(expected),
            "signed_error_rational": str(delta), "absolute_error_rational": str(abs(delta)),
            "absolute_error_display": float(abs(delta))}


def enclosure(record):
    require(record["purpose"] == "display only; never an arithmetic input", "Unexpected retained enclosure provenance")
    certificate = record["certificate"]
    bits = certificate["denominator_power"]
    require(type(bits) is int and bits > 0, "Invalid exact enclosure denominator power")
    lo = F(int(certificate["lower_integer"]), 1 << bits)
    hi = F(int(certificate["upper_integer"]), 1 << bits)
    require(lo <= hi, "A retained exact enclosure is reversed")
    return lo, hi


def interval_scale(interval, coefficient):
    values = interval[0] * coefficient, interval[1] * coefficient
    return min(values), max(values)


def interval_sum(intervals, constant=F(0)):
    pairs = list(intervals)
    return constant + sum(x[0] for x in pairs), constant + sum(x[1] for x in pairs)


def overlaps(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])


def interval_record(interval):
    return {"lower_rational": str(interval[0]), "upper_rational": str(interval[1]),
            "lower_display": float(interval[0]), "upper_display": float(interval[1])}


def raw_payload(raw, required_key):
    require(isinstance(raw, dict) and not raw.get("isError", False), "The retained raw tool call reports an error")
    candidates = []
    structured = raw.get("structuredContent")
    if isinstance(structured, dict) and required_key in structured:
        candidates.append(structured)
    require(isinstance(raw.get("content", []), list), "Raw tool content is not a list")
    for block in raw.get("content", []):
        require(isinstance(block, dict), "A raw tool content block is not an object")
        if block.get("type") != "text":
            continue
        text = block.get("text", "").strip()
        if not text.startswith("{"):
            continue
        value = parse_json(text)
        if isinstance(value, dict) and required_key in value:
            candidates.append(value)
    require(candidates, "No matching JSON payload exists in the retained raw tool response")
    require(all(x == candidates[0] for x in candidates), "Raw text and structured tool payloads disagree")
    return candidates[0]


def independently_reconstruct_batch(directory, batch, submitted):
    require(batch["qubit_count"] == 12 and len(batch["members"]) == 3, "Every concurrent batch must execute all three complete four-qubit circuits")
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[12];', 'creg c[12];']
    source_header = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[4];', 'creg c[4];']
    gate_counts = []
    for slot, member in enumerate(batch["members"]):
        key = identity(member)
        require(key in submitted, "A batch contains an undeclared logical circuit")
        item = submitted[key]
        require(member["slot"] == slot and member["qubits"] == list(range(4*slot,4*slot+4)), "A batch member uses the wrong disjoint register")
        require(all(member[name] == item[name] for name in ("qasm","qasm_sha256","expected","expected_sha256")), "A batch member changes a canonical circuit or reference")
        program = contained_path(directory,item["qasm"]).read_text(encoding="utf-8")
        original = program.splitlines()
        require(original[:4] == source_header and original[-1] == 'measure q -> c;', "A source circuit has unexpected declarations or measurements")
        statements = original[4:-1]
        require(statements, "A batch member contains no source gates")
        for statement in statements:
            match = re.fullmatch(r'(ry|rz)\(([^()\r\n]+)\) q\[([0-3])\];',statement)
            if match:
                gate, angle, qubit = match.groups()
                reconstructed = f'{gate}({angle}) q[{4*slot+int(qubit)}];'
            else:
                match = re.fullmatch(r'cx q\[([0-3])\],q\[([0-3])\];',statement)
                if match:
                    control, target = map(int,match.groups())
                    require(control != target, "A canonical CNOT has identical control and target")
                    reconstructed = f'cx q[{4*slot+control}],q[{4*slot+target}];'
                else:
                    match = re.fullmatch(r'(h|sdg) q\[([0-3])\];',statement)
                    require(match is not None, "An original gate was not independently understood: "+statement)
                    gate, qubit = match.groups()
                    reconstructed = f'{gate} q[{4*slot+int(qubit)}];'
            lines.append(reconstructed)
        gate_counts.append(len(statements))
    lines.append('measure q -> c;')
    reconstructed = '\n'.join(lines)+'\n'
    batch_path = contained_path(directory,batch["qasm"])
    require(batch_path.read_bytes() == reconstructed.encode("utf-8"), "The actual batch QASM differs from the independently reconstructed complete source gate sequence")
    require(digest(batch_path) == batch["qasm_sha256"], "The complete reconstructed batch QASM hash differs")
    return reconstructed, gate_counts


def verify_joint_batch(directory, batch, submitted, references):
    program, gate_counts = independently_reconstruct_batch(directory,batch,submitted)
    response_path = contained_path(directory,batch["response_file"])
    response = read_json(response_path)
    require(response["batch_id"] == batch["batch_id"] and response["status"] == "returned", "Batch response identity or execution status differs")
    require(response["source_commit"] == SOURCE_COMMIT and response["qubit_count"] == 12, "Batch response source or register size differs")
    require(response["qasm"] == program and response["qasm_sha256"] == batch["qasm_sha256"], "Actual submitted batch text differs from the complete verified program")
    require(response["response_file"] == batch["response_file"] and response["members"] == batch["members"], "Actual batch membership metadata differ")
    validation, simulation = response["validation"], response["simulation"]
    require(raw_payload(response["raw_validation"],"valid") == validation, "Batch validation differs from its raw tool payload")
    require(raw_payload(response["raw_simulation"],"probabilities") == simulation, "Batch simulation differs from its raw tool payload")
    require(validation["valid"] is True and validation["errors"] == [], "qBraid batch validation failed")
    require(validation["stats"]["numQubits"] == 12 and validation["stats"]["numClbits"] == 12 and simulation["num_qubits"] == 12, "The actual simulated batch is not twelve qubits and twelve bits")
    keys = tuple(f'{i:012b}' for i in range(4096))
    require(set(simulation["probabilities"]) == set(keys), "A batch omits or adds a joint bitstring outcome")
    reference_vectors = [[references[identity(member)][key] for key in BITSTRINGS] for member in batch["members"]]
    marginals = [{key:F(0) for key in BITSTRINGS} for _ in range(3)]
    total = F(0);expected_total = F(0);l1_error = F(0);maximum_error = F(0);worst = None;signed_errors = []
    for index, key in enumerate(keys):
        observed = exact_binary64(simulation["probabilities"][key])
        require(0 <= observed <= 1+PROBABILITY_TOLERANCE, "A returned joint probability is outside its allowed numerical range")
        local = [(index >> (4*slot)) & 15 for slot in range(3)]
        expected = reference_vectors[0][local[0]]*reference_vectors[1][local[1]]*reference_vectors[2][local[2]]
        error = observed-expected
        require(abs(error) <= PROBABILITY_TOLERANCE, "A complete joint probability differs from the exact three-circuit product by more than1/10^12: "+key)
        for slot in range(3):
            marginals[slot][BITSTRINGS[local[slot]]] += observed
        total += observed;expected_total += expected;l1_error += abs(error)
        if abs(error) > maximum_error or worst is None:
            maximum_error = abs(error);worst = {"bitstring":key,"returned_json_number":simulation["probabilities"][key],**exact_error(observed,expected)}
        signed_errors.append([key,str(error)])
    require(expected_total == 1, "The exact reference product distribution is not normalized")
    require(abs(total-1) <= PROBABILITY_TOLERANCE, "The returned twelve-qubit joint distribution has the wrong total probability")
    require(all(sum(marginal.values()) == total for marginal in marginals), "Exact marginal sums lost part of the complete joint distribution")
    receipt = {"schema":"grvq-qbraid-joint-rational-errors/v1","batch_id":batch["batch_id"],
               "source_commit":SOURCE_COMMIT,"response_sha256":digest(response_path),"qasm_sha256":batch["qasm_sha256"],
               "members":batch["members"],"outcomes":4096,"bit_order":"q11..q0; slot0 is bits q3..q0",
               "definition":"Fraction.from_float(returned_joint_probability) minus the product of the three saved exact probability fractions",
               "signed_probability_errors_rational":signed_errors}
    receipt_path = directory/"provider_joint_checks"/(batch["batch_id"]+".json.gz")
    receipt_path.parent.mkdir(exist_ok=True)
    temporary = receipt_path.with_name(receipt_path.name+".tmp")
    temporary.write_bytes(gzip.compress((json.dumps(receipt,separators=(',',':'))+'\n').encode(),compresslevel=6,mtime=0))
    temporary.replace(receipt_path)
    check = {"batch_id":batch["batch_id"],"status":"passed","members":batch["members"],
             "executed_program_qubits":12,"joint_outcomes_checked":4096,"raw_response_file":batch["response_file"],
             "raw_response_sha256":digest(response_path),"qasm":batch["qasm"],"qasm_sha256":batch["qasm_sha256"],
             "canonical_source_gate_counts":gate_counts,"independent_complete_qasm_reconstruction":True,
             "validation_stats":validation["stats"],"joint_probability_sum_rational":str(total),
             "joint_probability_sum_error_rational":str(abs(total-1)),"joint_probability_l1_error_rational":str(l1_error),
             "maximum_joint_probability_error_rational":str(maximum_error),"maximum_joint_probability_error":float(maximum_error),
             "worst_joint_probability_comparison":worst,"all4096_exact_product_comparisons_passed":True,
             "marginalization":"Sum256 exact binary64 Fractions for each of16 outcomes, separately for all three disjoint registers; no normalization",
             "joint_rational_error_receipt":str(receipt_path.relative_to(directory)),"joint_rational_error_receipt_sha256":digest(receipt_path)}
    return {"check":check,"response":response,"marginals":marginals}


def verify_z_frame(directory, case, step, probabilities, expected_probabilities, probability_l1_error):
    folder = contained_path(directory, case["result_file"]).parent
    quantum_path = folder / step["quantum_file"]
    result = case["_result"]
    require(step["quantum_file"] in result["artifact_sha256"], "The exact quantum record lacks its original artifact hash")
    require(digest(quantum_path) == result["artifact_sha256"][step["quantum_file"]], "The saved exact quantum state changed")
    quantum = read_json(quantum_path)
    require(len(quantum["probabilities"]) == 16, "The saved quantum state does not have sixteen probabilities")
    expected_vector = tuple(map(F, quantum["probabilities"]))
    require(tuple(expected_probabilities[k] for k in BITSTRINGS) == expected_vector, "The Z-basis reference differs from the saved exact quantum state")
    require(sum(expected_vector) == 1 and F(quantum["norm"]) == 1, "The saved exact quantum norm is not one")
    masks = [1 << q for q in range(4)] + [(1 << q) | (1 << (q+1)) for q in range(3)]
    exact_moments = [correlation(expected_probabilities, mask) for mask in masks]
    provider_moments = [correlation(probabilities, mask) for mask in masks]
    require(exact_moments[:4] == list(map(F, quantum["z"])), "Saved exact Z moments disagree with their full probability distribution")
    require(exact_moments[4:] == list(map(F, quantum["zz"])), "Saved exact ZZ moments disagree with their full probability distribution")
    errors = [a-b for a, b in zip(provider_moments, exact_moments)]
    require(all(abs(x) <= probability_l1_error and abs(x) <= MOMENT_TOLERANCE for x in errors), "A Z or ZZ discrepancy exceeds the complete probability-error bound")
    lambdas = list(map(enclosure, step["display"]["lambda_out"]))
    require(len(lambdas) == 4, "The current Hamiltonian does not retain four exact lambda bounds")
    target_energy = enclosure(step["display"]["energy_final"])
    exact_energy = interval_sum((interval_scale(x, c) for x, c in zip(lambdas, exact_moments[:4])), sum(exact_moments[4:])/2)
    require(overlaps(exact_energy, target_energy), "The retained exact energy disagrees with the independently reconstructed Pauli contraction")
    provider_energy = interval_sum((interval_scale(x, c) for x, c in zip(lambdas, provider_moments[:4])), sum(provider_moments[4:])/2)
    energy_error = interval_sum((interval_scale(x, c) for x, c in zip(lambdas, errors[:4])), sum(errors[4:])/2)
    direct_error = provider_energy[0]-target_energy[1], provider_energy[1]-target_energy[0]
    require(overlaps(energy_error, direct_error), "Independent provider-energy error enclosures disagree")
    maximum_lambda = [max(abs(x[0]), abs(x[1])) for x in lambdas]
    actual_error_bound = sum(a*abs(e) for a, e in zip(maximum_lambda, errors[:4])) + sum(map(abs, errors[4:]))/2
    allowed_error_bound = MOMENT_TOLERANCE * (sum(maximum_lambda) + F(3, 2))
    energy_error_upper = max(map(abs, energy_error))
    require(energy_error_upper <= actual_error_bound <= allowed_error_bound, "Current-Hamiltonian energy discrepancy exceeds its exact propagated probability tolerance")
    # Independently sum all sixteen measured basis energies. No state is dropped.
    basis_energies = []
    for index, key in enumerate(BITSTRINGS):
        signs = [F((-1) ** ((index & mask).bit_count())) for mask in masks]
        level = interval_sum((interval_scale(x, c) for x, c in zip(lambdas, signs[:4])), sum(signs[4:])/2)
        basis_energies.append(interval_scale(level, probabilities[key]))
    basis_energy = interval_sum(basis_energies)
    require(overlaps(basis_energy, provider_energy), "The complete sixteen-state provider energy disagrees with the Pauli contraction")
    computed = [-x/1000 for x in provider_moments[:4]]
    exact_computed = [-x/1000 for x in exact_moments[:4]]
    require(exact_computed == list(map(F, step["computed_feedback_deltas"])), "Saved feedback is not exactly minus Z divided by one thousand")
    enabled = case["configuration"]["feedback"]
    require(type(enabled) is bool, "Feedback control is not an explicit boolean")
    applied = computed if enabled else [F(0)]*4
    exact_applied = exact_computed if enabled else [F(0)]*4
    require(exact_applied == list(map(F, step["feedback_deltas"])), "Saved applied feedback does not respect the declared control")
    gram = [[sum((1 if i & (1 << q) else -1)*(1 if i & (1 << k) else -1) for i in range(16)) for k in range(4)] for q in range(4)]
    require(gram == [[16*int(q==k) for k in range(4)] for q in range(4)], "The complete signed injection basis does not have Gram matrix16I")
    field_increments = [sum((1 if i & (1 << q) else -1)*applied[q] for q in range(4)) for i in range(16)]
    projected = [sum((1 if i & (1 << q) else -1)*field_increments[i] for i in range(16))/16 for q in range(4)]
    require(projected == applied, "The provider-derived signed field increment does not project back to its four mode increments")
    computed_errors = [abs(a-b) for a, b in zip(computed, exact_computed)]
    applied_errors = [abs(a-b) for a, b in zip(projected, exact_applied)]
    feedback_bound = probability_l1_error/1000
    require(all(x <= feedback_bound for x in computed_errors+applied_errors), "A feedback discrepancy exceeds its exact probability-error bound")
    return {"quantum_file": str(quantum_path.relative_to(directory)), "quantum_sha256": digest(quantum_path),
            "z_comparisons": [exact_error(a,b) for a,b in zip(provider_moments[:4],exact_moments[:4])],
            "zz_comparisons": [exact_error(a,b) for a,b in zip(provider_moments[4:],exact_moments[4:])],
            "lambda_enclosures": [interval_record(x) for x in lambdas],
            "lambda_certificate_denominator_powers": [x["certificate"]["denominator_power"] for x in step["display"]["lambda_out"]],
            "provider_energy_enclosure": interval_record(provider_energy),
            "full_sixteen_state_energy_enclosure": interval_record(basis_energy),
            "saved_exact_energy_enclosure": interval_record(target_energy),
            "reference_pauli_energy_enclosure": interval_record(exact_energy),
            "signed_energy_error_enclosure": interval_record(energy_error),
            "absolute_energy_error_upper_bound_rational": str(energy_error_upper),
            "absolute_energy_error_upper_bound_display": float(energy_error_upper),
            "energy_tolerance_propagated_from_probability_limit_rational": str(allowed_error_bound),
            "energy_bound_from_actual_moment_errors_rational": str(actual_error_bound),
            "feedback_enabled": enabled,
            "computed_feedback_comparisons": [exact_error(a,b) for a,b in zip(computed,exact_computed)],
            "applied_feedback_comparisons": [exact_error(a,b) for a,b in zip(projected,exact_applied)],
            "provider_signed_field_increments_rational": list(map(str,field_increments)),
            "provider_projected_mode_increments_rational": list(map(str,projected)),
            "signed_basis_gram": gram,
            "projection_identity_exact": True,
            "maximum_computed_feedback_error_rational": str(max(computed_errors)),
            "maximum_applied_feedback_error_rational": str(max(applied_errors)),
            "feedback_error_bound_from_actual_probability_l1_rational": str(feedback_bound)}


def verify(directory):
    directory = Path(directory).resolve()
    comparisons, failures, actual_records, batch_records, batch_checks = [], [], [], [], []
    report = {"schema": "grvq-full-qbraid-verification/v2", "status": "incomplete",
              "comparisons": comparisons, "failures": failures,
              "batch_comparisons": batch_checks,
              "verifier_sha256": digest(Path(__file__)),
              "method": {"probability_arithmetic": "Fraction.from_float of every returned JSON binary64 value",
                         "reference_probabilities": "Unchanged saved exact fractions for all sixteen outcomes",
                         "batch_joint_reference": "Product of the three unchanged exact four-qubit reference distributions; every one of4096 outcomes checked",
                         "marginal_arithmetic": "Exact Fraction sums of all256 joint probabilities contributing to each four-bit outcome; never a float sum or normalization",
                         "probability_tolerance_rational": str(PROBABILITY_TOLERANCE),
                         "normalization_sum_tolerance_rational": str(PROBABILITY_TOLERANCE),
                         "moment_tolerance_rational": str(MOMENT_TOLERANCE),
                         "energy": "Exact Pauli and sixteen-state contractions of retained rational lambda enclosures; discrepancy bounded by exact binary64 probability errors",
                         "energy_error_summary": "Rigorous absolute upper bound, not subtraction of rounded display energies",
                         "feedback": "Exact -Z/1000 and signed-basis injection/projection; paired disabled controls apply exact zero",
                         "angle_precision": "Original saved QASM projection uses forty decimal digits; provider arithmetic is binary64",
                         "normalization_or_clamping": False, "omitted_outcomes": 0,
                         "field_or_optimizer_rerun": False, "provider_feedback_replaces_saved_trajectory": False}}
    per_case = {}
    try:
        manifest_path, suite_path = directory/"qbraid_manifest.json", directory/"results.json"
        manifest, suite = read_json(manifest_path), read_json(suite_path)
        require(manifest["schema"] == "grvq-full-qbraid-manifest/v1", "Unexpected provider manifest schema")
        require(manifest["source_commit"] == SOURCE_COMMIT, "Provider checks do not refer to the immutable completed sweep commit")
        require(manifest["sweep_results_sha256"] == digest(suite_path), "Provider manifest and exact sweep hashes disagree")
        require(suite["status"] == "complete" and len(suite["cases"]) == 36 and suite["total_frames"] == 72, "The source sweep is not the complete36case72frame experiment")
        require(manifest["expected_circuits"] == 288 and manifest["expected_frames"] == 72, "Provider coverage is not the complete requested grid")
        require(tuple(manifest["bases"]) == BASES and manifest["qpu"] is False, "Unexpected measurement bases or execution method")
        require(type(manifest["credits"]) in (int,float) and math.isfinite(manifest["credits"]), "Invalid provider credit metadata")
        source_audit_path = directory/"independent_output_audit.json"
        source_audit = read_json(source_audit_path)
        require(source_audit["status"] == "passed" and source_audit["suite_results_sha256"] == digest(suite_path), "The completed sweep's independent audit is not bound to these data")
        source_basis = source_audit["source_projection_basis"]
        require(len(source_basis["images"]) == len(source_basis["projection"]) == 4, "The source audit lacks the complete four-mode injection basis")
        for q in range(4):
            require(list(map(F,source_basis["images"][q])) == [F(1 if i & (1 << q) else -1) for i in range(16)], "The provider feedback projection uses a different basis from the executed source")
            require(list(map(F,source_basis["projection"][q])) == [F(q==k) for k in range(4)], "The executed source basis is not dual to its mode projection")
        report.update(source_commit=SOURCE_COMMIT, sweep_results_sha256=digest(suite_path),
                      manifest_sha256=digest(manifest_path), source_independent_audit_sha256=digest(source_audit_path),
                      provider=manifest["provider"], credits_spent=manifest["credits"], qpu=False)
        cases, exported, frame_records = {}, {}, {}
        for case in suite["cases"]:
            case_id = case["configuration"]["case_id"]
            require(case_id not in cases and case["status"] == "complete", "Duplicate or incomplete source case")
            case_path = contained_path(directory,case["result_file"])
            require(digest(case_path) == case["result_sha256"], "A source case result changed: "+case_id)
            result = read_json(case_path)
            require(result["status"] == "complete" and result["configuration"] == case["configuration"], "Source case metadata disagree")
            require(result["steps"] == case["steps"] and [x["step"] for x in result["steps"]] == [0,1], "Source case must retain both complete ordered steps")
            cases[case_id] = {**case,"_result":result}
            per_case[case_id] = {"expected_circuits":8,"checked_circuits":0,"z_frames_checked":0,"basis_counts":{},"errors":[],"status":"incomplete"}
            for step in result["steps"]:
                frame_records[case_id,step["step"]] = step
            qasm_manifest_path = case_path.parent/"qasm_manifest.json"
            qasm_manifest = read_json(qasm_manifest_path)
            require(len(qasm_manifest["circuits"]) == 8, "A source case does not export all eight circuits")
            for item in qasm_manifest["circuits"]:
                key = case_id,item["step"],item["basis"]
                require(key not in exported and item["step"] in (0,1) and item["basis"] in BASES, "Duplicate or unexpected source QASM export")
                exported[key] = {"qasm":str((case_path.parent/item["qasm"]).relative_to(directory)),
                                 "expected":str((case_path.parent/item["expected"]).relative_to(directory)),
                                 "qasm_sha256":item["qasm_sha256"],"case_qasm_manifest_sha256":digest(qasm_manifest_path)}
        expected = {(case_id,step,basis) for case_id in cases for step in (0,1) for basis in BASES}
        require(len(expected) == 288 and set(exported) == expected, "Original QASM export coverage is incomplete")
        submitted = {}
        for item in manifest["circuits"]:
            key = identity(item)
            require(key not in submitted, "Duplicate provider manifest circuit: "+label(key))
            submitted[key] = item
        require(set(submitted) == expected, "Provider manifest differs from the complete original export set")
        plan = manifest["execution_plan"]
        require((plan["individual_circuits"],plan["batched_circuits"],plan["batch_programs"],plan["expected_successful_simulator_responses"]) == (75,213,71,146), "The explicit direct/batched execution plan differs from the complete authorized plan")
        batches = {}
        membership = {}
        for batch in manifest["batches"]:
            ident = batch["batch_id"]
            require(re.fullmatch(r'batch_[0-9]{3}',ident) is not None and ident not in batches, "Duplicate or invalid batch identifier")
            batches[ident] = batch
            require(batch["qubit_count"] == 12 and len(batch["members"]) == 3, "A batch does not contain three complete independent circuits")
            for slot, member in enumerate(batch["members"]):
                key = identity(member)
                require(key in submitted and key not in membership, "Duplicate or unknown batched logical circuit")
                require(member["slot"] == slot and member["qubits"] == list(range(4*slot,4*slot+4)), "A batch changes its register assignment")
                membership[key] = ident,slot
        require(set(batches) == {f'batch_{i:03d}' for i in range(71)} and len(membership) == 213, "The complete71batch213circuit plan is not present")
        direct_keys = set()
        references = {}
        for key, item in submitted.items():
            original = exported[key]
            require(all(item[name] == original[name] for name in ("qasm","expected","qasm_sha256")), "Provider manifest changes an original circuit or reference path")
            require(digest(contained_path(directory,item["qasm"])) == item["qasm_sha256"], "An original circuit changed")
            expected_path = contained_path(directory,item["expected"])
            require(digest(expected_path) == item["expected_sha256"], "An original exact probability reference changed")
            reference = read_json(expected_path)
            require(reference["basis"] == key[2] and set(reference["probabilities_exact"]) == set(BITSTRINGS), "An exact reference has the wrong measurement basis or state set")
            references[key] = {bits:F(reference["probabilities_exact"][bits]) for bits in BITSTRINGS}
            require(all(0 <= x <= 1 for x in references[key].values()) and sum(references[key].values()) == 1, "An exact reference is not a complete normalized distribution")
            route = item["execution"]
            if route["kind"] == "individual":
                require(key not in membership and route["response_file"] == item["response_file"], "A direct response conflicts with the batch plan")
                direct_keys.add(key)
            else:
                require(route["kind"] == "batch_marginal" and key in membership, "A logical circuit has an unknown execution route")
                ident, slot = membership[key]
                require((route["batch_id"],route["slot"]) == (ident,slot) and route["qubits"] == list(range(4*slot,4*slot+4)), "A logical circuit's batch route differs from actual membership")
                require(route["response_file"] == batches[ident]["response_file"], "A logical circuit points to the wrong batch response")
        require(len(direct_keys) == 75 and direct_keys | set(membership) == expected and not direct_keys & set(membership), "The actual direct and batched logical circuit sets are incomplete or overlap")
        response_names = [submitted[key]["execution"]["response_file"] for key in direct_keys] + [batch["response_file"] for batch in batches.values()]
        require(len(set(response_names)) == 146, "Distinct executed programs share a provider response path")
        existing = {str(x.relative_to(directory)) for folder in ("provider_responses","provider_batches") for x in (directory/folder).rglob("*.json")}
        missing, unexpected = sorted(set(response_names)-existing), sorted(existing-set(response_names))
        report["coverage"] = {"expected_circuits":288,"manifest_circuits":len(submitted),"response_files_found":len(existing),
                              "expected_actual_program_responses":146,"direct_logical_circuits":75,"batched_logical_circuits":213,"batch_programs":71,
                              "missing_response_files":missing,"unexpected_response_files":unexpected,
                              "bases":list(BASES),"expected_circuits_per_basis":72,"expected_cases":36,"expected_frames":72}
        require(not missing and not unexpected, "Provider response set is incomplete or contains unexpected files")
        batch_cache = {}
        for ident in sorted(batches):
            try:
                entry = verify_joint_batch(directory,batches[ident],submitted,references)
                batch_cache[ident] = entry;batch_checks.append(entry["check"]);batch_records.append(entry["response"])
            except (VerificationError,KeyError,ValueError,TypeError,OSError,OverflowError) as error:
                message = f"{type(error).__name__}: {error}"
                batch_checks.append({"batch_id":ident,"status":"failed","error":message})
                failures.append({"batch_id":ident,"error":message})
        for key in sorted(expected):
            item, original = submitted[key], exported[key]
            row = {"case_id":key[0],"step":key[1],"basis":key[2],"status":"failed"}
            try:
                require(all(item[k] == original[k] for k in ("qasm","expected","qasm_sha256")), "Provider manifest changes an original circuit identity or path")
                qasm_path, expected_path = [contained_path(directory,item[name]) for name in ("qasm","expected")]
                require(digest(qasm_path) == item["qasm_sha256"], "Submitted QASM hash does not match the unchanged export")
                require(digest(expected_path) == item["expected_sha256"], "Saved exact probability reference hash changed")
                route = item["execution"]
                response_path = contained_path(directory,route["response_file"])
                if route["kind"] == "individual":
                    response = read_json(response_path)
                    require(identity(response) == key and response["status"] == "returned", "Returned circuit identity or execution status differs")
                    require(response["source_commit"] == SOURCE_COMMIT and response["qasm_sha256"] == item["qasm_sha256"], "Returned circuit provenance differs")
                    require(response["qasm"].encode("utf-8") == qasm_path.read_bytes(), "Actual submitted QASM text differs from the saved file")
                    validation, simulation = response["validation"], response["simulation"]
                    require(raw_payload(response["raw_validation"],"valid") == validation, "Parsed validation differs from the retained raw tool response")
                    require(raw_payload(response["raw_simulation"],"probabilities") == simulation, "Parsed simulation differs from the retained raw tool response")
                    require(validation["valid"] is True and validation["errors"] == [], "qBraid circuit validation failed")
                    require(validation["stats"]["numQubits"] == 4 and validation["stats"]["numClbits"] == 4 and simulation["num_qubits"] == 4, "The returned circuit is not the complete four-qubit/four-bit export")
                    require(set(simulation["probabilities"]) == set(BITSTRINGS), "Provider response omits or adds a bitstring outcome")
                    probabilities = {k:exact_binary64(simulation["probabilities"][k]) for k in BITSTRINGS}
                    raw_numbers = simulation["probabilities"]
                    executed_qubits = 4
                    provider_qasm_sha = item["qasm_sha256"]
                else:
                    ident, slot = route["batch_id"],route["slot"]
                    require(ident in batch_cache, "The required full joint batch failed verification: "+ident)
                    entry = batch_cache[ident]
                    response = entry["response"];validation = response["validation"]
                    probabilities = entry["marginals"][slot]
                    raw_numbers = None;executed_qubits = 12;provider_qasm_sha = batches[ident]["qasm_sha256"]
                    row["batch_marginal_provenance"] = {"batch_id":ident,"slot":slot,"qubits":route["qubits"],
                                                       "joint_outcomes":4096,"joint_terms_per_marginal_outcome":256,
                                                       "exact_fraction_sums":True,"manufactured_individual_tool_response":False}
                exact = references[key]
                require(all(0 <= x <= 1 for x in exact.values()) and sum(exact.values()) == 1, "Reference probabilities are not an exact normalized distribution")
                require(all(0 <= x <= 1+PROBABILITY_TOLERANCE for x in probabilities.values()), "Returned probabilities are outside their allowed numerical range")
                total = sum(probabilities.values());normalization_error = abs(total-1)
                require(normalization_error <= PROBABILITY_TOLERANCE, "Returned probability sum is outside the declared tolerance")
                errors = {k:abs(probabilities[k]-exact[k]) for k in BITSTRINGS}
                require(all(x <= PROBABILITY_TOLERANCE for x in errors.values()), "A bitstring probability differs by more than exact1/10^12")
                l1_error = sum(errors.values())
                row.update(response_file=route["response_file"],response_sha256=digest(response_path),qasm=item["qasm"],
                           qasm_sha256=digest(qasm_path),expected=item["expected"],expected_sha256=digest(expected_path),
                           case_qasm_manifest_sha256=original["case_qasm_manifest_sha256"],validation_stats=validation["stats"],
                           execution_kind=route["kind"],logical_qubits=4,executed_program_qubits=executed_qubits,
                           provider_program_qasm_sha256=provider_qasm_sha,outcomes_checked=16,probability_sum_rational=str(total),
                           probability_sum_error_rational=str(normalization_error),probability_sum_error_display=float(normalization_error),
                           probability_l1_error_rational=str(l1_error),maximum_probability_error_rational=str(max(errors.values())),
                           maximum_probability_error_display=float(max(errors.values())),
                           probability_comparisons={k:{**({"returned_json_number":raw_numbers[k]} if raw_numbers is not None else
                                                         {"derived_marginal_rational":str(probabilities[k])}),
                                                      **exact_error(probabilities[k],exact[k])} for k in BITSTRINGS})
                if key[2] == "ZZZZ":
                    row["z_frame_verification"] = verify_z_frame(directory,cases[key[0]],frame_records[key[:2]],probabilities,exact,l1_error)
                    per_case[key[0]]["z_frames_checked"] += 1
                row["status"] = "passed"
                per_case[key[0]]["checked_circuits"] += 1
                basis_counts = per_case[key[0]]["basis_counts"]
                basis_counts[key[2]] = basis_counts.get(key[2],0)+1
                if route["kind"] == "individual":
                    actual_records.append(response)
            except (VerificationError,KeyError,ValueError,TypeError,OSError,OverflowError) as error:
                message = f"{type(error).__name__}: {error}"
                row["error"] = message;failures.append({"circuit":label(key),"error":message})
                per_case[key[0]]["errors"].append({"step":key[1],"basis":key[2],"error":message})
            comparisons.append(row)
        require(not failures and len(comparisons) == 288, "One or more complete provider comparisons failed")
        require(len(actual_records) == 75 and len(batch_records) == len(batch_checks) == 71 and all(x["status"] == "passed" for x in batch_checks), "The full75direct71batch provider execution set did not pass")
        z_rows = [r["z_frame_verification"] for r in comparisons if r["basis"] == "ZZZZ"]
        require(len(z_rows) == 72, "Not all seventy-two Z frames were independently checked")
        for case_id, item in per_case.items():
            require(item["checked_circuits"] == 8 and item["z_frames_checked"] == 2 and item["basis_counts"] == {basis:2 for basis in BASES}, "A case lacks full four-basis/two-frame coverage")
            item["status"] = "complete"
            rows = [row for row in comparisons if row["case_id"] == case_id]
            item["execution_counts"] = dict(Counter(row["execution_kind"] for row in rows))
            z_checks = [row["z_frame_verification"] for row in rows if row["basis"] == "ZZZZ"]
            for name, values in (
                ("maximum_probability_error",[F(row["maximum_probability_error_rational"]) for row in rows]),
                ("maximum_probability_sum_error",[F(row["probability_sum_error_rational"]) for row in rows]),
                ("maximum_energy_error",[F(row["absolute_energy_error_upper_bound_rational"]) for row in z_checks]),
                ("maximum_feedback_increment_error",[F(row["maximum_computed_feedback_error_rational"]) for row in z_checks]),
                ("maximum_applied_feedback_increment_error",[F(row["maximum_applied_feedback_error_rational"]) for row in z_checks])):
                maximum = max(values)
                item[name] = float(maximum);item[name+"_rational"] = str(maximum)
        max_probability = max(F(r["maximum_probability_error_rational"]) for r in comparisons)
        max_energy = max(F(r["absolute_energy_error_upper_bound_rational"]) for r in z_rows)
        max_feedback = max(F(r["maximum_computed_feedback_error_rational"]) for r in z_rows)
        max_applied = max(F(r["maximum_applied_feedback_error_rational"]) for r in z_rows)
        max_joint = max(F(r["maximum_joint_probability_error_rational"]) for r in batch_checks)
        history = [{"file":str(p.relative_to(directory)),"sha256":digest(p)} for p in sorted((directory/"provider_attempt_history").rglob("*.json"))]
        aggregate = {"schema":"grvq-complete-qbraid-responses/v2","status":"complete", "source_commit":SOURCE_COMMIT,
                     "sweep_results_sha256":digest(suite_path),"manifest_sha256":digest(manifest_path),"provider":manifest["provider"],
                     "credits_spent":manifest["credits"],"logical_circuits":288,"actual_simulator_responses":146,
                     "individual_circuit_responses":actual_records,"batch_program_responses":batch_records,
                     "logical_execution_routes":[{"case_id":r["case_id"],"step":r["step"],"basis":r["basis"],"execution":r["execution"]} for r in manifest["circuits"]],
                     "response_file_sha256":{r["response_file"]:r["response_sha256"] for r in comparisons},
                     "historical_transport_attempts":history,
                     "record_semantics":"Complete actual75four-qubit and71twelve-qubit tool response records;213 logical distributions are exact derived marginals, never manufactured individual tool responses. All original raw files and failed transport attempt history remain unchanged."}
        aggregate_path = directory/"qbraid_results.json"
        atomic_json(aggregate_path,aggregate)
        report.update(status="complete",circuits=288,frames_checked_in_z=72,probability_values_checked=288*16,
                      logical_circuits=288,actual_simulator_calls=146,successful_validation_calls=146,
                      direct_circuits=75,batched_circuits=213,batch_programs=71,
                      direct_probability_values_checked=75*16,joint_probabilities_checked=71*4096,
                      derived_marginal_probabilities_checked=213*16,raw_probability_values_checked=75*16+71*4096,
                      maximum_joint_probability_error=float(max_joint),maximum_joint_probability_error_rational=str(max_joint),
                      historical_transport_attempt_files=len(history),historical_transport_attempts=history,
                      basis_counts=dict(Counter(r["basis"] for r in comparisons)),
                      circuits_by_basis=dict(Counter(r["basis"] for r in comparisons)),
                      maximum_probability_error=float(max_probability),maximum_probability_error_rational=str(max_probability),
                      maximum_energy_error=float(max_energy),maximum_energy_error_rational=str(max_energy),
                      maximum_feedback_increment_error=float(max_feedback),maximum_feedback_increment_error_rational=str(max_feedback),
                      maximum_applied_feedback_increment_error=float(max_applied),maximum_applied_feedback_increment_error_rational=str(max_applied),
                      maximum_probability_sum_error=max(r["probability_sum_error_display"] for r in comparisons),
                      maximum_probability_sum_error_rational=str(max(F(r["probability_sum_error_rational"]) for r in comparisons)),
                      qbraid_results_sha256=digest(aggregate_path))
    except (VerificationError,KeyError,ValueError,TypeError,OSError,OverflowError) as error:
        failures.append({"scope":"complete_suite","error":f"{type(error).__name__}: {error}"})
        report["status"] = "failed" if comparisons else "incomplete"
    report["per_case_coverage"] = per_case
    report["coverage_complete"] = report["status"] == "complete"
    atomic_json(directory/"qbraid_verification.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=BASE/"latest_hybrid_results")
    outcome = verify(parser.parse_args().results)
    print(json.dumps({k:outcome[k] for k in ("status","circuits","frames_checked_in_z","maximum_probability_error",
                                           "maximum_energy_error","maximum_feedback_increment_error","failures") if k in outcome},indent=2,allow_nan=False))
    raise SystemExit(0 if outcome["status"] == "complete" else 1)
