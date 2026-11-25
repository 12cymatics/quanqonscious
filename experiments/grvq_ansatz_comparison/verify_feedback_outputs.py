"""Independently audit the completed saved quantum–GRVQ trajectories.

Uses Python Fraction arithmetic for all circuit amplitudes, moments, feedback
increments and additive expression identities. It does not rerun optimization
or substitute decimal lambda values for the retained exact expression graph.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.set_int_max_str_digits(0)

HERE = Path(__file__).resolve().parent
MODES = ("SERIES", "PARALLEL", "SYMMETRIC_CONCURRENT")


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def independent_circuit(parameters):
    """Direct exact matrix-element action, independent of the JS simulator."""
    parameters = tuple(map(F, parameters))
    require(len(parameters) == 12, "The recovered source has twelve Ry coordinates")
    state = tuple(F(index == 0) for index in range(16))
    for layer in range(3):
        for qubit in range(4):
            t = parameters[4 * layer + qubit]
            c, s = (1-t*t)/(1+t*t), 2*t/(1+t*t)
            mask = 1 << qubit
            state = tuple(c*state[i] + (s if i & mask else -s)*state[i ^ mask]
                          for i in range(16))
        for control in range(3):
            state = tuple(state[i ^ (1 << (control+1))] if i & (1 << control)
                          else state[i] for i in range(16))
    return state


def expectation(probabilities, mask):
    return sum(((-1)**((i & mask).bit_count()))*p for i, p in enumerate(probabilities))


class Graph:
    def __init__(self, record):
        require(record["schema"] == "exact-rational-expression-dag/v1", "Unsupported expression schema")
        self.nodes = record["nodes"]
        self.hashes = []
        self.additive_cache = {}
        self.certificates = {(c["expression"], c["kind"], c["result"]): c for c in record["certificates"]}
        require(record["statistics"]["interval_values_in_state"] == 0, "An interval replaced an exact state")

    def scalar(self, value):
        if isinstance(value, str):
            return F(value)
        if "n" in value and "d" in value:
            return F(int(value["n"]), int(value["d"]))
        node = self.nodes[value["_expr"]]
        require(node[0] == "q", "A literal rational was expected")
        return F(int(node[1]), int(node[2]))

    def root_hash(self, value):
        target = value["_expr"]
        while len(self.hashes) <= target:
            node = self.nodes[len(self.hashes)]
            if node[0] == "q":
                data = node
            elif node[0] in ("add", "lin"):
                denominator, terms = ("1", node[1]) if node[0] == "add" else (node[1], node[2])
                data = [node[0], denominator, sorted([[self.hashes[i], c] for i, c in terms],key=lambda term:term[0])]
            else:
                children = [self.hashes[node[1]], self.hashes[node[2]]]
                data = [node[0], *(sorted(children) if node[0] == "mul" else children)]
            encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()
            self.hashes.append(hashlib.sha256(encoded).hexdigest())
        return self.hashes[target]

    def signature(self, value):
        """Exact additive form; nonlinear source nodes remain formal atoms.

        The only distributed scalar product is the source inverseL factor1/16.
        Thus source feedback can be checked without expanding its enormous
        rational inputs or assuming any identities between nonlinear atoms.
        """
        root = value["_expr"]
        if root in self.additive_cache:
            return self.additive_cache[root]
        node = self.nodes[root]
        if node[0] == "q":
            scalar = F(int(node[1]), int(node[2]))
            result = {} if not scalar else {None: scalar}
        elif node[0] == "add":
            result = {}
            for child, coefficient in node[1]:
                self.accumulate(result, self.signature({"_expr": child}), F(int(coefficient)))
        elif node[0] == "mul":
            a, b = self.nodes[node[1]], self.nodes[node[2]]
            if a[0] == "q" and F(int(a[1]), int(a[2])) == F(1,16):
                result = {k: c/16 for k,c in self.signature({"_expr":node[2]}).items()}
            elif b[0] == "q" and F(int(b[1]), int(b[2])) == F(1,16):
                result = {k: c/16 for k,c in self.signature({"_expr":node[1]}).items()}
            else:
                result = {root:F(1)}
        else:
            result = {root:F(1)}
        self.additive_cache[root] = result
        return result

    @staticmethod
    def accumulate(target, addition, factor=F(1)):
        for key, value in addition.items():
            updated = target.get(key,F(0)) + factor*value
            if updated:
                target[key] = updated
            else:
                target.pop(key,None)

    def equals_with_increment(self, after, before, increment):
        expected = dict(self.signature(before))
        self.accumulate(expected, {None:increment})
        return self.signature(after) == expected

    def recorded_decision(self, value, sign):
        node = self.nodes[value["_expr"]]
        if node[0] == "q":
            number = F(int(node[1]),int(node[2]))
            return ((number > 0)-(number < 0)) == sign
        certificate = self.certificates.get((value["_expr"],"sign",str(sign)))
        if certificate is None:
            return False
        if "proof" in certificate:
            return sign == 0 and certificate["proof"] == "exact_rational_linear_identity"
        lo, hi = int(certificate["lower_integer"]),int(certificate["upper_integer"])
        return lo > 0 if sign > 0 else hi < 0 if sign < 0 else lo == hi == 0


def source_projection_check():
    script = r'''
const E=require('./latest_hybrid_executor.cjs'),K=E.loadKernel(),Q=K.Q;
const gram=[],images=[];
for(let k=0;k<4;k++){
 const image=K.injectLambda(Array(16).fill(Q.ZERO),k,Q.ONE);
 images.push(image.map(Q.toString));
 gram.push(K.inverseL(image).map(Q.toString));
}
process.stdout.write(JSON.stringify({projection:gram,images}));
'''
    completed = subprocess.run(["node","-e",script],cwd=HERE,capture_output=True,text=True,check=True)
    result = json.loads(completed.stdout)
    for k in range(4):
        require(tuple(map(F,result["projection"][k])) == tuple(F(q==k) for q in range(4)), "Actual source projection/injection composition is not identity")
        require(tuple(map(F,result["images"][k])) == tuple(F(1 if i&(1<<k) else -1) for i in range(16)), "Actual source signed basis differs from the feedback proof")
    return result


def verify(directory):
    directory = Path(directory).resolve()
    sweep = read_json(directory/"results.json")
    require(sweep["status"] == "complete", "The trajectory sweep is not complete")
    require(sweep["expected_cases"] == len(sweep["cases"]) == 36, "The declared36trajectory sweep is incomplete")
    require(sweep["expected_frames"] == sweep["total_frames"] == 72, "The72frame sweep is incomplete")
    expected = {(m,s,seed,feedback) for m in MODES for s in ("1/29","1/58") for seed in (7,19,43) for feedback in (False,True)}
    actual = {(c["configuration"]["composition"],c["configuration"]["strength"],c["configuration"]["seed"],c["configuration"]["feedback"]) for c in sweep["cases"]}
    require(actual == expected, "The experiment grid differs from the declared36trajectories")
    source_basis = source_projection_check()
    counts=Counter();pairs={};case_checks=[];long_periods=[]
    for case in sweep["cases"]:
        cfg=case["configuration"];folder=directory/Path(case["result_file"]).parent
        result=read_json(folder/"result.json");require(result["status"]=="complete","A case did not complete")
        for filename,sha in result["code_sha256"].items():
            require(sha==sweep["configuration"]["code_sha256"][filename]==digest(HERE/filename),"A completed trajectory used different executable code: "+filename)
        require(digest(folder/"result.json")==case["result_sha256"],"A saved case result changed")
        for filename,sha in result["artifact_sha256"].items():
            require(digest(folder/filename)==sha,"A complete trajectory artifact changed: "+filename)
        graph=Graph(read_json(folder/"exact_graph.json.gz"))
        previous=None
        for summary in result["steps"]:
            frame=read_json(folder/summary["frame_file"]);quantum=read_json(folder/summary["quantum_file"])
            source=frame["source_formula"];feedback=frame["feedback"];qframe=frame["quantum"]
            require([s["id"] for s in source["active"]]==list(range(1,30)),"A frame omitted an active method")
            require(all(graph.scalar(s["strength"])==F(cfg["strength"])>0 for s in source["active"]),"An active method strength was zero or substituted")
            expected_order=list(range(1,30))+(list(range(29,0,-1)) if cfg["composition"]=="SYMMETRIC_CONCURRENT" else [])
            require([r["id"] for r in source["sutra_records"]]==expected_order,"A source method application is missing or reordered")
            require([r["id"] for r in source["distributed_grvq_records"]]==[i for i in range(1,30) if i not in (1,5,9,14)],"A distributed GRVQ contribution is missing")
            require(source["matrix_representation"]=="commuting_tensor","A trajectory used an undeclared matrix representation")
            for record in source["sutra_records"]:
                certificate=record["exact_matrix_representation"]
                zero_terms=sum(not graph.scalar(term) for term in record["terms"])
                nonzero_terms=len(record["terms"])-zero_terms
                require(certificate["visited_terms"]==len(record["terms"]),"The matrix did not consume every source coefficient")
                require(certificate["zero_identity_terms"]==zero_terms and certificate["nonzero_terms"]==nonzero_terms,"A source coefficient was silently treated as zero")
                require(sum(certificate["grouped_multiplicities"])==nonzero_terms and certificate["total_source_matching_rotations"]==8*nonzero_terms,"The exact grouped product changed a coefficient multiplicity")
                require(certificate["axis_multiplicities"]==[2,2,2,2] and certificate["exact_norm_identity"] and certificate["unreduced_source_denominator_preserved"],"The source matrix identity is unproven")
            for root_key,summary_key in (("input","input_field_digests"),("afterB","after_formula_field_digests")):
                require([graph.root_hash(x) for x in source[root_key]]==summary[summary_key],"A saved source field differs from its digest")
            next_hashes=[graph.root_hash(x) for x in feedback["next_field"]]
            require(next_hashes==summary["next_field_digests"],"Feedback output digest differs from the actual field")
            if previous is not None:
                require(summary["input_field_digests"]==previous,"A step did not consume the complete prior feedback output")
            previous=next_hashes
            amplitudes=independent_circuit(quantum["parameters"])
            require(amplitudes==tuple(map(F,quantum["amplitudes"])),"Actual quantum amplitudes differ from independent source gate evaluation")
            probs=tuple(a*a for a in amplitudes);require(sum(probs)==1,"Quantum norm differs from1")
            require(probs==tuple(map(F,quantum["probabilities"])),"Quantum probabilities differ from squared amplitudes")
            zs=tuple(expectation(probs,1<<q) for q in range(4))
            zzs=tuple(expectation(probs,(1<<q)|(1<<(q+1))) for q in range(3))
            require(zs==tuple(map(F,quantum["z"])) and zzs==tuple(map(F,quantum["zz"])),"Quantum Pauli moments differ from exact probabilities")
            require(quantum=={k:qframe[k] for k in quantum},"Quantum state and frame records disagree")
            computed=tuple(-z/1000 for z in zs);applied=computed if cfg["feedback"] else (F(0),)*4
            require(computed==tuple(map(F,feedback["computed_delta_rationals"])),"Feedback law is not exactly-Z/1000")
            require(applied==tuple(map(F,feedback["applied_delta_rationals"])),"The wrong feedback switch/delta was applied")
            for i in range(16):
                increment=sum((1 if i&(1<<q) else -1)*applied[q] for q in range(4))
                require(graph.equals_with_increment(feedback["next_field"][i],source["afterB"][i],increment),"Actual next field does not contain the exact signed feedback increment")
            for q in range(4):
                require(graph.equals_with_increment(feedback["actual_next_lambda"][q],source["lambdaOut"][q],applied[q]),"Actual projected lambda differs from its exact expected increment")
            lines=qframe["coordinate_records"]
            require(len(lines)==cfg["sweeps"]*12,"The optimizer skipped a declared coordinate or sweep")
            for number,line in enumerate(lines):
                require((line["sweep"],line["coordinate"])==divmod(number,12),"Coordinate search order differs from its declaration")
                require(len(line["trials"])==2*cfg["grid_denominator"]+1,"A declared rational grid candidate was omitted")
                for trial in line["trials"]:
                    require(graph.recorded_decision(trial["comparison_difference"],trial["comparison_with_incumbent"]),"A candidate comparison lacks its exact decision certificate")
            for record in source["sutra_records"]:
                if record["id"] not in (1,12):continue
                d=graph.scalar(record["operands"]["denominator"])
                if d<=999:continue
                require(d.denominator==1,"Repetend denominator must be an integer")
                denominator=d.numerator;remainder=1;digits=[]
                while True:
                    digit,remainder=divmod(10*remainder,denominator);digits.append(digit)
                    if remainder==1:break
                actual_digits=[graph.scalar(t) for t in record["terms"]]
                require(actual_digits==digits,"An extended repetend digit was changed or omitted")
                long_periods.append({"case_id":cfg["case_id"],"step":summary["step"],"id":record["id"],"denominator":str(denominator),"complete_digits":len(digits)})
            counts.update(frames=1,quantum_amplitude_equalities=16,field_feedback_equalities=16,lambda_feedback_equalities=4,
                          core_method_calls=len(source["sutra_records"]),distributed_grvq_updates=len(source["distributed_grvq_records"]),
                          coordinate_lines=len(lines),candidate_comparison_certificates=sum(len(line["trials"]) for line in lines))
            counts["complete_source_coefficients_consumed"]+=sum(len(record["terms"]) for record in source["sutra_records"])
        checkpoint=read_json(folder/"checkpoint.json")
        require(checkpoint["next_step"]==2,"The final checkpoint is not after both declared steps")
        require([graph.root_hash(v) for v in checkpoint["field"]]==previous,"Final checkpoint did not retain the feedback field")
        key=(cfg["composition"],cfg["strength"],cfg["seed"]);pairs.setdefault(key,{})[cfg["feedback"]]=result
        case_checks.append({"case_id":cfg["case_id"],"frames":len(result["steps"]),"exact_checks_passed":True})
    require(counts["core_method_calls"]==2784 and counts["distributed_grvq_updates"]==1800,"Aggregate all29 execution totals disagree")
    paired=[]
    for key,pair in pairs.items():
        require(set(pair)=={False,True},"A paired feedback control is missing")
        off,on=pair[False],pair[True]
        require(off["configuration"]["initial_parameters"]==on["configuration"]["initial_parameters"],"Paired controls used different quantum starts")
        require(off["steps"][0]["input_field_digests"]==on["steps"][0]["input_field_digests"],"Paired controls used different initial fields")
        require(off["steps"][0]["after_formula_field_digests"]==on["steps"][0]["after_formula_field_digests"],"Paired initial source formula differs before quantum feedback")
        require(any(map(F,on["steps"][0]["feedback_deltas"])),"The enabled control has no nonzero first feedback to test")
        require(off["steps"][1]["input_field_digests"]!=on["steps"][1]["input_field_digests"],"Enabled feedback did not change the next source input")
        paired.append({"composition":key[0],"strength":key[1],"seed":key[2],"identical_initial_conditions":True,"second_step_input_changed_by_feedback":True})
    require(len(paired)==18,"Expected18pairedfeedbacktrajectories")
    output={"schema":"grvq-feedback-independent-output-audit/v1","status":"passed","suite_results_sha256":digest(directory/"results.json"),
            "cases":case_checks,"counts":dict(counts),"paired_controls":paired,"source_projection_basis":source_basis,
            "extended_repetend_checks":long_periods,"arithmetic":"Python Fraction for quantum evolution and feedback identities; source exact-expression commitments and retained decision certificates"}
    target=directory/"independent_output_audit.json";target.write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path,nargs="?",default=HERE/"latest_hybrid_results")
    result=verify(parser.parse_args().directory)
    print(json.dumps({"status":result["status"],"counts":result["counts"],"pairs":len(result["paired_controls"]),"extended_repetends":len(result["extended_repetend_checks"])},indent=2))
