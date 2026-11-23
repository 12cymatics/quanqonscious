"""Compare saved actual two-frame PARALLEL worker and serial executions."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    with (gzip.open(path,"rt") if path.suffix==".gz" else path.open()) as stream:
        return json.load(stream)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def verify(serial,workers,output,*,require_same_code=True):
    a,b=read(serial/"result.json"),read(workers/"result.json")
    assert a["status"]==b["status"]=="complete"
    assert a["configuration"]["backend"]=="serial" and b["configuration"]["backend"]=="workers"
    assert a["configuration"]["composition"]==b["configuration"]["composition"]=="PARALLEL"
    ac,bc=a["configuration"].copy(),b["configuration"].copy()
    for obj in (ac,bc):
        for key in ("case_id","backend"):obj.pop(key)
    assert ac==bc,"The backend comparison changed the mathematical experiment"
    if require_same_code:assert a["code_sha256"]==b["code_sha256"]
    assert len(a["steps"])==len(b["steps"])==2
    records=[]
    for left,right in zip(a["steps"],b["steps"]):
        for key in ("input_field_digests","after_formula_field_digests","next_field_digests"):
            assert left[key]==right[key],key
        qa,qb=read(serial/left["quantum_file"]),read(workers/right["quantum_file"])
        assert qa==qb,"Actual rational circuit states differ between CPU backends"
        af,bf=read(serial/left["frame_file"]),read(workers/right["frame_file"])
        receipts=bf["source_formula"]["worker_receipts"]
        assert [x["id"] for x in receipts]==list(range(1,30))
        ids=sorted({x["worker_thread_id"] for x in receipts})
        assert len(ids)==2 and all(i>0 for i in ids)
        for key in ("computed_delta_rationals","applied_delta_rationals"):
            assert af["feedback"][key]==bf["feedback"][key]
        records.append({"step":left["step"],"exact_input_field_digest_equalities":16,
                        "exact_after_formula_field_digest_equalities":16,"exact_feedback_field_digest_equalities":16,
                        "actual_worker_thread_ids":ids,"all29_worker_receipts":True,
                        "identical_exact_parameters_amplitudes_probabilities_and_moments":True})
    result={"schema":"grvq-feedback-parallel-backend-equivalence/v1","status":"passed",
            "serial_directory":str(serial.relative_to(HERE)),"workers_directory":str(workers.relative_to(HERE)),
            "serial_result_sha256":sha(serial/"result.json"),"workers_result_sha256":sha(workers/"result.json"),
            "same_executable_code":a["code_sha256"]==b["code_sha256"],
            "serial_code_sha256":a["code_sha256"],"workers_code_sha256":b["code_sha256"],
            "frames":records,"complete_previous_feedback_state_consumed":a["steps"][1]["input_field_digests"]==a["steps"][0]["next_field_digests"]}
    assert result["complete_previous_feedback_state_consumed"]
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial",type=Path,default=HERE/"latest_hybrid_results/cases/PARALLEL__s1_29__seed7__feedback_on")
    parser.add_argument("--workers",type=Path,default=HERE/"latest_hybrid_backend_validation/workers_case")
    parser.add_argument("--output",type=Path,default=HERE/"latest_hybrid_backend_validation/backend_equivalence.json")
    parser.add_argument("--allow-source-matrix-version-comparison",action="store_true")
    args=parser.parse_args()
    result=verify(args.serial.resolve(),args.workers.resolve(),args.output.resolve(),require_same_code=not args.allow_source_matrix_version_comparison)
    print(json.dumps({"status":result["status"],"frames":len(result["frames"]),"same_executable_code":result["same_executable_code"]}))
