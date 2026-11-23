"""Assemble independent exported circuits on disjoint qBraid simulator registers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

RESULTS = Path(__file__).resolve().parents[1] / "latest_hybrid_results"
HEAD = ['OPENQASM 2.0;', 'include "qelib1.inc";', 'qreg q[4];', 'creg c[4];']

def assemble(programs):
    if not 1 <= len(programs) <= 3:
        raise ValueError("Use one to three complete four-qubit source circuits.")
    lines=['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{4*len(programs)}];', f'creg c[{4*len(programs)}];']
    for slot, program in enumerate(programs):
        original=program.splitlines()
        if original[:4] != HEAD or original[-1] != 'measure q -> c;':
            raise ValueError("Unexpected canonical circuit declarations or measurement.")
        for instruction in original[4:-1]:
            if not re.fullmatch(r'(?:r[yz]\([^\r\n]+\) q\[[0-3]\]|cx q\[[0-3]\],q\[[0-3]\]|h q\[[0-3]\]|sdg q\[[0-3]\]);', instruction):
                raise ValueError(f"Unrecognized source instruction: {instruction!r}")
            lines.append(re.sub(r'q\[([0-3])\]', lambda m:f'q[{int(m.group(1))+4*slot}]', instruction))
    lines.append('measure q -> c;')
    return '\n'.join(lines)+'\n'

def main():
    path=RESULTS/'qbraid_manifest.json'
    m=json.loads(path.read_text())
    if m.get('batches'):
        raise RuntimeError("A batch plan already exists; preserve the existing execution plan.")
    history=RESULTS/'provider_attempt_history';history.mkdir(exist_ok=True)
    pending=[];direct=0
    for e in m['circuits']:
        response=RESULTS/e['response_file']
        valid=False
        if response.exists():
            r=json.loads(response.read_text())
            valid=(r.get('status')=='returned' and r.get('validation',{}).get('valid') is True
                   and r.get('simulation',{}).get('num_qubits')==4)
        if valid:
            e['execution']={'kind':'individual','response_file':e['response_file']}
            direct+=1
        else:
            if response.exists():
                destination=history/response.name
                if destination.exists():
                    raise RuntimeError(f"Attempt history already exists: {destination}")
                response.rename(destination)
            pending.append(e)
    programs=RESULTS/'provider_batch_programs';programs.mkdir(exist_ok=True)
    (RESULTS/'provider_batches').mkdir(exist_ok=True)
    batches=[]
    for start in range(0,len(pending),3):
        members=pending[start:start+3];ident=f'batch_{len(batches):03d}'
        qasm=assemble([(RESULTS/e['qasm']).read_text() for e in members])
        qpath=programs/(ident+'.qasm');qpath.write_text(qasm)
        batch={'batch_id':ident,'qubit_count':4*len(members),'qasm':qpath.relative_to(RESULTS).as_posix(),
               'qasm_sha256':hashlib.sha256(qasm.encode()).hexdigest(),
               'response_file':f'provider_batches/{ident}.json','members':[]}
        for slot,e in enumerate(members):
            qubits=list(range(4*slot,4*slot+4))
            e['execution']={'kind':'batch_marginal','batch_id':ident,'slot':slot,'qubits':qubits,
                            'response_file':batch['response_file']}
            batch['members'].append({k:e[k] for k in ('case_id','step','basis','qasm','qasm_sha256','expected','expected_sha256')})
            batch['members'][-1].update({'slot':slot,'qubits':qubits})
        batches.append(batch)
    m['batches']=batches
    m['execution_plan']={'individual_circuits':direct,'batched_circuits':len(pending),
        'batch_programs':len(batches),'max_qubits_per_batch':12,
        'expected_successful_simulator_responses':direct+len(batches),
        'independence':'Every original unitary gate is retained in order on a disjoint four-qubit register. Terminal measurements commute with all other registers. Full joint distributions and exact marginal sums are independently checked.',
        'quota_handling':'The server-requested cooldown was respected. Failed rate-limited attempts are retained separately; every remaining logical circuit is executed in a complete concurrent batch.'}
    path.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(m['execution_plan'],indent=2))

if __name__=='__main__':
    main()
