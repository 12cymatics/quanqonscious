import copy
import hashlib
import importlib.util
import json
import pathlib
import tempfile

BASE = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('readiness_under_review', BASE / 'validate_readiness.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as folder:
    root = pathlib.Path(folder)
    evidence = root / 'unrelated.txt'
    evidence.write_text('Synthetic evidence: no foundation, physics, proof or runtime was assessed.\n')
    ref = {'path': evidence.name, 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(), 'subject_revision': 'synthetic'}
    record = json.loads((BASE / 'readiness.template.json').read_text())
    record.update(kind='project_record', task_state='complete', domain='cymatica', mathematical_claims=True, subject_revision='synthetic', completion_criterion='Synthetic consistency-only reproduction')
    for gate in record['gates']:
        gate.update(applicable=True, outcome='passed', evidence=[copy.deepcopy(ref)])
    result = m.validate(record, root, True)
    print(json.dumps({'case': 'Cymatica readiness without foundation records or typed scientific evidence', 'result': result, 'interpretation': 'Consistency accepts opaque evidence. This is not scientific certification and result explicitly says so.'}, indent=2))
