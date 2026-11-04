import argparse
import copy
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
bundle = parser.parse_args().bundle
spec = importlib.util.spec_from_file_location('validator', bundle / 'validate_contracts.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
policy = json.loads((bundle / 'policy.yaml').read_text())
memory = json.loads((bundle / 'project-memory.template.yaml').read_text())
results = {}
def accepts(record):
    try:
        return bool(validator.validate(policy, record))
    except (ValueError, KeyError, TypeError):
        return False
record = copy.deepcopy(memory)
record.update(task_state='complete', completion_criterion='Synthetic completion')
record['checks'] = [{'id': 'unrelated', 'scope': 'synthetic', 'outcome': 'passed', 'evidence': 'nonexistent.txt'}]
results['complete_without_required_gate_coverage'] = accepts(record)
record['active_operations'] = [{'state': 'uncertain', 'operation': 'publish'}]
results['complete_with_uncertain_operation'] = accepts(record)
record = copy.deepcopy(memory)
record['ownership'] = {'integrator': 'a', 'assignments': [{'owner': 'a', 'outcome': 'one', 'write_paths': ['src']}, {'owner': 'b', 'outcome': 'two', 'write_paths': ['src/main.py']}]}
results['overlapping_ancestor_ownership'] = accepts(record)
print(json.dumps(results, indent=2))
