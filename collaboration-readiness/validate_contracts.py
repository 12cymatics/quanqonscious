import argparse
import copy
import datetime
import json
import pathlib
import re
import sys

REQUIRED_GATES = {
    'source_and_input_provenance', 'foundation_traceability',
    'mathematical_integrity', 'targeted_regressions',
    'existing_aggregate_checks', 'actual_runtime_validation',
    'exact_unpacked_deliverable', 'authorization_before_publication',
    'remote_revision_and_ci'
}
CHECK_STATES = {'passed', 'failed', 'blocked', 'not_run'}
TASK_STATES = {'not_started', 'in_progress', 'blocked', 'complete', 'cancelled'}
EVIDENCE_CLASSES = {'observed', 'reported', 'inferred', 'proposed'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def timestamp(value, field):
    if value is None:
        return
    require(isinstance(value, str), field + ' must be a string or null')
    parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, field + ' requires a timezone')

def validate(policy, memory):
    require(policy.get('schema_version') == 1, 'Unsupported policy schema')
    require(memory.get('schema_version') == 1, 'Unsupported memory schema')
    require(policy.get('status') == 'draft', 'This validator supports the draft contract only')
    activation = policy['activation']
    for key in ['automatically_enforced', 'installed_in_repository', 'external_settings_changed']:
        require(activation.get(key) is False, 'Draft activation flag must be false: ' + key)
    foundation = policy['foundation']
    require(foundation.get('require_traceable_formalization') is True, 'Formalization must remain required')
    require(foundation.get('historical_attribution_is_proof') is False, 'Attribution cannot be treated as proof')
    require(foundation.get('silent_model_substitution_allowed') is False, 'Silent substitution must remain prohibited')
    require(set(foundation['companion_contracts']) == {'MATHEMATICAL_FOUNDATION.md', 'INTEGRITY_RULES.md'}, 'Foundation references differ')
    require(set(foundation['evidence_classes']) == EVIDENCE_CLASSES, 'Evidence classes differ')
    require(set(policy['outcomes']['check_states']) == CHECK_STATES, 'Check outcome set differs')
    require(set(policy['outcomes']['task_states']) == TASK_STATES, 'Task state set differs')
    require(policy['outcomes']['uncertain_operations_require_reconciliation'] is True, 'Uncertain operations need reconciliation')
    gate_ids = [gate['id'] for gate in policy['gates']]
    require(len(gate_ids) == len(set(gate_ids)), 'Duplicate gate IDs')
    require(set(gate_ids) == REQUIRED_GATES, 'Required gates are missing or unknown')
    require(all(gate.get('required') is True for gate in policy['gates']), 'Draft gates cannot silently be disabled')
    bindings = policy['project_command_bindings']
    require(bindings.get('configured') is False and bindings.get('commands') == [], 'Draft cannot claim bound repository commands')
    release = policy['release']
    for key in ['permitted_by_this_file', 'mock_execution_equals_runtime_validation', 'source_overlay_equals_runnable_package']:
        require(release.get(key) is False, 'Unsafe release flag: ' + key)
    for key in ['requires_applicable_gates_passed', 'requires_no_unresolved_blocking_failure', 'applicability_exceptions_require_reason_and_review']:
        require(release.get(key) is True, 'Required release guard absent: ' + key)
    require(release.get('unsupported_gate_handling') == 'report_blocked', 'Unsupported gates must be reported blocked')
    require(memory.get('status') in {'template', 'project_record'}, 'Unknown memory record status')
    require(memory.get('task_state') in TASK_STATES, 'Unknown task state')
    require(memory.get('task_state') != 'complete', 'Draft checker cannot attest completion; use validate_readiness.py')
    require(memory['source']['dirty_state'] in {'unknown', 'clean', 'dirty'}, 'Unknown dirty state')
    timestamp(memory.get('updated_at'), 'updated_at')
    timestamp(memory['source'].get('inspected_at'), 'source.inspected_at')
    digest = memory['candidate_build'].get('sha256')
    require(digest is None or isinstance(digest, str) and re.fullmatch(r'[a-f0-9]{64}', digest), 'Invalid candidate SHA-256')
    for field in ['inputs', 'foundation_records', 'claims', 'changes', 'checks', 'active_operations', 'blockers', 'approvals']:
        require(isinstance(memory[field], list), field + ' must be a list')
    for claim in memory['claims']:
        require(claim.get('kind') in EVIDENCE_CLASSES, 'Unknown claim evidence class')
        require(bool(claim.get('text')) and bool(claim.get('source')), 'Claims need text and source')
        require(claim.get('time_basis') in {'event', 'observation', 'document_metadata', 'unknown'}, 'Unknown claim time basis')
        timestamp(claim.get('timestamp'), 'claim.timestamp')
        if claim['kind'] == 'inferred':
            require(bool(claim.get('reasoning')), 'Inferred claims need reasoning')
    checks = memory['checks']
    ids = [check['id'] for check in checks]
    require(len(ids) == len(set(ids)), 'Duplicate check IDs')
    for check in checks:
        require(check.get('outcome') in CHECK_STATES, 'Unknown check outcome')
        require(bool(check.get('scope')), 'Check scope is required')
        if check['outcome'] in {'passed', 'failed'}:
            require(bool(check.get('evidence')), 'Observed outcomes need evidence')
        else:
            require(bool(check.get('reason')), 'Blocked or unrun checks need a reason')
    assignments = memory['ownership']['assignments']
    require(isinstance(assignments, list), 'Assignments must be a list')
    owned = {}
    for assignment in assignments:
        require(bool(assignment.get('owner')) and bool(assignment.get('outcome')), 'Assignment owner and outcome are required')
        require(isinstance(assignment.get('write_paths'), list), 'Assignment write_paths must be a list')
        for raw in assignment['write_paths']:
            require(isinstance(raw, str) and raw.strip(), 'Ownership path must be a nonempty string')
            require('\\' not in raw and not any(c in raw for c in '*?['), 'Ownership paths must be literal POSIX paths')
            path = pathlib.PurePosixPath(raw)
            require(not path.is_absolute() and '..' not in path.parts and path.parts, 'Ownership path must remain within root')
            require(all(not (path == prior or path in prior.parents or prior in path.parents) for prior in owned), 'Overlapping write ownership: ' + raw)
            owned[path] = assignment['owner']
    if assignments:
        require(bool(memory['ownership']['integrator']), 'Parallel assignments need an integrator')
    return True

def self_test(policy, memory):
    validate(policy, memory)
    cases = []
    def case(name, edit):
        p, m = copy.deepcopy(policy), copy.deepcopy(memory)
        edit(p, m)
        try:
            validate(p, m)
        except (ValueError, KeyError, TypeError):
            cases.append(name)
            return
        raise AssertionError('Invalid case was accepted: ' + name)
    case('automatic enforcement claim', lambda p, m: p['activation'].update(automatically_enforced=True))
    case('attribution as proof', lambda p, m: p['foundation'].update(historical_attribution_is_proof=True))
    case('silent model substitution', lambda p, m: p['foundation'].update(silent_model_substitution_allowed=True))
    case('duplicate gate', lambda p, m: p['gates'].append(copy.deepcopy(p['gates'][0])))
    case('disabled gate', lambda p, m: p['gates'][0].update(required=False))
    case('invented command activation', lambda p, m: p['project_command_bindings'].update(configured=True))
    case('mock as runtime', lambda p, m: p['release'].update(mock_execution_equals_runtime_validation=True))
    case('unknown check result', lambda p, m: m['checks'].append({'id': 'x', 'scope': 'synthetic', 'outcome': 'green'}))
    case('missing pass evidence', lambda p, m: m['checks'].append({'id': 'x', 'scope': 'synthetic', 'outcome': 'passed'}))
    case('missing blocked reason', lambda p, m: m['checks'].append({'id': 'x', 'scope': 'synthetic', 'outcome': 'blocked'}))
    case('unsupported inference', lambda p, m: m['claims'].append({'kind': 'inferred', 'text': 'synthetic', 'source': 'synthetic', 'time_basis': 'unknown'}))
    case('timestamp without timezone', lambda p, m: m.update(updated_at='2026-10-23T02:00:00'))
    case('invalid digest', lambda p, m: m['candidate_build'].update(sha256='unknown'))
    case('premature completion', lambda p, m: m.update(task_state='complete'))
    case('missing integration owner', lambda p, m: m['ownership']['assignments'].append({'owner': 'synthetic', 'outcome': 'synthetic', 'write_paths': ['file']}))
    case('duplicate write ownership', lambda p, m: m['ownership'].update(integrator='synthetic', assignments=[{'owner': 'a', 'outcome': 'synthetic', 'write_paths': ['same']}, {'owner': 'b', 'outcome': 'synthetic', 'write_paths': ['same']}]))
    return cases

def main():
    parser = argparse.ArgumentParser()
    here = pathlib.Path(__file__).resolve().parent
    parser.add_argument('--policy', type=pathlib.Path, default=here / 'policy.yaml')
    parser.add_argument('--memory', type=pathlib.Path, default=here / 'project-memory.template.yaml')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    memory = json.loads(args.memory.read_text())
    validate(policy, memory)
    rejected = self_test(policy, memory) if args.self_test else []
    print(json.dumps({'contract_consistency': 'passed', 'negative_cases_rejected': rejected, 'scientific_or_runtime_validation': 'not_performed', 'workflow_activation': 'not_performed'}, indent=2))

if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
