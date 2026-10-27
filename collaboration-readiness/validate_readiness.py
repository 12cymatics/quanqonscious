import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import uuid

from validation_identity import digest_bytes, read_json, relative_path, require, validate_ownership, verify_signed_contract, verify_source_manifest

GATES = {'source_and_input_provenance', 'foundation_traceability', 'mathematical_integrity', 'targeted_regressions', 'existing_aggregate_checks', 'actual_runtime_validation', 'exact_unpacked_deliverable', 'authorization_before_publication', 'remote_revision_and_ci'}
PROFILE = ('domain', 'mathematical_claims', 'deliverable_kind', 'completion_criterion')


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + ' has missing or unknown fields')


def identifiers(values, label, allow_empty=False):
    require(isinstance(values, list) and (allow_empty or bool(values)), label + ' must be a list')
    require(all(nonempty(value) for value in values), label + ' contains an invalid identifier')
    require(len(set(values)) == len(values), label + ' contains duplicate identifiers')
    return set(values)


def profile_gates(profile):
    require(profile.get('domain') in {'general', 'cymatica', 'eternal', 'sites', 'solver'}, 'Unknown domain')
    require(type(profile.get('mathematical_claims')) is bool, 'mathematical_claims must be a boolean')
    require(profile.get('deliverable_kind') in {'documentation', 'proof', 'software', 'package', 'deployment'}, 'Unknown deliverable kind')
    require(nonempty(profile.get('completion_criterion')), 'Completion criterion required')
    required = {'source_and_input_provenance', 'foundation_traceability', 'existing_aggregate_checks'}
    if profile['mathematical_claims'] or profile['domain'] in {'cymatica', 'solver'} or profile['deliverable_kind'] == 'proof':
        required.add('mathematical_integrity')
    if profile['deliverable_kind'] in {'software', 'package', 'deployment'}:
        required |= {'targeted_regressions', 'actual_runtime_validation'}
    if profile['deliverable_kind'] == 'package':
        required.add('exact_unpacked_deliverable')
    if profile['deliverable_kind'] == 'deployment':
        required |= {'authorization_before_publication', 'remote_revision_and_ci'}
    return required


def load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root):
    require(all(value is not None for value in (contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)), 'Signed contract, independently pinned public key and contract digests, and source root are required')
    payload, contract_hash = verify_signed_contract(contract_path, public_key_path, public_key_sha256)
    require(contract_hash == expected_contract_sha256, 'Contract differs from independently pinned contract digest')
    exact_keys(payload, {'schema_version', 'task_id', 'user_request', 'user_request_sha256', 'generation', *PROFILE, 'source_root', 'source_manifest', 'ownership', 'equations', 'checks', 'gates'}, 'Contract')
    require(payload['schema_version'] == 1, 'Unsupported signed contract schema')
    require(nonempty(payload['task_id']) and nonempty(payload['user_request']), 'Task and actual user request are required')
    require(digest_bytes(payload['user_request'].encode()) == payload['user_request_sha256'], 'User request digest mismatch')
    require(type(payload['generation']) is int and payload['generation'] > 0, 'Positive contract generation required')
    source = pathlib.Path(source_root).resolve(strict=True)
    require(source.is_dir(), 'Source root must be a directory')
    require(isinstance(payload['source_root'], str) and pathlib.Path(payload['source_root']).is_absolute(), 'Signed source root must be absolute')
    require(str(source) == payload['source_root'], 'Source root differs from signed root')
    revision = 'sha256:' + verify_source_manifest(payload['source_manifest'], source)
    required = profile_gates(payload)
    ownership = payload['ownership']
    exact_keys(ownership, {'integrator', 'assignments'}, 'Contract ownership')
    require(nonempty(ownership['integrator']), 'Named integration owner required')
    validate_ownership(ownership['assignments'], source)
    files = {item['path']: item['sha256'] for item in payload['source_manifest']['files']}
    require(isinstance(payload['checks'], list) and payload['checks'], 'Executable acceptance checks required')
    checks = {}
    for check in payload['checks']:
        exact_keys(check, {'id', 'gate_id', 'argv', 'executable_sha256', 'entrypoint', 'assertions', 'equation_ids', 'timeout_seconds'}, 'Acceptance check')
        require(nonempty(check['id']) and check['id'] not in checks, 'Duplicate or missing acceptance check id')
        require(check['gate_id'] in GATES, 'Unknown acceptance check gate')
        identifiers(check['assertions'], 'Required assertions')
        identifiers(check['equation_ids'], 'Equation references', True)
        require(type(check['timeout_seconds']) in {int, float} and check['timeout_seconds'] > 0, 'Positive explicit check timeout required')
        argv = check['argv']
        require(isinstance(argv, list) and len(argv) >= 2 and all(nonempty(arg) for arg in argv), 'Check requires an executable and source entrypoint')
        executable = pathlib.Path(argv[0])
        require(executable.is_absolute() and executable.is_file(), 'Check executable must be an existing absolute path')
        require(hashlib.sha256(executable.read_bytes()).hexdigest() == check['executable_sha256'], 'Check executable digest mismatch')
        entry = str(relative_path(check['entrypoint']))
        require(entry in files and entry in argv[1:], 'Check entrypoint must be a bound source file and literal command argument')
        checks[check['id']] = check
    require(isinstance(payload['gates'], list), 'Contract gates must be a list')
    gates = {}
    selected_checks = set()
    for gate in payload['gates']:
        exact_keys(gate, {'id', 'applicable', 'check_ids', 'reason'}, 'Contract gate')
        require(gate['id'] in GATES and gate['id'] not in gates, 'Unknown or duplicate contract gate')
        require(type(gate['applicable']) is bool, 'Contract applicability must be resolved')
        refs = identifiers(gate['check_ids'], 'Gate check identifiers', not gate['applicable'])
        if gate['applicable']:
            require(gate['reason'] is None, 'Applicable gate cannot have an exclusion reason')
            require(refs <= set(checks), 'Gate references an unknown check')
            require(all(checks[name]['gate_id'] == gate['id'] for name in refs), 'Acceptance check bound to a different gate')
            require(not (selected_checks & refs), 'Acceptance check reused across gates')
            selected_checks |= refs
        else:
            require(gate['id'] not in required, 'Required profile gate cannot be excluded')
            require(not refs and nonempty(gate['reason']), 'Excluded gate requires a signed reason and no checks')
        gates[gate['id']] = gate
    require(set(gates) == GATES, 'Exactly one signed decision per required gate is required')
    require(selected_checks == set(checks), 'Unused or unexecuted acceptance check in contract')
    equations = payload['equations']
    require(isinstance(equations, list), 'Equation inventory must be a list')
    if 'mathematical_integrity' in required:
        require(bool(equations), 'Full equation inventory required for mathematical acceptance')
    eqs = {}
    for equation in equations:
        exact_keys(equation, {'id', 'statement', 'implementation_paths', 'check_ids'}, 'Equation')
        require(nonempty(equation['id']) and equation['id'] not in eqs, 'Duplicate or missing equation id')
        exact_keys(equation['statement'], {'path', 'sha256'}, 'Equation statement')
        statement = str(relative_path(equation['statement']['path']))
        require(statement in files and files[statement] == equation['statement']['sha256'], 'Equation statement is not bound to actual source bytes')
        paths = identifiers(equation['implementation_paths'], 'Equation implementation paths')
        require(all(str(relative_path(path)) in files for path in paths), 'Equation implementation is not included in source manifest')
        refs = identifiers(equation['check_ids'], 'Equation check identifiers')
        require(refs <= set(checks), 'Equation references an unknown check')
        require(any(checks[name]['gate_id'] == 'mathematical_integrity' for name in refs), 'Equation has no mathematical integrity check')
        require(all(equation['id'] in checks[name]['equation_ids'] for name in refs), 'Equation/check linkage is not reciprocal')
        eqs[equation['id']] = equation
    for check in checks.values():
        for name in check['equation_ids']:
            require(name in eqs and check['id'] in eqs[name]['check_ids'], 'Check references an unbound equation')
    return payload, contract_hash, revision, source, gates, checks


def check_result(value, check):
    exact_keys(value, {'schema_version', 'check_id', 'outcome', 'assertions'}, 'Executed check result')
    require(value['schema_version'] == 1 and value['check_id'] == check['id'], 'Check result identity mismatch')
    require(value['outcome'] == 'passed', 'Acceptance check reported failure: ' + check['id'])
    assertions = value['assertions']
    require(isinstance(assertions, list), 'Check assertions must be a list')
    seen = set()
    for assertion in assertions:
        exact_keys(assertion, {'id', 'passed'}, 'Check assertion')
        require(nonempty(assertion['id']) and assertion['id'] not in seen, 'Duplicate or missing assertion id')
        require(assertion['passed'] is True, 'Required assertion failed: ' + assertion['id'])
        seen.add(assertion['id'])
    require(seen == set(check['assertions']), 'Required assertions are missing or changed: ' + check['id'])
    return value


def execute(payload, source, checks):
    results = {}
    environment = {'PATH': os.defpath, 'LC_ALL': 'C', 'TZ': 'UTC', 'PYTHONDONTWRITEBYTECODE': '1'}
    for check in checks.values():
        verify_source_manifest(payload['source_manifest'], source)
        executable = pathlib.Path(check['argv'][0])
        require(hashlib.sha256(executable.read_bytes()).hexdigest() == check['executable_sha256'], 'Check executable changed before execution')
        try:
            process = subprocess.run(check['argv'], cwd=source, env=environment, capture_output=True, timeout=check['timeout_seconds'])
        except subprocess.TimeoutExpired as error:
            raise ValueError('Acceptance check timed out: ' + check['id']) from error
        require(process.returncode == 0, 'Acceptance check failed: ' + check['id'] + ' exit=' + str(process.returncode) + ' stderr=' + process.stderr.decode(errors='replace'))
        try:
            value = json.loads(process.stdout, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError('Acceptance check did not emit valid structured results: ' + check['id']) from error
        results[check['id']] = check_result(value, check)
        require(hashlib.sha256(executable.read_bytes()).hexdigest() == check['executable_sha256'], 'Check executable changed during execution')
        verify_source_manifest(payload['source_manifest'], source)
    return results


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def report_for(gate, revision, contract_hash, results):
    return {'schema_version': 1, 'gate_id': gate['id'], 'subject_revision': revision, 'contract_sha256': contract_hash, 'outcome': 'passed', 'checks': [results[name] for name in gate['check_ids']]}


def local_evidence(item, root, revision, contract_hash, gate, checks):
    exact_keys(item, {'path', 'sha256', 'subject_revision'}, 'Evidence reference')
    path = (root / relative_path(item['path'])).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'Evidence escapes root or is not a file')
    content = path.read_bytes()
    require(content and digest_bytes(content) == item['sha256'], 'Evidence is empty or digest differs')
    require(item['subject_revision'] == revision, 'Evidence subject revision mismatch')
    value = json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)
    exact_keys(value, {'schema_version', 'gate_id', 'subject_revision', 'contract_sha256', 'outcome', 'checks'}, 'Gate evidence')
    require(value['schema_version'] == 1 and value['gate_id'] == gate['id'], 'Evidence gate identity mismatch')
    require(value['subject_revision'] == revision and value['contract_sha256'] == contract_hash, 'Evidence is not bound to current source and contract')
    require(value['outcome'] == 'passed', 'Evidence explicitly reports a nonpassing result')
    require(isinstance(value['checks'], list), 'Evidence checks must be a list')
    seen = set()
    for result in value['checks']:
        require(isinstance(result, dict) and result.get('check_id') in gate['check_ids'], 'Evidence includes an unrelated check')
        name = result['check_id']
        require(name not in seen, 'Duplicate evidence check')
        check_result(result, checks[name])
        seen.add(name)
    require(seen == set(gate['check_ids']), 'Evidence omits required checks')
    return value


def validate_record(record, root, payload, contract_hash, revision, source, gates, checks):
    exact_keys(record, {'schema_version', 'kind', 'task_state', *PROFILE, 'subject_revision', 'contract_sha256', 'contract_generation', 'blockers', 'active_operations', 'ownership', 'gates'}, 'Readiness record')
    require(record['schema_version'] == 3 and record['kind'] == 'project_record', 'Expected schema 3 project record')
    require(record['task_state'] in {'in_progress', 'complete'}, 'Readiness requires in-progress or complete state')
    require(record['subject_revision'] == revision, 'Record subject revision differs from actual source manifest')
    require(record['contract_sha256'] == contract_hash and type(record['contract_generation']) is int and record['contract_generation'] == payload['generation'], 'Record contract identity or generation mismatch')
    require(all(record[key] == payload[key] and type(record[key]) is type(payload[key]) for key in PROFILE), 'Record scope differs from signed user scope')
    require(record['blockers'] == [] and record['active_operations'] == [], 'Unresolved blockers or operations')
    require(record['ownership'] == payload['ownership'], 'Record ownership differs from signed ownership')
    validate_ownership(record['ownership']['assignments'], source)
    require(isinstance(record['gates'], list), 'Record gates must be a list')
    seen = set()
    reports = {}
    for gate in record['gates']:
        exact_keys(gate, {'id', 'applicable', 'outcome', 'reason', 'evidence'}, 'Record gate')
        require(gate['id'] in gates and gate['id'] not in seen, 'Unknown or duplicate record gate')
        seen.add(gate['id'])
        signed = gates[gate['id']]
        require(type(gate['applicable']) is bool and gate['applicable'] == signed['applicable'], 'Gate applicability differs from signed scope')
        require(gate['reason'] == signed['reason'], 'Gate exclusion reason differs from signed scope')
        if gate['applicable']:
            require(gate['outcome'] == 'passed', 'Applicable gate has not passed')
            require(isinstance(gate['evidence'], list) and gate['evidence'], 'Observed outcome requires evidence')
            reports[gate['id']] = [local_evidence(item, root, revision, contract_hash, signed, checks) for item in gate['evidence']]
        else:
            require(gate['outcome'] == 'not_applicable' and gate['evidence'] == [], 'Excluded gate cannot claim execution')
    require(seen == GATES, 'Exactly one record decision per gate is required')
    return reports


def accepted(payload, contract_hash, revision, results):
    return {'record_consistency': 'passed', 'readiness_attestation': 'verified', 'contract_sha256': contract_hash, 'contract_generation': payload['generation'], 'subject_revision': revision, 'task_id': payload['task_id'], 'executed_checks': list(results), 'checked_equation_ids': [item['id'] for item in payload['equations']], 'publication_performed': False}


def validate(record, root, readiness=False, ownership_root=None, *, contract_path=None, public_key_path=None, public_key_sha256=None, expected_contract_sha256=None, source_root=None, execute_checks=False):
    require(isinstance(record, dict), 'Record must be an object')
    require(record.get('schema_version') == 3, 'Legacy schema cannot attest readiness; signed schema 3 required')
    if record.get('kind') == 'template':
        require(not readiness and record.get('task_state') == 'not_started', 'Template cannot claim readiness or completion')
        require(record.get('subject_revision') is None and record.get('contract_sha256') is None and record.get('contract_generation') is None, 'Template cannot claim source or contract identity')
        require(isinstance(record.get('gates'), list), 'Template gates required')
        require(len(record['gates']) == len(GATES) and {g.get('id') for g in record['gates']} == GATES, 'Template gate coverage mismatch')
        require(all(g.get('applicable') is None and g.get('outcome') == 'not_run' and g.get('evidence') == [] for g in record['gates']), 'Template cannot claim a passed or excluded gate')
        return {'record_consistency': 'passed', 'readiness_attestation': 'not_evaluated'}
    require(readiness or record.get('task_state') == 'complete', 'Project acceptance requires an explicit readiness operation')
    require(execute_checks, 'Readiness requires execution of the signed acceptance checks')
    root = pathlib.Path(root).resolve(strict=True)
    require(root.is_dir(), 'Evidence root must be a directory')
    payload, contract_hash, revision, source, gates, checks = load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)
    require(ownership_root is None or pathlib.Path(ownership_root).resolve(strict=True) == source, 'Ownership root differs from signed source root')
    reports = validate_record(record, root, payload, contract_hash, revision, source, gates, checks)
    results = execute(payload, source, checks)
    for name, evidence in reports.items():
        expected = report_for(gates[name], revision, contract_hash, results)
        require(all(canonical(item) == canonical(expected) for item in evidence), 'Recorded evidence differs from current execution: ' + name)
    verify_source_manifest(payload['source_manifest'], source)
    return accepted(payload, contract_hash, revision, results)


def run_checks(*, contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root, evidence_root):
    payload, contract_hash, revision, source, gates, checks = load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)
    root = pathlib.Path(evidence_root).resolve()
    require(not root.is_relative_to(source), 'Evidence output must be outside the bound source root')
    root.mkdir(parents=True, exist_ok=True)
    results = execute(payload, source, checks)
    record = {'schema_version': 3, 'kind': 'project_record', 'task_state': 'complete', **{key: payload[key] for key in PROFILE}, 'subject_revision': revision, 'contract_sha256': contract_hash, 'contract_generation': payload['generation'], 'blockers': [], 'active_operations': [], 'ownership': payload['ownership'], 'gates': []}
    run = root / ('run-' + uuid.uuid4().hex)
    run.mkdir()
    for gate in gates.values():
        references = []
        if gate['applicable']:
            path = run / (gate['id'] + '.json')
            data = json.dumps(report_for(gate, revision, contract_hash, results), indent=2, allow_nan=False).encode() + b'\n'
            with path.open('xb') as handle:
                handle.write(data)
            references.append({'path': str(path.relative_to(root)), 'sha256': digest_bytes(data), 'subject_revision': revision})
        record['gates'].append({'id': gate['id'], 'applicable': gate['applicable'], 'outcome': 'passed' if gate['applicable'] else 'not_applicable', 'reason': gate['reason'], 'evidence': references})
    verify_source_manifest(payload['source_manifest'], source)
    validate_record(record, root, payload, contract_hash, revision, source, gates, checks)
    return record, accepted(payload, contract_hash, revision, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=pathlib.Path)
    parser.add_argument('--evidence-root', required=True, type=pathlib.Path)
    parser.add_argument('--ownership-root', type=pathlib.Path)
    parser.add_argument('--source-root', type=pathlib.Path)
    parser.add_argument('--contract', type=pathlib.Path)
    parser.add_argument('--trusted-public-key', type=pathlib.Path)
    parser.add_argument('--trusted-key-sha256')
    parser.add_argument('--contract-sha256')
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--execute-checks', action='store_true')
    parser.add_argument('--run-checks', action='store_true')
    parser.add_argument('--output-record', type=pathlib.Path)
    args = parser.parse_args()
    trust = {'contract_path': args.contract, 'public_key_path': args.trusted_public_key, 'public_key_sha256': args.trusted_key_sha256, 'expected_contract_sha256': args.contract_sha256, 'source_root': args.source_root}
    if args.run_checks:
        require(args.record is None and args.output_record is not None, '--run-checks requires --output-record and no submitted --record')
        require(not args.output_record.exists(), 'Output record exists; preserve prior evidence')
        if args.source_root is not None:
            require(not args.output_record.resolve().is_relative_to(args.source_root.resolve()), 'Output record must be outside source root')
        record, result = run_checks(evidence_root=args.evidence_root, **trust)
        with args.output_record.open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
    else:
        require(args.record is not None and args.output_record is None, '--record required for validation')
        result = validate(read_json(args.record), args.evidence_root, args.readiness, args.ownership_root, execute_checks=args.execute_checks, **trust)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
