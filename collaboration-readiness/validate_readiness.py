import argparse
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid

from validation_identity import digest_bytes, read_json, relative_path, require, source_state, validate_ownership, verify_signed_contract, verify_source_manifest

GATES = {'source_and_input_provenance', 'foundation_traceability', 'mathematical_integrity', 'targeted_regressions', 'existing_aggregate_checks', 'actual_runtime_validation', 'exact_unpacked_deliverable', 'authorization_before_publication', 'remote_revision_and_ci'}
PROFILE = ('domain', 'mathematical_claims', 'deliverable_kind', 'completion_criterion')
REQUIREMENT_KINDS = {'equation', 'term', 'coupling', 'mode', 'domain', 'initial_condition', 'boundary_condition', 'arithmetic'}
RECORD_KEYS = {'schema_version', 'kind', 'task_state', *PROFILE, 'subject_revision', 'contract_sha256', 'contract_generation', 'blockers', 'active_operations', 'ownership', 'gates'}


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
    require(isinstance(profile.get('domain'), str) and profile['domain'] in {'general', 'cymatica', 'eternal', 'sites', 'solver'}, 'Unknown domain')
    require(type(profile.get('mathematical_claims')) is bool, 'mathematical_claims must be a boolean')
    require(isinstance(profile.get('deliverable_kind'), str) and profile['deliverable_kind'] in {'documentation', 'proof', 'software', 'package', 'deployment'}, 'Unknown deliverable kind')
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
    exact_keys(payload, {'schema_version', 'task_id', 'user_request', 'user_request_sha256', 'generation', *PROFILE, 'source_root', 'source_manifest', 'ownership', 'equations', 'foundation_register', 'geometry_index', 'checks', 'gates'}, 'Contract')
    require(type(payload['schema_version']) is int and payload['schema_version'] == 2, 'Unsupported signed contract schema')
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
    require(bool(files), 'Source manifest cannot be empty for project acceptance')
    require(isinstance(payload['checks'], list) and payload['checks'], 'Executable acceptance checks required')
    checks = {}
    for check in payload['checks']:
        exact_keys(check, {'id', 'gate_id', 'argv', 'executable_sha256', 'entrypoint', 'assertions', 'equation_ids', 'timeout_seconds'}, 'Acceptance check')
        require(nonempty(check['id']) and check['id'] not in checks, 'Duplicate or missing acceptance check id')
        require(isinstance(check['gate_id'], str) and check['gate_id'] in GATES, 'Unknown acceptance check gate')
        identifiers(check['assertions'], 'Required assertions')
        identifiers(check['equation_ids'], 'Equation references', True)
        require(type(check['timeout_seconds']) in {int, float} and math.isfinite(check['timeout_seconds']) and check['timeout_seconds'] > 0, 'Positive finite explicit check timeout required')
        argv = check['argv']
        require(isinstance(argv, list) and len(argv) >= 2 and all(nonempty(arg) for arg in argv), 'Check requires an executable and source entrypoint')
        executable = pathlib.Path(argv[0])
        require(executable.is_absolute() and executable.is_file(), 'Check executable must be an existing absolute path')
        require(hashlib.sha256(executable.read_bytes()).hexdigest() == check['executable_sha256'], 'Check executable digest mismatch')
        entry = str(relative_path(check['entrypoint']))
        require(entry in files and argv[1] == entry, 'Check entrypoint must be a bound source file and the first executable argument')
        checks[check['id']] = check
    require(isinstance(payload['gates'], list), 'Contract gates must be a list')
    gates = {}
    selected_checks = set()
    for gate in payload['gates']:
        exact_keys(gate, {'id', 'applicable', 'check_ids', 'reason'}, 'Contract gate')
        require(isinstance(gate['id'], str) and gate['id'] in GATES and gate['id'] not in gates, 'Unknown or duplicate contract gate')
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
    mathematical = gates['mathematical_integrity']['applicable']
    if mathematical:
        require(bool(equations), 'Full equation inventory required for mathematical acceptance')
    else:
        require(not equations, 'Equation inventory requires an applicable mathematical integrity gate')
    eqs = {}
    requirement_ids = set()
    for equation in equations:
        exact_keys(equation, {'id', 'statement', 'implementation_paths', 'check_ids', 'requirements', 'requirement_applicability'}, 'Equation')
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
        if payload['deliverable_kind'] in {'software', 'package', 'deployment'}:
            require(any(checks[name]['gate_id'] == 'actual_runtime_validation' for name in refs), 'Equation has no actual runtime check')
        applicability = equation['requirement_applicability']
        exact_keys(applicability, REQUIREMENT_KINDS, 'Equation requirement applicability')
        required_kinds = set()
        for kind, decision in applicability.items():
            exact_keys(decision, {'applicable', 'reason'}, 'Requirement category decision')
            require(type(decision['applicable']) is bool, 'Requirement category applicability must be resolved')
            if decision['applicable']:
                require(decision['reason'] is None, 'Applicable requirement category cannot have an exclusion reason')
                required_kinds.add(kind)
            else:
                require(kind not in {'equation', 'domain', 'arithmetic'} and nonempty(decision['reason']), 'Required equation category cannot be excluded without a valid signed scope decision')
        require(isinstance(equation['requirements'], list) and equation['requirements'], 'Equation requirements must be a nonempty list')
        seen_kinds = set()
        requirement_checks = set()
        requirement_paths = set()
        for obligation in equation['requirements']:
            exact_keys(obligation, {'id', 'kind', 'statement', 'implementation_paths', 'check_assertions'}, 'Equation requirement')
            require(nonempty(obligation['id']) and obligation['id'] not in requirement_ids, 'Duplicate or missing equation requirement id')
            requirement_ids.add(obligation['id'])
            require(isinstance(obligation['kind'], str) and obligation['kind'] in required_kinds, 'Requirement belongs to an excluded or unknown category')
            require(nonempty(obligation['statement']), 'Every equation requirement needs its actual specified statement')
            seen_kinds.add(obligation['kind'])
            bound_paths = identifiers(obligation['implementation_paths'], 'Requirement implementation paths')
            require(bound_paths <= paths, 'Requirement implementation is outside its equation implementation')
            requirement_paths |= bound_paths
            require(isinstance(obligation['check_assertions'], list) and obligation['check_assertions'], 'Requirement needs executable assertion references')
            linked = set()
            for assertion in obligation['check_assertions']:
                exact_keys(assertion, {'check_id', 'assertion_id'}, 'Requirement assertion reference')
                require(nonempty(assertion['check_id']) and assertion['check_id'] in refs, 'Requirement references an unbound equation check')
                check = checks[assertion['check_id']]
                require(nonempty(assertion['assertion_id']) and assertion['assertion_id'] in check['assertions'], 'Requirement references an unknown executable assertion')
                pair = (assertion['check_id'], assertion['assertion_id'])
                require(pair not in linked, 'Duplicate requirement assertion reference')
                linked.add(pair)
                requirement_checks.add(assertion['check_id'])
            require(any(checks[name]['gate_id'] == 'mathematical_integrity' for name, assertion_id in linked), 'Requirement lacks a mathematical integrity assertion')
            if payload['deliverable_kind'] in {'software', 'package', 'deployment'}:
                require(any(checks[name]['gate_id'] == 'actual_runtime_validation' for name, assertion_id in linked), 'Requirement lacks an actual runtime assertion')
        require(seen_kinds == required_kinds, 'Applicable equation requirement category has no obligations')
        require(requirement_checks == refs and requirement_paths == paths, 'Equation implementation or checks lack requirement linkage')
        eqs[equation['id']] = equation
    for check in checks.values():
        for name in check['equation_ids']:
            require(name in eqs and check['id'] in eqs[name]['check_ids'], 'Check references an unbound equation')
    for field in ('foundation_register', 'geometry_index'):
        reference = payload[field]
        if not mathematical:
            require(reference is None, field + ' requires mathematical acceptance')
            continue
        exact_keys(reference, {'path', 'sha256'}, field)
        relative = str(relative_path(reference['path']))
        require(relative in files and files[relative] == reference['sha256'], field + ' is not bound to actual source bytes')
        content = read_json(source / relative)
        exact_keys(content, {'schema_version', 'kind', 'task_id', 'equations'}, field + ' contents')
        require(type(content['schema_version']) is int and content['schema_version'] == 1 and content['kind'] == 'project_record' and content['task_id'] == payload['task_id'], field + ' identity differs from signed task')
        expected = equations if field == 'foundation_register' else [{'id': eq['id'], 'statement_path': eq['statement']['path'], 'implementation_paths': eq['implementation_paths'], 'requirement_ids': [item['id'] for item in eq['requirements']], 'check_ids': eq['check_ids']} for eq in equations]
        require(canonical(content['equations']) == canonical(expected), field + ' differs from signed equation and requirement inventory')
    verify_source_manifest(payload['source_manifest'], source)
    return payload, contract_hash, revision, source, gates, checks


def check_result(value, check):
    exact_keys(value, {'schema_version', 'check_id', 'outcome', 'assertions'}, 'Executed check result')
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and value['check_id'] == check['id'], 'Check result identity mismatch')
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


def write_observation(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def execution_directory(root, source):
    root = pathlib.Path(root).resolve()
    require(not root.is_relative_to(source), 'Evidence output must be outside the bound source root')
    root.mkdir(parents=True, exist_ok=True)
    run = root / ('run-' + uuid.uuid4().hex)
    run.mkdir(mode=0o700)
    return root, run


def execute(payload, source, checks, run):
    results = {}
    environment = {'PATH': os.defpath, 'LC_ALL': 'C', 'TZ': 'UTC', 'PYTHONDONTWRITEBYTECODE': '1'}
    initial = source_state(source, payload['source_manifest'])
    for index, check in enumerate(checks.values()):
        verify_source_manifest(payload['source_manifest'], source)
        require(source_state(source, payload['source_manifest']) == initial, 'Source filesystem state changed between acceptance checks')
        executable = pathlib.Path(check['argv'][0])
        require(hashlib.sha256(executable.read_bytes()).hexdigest() == check['executable_sha256'], 'Check executable changed before execution')
        started = time.time_ns()
        observed = {'schema_version': 1, 'check_id': check['id'], 'argv': check['argv'], 'executable_sha256': check['executable_sha256'], 'source_manifest_sha256': digest_bytes(canonical(payload['source_manifest'])), 'started_at_unix_ns': started, 'finished_at_unix_ns': None, 'return_code': None, 'timed_out': False, 'background_processes': False, 'stdout_sha256': None, 'stderr_sha256': None, 'outcome': 'failed', 'failure': None}
        stdout, stderr = b'', b''
        process = None
        try:
            process = subprocess.Popen(check['argv'], cwd=source, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=check['timeout_seconds'])
            except subprocess.TimeoutExpired:
                observed['timed_out'] = True
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            observed['return_code'] = process.returncode
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                observed['background_processes'] = True
                os.killpg(process.pid, signal.SIGKILL)
            require(not observed['timed_out'], 'Acceptance check timed out: ' + check['id'])
            require(not observed['background_processes'], 'Acceptance check left background processes: ' + check['id'])
            require(process.returncode == 0, 'Acceptance check failed: ' + check['id'] + ' exit=' + str(process.returncode))
            try:
                value = json.loads(stdout.decode('utf-8'), object_pairs_hook=unique_object, parse_constant=reject_constant, parse_float=finite_float)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError('Acceptance check did not emit valid structured results: ' + check['id']) from error
            results[check['id']] = check_result(value, check)
            require(hashlib.sha256(executable.read_bytes()).hexdigest() == check['executable_sha256'], 'Check executable changed during execution')
            verify_source_manifest(payload['source_manifest'], source)
            require(source_state(source, payload['source_manifest']) == initial, 'Source filesystem state changed during execution: ' + check['id'])
            observed['outcome'] = 'passed'
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            observed['failure'] = str(error)
            raise ValueError(str(error) + '; execution evidence: ' + str(run)) from error
        except KeyboardInterrupt:
            observed['failure'] = 'Acceptance check cancelled: ' + check['id']
            observed['outcome'] = 'cancelled'
            raise
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                observed['return_code'] = process.returncode
            observed['finished_at_unix_ns'] = time.time_ns()
            observed['stdout_sha256'] = digest_bytes(stdout)
            observed['stderr_sha256'] = digest_bytes(stderr)
            for suffix, content in (('stdout', stdout), ('stderr', stderr)):
                with (run / (str(index) + '.' + suffix)).open('xb') as handle:
                    handle.write(content)
            write_observation(run / (str(index) + '.execution.json'), observed)
    return results


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def finite_float(value):
    parsed = float(value)
    require(math.isfinite(parsed), 'Non-finite JSON number: ' + value)
    return parsed


def report_for(gate, revision, contract_hash, results):
    return {'schema_version': 1, 'gate_id': gate['id'], 'subject_revision': revision, 'contract_sha256': contract_hash, 'outcome': 'passed', 'checks': [results[name] for name in gate['check_ids']]}


def local_evidence(item, root, revision, contract_hash, gate, checks):
    exact_keys(item, {'path', 'sha256', 'subject_revision'}, 'Evidence reference')
    path = (root / relative_path(item['path'])).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), 'Evidence escapes root or is not a file')
    content = path.read_bytes()
    require(content and digest_bytes(content) == item['sha256'], 'Evidence is empty or digest differs')
    require(item['subject_revision'] == revision, 'Evidence subject revision mismatch')
    value = json.loads(content.decode('utf-8'), object_pairs_hook=unique_object, parse_constant=reject_constant, parse_float=finite_float)
    exact_keys(value, {'schema_version', 'gate_id', 'subject_revision', 'contract_sha256', 'outcome', 'checks'}, 'Gate evidence')
    require(type(value['schema_version']) is int and value['schema_version'] == 1 and value['gate_id'] == gate['id'], 'Evidence gate identity mismatch')
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
    exact_keys(record, RECORD_KEYS, 'Readiness record')
    require(type(record['schema_version']) is int and record['schema_version'] == 3 and record['kind'] == 'project_record', 'Expected schema 3 project record')
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
        require(isinstance(gate['id'], str) and gate['id'] in gates and gate['id'] not in seen, 'Unknown or duplicate record gate')
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


def accepted(payload, contract_hash, revision, results, run):
    return {'record_consistency': 'passed', 'readiness_attestation': 'verified', 'contract_sha256': contract_hash, 'contract_generation': payload['generation'], 'subject_revision': revision, 'task_id': payload['task_id'], 'executed_checks': list(results), 'checked_equation_ids': [item['id'] for item in payload['equations']], 'checked_requirement_ids': [requirement['id'] for equation in payload['equations'] for requirement in equation['requirements']], 'execution_evidence': str(run), 'publication_performed': False}


def run_summary(run, payload, contract_hash, revision, outcome, failure=None):
    write_observation(run / 'run-summary.json', {'schema_version': 1, 'task_id': payload['task_id'], 'contract_sha256': contract_hash, 'contract_generation': payload['generation'], 'subject_revision': revision, 'outcome': outcome, 'failure': failure, 'finished_at_unix_ns': time.time_ns()})


def validate(record, root, readiness=False, ownership_root=None, *, contract_path=None, public_key_path=None, public_key_sha256=None, expected_contract_sha256=None, source_root=None, execute_checks=False):
    require(isinstance(record, dict), 'Record must be an object')
    require(type(record.get('schema_version')) is int and record['schema_version'] == 3, 'Legacy schema cannot attest readiness; signed schema 3 required')
    if record.get('kind') == 'template':
        exact_keys(record, RECORD_KEYS, 'Readiness template')
        require(not readiness and record.get('task_state') == 'not_started', 'Template cannot claim readiness or completion')
        require(record.get('subject_revision') is None and record.get('contract_sha256') is None and record.get('contract_generation') is None, 'Template cannot claim source or contract identity')
        require(record['ownership'] == {'integrator': None, 'assignments': []} and record['blockers'] == [] and record['active_operations'] == [], 'Template cannot claim assigned ownership or assessed project state')
        require(record['completion_criterion'] is None, 'Template cannot claim an established completion criterion')
        require(record['domain'] == 'general' and record['mathematical_claims'] is False and record['deliverable_kind'] == 'software', 'Template profile must remain explicitly unassigned defaults')
        require(isinstance(record.get('gates'), list), 'Template gates required')
        for gate in record['gates']:
            exact_keys(gate, {'id', 'applicable', 'outcome', 'reason', 'evidence'}, 'Template gate')
            require(nonempty(gate['id']) and nonempty(gate['reason']), 'Template gate requires an identifier and unresolved reason')
        require(len(record['gates']) == len(GATES) and {g['id'] for g in record['gates']} == GATES, 'Template gate coverage mismatch')
        require(all(g.get('applicable') is None and g.get('outcome') == 'not_run' and g.get('evidence') == [] for g in record['gates']), 'Template cannot claim a passed or excluded gate')
        return {'record_consistency': 'passed', 'readiness_attestation': 'not_evaluated'}
    require(readiness, 'Project acceptance requires an explicit readiness operation')
    require(execute_checks, 'Readiness requires execution of the signed acceptance checks')
    root = pathlib.Path(root).resolve(strict=True)
    require(root.is_dir(), 'Evidence root must be a directory')
    payload, contract_hash, revision, source, gates, checks = load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)
    require(ownership_root is None or pathlib.Path(ownership_root).resolve(strict=True) == source, 'Ownership root differs from signed source root')
    reports = validate_record(record, root, payload, contract_hash, revision, source, gates, checks)
    root, run = execution_directory(root, source)
    try:
        results = execute(payload, source, checks, run)
        for name, evidence in reports.items():
            expected = report_for(gates[name], revision, contract_hash, results)
            require(all(canonical(item) == canonical(expected) for item in evidence), 'Recorded evidence differs from current execution: ' + name)
        current = load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)
        require(current[1] == contract_hash and current[2] == revision, 'Contract or source identity changed during execution')
    except KeyboardInterrupt:
        run_summary(run, payload, contract_hash, revision, 'cancelled', 'Acceptance operation interrupted')
        raise
    except Exception as error:
        run_summary(run, payload, contract_hash, revision, 'failed', str(error))
        raise
    run_summary(run, payload, contract_hash, revision, 'passed')
    return accepted(payload, contract_hash, revision, results, run)


def run_checks(*, contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root, evidence_root):
    payload, contract_hash, revision, source, gates, checks = load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)
    root, run = execution_directory(evidence_root, source)
    try:
        results = execute(payload, source, checks, run)
        record = {'schema_version': 3, 'kind': 'project_record', 'task_state': 'complete', **{key: payload[key] for key in PROFILE}, 'subject_revision': revision, 'contract_sha256': contract_hash, 'contract_generation': payload['generation'], 'blockers': [], 'active_operations': [], 'ownership': payload['ownership'], 'gates': []}
        for gate in gates.values():
            references = []
            if gate['applicable']:
                path = run / (gate['id'] + '.json')
                data = json.dumps(report_for(gate, revision, contract_hash, results), indent=2, allow_nan=False).encode() + b'\n'
                with path.open('xb') as handle:
                    handle.write(data)
                references.append({'path': str(path.relative_to(root)), 'sha256': digest_bytes(data), 'subject_revision': revision})
            record['gates'].append({'id': gate['id'], 'applicable': gate['applicable'], 'outcome': 'passed' if gate['applicable'] else 'not_applicable', 'reason': gate['reason'], 'evidence': references})
        current = load_contract(contract_path, public_key_path, public_key_sha256, expected_contract_sha256, source_root)
        require(current[1] == contract_hash and current[2] == revision, 'Contract or source identity changed during execution')
        validate_record(record, root, payload, contract_hash, revision, source, gates, checks)
    except KeyboardInterrupt:
        run_summary(run, payload, contract_hash, revision, 'cancelled', 'Acceptance operation interrupted')
        raise
    except Exception as error:
        run_summary(run, payload, contract_hash, revision, 'failed', str(error))
        raise
    run_summary(run, payload, contract_hash, revision, 'passed')
    return record, accepted(payload, contract_hash, revision, results, run)


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
    except KeyboardInterrupt:
        print('Acceptance operation cancelled; no readiness attestation issued', file=sys.stderr)
        sys.exit(130)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
