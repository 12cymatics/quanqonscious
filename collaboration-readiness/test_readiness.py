import base64
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from validate_readiness import GATES, REQUIREMENT_KINDS, canonical, load_contract, profile_gates, run_checks, validate
from validation_identity import OPENSSL


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='readiness-protocol-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name).resolve()
        self.source = self.root / 'source'
        self.evidence = self.root / 'evidence'
        self.source.mkdir()
        self.evidence.mkdir()
        package = pathlib.Path(__file__).resolve().parent
        for name in ('validate_contracts.py', 'policy.yaml', 'project-memory.template.yaml'):
            shutil.copyfile(package / name, self.source / name)
        self.runner = '''import json
import pathlib
import sys
import validate_contracts

policy = json.loads(pathlib.Path('policy.yaml').read_text())
memory = json.loads(pathlib.Path('project-memory.template.yaml').read_text())
valid = validate_contracts.validate(policy, memory)
rejected = validate_contracts.self_test(policy, memory)
assertions = [{'id': 'draft_relationships', 'passed': valid is True}, {'id': 'invalid_contracts_rejected', 'passed': len(rejected) >= 16 and len(set(rejected)) == len(rejected)}]
print(json.dumps({'schema_version': 1, 'check_id': sys.argv[1], 'outcome': 'passed' if all(item['passed'] for item in assertions) else 'failed', 'assertions': assertions}))
'''
        (self.source / 'runner.py').write_text(self.runner)
        self.environment = dict(os.environ, GIT_AUTHOR_NAME='Readiness Protocol Test', GIT_AUTHOR_EMAIL='test@example.invalid', GIT_COMMITTER_NAME='Readiness Protocol Test', GIT_COMMITTER_EMAIL='test@example.invalid', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        self.command(['git', 'init', '-q', str(self.source)])
        self.command(['git', '-C', str(self.source), 'add', '.'])
        self.command(['git', '-C', str(self.source), 'commit', '-q', '-m', 'Protocol test source'])
        self.head = self.command(['git', '-C', str(self.source), 'rev-parse', 'HEAD']).stdout.strip()
        self.assertIsNotNone(OPENSSL, 'OpenSSL 3 is required; unsupported crypto must fail visibly')
        self.private = self.root / 'test-private.pem'
        self.public = self.root / 'test-public.pem'
        self.command([OPENSSL, 'genpkey', '-algorithm', 'ED25519', '-out', str(self.private)])
        self.command([OPENSSL, 'pkey', '-in', str(self.private), '-pubout', '-out', str(self.public)])
        self.contract = self.root / 'contract.json'
        request = 'Exercise the actual administrative contract validator and its negative checks. No scientific calculation is accepted by this fixture.'
        self.payload = {'schema_version': 2, 'task_id': 'readiness-package-test', 'user_request': request, 'user_request_sha256': hashlib.sha256(request.encode()).hexdigest(), 'generation': 1, 'domain': 'general', 'mathematical_claims': False, 'deliverable_kind': 'software', 'completion_criterion': 'Actual package contract checks execute and reject their invalid inputs', 'source_root': str(self.source), 'source_manifest': None, 'ownership': {'integrator': 'test-integrator', 'assignments': []}, 'equations': [], 'foundation_register': None, 'geometry_index': None, 'checks': [], 'gates': []}
        required = profile_gates(self.payload)
        for name in sorted(GATES):
            applicable = name in required
            self.payload['gates'].append({'id': name, 'applicable': applicable, 'check_ids': [name] if applicable else [], 'reason': None if applicable else 'Outside the signed administrative software test scope'})
            if applicable:
                self.payload['checks'].append(self.make_check(name, ['draft_relationships', 'invalid_contracts_rejected']))
        self.rebind_source()
        self.sign()
        self.record, self.receipt = run_checks(evidence_root=self.evidence, **self.trust)

    def command(self, argv):
        return subprocess.run(argv, capture_output=True, text=True, check=True, env=self.environment)

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def make_check(self, name, assertions, equation_ids=None):
        executable = pathlib.Path(sys.executable).resolve()
        return {'id': name, 'gate_id': name, 'argv': [str(executable), 'runner.py', name], 'executable_sha256': self.digest(executable), 'entrypoint': 'runner.py', 'assertions': assertions, 'equation_ids': equation_ids or [], 'timeout_seconds': 30}

    def rebind_source(self):
        files = []
        for path in sorted(self.source.rglob('*')):
            relative = path.relative_to(self.source)
            if relative.parts[0] != '.git' and path.is_file() and not path.is_symlink():
                files.append({'path': relative.as_posix(), 'sha256': self.digest(path)})
        self.payload['source_manifest'] = {'schema_version': 1, 'base_head': self.head, 'files': files, 'excluded_paths': [], 'external_assets': []}

    def sign(self):
        message = self.root / 'contract-payload.bin'
        signature = self.root / 'signature.bin'
        message.write_bytes(canonical(self.payload))
        self.command([OPENSSL, 'pkeyutl', '-sign', '-rawin', '-inkey', str(self.private), '-in', str(message), '-out', str(signature)])
        self.contract.write_text(json.dumps({'payload': self.payload, 'signature': base64.b64encode(signature.read_bytes()).decode()}, indent=2) + '\n')
        self.trust = {'contract_path': self.contract, 'public_key_path': self.public, 'public_key_sha256': self.digest(self.public), 'expected_contract_sha256': hashlib.sha256(canonical(self.payload)).hexdigest(), 'source_root': self.source}

    def submit(self, record=None, **overrides):
        args = dict(readiness=True, ownership_root=self.source, execute_checks=True, **self.trust)
        args.update(overrides)
        return validate(self.record if record is None else record, self.evidence, **args)

    def rejected_record(self, pattern, change):
        record = copy.deepcopy(self.record)
        change(record)
        with self.assertRaisesRegex(ValueError, pattern):
            self.submit(record)

    def rejected_contract(self, pattern, change):
        change(self.payload)
        self.sign()
        with self.assertRaisesRegex(ValueError, pattern):
            load_contract(**self.trust)

    def gate(self, record=None, name='existing_aggregate_checks'):
        return next(item for item in (self.record if record is None else record)['gates'] if item['id'] == name)

    def change_evidence(self, content, gate_name='existing_aggregate_checks', append=False):
        path = self.evidence / ('changed-' + str(len(list(self.evidence.glob('changed-*')))) + '.json')
        path.write_bytes(content)
        reference = {'path': path.name, 'sha256': self.digest(path), 'subject_revision': self.record['subject_revision']}
        if append:
            self.gate(name=gate_name)['evidence'].append(reference)
        else:
            self.gate(name=gate_name)['evidence'] = [reference]

    def evidence_value(self, gate_name='existing_aggregate_checks'):
        return json.loads((self.evidence / self.gate(name=gate_name)['evidence'][0]['path']).read_text())

    def write_runner(self, text):
        (self.source / 'runner.py').write_text(text)
        self.rebind_source()
        self.sign()

    def assert_run_rejected(self, pattern):
        before = set(self.evidence.iterdir())
        with self.assertRaisesRegex(ValueError, pattern):
            run_checks(evidence_root=self.evidence, **self.trust)
        after = set(self.evidence.iterdir()) - before
        self.assertEqual(len(after), 1)
        run = after.pop()
        observed = json.loads((run / '0.execution.json').read_text())
        self.assertEqual(observed['outcome'], 'failed')
        self.assertTrue(observed['failure'])
        self.assertEqual(self.digest(run / '0.stdout'), observed['stdout_sha256'])
        self.assertEqual(self.digest(run / '0.stderr'), observed['stderr_sha256'])
        summary = json.loads((run / 'run-summary.json').read_text())
        self.assertEqual(summary['outcome'], 'failed')
        self.assertTrue(summary['failure'])
        return run, observed

    def enable_equation(self):
        self.payload.update(domain='solver', mathematical_claims=True)
        (self.source / 'statement.txt').write_text('Protocol-only exact-function fixture: for integer x, f(x) = x + 2. No physical claim.\n')
        (self.source / 'kernel.py').write_text("def calculate(value):\n    if type(value) is not int:\n        raise ValueError('integer domain required')\n    return value + 2\n")
        self.equation_assertions = ['equation_exact', 'term_present', 'domain_enforced', 'integer_preserved']
        equation_runner = '''import json
import sys
from kernel import calculate

domain_rejected = False
try:
    calculate('1')
except ValueError:
    domain_rejected = True
values = [-(10 ** 100), -2, -1, 0, 1, 2, 10 ** 100]
expected = [-(10 ** 100) + 2, 0, 1, 2, 3, 4, 10 ** 100 + 2]
observed = [calculate(value) for value in values]
assertions = [{'id': 'equation_exact', 'passed': observed == expected}, {'id': 'term_present', 'passed': calculate(0) == 2}, {'id': 'domain_enforced', 'passed': domain_rejected}, {'id': 'integer_preserved', 'passed': all(type(value) is int for value in observed)}]
print(json.dumps({'schema_version': 1, 'check_id': sys.argv[1], 'outcome': 'passed' if all(item['passed'] for item in assertions) else 'failed', 'assertions': assertions}))
'''
        (self.source / 'equation_runner.py').write_text(equation_runner)
        check_ids = ['mathematical_integrity', 'actual_runtime_validation']
        for name in check_ids:
            check = self.make_check(name, self.equation_assertions, ['integer-addition-fixture'])
            check['entrypoint'] = 'equation_runner.py'
            check['argv'][1] = 'equation_runner.py'
            self.payload['checks'] = [item for item in self.payload['checks'] if item['id'] != name]
            self.payload['checks'].append(check)
            next(item for item in self.payload['gates'] if item['id'] == name).update(applicable=True, reason=None, check_ids=[name])
        required = {'equation', 'term', 'domain', 'arithmetic'}
        applicability = {kind: {'applicable': kind in required, 'reason': None if kind in required else 'Absent from the supplied scalar algebraic test specification'} for kind in sorted(REQUIREMENT_KINDS)}
        requirements = []
        for kind, assertion, statement in [('equation', 'equation_exact', 'f(x) = x + 2'), ('term', 'term_present', 'The additive term is exactly +2'), ('domain', 'domain_enforced', 'Input domain is integers'), ('arithmetic', 'integer_preserved', 'Use unbounded integer operations throughout')]:
            requirements.append({'id': 'fixture-' + kind, 'kind': kind, 'statement': statement, 'implementation_paths': ['kernel.py'], 'check_assertions': [{'check_id': check_id, 'assertion_id': assertion} for check_id in check_ids]})
        self.payload['equations'] = [{'id': 'integer-addition-fixture', 'statement': {'path': 'statement.txt', 'sha256': self.digest(self.source / 'statement.txt')}, 'implementation_paths': ['kernel.py'], 'check_ids': check_ids, 'requirement_applicability': applicability, 'requirements': requirements}]
        self.sync_registers()

    def sync_registers(self):
        equations = self.payload['equations']
        foundation = {'schema_version': 1, 'kind': 'project_record', 'task_id': self.payload['task_id'], 'equations': equations}
        geometry = {'schema_version': 1, 'kind': 'project_record', 'task_id': self.payload['task_id'], 'equations': [{'id': eq['id'], 'statement_path': eq['statement']['path'], 'implementation_paths': eq['implementation_paths'], 'requirement_ids': [item['id'] for item in eq['requirements']], 'check_ids': eq['check_ids']} for eq in equations]}
        for field, path, value in [('foundation_register', 'foundation.json', foundation), ('geometry_index', 'geometry-index.json', geometry)]:
            (self.source / path).write_text(json.dumps(value, indent=2) + '\n')
            self.payload[field] = {'path': path, 'sha256': self.digest(self.source / path)}
        self.rebind_source()
        self.sign()

    def test_valid_real_package_entrypoint_and_rerun(self):
        self.assertEqual(self.receipt['readiness_attestation'], 'verified')
        self.assertEqual(set(self.receipt['executed_checks']), profile_gates(self.payload))
        self.assertEqual(self.receipt['checked_equation_ids'], [])
        actual = self.submit()
        self.assertEqual(actual['subject_revision'], self.receipt['subject_revision'])
        self.assertNotEqual(actual['execution_evidence'], self.receipt['execution_evidence'])
        run = pathlib.Path(actual['execution_evidence'])
        observed = json.loads((run / '0.execution.json').read_text())
        output = json.loads((run / '0.stdout').read_text())
        self.assertEqual(observed['return_code'], 0)
        self.assertEqual(observed['outcome'], 'passed')
        self.assertTrue(all(item['passed'] for item in output['assertions']))
        self.assertEqual(json.loads((run / 'run-summary.json').read_text())['outcome'], 'passed')

    def test_empty_evidence(self):
        self.change_evidence(b'')
        with self.assertRaisesRegex(ValueError, 'Evidence is empty or digest differs'):
            self.submit()

    def test_exact_audited_failure_evidence(self):
        self.change_evidence(b'{"outcome":"failed","full_equations_implemented":false,"runtime_executed":false,"authorization_granted":false}')
        with self.assertRaisesRegex(ValueError, 'Gate evidence has missing or unknown fields'):
            self.submit()

    def test_typed_failure_evidence(self):
        value = self.evidence_value()
        value['outcome'] = 'failed'
        self.change_evidence(canonical(value))
        with self.assertRaisesRegex(ValueError, 'Evidence explicitly reports a nonpassing result'):
            self.submit()

    def test_mixed_empty_and_passing_evidence(self):
        self.change_evidence(b'', append=True)
        with self.assertRaisesRegex(ValueError, 'Evidence is empty or digest differs'):
            self.submit()

    def test_mixed_failure_and_passing_evidence(self):
        value = self.evidence_value()
        value['outcome'] = 'failed'
        self.change_evidence(canonical(value), append=True)
        with self.assertRaisesRegex(ValueError, 'Evidence explicitly reports a nonpassing result'):
            self.submit()

    def test_opaque_evidence(self):
        self.change_evidence(b'Everything is verified')
        with self.assertRaises(json.JSONDecodeError):
            self.submit()

    def test_missing_evidence(self):
        record = copy.deepcopy(self.record)
        self.gate(record)['evidence'][0]['path'] = 'absent.json'
        with self.assertRaises(FileNotFoundError):
            self.submit(record)

    def test_empty_evidence_list(self):
        self.rejected_record('Observed outcome requires evidence', lambda r: self.gate(r).update(evidence=[]))

    def test_evidence_hash(self):
        self.rejected_record('Evidence is empty or digest differs', lambda r: self.gate(r)['evidence'][0].update(sha256='0' * 64))

    def test_evidence_revision(self):
        self.rejected_record('Evidence subject revision mismatch', lambda r: self.gate(r)['evidence'][0].update(subject_revision='stale'))

    def test_invented_revision(self):
        def change(record):
            record['subject_revision'] = 'invented-revision-not-bound-to-source'
            for gate in record['gates']:
                for item in gate['evidence']:
                    item['subject_revision'] = record['subject_revision']
        self.rejected_record('Record subject revision differs from actual source manifest', change)

    def test_scope_self_classification(self):
        self.rejected_record('Record scope differs from signed user scope', lambda r: r.update(domain='general', mathematical_claims=False, deliverable_kind='documentation'))

    def test_criterion_narrowing(self):
        self.rejected_record('Record scope differs from signed user scope', lambda r: r.update(completion_criterion='Only parse the fixture'))

    def test_aggregate_excluded_by_record_author(self):
        self.rejected_record('Gate applicability differs from signed scope', lambda r: self.gate(r).update(applicable=False, outcome='not_applicable', reason='Too expensive to run', evidence=[]))

    def test_aggregate_cannot_be_excluded_in_signed_profile(self):
        def change(payload):
            next(gate for gate in payload['gates'] if gate['id'] == 'existing_aggregate_checks').update(applicable=False, check_ids=[], reason='Too expensive to run')
            payload['checks'] = [item for item in payload['checks'] if item['id'] != 'existing_aggregate_checks']
        self.rejected_contract('Required profile gate cannot be excluded', change)

    def test_fabricated_reviewer(self):
        self.rejected_record('Record gate has missing or unknown fields', lambda r: self.gate(r).update(applicability_review={'reason': 'Too expensive', 'reviewer': 'same-record-author'}))

    def test_missing_gate(self):
        self.rejected_record('Exactly one record decision per gate', lambda r: r['gates'].pop())

    def test_unknown_gate(self):
        self.rejected_record('Unknown or duplicate record gate', lambda r: r['gates'][0].update(id='unrelated'))

    def test_bad_gate_id(self):
        self.rejected_record('Unknown or duplicate record gate', lambda r: r['gates'][0].update(id=[]))

    def test_duplicate_gate(self):
        self.rejected_record('Unknown or duplicate record gate', lambda r: r['gates'].append(copy.deepcopy(r['gates'][0])))

    def test_unresolved_applicability(self):
        self.rejected_record('Gate applicability differs from signed scope', lambda r: self.gate(r).update(applicable=None))

    def test_unreviewed_na(self):
        self.rejected_record('Gate exclusion reason differs from signed scope', lambda r: self.gate(r, 'authorization_before_publication').update(reason=None))

    def test_failed_gate(self):
        self.rejected_record('Applicable gate has not passed', lambda r: self.gate(r).update(outcome='failed'))

    def test_cancelled(self):
        self.rejected_record('Readiness requires in-progress or complete state', lambda r: r.update(task_state='cancelled'))

    def test_active_operations(self):
        self.rejected_record('Unresolved blockers or operations', lambda r: r.update(active_operations=['running']))

    def test_blockers(self):
        self.rejected_record('Unresolved blockers or operations', lambda r: r.update(blockers=['Required calculation missing']))

    def test_unknown_revision(self):
        self.rejected_record('Record subject revision differs', lambda r: r.update(subject_revision=None))

    def test_contract_generation(self):
        self.rejected_record('Record contract identity or generation mismatch', lambda r: r.update(contract_generation=2))

    def test_missing_signature_trust(self):
        with self.assertRaisesRegex(ValueError, 'Signed contract, independently pinned'):
            self.submit(contract_path=None)

    def test_stale_contract_pin(self):
        with self.assertRaisesRegex(ValueError, 'Contract differs from independently pinned'):
            self.submit(expected_contract_sha256='0' * 64)

    def test_self_signed_key_replacement(self):
        other = self.root / 'other-public.pem'
        other.write_bytes(self.public.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Public-key digest does not match externally pinned'):
            self.submit(public_key_path=other)

    def test_missing_explicit_readiness(self):
        with self.assertRaisesRegex(ValueError, 'explicit readiness operation'):
            self.submit(readiness=False)

    def test_missing_execution(self):
        with self.assertRaisesRegex(ValueError, 'Readiness requires execution'):
            self.submit(execute_checks=False)

    def test_source_mutation(self):
        (self.source / 'runner.py').write_text(self.runner + '\n')
        with self.assertRaisesRegex(ValueError, 'Source file digest mismatch: runner.py'):
            self.submit()

    def test_untracked_input(self):
        (self.source / 'untracked.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Source files are not accounted for: untracked.json'):
            self.submit()

    def test_invented_head(self):
        self.rejected_contract('Source base_head does not match', lambda p: p['source_manifest'].update(base_head='0' * 40))

    def test_ownership_cannot_be_reassigned_in_record(self):
        self.rejected_record('Record ownership differs from signed ownership', lambda r: r['ownership'].update(assignments=[{'owner': 'worker', 'write_paths': ['runner.py']}]))

    def test_bad_assignment(self):
        self.rejected_contract('Assignment must be an object', lambda p: p['ownership'].update(assignments=[None]))

    def test_parent_overlap(self):
        self.rejected_contract('Overlapping, duplicate or aliased ownership', lambda p: p['ownership'].update(assignments=[{'owner': 'a', 'write_paths': ['src']}, {'owner': 'b', 'write_paths': ['src/module.py']}]))

    def test_alias_overlap(self):
        self.rejected_contract('Path must be canonical and relative', lambda p: p['ownership'].update(assignments=[{'owner': 'a', 'write_paths': ['src/file']}, {'owner': 'b', 'write_paths': ['src/./file']}]))

    def test_case_alias_existing_directory(self):
        (self.source / 'CaseOwned').mkdir()
        self.rejected_contract('Overlapping, duplicate or aliased ownership|Case or Unicode ownership alias', lambda p: p['ownership'].update(assignments=[{'owner': 'a', 'write_paths': ['CaseOwned']}, {'owner': 'b', 'write_paths': ['caseowned']}]))

    def test_case_alias_planned_descendant(self):
        self.rejected_contract('Overlapping, duplicate or aliased ownership', lambda p: p['ownership'].update(assignments=[{'owner': 'a', 'write_paths': ['Future']}, {'owner': 'b', 'write_paths': ['future/file.py']}]))

    def test_distinct_source_escape(self):
        (self.source / 'outside').symlink_to(self.evidence, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink source or ownership path is not supported'):
            self.submit()

    def test_distinct_source_alias(self):
        (self.source / 'src').mkdir()
        (self.source / 'alias').symlink_to(self.source / 'src', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink source or ownership path is not supported'):
            self.submit()

    def test_ownership_root_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Ownership root differs from signed source root'):
            self.submit(ownership_root=self.evidence)

    def test_traversal(self):
        self.rejected_record('Path must be canonical and relative', lambda r: self.gate(r)['evidence'][0].update(path='../evidence.json'))

    def test_symlink_escape(self):
        outside = self.root / 'outside-evidence.json'
        outside.write_text('{}')
        (self.evidence / 'escape').symlink_to(outside)
        self.rejected_record('Evidence escapes root or is not a file', lambda r: self.gate(r)['evidence'][0].update(path='escape'))

    def test_evidence_cannot_be_reused_for_unrelated_gate(self):
        other = self.gate(name='source_and_input_provenance')['evidence']
        self.rejected_record('Evidence gate identity mismatch', lambda r: self.gate(r).update(evidence=copy.deepcopy(other)))

    def test_evidence_failed_assertion(self):
        value = self.evidence_value()
        value['checks'][0]['assertions'][0]['passed'] = False
        self.change_evidence(canonical(value))
        with self.assertRaisesRegex(ValueError, 'Required assertion failed: draft_relationships'):
            self.submit()

    def test_evidence_missing_assertion(self):
        value = self.evidence_value()
        value['checks'][0]['assertions'].pop()
        self.change_evidence(canonical(value))
        with self.assertRaisesRegex(ValueError, 'Required assertions are missing or changed'):
            self.submit()

    def test_evidence_duplicate_assertion(self):
        value = self.evidence_value()
        value['checks'][0]['assertions'].append(copy.deepcopy(value['checks'][0]['assertions'][0]))
        self.change_evidence(canonical(value))
        with self.assertRaisesRegex(ValueError, 'Duplicate or missing assertion id'):
            self.submit()

    def test_evidence_duplicate_json_key(self):
        self.change_evidence(b'{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key: schema_version'):
            self.submit()

    def test_evidence_nonfinite_number(self):
        self.change_evidence(b'{"schema_version":1e999}')
        with self.assertRaisesRegex(ValueError, 'Non-finite JSON number'):
            self.submit()

    def test_entrypoint_cannot_be_unused_interpreter_argument(self):
        self.rejected_contract('first executable argument', lambda p: p['checks'][0].update(argv=[p['checks'][0]['argv'][0], '-c', 'print(1)', 'runner.py']))

    def test_unbound_entrypoint(self):
        self.rejected_contract('Check entrypoint must be a bound source file', lambda p: p['checks'][0].update(entrypoint='absent.py', argv=[p['checks'][0]['argv'][0], 'absent.py']))

    def test_executable_digest(self):
        self.rejected_contract('Check executable digest mismatch', lambda p: p['checks'][0].update(executable_sha256='0' * 64))

    def test_timeout_boolean(self):
        self.rejected_contract('Positive finite explicit check timeout required', lambda p: p['checks'][0].update(timeout_seconds=True))

    def test_exit_zero_with_failed_assertion(self):
        self.write_runner(self.runner.replace("'passed': valid is True", "'passed': False"))
        self.assert_run_rejected('Acceptance check reported failure')

    def test_nonzero_exit_preserves_logs(self):
        self.write_runner("import sys\nprint('actual stdout')\nprint('actual stderr', file=sys.stderr)\nsys.exit(7)\n")
        run, observed = self.assert_run_rejected('exit=7')
        self.assertEqual(observed['return_code'], 7)
        self.assertEqual((run / '0.stdout').read_text(), 'actual stdout\n')
        self.assertEqual((run / '0.stderr').read_text(), 'actual stderr\n')

    def test_timeout_preserves_terminal_failure(self):
        self.write_runner("import time\nprint('started', flush=True)\ntime.sleep(10)\n")
        self.payload['checks'][0]['timeout_seconds'] = 0.05
        self.sign()
        run, observed = self.assert_run_rejected('Acceptance check timed out')
        self.assertTrue(observed['timed_out'])
        self.assertIsInstance(observed['return_code'], int)

    def test_empty_runner_output(self):
        self.write_runner('import sys\nsys.exit(0)\n')
        self.assert_run_rejected('did not emit valid structured results')

    def test_wrong_runner_check_identity(self):
        self.write_runner(self.runner.replace("'check_id': sys.argv[1]", "'check_id': 'unrelated'"))
        self.assert_run_rejected('Check result identity mismatch')

    def test_runner_modifies_then_restores_source(self):
        mutation = "import pathlib\np = pathlib.Path('policy.yaml')\noriginal = p.read_bytes()\np.write_bytes(original + b'\\n')\np.write_bytes(original)\n"
        self.write_runner(mutation + self.runner)
        self.assert_run_rejected('Source filesystem state changed during execution')

    def test_runner_changes_contract_after_loading(self):
        mutation = "import pathlib\npathlib.Path(" + repr(str(self.contract)) + ").write_text('{}')\n"
        self.write_runner(mutation + self.runner)
        before = set(self.evidence.iterdir())
        with self.assertRaisesRegex(ValueError, 'Signed contract must contain exactly payload and signature'):
            run_checks(evidence_root=self.evidence, **self.trust)
        run = (set(self.evidence.iterdir()) - before).pop()
        self.assertEqual(json.loads((run / '0.execution.json').read_text())['outcome'], 'passed')
        summary = json.loads((run / 'run-summary.json').read_text())
        self.assertEqual(summary['outcome'], 'failed')
        self.assertIn('Signed contract must contain exactly payload and signature', summary['failure'])

    def test_relabelled_manual_success_still_executes_failure(self):
        self.write_runner(self.runner.replace("'passed': valid is True", "'passed': False"))
        self.record['subject_revision'] = 'sha256:' + hashlib.sha256(canonical(self.payload['source_manifest'])).hexdigest()
        self.record['contract_sha256'] = self.trust['expected_contract_sha256']
        for gate in self.record['gates']:
            if gate['applicable']:
                old = json.loads((self.evidence / gate['evidence'][0]['path']).read_text())
                old['subject_revision'] = self.record['subject_revision']
                old['contract_sha256'] = self.record['contract_sha256']
                self.change_evidence(canonical(old), gate_name=gate['id'])
        with self.assertRaisesRegex(ValueError, 'Acceptance check reported failure'):
            self.submit()

    def test_mathematical_registers_and_real_function_execute(self):
        self.enable_equation()
        record, receipt = run_checks(evidence_root=self.evidence, **self.trust)
        self.assertEqual(receipt['checked_equation_ids'], ['integer-addition-fixture'])
        self.assertEqual(set(receipt['checked_requirement_ids']), {'fixture-equation', 'fixture-term', 'fixture-domain', 'fixture-arithmetic'})
        self.assertEqual(self.submit(record)['readiness_attestation'], 'verified')

    def test_cymatica_cannot_omit_equations(self):
        def change(payload):
            payload['domain'] = 'cymatica'
            gate = next(item for item in payload['gates'] if item['id'] == 'mathematical_integrity')
            gate.update(applicable=True, reason=None, check_ids=['mathematical_integrity'])
            payload['checks'].append(self.make_check('mathematical_integrity', ['draft_relationships', 'invalid_contracts_rejected']))
        self.rejected_contract('Full equation inventory required for mathematical acceptance', change)

    def test_cymatica_integrity_cannot_be_excluded(self):
        self.rejected_contract('Required profile gate cannot be excluded', lambda p: p.update(domain='cymatica'))

    def test_foundation_always(self):
        def change(payload):
            next(item for item in payload['gates'] if item['id'] == 'foundation_traceability').update(applicable=False, reason='Avoid foundation', check_ids=[])
            payload['checks'] = [item for item in payload['checks'] if item['id'] != 'foundation_traceability']
        self.rejected_contract('Required profile gate cannot be excluded', change)

    def test_equation_register_mismatch(self):
        self.enable_equation()
        content = json.loads((self.source / 'foundation.json').read_text())
        content['equations'][0]['requirements'].pop()
        (self.source / 'foundation.json').write_text(json.dumps(content))
        self.payload['foundation_register']['sha256'] = self.digest(self.source / 'foundation.json')
        self.rebind_source()
        self.sign()
        with self.assertRaisesRegex(ValueError, 'foundation_register differs from signed equation and requirement inventory'):
            load_contract(**self.trust)

    def test_geometry_index_mismatch(self):
        self.enable_equation()
        content = json.loads((self.source / 'geometry-index.json').read_text())
        content['equations'][0]['requirement_ids'] = []
        (self.source / 'geometry-index.json').write_text(json.dumps(content))
        self.payload['geometry_index']['sha256'] = self.digest(self.source / 'geometry-index.json')
        self.rebind_source()
        self.sign()
        with self.assertRaisesRegex(ValueError, 'geometry_index differs from signed equation and requirement inventory'):
            load_contract(**self.trust)

    def test_missing_requirement_category(self):
        self.enable_equation()
        self.payload['equations'][0]['requirement_applicability'].pop('boundary_condition')
        self.sync_registers()
        with self.assertRaisesRegex(ValueError, 'Equation requirement applicability has missing or unknown fields'):
            load_contract(**self.trust)

    def test_applicable_category_without_requirements(self):
        self.enable_equation()
        self.payload['equations'][0]['requirement_applicability']['boundary_condition'] = {'applicable': True, 'reason': None}
        self.sync_registers()
        with self.assertRaisesRegex(ValueError, 'Applicable equation requirement category has no obligations'):
            load_contract(**self.trust)

    def test_unknown_requirement_assertion(self):
        self.enable_equation()
        self.payload['equations'][0]['requirements'][0]['check_assertions'][0]['assertion_id'] = 'missing'
        self.sync_registers()
        with self.assertRaisesRegex(ValueError, 'Requirement references an unknown executable assertion'):
            load_contract(**self.trust)

    def test_missing_requirement_runtime_assertion(self):
        self.enable_equation()
        obligation = self.payload['equations'][0]['requirements'][0]
        obligation['check_assertions'] = [item for item in obligation['check_assertions'] if item['check_id'] != 'actual_runtime_validation']
        self.sync_registers()
        with self.assertRaisesRegex(ValueError, 'Requirement lacks an actual runtime assertion'):
            load_contract(**self.trust)

    def test_unbound_implementation(self):
        self.enable_equation()
        self.payload['equations'][0]['implementation_paths'].append('missing.py')
        self.sync_registers()
        with self.assertRaisesRegex(ValueError, 'Equation implementation is not included in source manifest'):
            load_contract(**self.trust)

    def test_removed_equation_term_fails_execution(self):
        self.enable_equation()
        kernel = self.source / 'kernel.py'
        kernel.write_text(kernel.read_text().replace('value + 2', 'value + 1'))
        self.rebind_source()
        self.sign()
        with self.assertRaisesRegex(ValueError, 'Acceptance check reported failure: mathematical_integrity'):
            run_checks(evidence_root=self.evidence, **self.trust)

    def test_disabled_function_fails_execution(self):
        self.enable_equation()
        kernel = self.source / 'kernel.py'
        kernel.write_text(kernel.read_text().replace('return value + 2', "raise RuntimeError('Required calculation disabled')"))
        self.rebind_source()
        self.sign()
        with self.assertRaisesRegex(ValueError, 'Acceptance check failed: mathematical_integrity'):
            run_checks(evidence_root=self.evidence, **self.trust)

    def template(self):
        return {'schema_version': 3, 'kind': 'template', 'task_state': 'not_started', 'domain': 'general', 'mathematical_claims': False, 'deliverable_kind': 'software', 'completion_criterion': None, 'subject_revision': None, 'contract_sha256': None, 'contract_generation': None, 'blockers': [], 'active_operations': [], 'ownership': {'integrator': None, 'assignments': []}, 'gates': [{'id': name, 'applicable': None, 'outcome': 'not_run', 'reason': 'Unassigned template', 'evidence': []} for name in sorted(GATES)]}

    def test_template_shape_only(self):
        self.assertEqual(validate(self.template(), self.evidence), {'record_consistency': 'passed', 'readiness_attestation': 'not_evaluated'})

    def test_template_completion(self):
        with self.assertRaisesRegex(ValueError, 'Template cannot claim readiness or completion'):
            validate(self.template(), self.evidence, readiness=True)

    def test_template_cannot_hide_unknown_claims(self):
        value = self.template()
        value['authorization_granted'] = True
        with self.assertRaisesRegex(ValueError, 'Readiness template has missing or unknown fields'):
            validate(value, self.evidence)

    def test_template_cannot_claim_ownership(self):
        value = self.template()
        value['ownership']['integrator'] = 'self-assigned'
        with self.assertRaisesRegex(ValueError, 'Template cannot claim assigned ownership'):
            validate(value, self.evidence)

    def test_legacy_schema_rejected_explicitly(self):
        self.rejected_record('Legacy schema cannot attest readiness', lambda r: r.update(schema_version=2))

    def test_cli_run_and_validate(self):
        record_path = self.root / 'cli-record.json'
        script = str(pathlib.Path(__file__).with_name('validate_readiness.py').resolve())
        base = [sys.executable, '-B', script, '--evidence-root', str(self.evidence), '--source-root', str(self.source), '--contract', str(self.contract), '--trusted-public-key', str(self.public), '--trusted-key-sha256', self.trust['public_key_sha256'], '--contract-sha256', self.trust['expected_contract_sha256']]
        first = subprocess.run(base + ['--run-checks', '--output-record', str(record_path)], text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)['readiness_attestation'], 'verified')
        second = subprocess.run(base + ['--record', str(record_path), '--readiness', '--execute-checks'], text=True, capture_output=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)['readiness_attestation'], 'verified')
        absent_execution = subprocess.run(base + ['--record', str(record_path), '--readiness'], text=True, capture_output=True)
        self.assertEqual(absent_execution.returncode, 1)
        self.assertIn('Readiness requires execution', absent_execution.stderr)


if __name__ == '__main__':
    unittest.main()
