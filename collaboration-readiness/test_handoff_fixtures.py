import base64
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import project_handoff
from validate_readiness import GATES, canonical, profile_gates
from validation_identity import OPENSSL


class HandoffFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='project-handoff-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / 'source'
        self.candidate = self.root / 'candidate'
        self.journals = self.root / 'journals'
        self.source.mkdir()
        self.environment = {name: value for name, value in os.environ.items() if not name.startswith('GIT_')}
        self.environment.update(GIT_AUTHOR_NAME='Handoff Test', GIT_AUTHOR_EMAIL='handoff@example.invalid', GIT_COMMITTER_NAME='Handoff Test', GIT_COMMITTER_EMAIL='handoff@example.invalid', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, PYTHONDONTWRITEBYTECODE='1')
        package = Path(__file__).resolve().parent
        for name in ('validate_contracts.py', 'policy.yaml', 'project-memory.template.yaml'):
            shutil.copyfile(package / name, self.source / name)
        runner = "import json\nfrom pathlib import Path\nimport sys\nimport validate_contracts\n\npolicy = json.loads(Path('policy.yaml').read_text())\nmemory = json.loads(Path('project-memory.template.yaml').read_text())\nvalid = validate_contracts.validate(policy, memory)\nrejected = validate_contracts.self_test(policy, memory)\nassertions = [{'id': 'actual_contract_checked', 'passed': valid is True}, {'id': 'invalid_inputs_rejected', 'passed': len(rejected) >= 16 and len(rejected) == len(set(rejected))}]\nprint(json.dumps({'schema_version': 1, 'check_id': sys.argv[1], 'outcome': 'passed' if all(item['passed'] for item in assertions) else 'failed', 'assertions': assertions}))\n"
        (self.source / 'runner.py').write_text(runner)
        (self.source / 'src').mkdir()
        (self.source / 'src/replace.txt').write_bytes(b'original replacement target\n')
        (self.source / 'src/delete.txt').write_bytes(b'original deletion target\n')
        (self.source / 'unowned.txt').write_bytes(b'outside assigned write ownership\n')
        self.command(['git', 'init', '-q', str(self.source)])
        self.command(['git', '-C', str(self.source), 'add', '.'])
        self.command(['git', '-C', str(self.source), '-c', 'commit.gpgsign=false', 'commit', '-q', '-m', 'Handoff protocol fixture'])
        self.head = self.command(['git', '-C', str(self.source), 'rev-parse', 'HEAD']).stdout.strip()
        shutil.copytree(self.source, self.candidate)
        (self.candidate / 'src/replace.txt').write_bytes(b'candidate replacement bytes\n')
        (self.candidate / 'src/delete.txt').unlink()
        (self.candidate / 'src/new/deep').mkdir(parents=True)
        (self.candidate / 'src/new/deep/new.txt').write_bytes(b'candidate created bytes\n')
        (self.candidate / 'src/new/deep/new.txt').chmod(0o644)
        self.assertIsNotNone(OPENSSL, 'OpenSSL 3 is required for real Ed25519 protocol tests')
        self.private, self.public = self.keypair('integrator')
        self.worker_private, self.worker_public = self.keypair('worker')
        self.contract_path = self.root / 'contract.json'
        self.registry_path = self.root / 'registry.json'
        self.handoff_path = self.root / 'handoff.json'
        request = 'Apply the signed fixture documentation create, replace and delete changes under src, preserving every unassigned input.'
        self.payload = {'schema_version': 2, 'task_id': 'handoff-protocol-test', 'user_request': request, 'user_request_sha256': hashlib.sha256(request.encode()).hexdigest(), 'generation': 1, 'domain': 'general', 'mathematical_claims': False, 'deliverable_kind': 'documentation', 'completion_criterion': 'Assigned fixture bytes are integrated and the real contract validator runs on those bytes', 'source_root': str(self.source), 'source_manifest': self.manifest(self.source), 'ownership': {'integrator': 'integrator', 'assignments': [{'owner': 'worker', 'write_paths': ['src']}]}, 'equations': [], 'foundation_register': None, 'geometry_index': None, 'checks': [], 'gates': []}
        executable = Path(sys.executable).resolve()
        required = profile_gates(self.payload)
        for name in sorted(GATES):
            applicable = name in required
            self.payload['gates'].append({'id': name, 'applicable': applicable, 'check_ids': [name] if applicable else [], 'reason': None if applicable else 'Outside the signed documentation integration fixture'})
            if applicable:
                self.payload['checks'].append({'id': name, 'gate_id': name, 'argv': [str(executable), 'runner.py', name], 'executable_sha256': self.digest(executable), 'entrypoint': 'runner.py', 'assertions': ['actual_contract_checked', 'invalid_inputs_rejected'], 'equation_ids': [], 'timeout_seconds': 30})
        self.contract_hash = self.sign(self.contract_path, self.payload, self.private)
        now = datetime.datetime.now(datetime.timezone.utc)
        self.issued = (now - datetime.timedelta(minutes=5)).isoformat()
        self.expiry = (now + datetime.timedelta(hours=2)).isoformat()
        self.assignment_expiry = (now + datetime.timedelta(hours=1)).isoformat()
        self.registry = {'schema_version': 1, 'task_id': self.payload['task_id'], 'contract_sha256': self.contract_hash, 'contract_generation': self.payload['generation'], 'source_manifest_sha256': hashlib.sha256(canonical(self.payload['source_manifest'])).hexdigest(), 'generation': 1, 'writer': 'integrator', 'issued_at': self.issued, 'expires_at': self.expiry, 'previous_registry_sha256': None, 'reconciliations': [], 'assignments': [{'id': 'assignment-1', 'owner': 'worker', 'write_paths': ['src'], 'state': 'active', 'expires_at': self.assignment_expiry, 'pending_operations': [], 'worker_public_key_sha256': self.digest(self.worker_public)}], 'exclusive_resources': [{'kind': 'branch', 'id': 'repair-fixture', 'assignment_id': 'assignment-1'}, {'kind': 'live_session', 'id': 'fixture-editor', 'assignment_id': 'assignment-1'}], 'authorization': {'action': 'apply_project_changes', 'user_request_sha256': self.payload['user_request_sha256']}}
        self.registry_hash = self.sign(self.registry_path, self.registry, self.private)
        result_manifest = self.manifest(self.candidate)
        self.handoff = {'schema_version': 1, 'task_id': self.payload['task_id'], 'contract_sha256': self.contract_hash, 'contract_generation': self.payload['generation'], 'registry_sha256': self.registry_hash, 'registry_generation': 1, 'assignment_id': 'assignment-1', 'worker': 'worker', 'base_manifest_sha256': self.registry['source_manifest_sha256'], 'result_manifest': result_manifest, 'changes': project_handoff.manifest_delta(self.payload['source_manifest'], result_manifest), 'outcome': 'ready_for_integration', 'pending_operations': []}
        self.handoff_hash = self.sign(self.handoff_path, self.handoff, self.worker_private)
        self.options = {'contract_path': self.contract_path, 'public_key_path': self.public, 'public_key_sha256': self.digest(self.public), 'expected_contract_sha256': self.contract_hash, 'source_root': self.source, 'registry_path': self.registry_path, 'expected_registry_sha256': self.registry_hash, 'registry_generation': 1, 'worker': 'worker', 'assignment_id': 'assignment-1', 'requested_paths': [item['path'] for item in self.handoff['changes']]}
        self.handoff_options = {'handoff_path': self.handoff_path, 'worker_public_key_path': self.worker_public, 'artifact_root': self.candidate}
        self.baseline_bytes = self.bytes_of(self.source)
        self.assertEqual(self.validate()[0]['state'], 'handoff_validated')

    def command(self, argv):
        return subprocess.run(argv, capture_output=True, text=True, check=True, env=self.environment)

    def digest(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def bytes_of(self, root):
        return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file() and '.git' not in path.relative_to(root).parts}

    def manifest(self, root):
        return {'schema_version': 1, 'base_head': self.head, 'files': [{'path': name, 'sha256': hashlib.sha256(data).hexdigest()} for name, data in self.bytes_of(root).items()], 'excluded_paths': [], 'external_assets': []}

    def keypair(self, name):
        private, public = self.root / (name + '-private.pem'), self.root / (name + '-public.pem')
        self.command([OPENSSL, 'genpkey', '-algorithm', 'ED25519', '-out', str(private)])
        self.command([OPENSSL, 'pkey', '-in', str(private), '-pubout', '-out', str(public)])
        return private, public

    def sign(self, path, payload, private):
        message = path.with_suffix('.payload.bin')
        signature = path.with_suffix('.signature.bin')
        message.write_bytes(canonical(payload))
        self.command([OPENSSL, 'pkeyutl', '-sign', '-rawin', '-inkey', str(private), '-in', str(message), '-out', str(signature)])
        path.write_text(json.dumps({'payload': payload, 'signature': base64.b64encode(signature.read_bytes()).decode('ascii')}, indent=2) + '\n')
        return hashlib.sha256(canonical(payload)).hexdigest()

    def seal_registry(self):
        self.registry_hash = self.sign(self.registry_path, self.registry, self.private)
        self.options['expected_registry_sha256'] = self.registry_hash
        self.handoff['registry_sha256'] = self.registry_hash
        self.seal_handoff()

    def seal_handoff(self):
        self.handoff_hash = self.sign(self.handoff_path, self.handoff, self.worker_private)

    def seal_all(self):
        self.contract_hash = self.sign(self.contract_path, self.payload, self.private)
        self.options['expected_contract_sha256'] = self.contract_hash
        self.registry.update(task_id=self.payload['task_id'], contract_sha256=self.contract_hash, contract_generation=self.payload['generation'], source_manifest_sha256=hashlib.sha256(canonical(self.payload['source_manifest'])).hexdigest())
        self.handoff.update(task_id=self.payload['task_id'], contract_sha256=self.contract_hash, contract_generation=self.payload['generation'], base_manifest_sha256=self.registry['source_manifest_sha256'])
        self.seal_registry()

    def admit(self, **overrides):
        options = dict(self.options)
        options.update(overrides)
        return project_handoff.admit(**options)

    def validate(self, **overrides):
        options = dict(self.options, **self.handoff_options)
        options.update(overrides)
        return project_handoff.validate_handoff(**options)

    def apply(self, **overrides):
        options = dict(self.options, **self.handoff_options)
        options.update(overrides)
        return project_handoff.apply_handoff(journal_root=self.journals, **options)

    def registry_rejected(self, pattern, change):
        change(self.registry)
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, pattern):
            self.admit()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def handoff_rejected(self, pattern, change):
        change(self.handoff)
        self.seal_handoff()
        with self.assertRaisesRegex(ValueError, pattern):
            self.validate()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def journal(self):
        paths = list(self.journals.glob('integration-*/journal.json'))
        self.assertEqual(len(paths), 1)
        return paths[0], json.loads(paths[0].read_text())

    def transfer(self):
        previous = copy.deepcopy(self.registry)
        previous_path = self.root / 'previous-registry.json'
        previous_hash = self.sign(previous_path, previous, self.private)
        self.worker_private, self.worker_public = self.keypair('replacement-worker')
        self.payload['generation'] += 1
        self.payload['ownership']['assignments'] = [{'owner': 'replacement-worker', 'write_paths': ['src']}]
        self.registry['generation'] = 2
        self.registry['previous_registry_sha256'] = previous_hash
        self.registry['assignments'][0].update(owner='replacement-worker', worker_public_key_sha256=self.digest(self.worker_public))
        self.registry['reconciliations'] = [{'assignment_id': 'assignment-1', 'owner': 'worker', 'write_paths': ['src'], 'outcome': 'reconciled', 'pending_operations': [], 'final_source_manifest_sha256': self.registry['source_manifest_sha256']}]
        self.options.update(registry_generation=2, previous_registry=previous_path, worker='replacement-worker')
        self.handoff.update(worker='replacement-worker', registry_generation=2)
        self.handoff_options['worker_public_key_path'] = self.worker_public
        self.seal_all()
        return previous_path, previous
