import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from worker_programs import BOUNDARY_PROBE, BYTE_ECHO, EXEC_PROBE, FAIL_WITH_OUTPUT, TIMEOUT


class GateAcceptance(unittest.TestCase):
    def setUp(self):
        from gate3 import Gate
        self.Gate = Gate
        self.temp = tempfile.TemporaryDirectory(prefix='gate-acceptance-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.state = self.base / 'state'
        self.spec = ROOT / 'source' / 'verified_delegation_instruction.txt'
        self.original = self.spec.read_bytes()
        self.authorization = self.base / 'authorization.txt'
        self.authorization.write_bytes((ROOT / 'tests' / 'authorization.txt').read_bytes())
        self.key = Ed25519PrivateKey.generate()
        self.private = self.base / 'authority.key'
        self.public = self.base / 'authority.pub'
        self.private.write_bytes(self.key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
        self.private.chmod(0o600)
        self.public.write_bytes(self.key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))
        self.fingerprint = hashlib.sha256(self.public.read_bytes()).hexdigest()
        self.cli('init', '--state', self.state, '--spec', self.spec,
                 '--authority-key', self.private, '--expected-fingerprint', self.fingerprint,
                 '--authorization', self.authorization)
        self.gate = Gate(self.state)
        self.counter = 0

    def cli(self, *args, check=True, timeout=30):
        result = subprocess.run([sys.executable, str(ROOT / 'gate3.py'), *map(str, args)],
                                capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            self.fail(f'gate CLI rejected an expected valid operation: {result.stderr}\n{result.stdout}')
        return result

    def issue(self, source=BYTE_ECHO, *, parent=None, argv=None, limits=None):
        self.counter += 1
        job = self.base / f'input-{self.counter}'
        job.mkdir()
        (job / 'work.py').write_text(source, encoding='utf-8')
        envelope = Path(self.gate.create(job, 'work.py', 'Verify authoritative byte delivery',
                                         argv=argv or [], parent=parent, limits=limits))
        return envelope, job

    def output(self, name):
        matches = list(self.state.rglob(name))
        self.assertEqual(len(matches), 1, f'Expected one preserved job output named {name}: {matches}')
        return matches[0]

    def altered(self, envelope, mutate):
        obj = json.loads(envelope.read_bytes())
        mutate(obj)
        path = self.base / f'altered-{self.counter}.json'
        path.write_text(json.dumps(obj, sort_keys=True, separators=(',', ':')))
        return path

    def rejected(self, envelope):
        previous = {str(path): path.read_bytes() for path in self.state.rglob('result.json')}
        with self.assertRaises(Exception):
            self.gate.dispatch(envelope)
        self.assertEqual({str(path): path.read_bytes() for path in self.state.rglob('result.json')}, previous)

    def test_full_source_reaches_actual_process(self):
        envelope, _ = self.issue(argv=['byte-exact', '\u03c0'])
        self.gate.dispatch(envelope)
        report = json.loads(self.output('result.json').read_bytes())
        self.assertEqual(base64.b64decode(report['instruction_b64']), self.original)
        self.assertEqual(report['instruction_length'], len(self.original))
        self.assertEqual(report['instruction_sha256'], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(report['argv'], ['byte-exact', '\u03c0'])
        self.assertTrue(report['envelope_present'])
        self.assertTrue(report['manifest_present'])
        self.assertEqual(base64.b64decode(report['global_instruction_b64']), self.original)
        self.assertEqual(report['global_instruction_sha256'], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(report['assigned_operation'], 'Verify authoritative byte delivery')
        self.assertEqual(report['handoff_id'], envelope.parent.name)
        self.assertIsNone(report['parent_sha256'])
        self.gate.audit_verify()

    def test_invalid_signature_stops_actual_dispatch(self):
        envelope, _ = self.issue()
        def mutation(obj):
            signature = bytearray(base64.b64decode(obj['signature_b64']))
            signature[0] ^= 1
            obj['signature_b64'] = base64.b64encode(signature).decode('ascii')
        self.rejected(self.altered(envelope, mutation))

    def test_signature_from_an_unenrolled_key_rejected(self):
        from gate3 import HANDOFF_DOMAIN
        envelope, _ = self.issue()
        other = Ed25519PrivateKey.generate()
        def mutation(obj):
            payload = base64.b64decode(obj['payload_b64'])
            obj['signature_b64'] = base64.b64encode(other.sign(HANDOFF_DOMAIN + payload)).decode('ascii')
        self.rejected(self.altered(envelope, mutation))

    def test_changed_instruction_stops_actual_dispatch(self):
        envelope, _ = self.issue()
        def mutation(obj):
            payload = base64.b64decode(obj['payload_b64'])
            self.assertIn(self.original, payload)
            modified = bytes([self.original[0] ^ 1]) + self.original[1:]
            obj['payload_b64'] = base64.b64encode(payload.replace(self.original, modified, 1)).decode('ascii')
        self.rejected(self.altered(envelope, mutation))

    def test_changed_operation_stops_actual_dispatch(self):
        envelope, _ = self.issue()
        def mutation(obj):
            payload = base64.b64decode(obj['payload_b64'])
            original = b'Verify authoritative byte delivery'
            self.assertIn(original, payload)
            obj['payload_b64'] = base64.b64encode(payload.replace(original, b'verify authoritative byte delivery', 1)).decode('ascii')
        self.rejected(self.altered(envelope, mutation))

    def test_missing_original_stops_actual_dispatch(self):
        envelope, _ = self.issue()
        def mutation(obj):
            payload = base64.b64decode(obj['payload_b64'])
            self.assertIn(self.original, payload)
            obj['payload_b64'] = base64.b64encode(payload.replace(self.original, b'', 1)).decode('ascii')
        self.rejected(self.altered(envelope, mutation))

    def test_replay_does_not_start_second_process(self):
        envelope, _ = self.issue()
        self.gate.dispatch(envelope)
        first = self.output('result.json').read_bytes()
        with self.assertRaises(Exception):
            self.gate.dispatch(envelope)
        self.assertEqual(self.output('result.json').read_bytes(), first)
        self.gate.audit_verify()

    def test_concurrent_replay_has_one_execution(self):
        envelope, _ = self.issue()
        command = [sys.executable, str(ROOT / 'gate3.py'), 'dispatch',
                   '--state', str(self.state), '--envelope', str(envelope)]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(2)]
        for process in processes:
            process.communicate(timeout=15)
        self.assertEqual(sorted(process.returncode for process in processes), [0, 1])
        self.assertEqual(self.gate.audit_verify()['events'].count('started'), 1)
        self.assertEqual(self.gate.audit_verify()['events'].count('completed'), 1)
        self.output('result.json')

    def test_interrupted_dispatch_cannot_be_replayed(self):
        envelope, _ = self.issue(TIMEOUT, limits={'timeout_seconds': 3})
        command = [sys.executable, str(ROOT / 'gate3.py'), 'dispatch',
                   '--state', str(self.state), '--envelope', str(envelope)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if b'"event":"started"' in (self.state / 'audit.jsonl').read_bytes():
                    break
                if process.poll() is not None:
                    self.fail('dispatch ended before its durable execution record')
                time.sleep(0.01)
            else:
                self.fail('dispatch did not durably reserve its handoff')
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)
        resumed = self.Gate(self.state)
        with self.assertRaises(Exception):
            resumed.dispatch(envelope)
        events = resumed.audit_verify()['events']
        self.assertEqual(events.count('started'), 1)

    def test_edits_to_source_directory_do_not_change_signed_execution(self):
        envelope, source = self.issue()
        (source / 'work.py').write_text("raise RuntimeError('source changed after signing')\n")
        self.gate.dispatch(envelope)
        report = json.loads(self.output('result.json').read_bytes())
        self.assertEqual(base64.b64decode(report['instruction_b64']), self.original)

    def test_changed_snapshot_code_is_rejected(self):
        envelope, _ = self.issue()
        snapshot = envelope.parent / 'code' / 'work.py'
        snapshot.chmod(0o600)
        snapshot.write_text("print('modified')\n")
        self.rejected(envelope)

    def test_missing_delivered_instruction_is_rejected(self):
        envelope, _ = self.issue()
        (envelope.parent / 'code' / 'instruction.bin').unlink()
        self.rejected(envelope)

    def test_child_cannot_be_issued_before_parent_execution(self):
        parent, _ = self.issue()
        parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
        with self.assertRaises(Exception):
            self.issue(parent=parent_hash)

    def test_successful_parent_can_issue_authenticated_child(self):
        parent, _ = self.issue()
        self.gate.dispatch(parent)
        parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
        child, _ = self.issue(parent=parent_hash)
        self.gate.verify(child)
        self.gate.dispatch(child)
        results = list(self.state.rglob('result.json'))
        self.assertEqual(len(results), 2)
        for path in results:
            self.assertEqual(base64.b64decode(json.loads(path.read_bytes())['instruction_b64']), self.original)
        child_output = self.state / 'results' / child.parent.name / 'output' / 'result.json'
        child_report = json.loads(child_output.read_bytes())
        self.assertEqual(child_report['handoff_id'], child.parent.name)
        self.assertEqual(child_report['parent_sha256'], parent_hash)
        self.assertEqual(child_report['assigned_operation'], 'Verify authoritative byte delivery')
        self.gate.audit_verify()

    def test_changed_parent_rejected_before_child_execution(self):
        parent, _ = self.issue()
        self.gate.dispatch(parent)
        parent_digest = hashlib.sha256(parent.read_bytes()).digest()
        child, _ = self.issue(parent=parent_digest.hex())
        def mutation(obj):
            payload = base64.b64decode(obj['payload_b64'])
            self.assertIn(parent_digest, payload)
            changed = bytes([parent_digest[0] ^ 1]) + parent_digest[1:]
            obj['payload_b64'] = base64.b64encode(payload.replace(parent_digest, changed, 1)).decode('ascii')
        self.rejected(self.altered(child, mutation))

    def test_unregistered_parent_rejected(self):
        with self.assertRaises(Exception):
            self.issue(parent='9' * 64)

    def test_worker_cannot_escape_execution_boundary(self):
        secret = self.state / 'acceptance-private-canary'
        secret.write_bytes(b'controller-private-canary')
        envelope, _ = self.issue(BOUNDARY_PROBE, argv=[str(secret)])
        self.gate.dispatch(envelope)
        result = json.loads(self.output('boundary.json').read_bytes())
        self.assertEqual(set(result), {'host_state_read', 'job_write', 'bootstrap_write',
                                      'network_inet', 'network_unix', 'subprocess', 'fork', 'libc_system'})
        for name, outcome in result.items():
            self.assertTrue(outcome['denied'], (name, outcome))
        self.assertEqual(secret.read_bytes(), b'controller-private-canary')

    def test_worker_cannot_exec_a_second_program(self):
        envelope, _ = self.issue(EXEC_PROBE)
        self.gate.dispatch(envelope)
        result = json.loads(self.output('exec-result.json').read_bytes())
        self.assertTrue(result['denied'])
        self.assertFalse(list(self.state.rglob('exec-escaped.txt')))

    def test_failed_worker_has_signed_exit_receipt_and_preserved_output(self):
        from gate3 import AUDIT_DOMAIN, canonical
        envelope, _ = self.issue(FAIL_WITH_OUTPUT)
        receipt = self.gate.dispatch(envelope)
        payload = receipt['payload']
        self.assertEqual(payload['event'], 'completed')
        self.assertEqual(payload['data']['exit_code'], 7)
        self.assertEqual(self.output('before-failure.txt').read_text(), 'retained output')
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        audit_public = base64.b64decode(self.gate.config['audit_public_b64'])
        Ed25519PublicKey.from_public_bytes(audit_public).verify(
            base64.b64decode(receipt['signature_b64']), AUDIT_DOMAIN + canonical(payload))
        with self.assertRaises(Exception):
            self.issue(parent=hashlib.sha256(envelope.read_bytes()).hexdigest())

    def test_timeout_is_enforced_and_recorded(self):
        envelope, _ = self.issue(TIMEOUT, limits={'timeout_seconds': 1, 'cpu_seconds': 2})
        started = time.monotonic()
        receipt = self.gate.dispatch(envelope)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(receipt['payload']['data']['exit_code'], 124)
        self.assertEqual(self.output('started.txt').read_text(), 'started')
        self.assertFalse(list(self.state.rglob('should-not-exist.txt')))

    def test_audit_edit_blocks_dispatch(self):
        envelope, _ = self.issue()
        path = self.state / 'audit.jsonl'
        original = path.read_bytes()
        self.assertIn(b'"event":"issued"', original)
        path.write_bytes(original.replace(b'"event":"issued"', b'"event":"forged"', 1))
        with self.assertRaises(Exception):
            self.gate.audit_verify()
        self.rejected(envelope)

    def test_audit_suffix_truncation_detected_by_checkpoint(self):
        envelope, _ = self.issue()
        path = self.state / 'audit.jsonl'
        original = path.read_bytes().splitlines(keepends=True)
        self.assertGreater(len(original), 1)
        path.write_bytes(b''.join(original[:-1]))
        with self.assertRaises(Exception):
            self.gate.audit_verify()
        self.rejected(envelope)

    def test_unsafe_outputs_cannot_authorize_a_child(self):
        source = "from pathlib import Path\nPath('/output/unsafe-link').symlink_to('/job/instruction.bin')\n"
        envelope, _ = self.issue(source)
        receipt = self.gate.dispatch(envelope)
        self.assertTrue(receipt['payload']['data']['unsafe_output'])
        with self.assertRaises(Exception):
            self.issue(parent=hashlib.sha256(envelope.read_bytes()).hexdigest())

    def retained_artifact_corruption(self, artifact, *, delete=False, symlink=False):
        envelope, _ = self.issue()
        receipt = self.gate.dispatch(envelope)
        handoff = receipt['payload']['data']['handoff_id']
        result = self.state / 'results' / handoff
        paths = {'receipt': self.state / 'receipts' / (handoff + '.json'),
                 'stdout': result / 'stdout.bin', 'stderr': result / 'stderr.bin',
                 'output': result / 'output' / 'result.json'}
        self.gate.audit_verify()
        if symlink:
            (result / 'output' / 'injected-link').symlink_to('/etc/passwd')
        else:
            target = paths[artifact]
            if delete:
                target.unlink()
            else:
                target.chmod(0o600)
                target.write_bytes(target.read_bytes() + b'changed-after-completion')
        with self.assertRaises(Exception):
            self.gate.audit_verify()
        before = set((self.state / 'jobs').iterdir())
        with self.assertRaises(Exception):
            self.issue(parent=hashlib.sha256(envelope.read_bytes()).hexdigest())
        self.assertEqual(set((self.state / 'jobs').iterdir()), before)

    def test_modified_detached_receipt_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('receipt')

    def test_deleted_detached_receipt_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('receipt', delete=True)

    def test_modified_stdout_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('stdout')

    def test_deleted_stdout_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('stdout', delete=True)

    def test_modified_stderr_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('stderr')

    def test_deleted_stderr_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('stderr', delete=True)

    def test_modified_output_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('output')

    def test_deleted_output_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('output', delete=True)

    def test_injected_output_symlink_invalidates_audit_and_parent(self):
        self.retained_artifact_corruption('output', symlink=True)

    def test_nonregular_job_inputs_rejected(self):
        job = self.base / 'symlink-job'
        job.mkdir()
        (job / 'work.py').write_text(BYTE_ECHO)
        (job / 'outside').symlink_to(self.private)
        with self.assertRaises(Exception):
            self.gate.create(job, 'work.py', 'Verify authoritative byte delivery')

    def test_controller_state_cannot_be_snapshotted_as_job_input(self):
        (self.state / 'work.py').write_text(BYTE_ECHO)
        (self.base / 'work.py').write_text(BYTE_ECHO)
        before = set((self.state / 'jobs').iterdir())
        with self.assertRaises(Exception):
            self.gate.create(self.state, 'work.py', 'Attempt to include controller state')
        with self.assertRaises(Exception):
            self.gate.create(self.base, 'work.py', 'Attempt to include controller ancestor')
        self.assertEqual(set((self.state / 'jobs').iterdir()), before)

    def test_wrong_trust_fingerprint_rejected_before_enrollment(self):
        result = self.cli('init', '--state', self.base / 'bad-state', '--spec', self.spec,
                          '--authority-key', self.private, '--expected-fingerprint', '0' * 64,
                          '--authorization', self.authorization, check=False)
        self.assertNotEqual(result.returncode, 0)

    def test_out_of_range_limits_rejected_before_snapshot(self):
        before = set((self.state / 'jobs').iterdir())
        for limits in [{'memory_mib': 63}, {'memory_mib': 8193},
                       {'timeout_seconds': 0}, {'timeout_seconds': 3601}]:
            with self.subTest(limits=limits):
                with self.assertRaises(Exception):
                    self.issue(limits=limits)
                self.assertEqual(set((self.state / 'jobs').iterdir()), before)

    def test_oversize_argument_list_rejected_before_snapshot(self):
        before = set((self.state / 'jobs').iterdir())
        with self.assertRaises(Exception):
            self.issue(argv=['argument'] * 129)
        self.assertEqual(set((self.state / 'jobs').iterdir()), before)

    def test_worker_visible_system_state_location_rejected(self):
        target = Path('/usr/lib') / self.base.name
        result = self.cli('init', '--state', target, '--spec', self.spec,
                          '--authority-key', self.private, '--expected-fingerprint', self.fingerprint,
                          '--authorization', self.authorization, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('worker-visible system mounts', result.stderr)
        self.assertFalse(target.exists())

    def test_worker_visible_system_private_key_location_rejected(self):
        result = self.cli('init', '--state', self.base / 'bad-key-state', '--spec', self.spec,
                          '--authority-key', Path('/usr/lib') / (self.base.name + '.key'),
                          '--expected-fingerprint', self.fingerprint,
                          '--authorization', self.authorization, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('worker-visible system mounts', result.stderr)
        self.assertFalse((self.base / 'bad-key-state').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
