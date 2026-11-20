import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import gate
import snapshot
class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.area = tempfile.TemporaryDirectory()
        cls.root = Path(cls.area.name)
        cls.key = cls.root / 'execution.key'
        cls.pin = gate.init_authority(cls.root / 'trust', cls.key, Path(__file__).parent / '.lake/build/bin/bytegate')
        cls.authority = gate.Authority(cls.root / 'trust', cls.pin)
    @classmethod
    def tearDownClass(cls):
        cls.area.cleanup()
    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.root)
        self.path = Path(self.case.name)
        self.marker = self.path / 'executed.txt'
        self.script = self.path / 'work.py'
        self.script.write_text('from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text("original")\nprint("real execution")\n')
        self.journal = self.path / 'journal'
        self.raw = self.make()
    def tearDown(self):
        self.case.cleanup()
    def make(self, parent=None, identifier=None):
        return gate.create_envelope(self.authority, self.key, b'Execute the actual marker-writing program', [sys.executable, '-I', '-S', '-B', str(self.script), str(self.marker)], [4], parent, str(self.path), identifier=identifier)
    def run_gate(self, raw=None, parent=None, head=gate.GENESIS):
        raw = self.raw if raw is None else raw
        return gate.dispatch(self.authority, self.key, raw, gate.sha(raw), parent, self.journal, head)
    def resign(self, mutate):
        value = gate.parse(self.raw)['payload']
        mutate(value)
        return gate.signed(self.authority.signer(self.key), value)
    def rejected(self, raw=None, parent=None, head=gate.GENESIS):
        with self.assertRaises(Exception):
            self.run_gate(raw, parent, head)
        self.assertFalse(self.marker.exists())
    def test_root_executes_and_signs_receipts(self):
        result = self.run_gate()
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['stdout'], b'real execution\n')
        self.assertEqual(self.marker.read_text(), 'original')
        journal = gate.Journal(self.journal, self.authority)
        try:
            _, accepted, completed, receipts = journal.read(result['head'])
            self.assertEqual((len(accepted), len(completed), len(receipts)), (1, 1, 2))
            self.assertEqual(receipts[-1]['stdout_sha256'], gate.sha(result['stdout']))
        finally:
            journal.close()
    def test_child_requires_actual_accepted_parent(self):
        child = self.make(gate.sha(self.raw))
        self.rejected(child, gate.sha(self.raw))
        root_result = self.run_gate()
        self.marker.unlink()
        child_result = self.run_gate(child, gate.sha(self.raw), root_result['head'])
        self.assertEqual(child_result['exit_code'], 0)
        self.assertTrue(self.marker.exists())
    def test_changed_instruction_even_with_valid_signature_rejected(self):
        def mutate(value):
            value['sources'][1] = gate.source('verified_delegation_instruction.txt', b'Changed instructions')
        self.rejected(self.resign(mutate))
    def test_missing_original_rejected(self):
        self.rejected(self.resign(lambda value: value.update(sources=[])))
    def test_changed_operation_without_signature_rejected(self):
        value = gate.parse(self.raw)
        value['payload']['operation'] = gate.source('assigned_operation.txt', b'Changed task')
        self.rejected(gate.canonical(value) + b'\n')
    def test_changed_command_without_signature_rejected(self):
        value = gate.parse(self.raw)
        value['payload']['command'].append('changed')
        self.rejected(gate.canonical(value) + b'\n')
    def test_wrong_expected_digest_rejected(self):
        with self.assertRaisesRegex(ValueError, 'envelope digest'):
            gate.dispatch(self.authority, self.key, self.raw, 'f' * 64, None, self.journal, gate.GENESIS)
        self.assertFalse(self.marker.exists())
    def test_missing_external_head_rejected(self):
        with self.assertRaisesRegex(ValueError, 'journal head'):
            self.run_gate(head=None)
        self.assertFalse(self.marker.exists())
    def test_changed_parent_rejected(self):
        self.rejected(self.resign(lambda value: value.update(parent='f' * 64)))
    def test_invalid_signature_rejected(self):
        value = gate.parse(self.raw)
        value['signature'] = gate.b64(bytes(64))
        self.rejected(gate.canonical(value) + b'\n')
    def test_noncanonical_envelope_rejected(self):
        self.rejected(json.dumps(gate.parse(self.raw), indent=2).encode() + b'\n')
    def test_duplicate_json_key_rejected(self):
        self.rejected(self.raw[:-2] + b',"signature":"duplicate"}\n')
    def test_replay_rejected_after_real_execution(self):
        first = self.run_gate()
        self.marker.unlink()
        self.rejected(head=first['head'])
    def test_resigned_duplicate_id_rejected(self):
        first = self.run_gate()
        identifier = gate.parse(self.raw)['payload']['id']
        new = self.make(identifier=identifier)
        self.marker.unlink()
        self.rejected(new, head=first['head'])
    def test_modified_script_rejected_before_execution(self):
        self.script.write_text('raise RuntimeError("changed")\n')
        self.rejected()
    def test_sealed_script_is_actual_executed_script(self):
        original = gate.lean_accept
        def concurrent_change(*args):
            original(*args)
            self.script.write_text('raise RuntimeError("path replaced after validation")\n')
        with patch.object(gate, 'lean_accept', side_effect=concurrent_change):
            result = self.run_gate()
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(self.marker.read_text(), 'original')
    def test_seals_prevent_mutating_executed_bytes(self):
        fd = gate.sealed_file(b'bound', 'adversarial-write')
        try:
            with self.assertRaises(PermissionError):
                os.write(fd, b'evil')
            with self.assertRaises(PermissionError):
                os.ftruncate(fd, 0)
        finally:
            os.close(fd)
    def test_journal_truncation_rejected_by_external_head(self):
        result = self.run_gate()
        self.marker.unlink()
        (self.journal / 'receipts.jsonl').write_bytes(b'')
        self.rejected(head=result['head'])
    def test_receipt_tamper_rejected(self):
        result = self.run_gate()
        self.marker.unlink()
        path = self.journal / 'receipts.jsonl'
        rows = path.read_bytes().splitlines(keepends=True)
        value = gate.parse(rows[-1])
        value['payload']['exit_code'] = 7
        rows[-1] = gate.canonical(value) + b'\n'
        path.write_bytes(b''.join(rows))
        self.rejected(head=result['head'])
    def test_wrong_authority_pin_rejected(self):
        with self.assertRaisesRegex(ValueError, 'public key'):
            gate.Authority(self.root / 'trust', 'f' * 64)
    def test_lean_rejects_false_claims(self):
        valid = {'source': [65], 'expected': [65], 'claimedLength': 1, 'signatureOK': True, 'digestOK': True, 'canonicalOK': True, 'filesOK': True, 'id': 'id', 'parent': None, 'expectedParent': None, 'used': [], 'parents': []}
        mutations = [{'source': [66]}, {'claimedLength': 2}, {'signatureOK': False}, {'digestOK': False}, {'canonicalOK': False}, {'filesOK': False}, {'used': ['id']}, {'parent': 'other'}, {'parent': 'p', 'expectedParent': 'p'}]
        binary = str(Path(__file__).parent / '.lake/build/bin/bytegate')
        for mutation in [{}] + mutations:
            with self.subTest(mutation=mutation):
                value = valid | mutation
                result = subprocess.run([binary], input=gate.canonical(value), capture_output=True)
                self.assertEqual(result.returncode, 1 if mutation else 0)
    def test_failed_command_is_consumed(self):
        self.script.write_text('import sys\nsys.exit(9)\n')
        raw = self.make()
        result = self.run_gate(raw)
        self.assertEqual(result['exit_code'], 9)
        self.rejected(raw, head=result['head'])
    def test_source_snapshot_preserves_real_relative_imports(self):
        source = self.path / 'source'
        source.mkdir()
        (source / 'adjacent.py').write_text('VALUE="computed through actual adjacent import"\n')
        (source / 'main.py').write_text('from adjacent import VALUE\nfrom pathlib import Path\nPath("output/result.txt").write_text(VALUE)\nprint(VALUE)\n')
        manifest = snapshot.create(source, ['main.py', 'adjacent.py'], self.path / 'snapshot', ['output'])
        raw = gate.create_envelope(self.authority, self.key, b'Execute the actual source snapshot', [sys.executable, '-S', '-B', str(self.path / 'snapshot/main.py')], snapshot=manifest, snapshot_mode='trusted-host-readonly')
        result = self.run_gate(raw)
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual((self.path / 'snapshot/output/result.txt').read_text(), 'computed through actual adjacent import')
    def test_source_snapshot_tamper_prevents_execution(self):
        source = self.path / 'source'
        source.mkdir()
        (source / 'main.py').write_text('print("original")\n')
        manifest = snapshot.create(source, ['main.py'], self.path / 'snapshot')
        raw = gate.create_envelope(self.authority, self.key, b'Execute the actual source snapshot', [sys.executable, '-S', '-B', str(self.path / 'snapshot/main.py')], snapshot=manifest, snapshot_mode='trusted-host-readonly')
        script = self.path / 'snapshot/main.py'
        script.chmod(0o644)
        script.write_text('raise RuntimeError("altered")\n')
        script.chmod(0o444)
        self.rejected(raw)
    def test_snapshot_mode_change_is_rejected(self):
        value = gate.parse(self.raw)
        value['payload']['snapshot_mode'] = 'trusted-host-readonly'
        self.rejected(gate.canonical(value) + b'\n')
if __name__ == '__main__':
    unittest.main(verbosity=2)
