import base64
import copy
import datetime
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import project_handoff
from test_handoff_fixtures import HandoffFixture
from validate_readiness import run_checks
from validation_identity import verify_source_manifest


class HandoffTests(HandoffFixture, unittest.TestCase):
    def test_valid_admission_does_not_write(self):
        result = self.admit()[0]
        self.assertEqual(result['state'], 'admitted')
        self.assertFalse(result['writes_performed'])
        self.assertFalse(result['publication_performed'])
        self.assertEqual(result['source_manifest_sha256'], self.registry['source_manifest_sha256'])
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_valid_handoff_binds_all_three_operations(self):
        result = self.validate()[0]
        self.assertEqual(result['handoff_sha256'], self.handoff_hash)
        self.assertEqual({item['operation'] for item in result['changes']}, {'create', 'replace', 'delete'})
        self.assertEqual(result['changes'], self.handoff['changes'])
        self.assertFalse(result['writes_performed'])

    def test_valid_create_replace_delete_and_current_acceptance(self):
        result = self.apply()
        self.assertEqual(result['state'], 'applied_requires_acceptance')
        self.assertTrue(result['writes_performed'])
        self.assertFalse(result['publication_performed'])
        self.assertEqual(self.bytes_of(self.source), self.bytes_of(self.candidate))
        self.assertFalse((self.source / 'src/delete.txt').exists())
        self.assertEqual((self.source / 'src/replace.txt').read_bytes(), b'candidate replacement bytes\n')
        self.assertEqual((self.source / 'src/new/deep/new.txt').read_bytes(), b'candidate created bytes\n')
        self.assertEqual((self.source / 'unowned.txt').read_bytes(), self.baseline_bytes['unowned.txt'])
        path, journal = self.journal()
        self.assertEqual(result['journal'], str(path))
        self.assertEqual(journal['state'], 'applied_requires_acceptance')
        self.assertEqual(journal['rollback_errors'], [])
        self.assertEqual(set(journal['applied_paths']), set(self.options['requested_paths']))
        for index, change in enumerate(journal['changes']):
            if change['operation'] != 'create':
                self.assertEqual((path.parent / ('before-' + str(index))).read_bytes(), self.baseline_bytes[change['path']])
        self.payload['generation'] += 1
        self.payload['source_manifest'] = copy.deepcopy(self.handoff['result_manifest'])
        contract_hash = self.sign(self.contract_path, self.payload, self.private)
        record, receipt = run_checks(contract_path=self.contract_path, public_key_path=self.public, public_key_sha256=self.digest(self.public), expected_contract_sha256=contract_hash, source_root=self.source, evidence_root=self.root / 'acceptance')
        self.assertEqual(record['task_state'], 'complete')
        self.assertEqual(receipt['subject_revision'], result['subject_revision'])
        self.assertEqual(set(receipt['executed_checks']), {'source_and_input_provenance', 'foundation_traceability', 'existing_aggregate_checks'})

    def test_duplicate_requested_path(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate requested path'):
            self.admit(requested_paths=self.options['requested_paths'] * 2)

    def test_empty_requested_paths(self):
        with self.assertRaisesRegex(ValueError, 'At least one requested path'):
            self.admit(requested_paths=[])

    def test_requested_path_is_unowned(self):
        with self.assertRaisesRegex(ValueError, 'outside assigned ownership: unowned.txt'):
            self.admit(requested_paths=['unowned.txt'])

    def test_requested_path_traversal(self):
        with self.assertRaisesRegex(ValueError, 'canonical and relative'):
            self.admit(requested_paths=['src/../unowned.txt'])

    def test_requested_write_cannot_enter_excluded_scope(self):
        excluded = [{'path': 'src/excluded', 'reason': 'Outside this signed task'}]
        self.payload['source_manifest']['excluded_paths'] = excluded
        self.handoff['result_manifest']['excluded_paths'] = copy.deepcopy(excluded)
        self.seal_all()
        with self.assertRaisesRegex(ValueError, 'Requested write overlaps excluded source'):
            self.admit(requested_paths=['src/excluded/planned.txt'])

    def test_wrong_worker(self):
        with self.assertRaisesRegex(ValueError, 'Worker does not own this assignment'):
            self.admit(worker='another-worker')

    def test_unknown_assignment(self):
        with self.assertRaisesRegex(ValueError, 'Unknown assignment'):
            self.admit(assignment_id='unknown')

    def test_wrong_registry_pin(self):
        with self.assertRaisesRegex(ValueError, 'independently pinned registry digest'):
            self.admit(expected_registry_sha256='0' * 64)

    def test_wrong_registry_generation(self):
        with self.assertRaisesRegex(ValueError, 'current trusted generation'):
            self.admit(registry_generation=2)

    def test_boolean_registry_generation(self):
        with self.assertRaisesRegex(ValueError, 'current trusted generation'):
            self.admit(registry_generation=True)

    def test_registry_expired(self):
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
        self.registry_rejected('Registry is future-dated or expired', lambda r: r.update(expires_at=past))

    def test_registry_future_issue(self):
        future = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)).isoformat()
        self.registry_rejected('Registry is future-dated or expired', lambda r: r.update(issued_at=future))

    def test_registry_timezone_missing(self):
        self.registry_rejected('Registry expiry requires a timezone', lambda r: r.update(expires_at='2099-01-01T00:00:00'))

    def test_assignment_expired(self):
        past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
        self.registry_rejected('Assignment is expired or outlives registry', lambda r: r['assignments'][0].update(expires_at=past))

    def test_assignment_outlives_registry(self):
        future = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3)).isoformat()
        self.registry_rejected('Assignment is expired or outlives registry', lambda r: r['assignments'][0].update(expires_at=future))

    def test_assignment_inactive(self):
        self.registry_rejected('Assignment is not active', lambda r: r['assignments'][0].update(state='expired'))

    def test_assignment_pending_operations(self):
        self.registry_rejected('unreconciled pending operations', lambda r: r['assignments'][0].update(pending_operations=['write-in-flight']))

    def test_registry_unassigned_template(self):
        self.registry_rejected('Assigned production ownership is required', lambda r: r.update(assignments=[]))

    def test_registry_wrong_integrator(self):
        self.registry_rejected('not the selected integrator', lambda r: r.update(writer='record-author'))

    def test_registry_wrong_task(self):
        self.registry_rejected('Registry contract identity', lambda r: r.update(task_id='unrelated-task'))

    def test_registry_stale_contract(self):
        self.registry_rejected('Registry contract identity', lambda r: r.update(contract_sha256='0' * 64))

    def test_registry_stale_source(self):
        self.registry_rejected('Registry source identity', lambda r: r.update(source_manifest_sha256='0' * 64))

    def test_registry_unauthorized_action(self):
        self.registry_rejected('task-bound project-change authorization', lambda r: r['authorization'].update(action='claim_success'))

    def test_registry_wrong_request_binding(self):
        self.registry_rejected('task-bound project-change authorization', lambda r: r['authorization'].update(user_request_sha256='0' * 64))

    def test_registry_duplicate_assignment(self):
        self.registry_rejected('Duplicate or missing assignment identity', lambda r: r['assignments'].append(copy.deepcopy(r['assignments'][0])))

    def test_registry_ownership_cannot_expand_contract(self):
        self.registry_rejected('Registry ownership differs', lambda r: r['assignments'][0].update(write_paths=['src', 'unowned.txt']))

    def test_registry_duplicate_exclusive_resource(self):
        self.registry_rejected('Exclusive resource has duplicate or unknown ownership', lambda r: r['exclusive_resources'].append(copy.deepcopy(r['exclusive_resources'][0])))

    def test_registry_unknown_resource_owner(self):
        self.registry_rejected('Exclusive resource has duplicate or unknown ownership', lambda r: r['exclusive_resources'][0].update(assignment_id='unknown'))

    def test_registry_unknown_resource_kind(self):
        self.registry_rejected('Unknown or missing exclusive resource', lambda r: r['exclusive_resources'][0].update(kind='everything'))

    def test_registry_signature_tampered(self):
        envelope = json.loads(self.registry_path.read_text())
        signature = bytearray(base64.b64decode(envelope['signature']))
        signature[0] ^= 1
        envelope['signature'] = base64.b64encode(signature).decode('ascii')
        self.registry_path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, 'signature verification failed'):
            self.admit()

    def test_registry_signed_by_untrusted_key(self):
        self.sign(self.registry_path, self.registry, self.worker_private)
        with self.assertRaisesRegex(ValueError, 'signature verification failed'):
            self.admit()

    def test_handoff_signature_tampered(self):
        envelope = json.loads(self.handoff_path.read_text())
        envelope['payload']['outcome'] = 'already_integrated'
        self.handoff_path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, 'signature verification failed'):
            self.validate()

    def test_handoff_wrong_worker_public_key(self):
        with self.assertRaisesRegex(ValueError, 'externally pinned identity'):
            self.validate(worker_public_key_path=self.public)

    def test_handoff_stale_contract_generation(self):
        self.handoff_rejected('Handoff identity mismatch: contract_generation', lambda h: h.update(contract_generation=0))

    def test_handoff_stale_registry_generation(self):
        self.handoff_rejected('Handoff identity mismatch: registry_generation', lambda h: h.update(registry_generation=0))

    def test_handoff_wrong_registry_digest(self):
        self.handoff_rejected('Handoff identity mismatch: registry_sha256', lambda h: h.update(registry_sha256='0' * 64))

    def test_handoff_wrong_base(self):
        self.handoff_rejected('Handoff base source is stale', lambda h: h.update(base_manifest_sha256='0' * 64))

    def test_handoff_wrong_worker_identity(self):
        self.handoff_rejected('Handoff identity mismatch: worker', lambda h: h.update(worker='another-worker'))

    def test_handoff_pending_operation(self):
        self.handoff_rejected('Handoff has unresolved work or operations', lambda h: h.update(pending_operations=['write-in-flight']))

    def test_handoff_failed_outcome(self):
        self.handoff_rejected('Handoff has unresolved work or operations', lambda h: h.update(outcome='failed'))

    def test_handoff_empty_declared_delta(self):
        self.handoff_rejected('Declared handoff changes differ', lambda h: h.update(changes=[]))

    def test_handoff_mismatched_declared_delta(self):
        self.handoff_rejected('Declared handoff changes differ', lambda h: h['changes'][0].update(before_sha256='0' * 64))

    def test_handoff_duplicate_declared_delta(self):
        self.handoff_rejected('Declared handoff changes differ', lambda h: h['changes'].append(copy.deepcopy(h['changes'][0])))

    def test_handoff_empty_actual_delta_cannot_claim_integration(self):
        (self.candidate / 'src/replace.txt').write_bytes(self.baseline_bytes['src/replace.txt'])
        (self.candidate / 'src/delete.txt').write_bytes(self.baseline_bytes['src/delete.txt'])
        (self.candidate / 'src/new/deep/new.txt').unlink()
        self.handoff['result_manifest'] = copy.deepcopy(self.payload['source_manifest'])
        self.handoff['changes'] = []
        self.seal_handoff()
        with self.assertRaisesRegex(ValueError, 'Declared handoff changes differ'):
            self.validate()

    def test_handoff_cannot_change_excluded_scope(self):
        self.handoff['result_manifest']['excluded_paths'] = [{'path': 'other-inputs', 'reason': 'Unapproved scope change'}]
        self.seal_handoff()
        with self.assertRaisesRegex(ValueError, 'Handoff changes source scope or repository base: excluded_paths'):
            self.validate()

    def test_requested_paths_must_equal_delta(self):
        with self.assertRaisesRegex(ValueError, 'Requested paths do not equal actual handoff changes'):
            self.validate(requested_paths=['src/replace.txt'])

    def test_candidate_must_be_separate_from_source(self):
        with self.assertRaisesRegex(ValueError, 'Candidate and live source must be separate trees'):
            self.validate(artifact_root=self.source)

    def test_candidate_actual_bytes_changed(self):
        (self.candidate / 'src/replace.txt').write_bytes(b'unapproved candidate bytes\n')
        with self.assertRaisesRegex(ValueError, 'Source file digest mismatch: src/replace.txt'):
            self.validate()

    def test_candidate_unmanifested_file(self):
        (self.candidate / 'src/unmanifested.txt').write_bytes(b'omitted input\n')
        with self.assertRaisesRegex(ValueError, 'Source files are not accounted for'):
            self.validate()

    def test_candidate_unowned_change(self):
        (self.candidate / 'unowned.txt').write_bytes(b'not permitted by assignment\n')
        self.handoff['result_manifest'] = self.manifest(self.candidate)
        self.handoff['changes'] = project_handoff.manifest_delta(self.payload['source_manifest'], self.handoff['result_manifest'])
        self.options['requested_paths'] = [item['path'] for item in self.handoff['changes']]
        self.seal_handoff()
        with self.assertRaisesRegex(ValueError, 'outside assigned ownership: unowned.txt'):
            self.validate()

    def test_candidate_new_git_head_rejected(self):
        self.command(['git', '-C', str(self.candidate), 'add', '.'])
        self.command(['git', '-C', str(self.candidate), '-c', 'commit.gpgsign=false', 'commit', '-q', '-m', 'Different repository base'])
        self.handoff['result_manifest']['base_head'] = self.command(['git', '-C', str(self.candidate), 'rev-parse', 'HEAD']).stdout.strip()
        self.seal_handoff()
        with self.assertRaisesRegex(ValueError, 'Handoff changes source scope or repository base: base_head'):
            self.validate()

    def test_live_source_changed_since_dispatch(self):
        (self.source / 'src/replace.txt').write_bytes(b'new user edit must not be overwritten\n')
        with self.assertRaisesRegex(ValueError, 'Source file digest mismatch: src/replace.txt'):
            self.apply()
        self.assertEqual((self.source / 'src/replace.txt').read_bytes(), b'new user edit must not be overwritten\n')

    def test_case_alias_ownership_rejected_on_actual_filesystem(self):
        (self.source / 'CaseOwned').mkdir()
        self.payload['ownership']['assignments'] = [{'owner': 'worker', 'write_paths': ['CaseOwned']}, {'owner': 'worker-b', 'write_paths': ['caseowned']}]
        self.contract_hash = self.sign(self.contract_path, self.payload, self.private)
        self.options['expected_contract_sha256'] = self.contract_hash
        with self.assertRaisesRegex(ValueError, 'case/Unicode ownership collision'):
            self.admit(requested_paths=['CaseOwned/new.txt'])

    def test_case_alias_planned_descendant_rejected(self):
        self.payload['ownership']['assignments'] = [{'owner': 'worker', 'write_paths': ['Future']}, {'owner': 'worker-b', 'write_paths': ['future/new.txt']}]
        self.contract_hash = self.sign(self.contract_path, self.payload, self.private)
        self.options['expected_contract_sha256'] = self.contract_hash
        with self.assertRaisesRegex(ValueError, 'case/Unicode ownership collision'):
            self.admit(requested_paths=['Future/new.txt'])

    def test_actual_samefile_hardlink_is_not_independent_ownership(self):
        first = self.source / 'src/replace.txt'
        second = self.source / 'hardlink.txt'
        os.link(first, second)
        self.assertTrue(first.samefile(second))
        with self.assertRaisesRegex(ValueError, 'Hard-linked source or ownership path'):
            self.admit()

    def test_valid_signed_ownership_transfer(self):
        self.transfer()
        result = self.validate()[0]
        self.assertEqual(result['worker'], 'replacement-worker')
        self.assertEqual(result['registry_generation'], 2)

    def test_transfer_requires_prior_signed_registry(self):
        self.transfer()
        self.options.pop('previous_registry')
        with self.assertRaisesRegex(ValueError, 'requires the previous signed registry'):
            self.admit()

    def test_transfer_requires_reconciliation(self):
        self.transfer()
        self.registry_rejected('lacks signed reconciliation', lambda r: r.update(reconciliations=[]))

    def test_transfer_pending_operations_rejected(self):
        self.transfer()
        self.registry_rejected('Transfer has unreconciled pending operations', lambda r: r['reconciliations'][0].update(pending_operations=['pending-write']))

    def test_transfer_reconciliation_stale_source(self):
        self.transfer()
        self.registry_rejected('Reconciliation source is stale', lambda r: r['reconciliations'][0].update(final_source_manifest_sha256='0' * 64))

    def test_transfer_reconciliation_wrong_owner(self):
        self.transfer()
        self.registry_rejected('Reconciliation differs from prior ownership', lambda r: r['reconciliations'][0].update(owner='replacement-worker'))

    def test_transfer_previous_digest_mismatch(self):
        self.transfer()
        self.registry_rejected('Previous registry digest mismatch', lambda r: r.update(previous_registry_sha256='0' * 64))

    def test_transfer_rejects_unknown_previous_schema(self):
        path, previous = self.transfer()
        previous['schema_version'] = 999
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, '[Pp]revious (coordination )?registry|[Hh]istorical registry'):
            self.admit()

    def test_transfer_rejects_malformed_previous_registry(self):
        path, previous = self.transfer()
        previous.pop('authorization')
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, '[Pp]revious (coordination )?registry|[Hh]istorical registry'):
            self.admit()

    def test_transfer_rejects_duplicate_previous_assignment(self):
        path, previous = self.transfer()
        previous['assignments'].append(copy.deepcopy(previous['assignments'][0]))
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, '[Pp]revious.*assignment|[Hh]istorical.*assignment'):
            self.admit()

    def test_valid_reconciliation_of_expired_previous_registry(self):
        path, previous = self.transfer()
        now = datetime.datetime.now(datetime.timezone.utc)
        previous['issued_at'] = (now - datetime.timedelta(days=3)).isoformat()
        previous['expires_at'] = (now - datetime.timedelta(days=1)).isoformat()
        previous['assignments'][0]['expires_at'] = (now - datetime.timedelta(days=2)).isoformat()
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.seal_registry()
        self.assertEqual(self.validate()[0]['state'], 'handoff_validated')

    def test_unchanged_owner_must_reconcile_prior_pending_operations(self):
        previous = copy.deepcopy(self.registry)
        previous['assignments'][0]['pending_operations'] = ['uncertain-write']
        path = self.root / 'previous-registry.json'
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.registry['generation'] = 2
        self.options.update(registry_generation=2, previous_registry=path)
        self.handoff['registry_generation'] = 2
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, 'lacks signed reconciliation'):
            self.admit()
        self.registry['reconciliations'] = [{'assignment_id': 'assignment-1', 'owner': 'worker', 'write_paths': ['src'], 'outcome': 'reconciled', 'pending_operations': [], 'final_source_manifest_sha256': self.registry['source_manifest_sha256']}]
        self.seal_registry()
        self.assertEqual(self.admit()[0]['state'], 'admitted')

    def test_resource_transfer_requires_reconciliation_even_if_file_owner_unchanged(self):
        previous = copy.deepcopy(self.registry)
        path = self.root / 'previous-registry.json'
        previous_hash = self.sign(path, previous, self.private)
        other_private, other_public = self.keypair('other-owner')
        self.payload['generation'] += 1
        self.payload['ownership']['assignments'].append({'owner': 'other-worker', 'write_paths': ['other']})
        self.registry['assignments'].append({'id': 'assignment-2', 'owner': 'other-worker', 'write_paths': ['other'], 'state': 'active', 'expires_at': self.assignment_expiry, 'pending_operations': [], 'worker_public_key_sha256': self.digest(other_public)})
        self.registry['exclusive_resources'][0]['assignment_id'] = 'assignment-2'
        self.registry.update(generation=2, previous_registry_sha256=previous_hash)
        self.options.update(registry_generation=2, previous_registry=path)
        self.handoff['registry_generation'] = 2
        self.seal_all()
        with self.assertRaisesRegex(ValueError, 'lacks signed reconciliation'):
            self.admit()
        self.registry['reconciliations'] = [{'assignment_id': 'assignment-1', 'owner': 'worker', 'write_paths': ['src'], 'outcome': 'reconciled', 'pending_operations': [], 'final_source_manifest_sha256': self.registry['source_manifest_sha256']}]
        self.seal_registry()
        self.assertEqual(self.admit()[0]['state'], 'admitted')

    def test_prior_assignment_cannot_continue_after_transfer(self):
        old_public = self.worker_public
        self.transfer()
        with self.assertRaisesRegex(ValueError, 'Worker does not own this assignment'):
            self.admit(worker='worker')
        with self.assertRaisesRegex(ValueError, 'externally pinned identity'):
            self.validate(worker_public_key_path=old_public)

    def test_failure_before_second_write_rolls_back_first(self):
        original = project_handoff.replace_at
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise OSError('injected second-write failure')
            return original(*args, **kwargs)
        with patch.object(project_handoff, 'replace_at', side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, 'Integration failed; rolled_back'):
                self.apply()
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)
        path, journal = self.journal()
        self.assertEqual(journal['state'], 'rolled_back')
        self.assertIn('injected second-write failure', journal['error'])
        self.assertEqual(journal['rollback_errors'], [])

    def test_failure_after_real_write_does_not_leave_it_untracked(self):
        original = project_handoff.replace_at
        calls = []
        def fail_after_first(*args, **kwargs):
            calls.append(args[1])
            result = original(*args, **kwargs)
            if len(calls) == 1:
                raise OSError('injected failure after filesystem mutation')
            return result
        with patch.object(project_handoff, 'replace_at', side_effect=fail_after_first):
            with self.assertRaisesRegex(ValueError, 'Integration failed; rolled_back'):
                self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)
        self.assertEqual(self.journal()[1]['state'], 'rolled_back')

    def test_special_permission_bits_on_candidate_create_rejected(self):
        (self.candidate / 'src/new/deep/new.txt').chmod(0o4755)
        with self.assertRaisesRegex(ValueError, '[Ss]pecial permission'):
            self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_new_file_executable_mode_rejected(self):
        (self.candidate / 'src/new/deep/new.txt').chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'Permission changes are not supported'):
            self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_new_file_mode_is_deterministic_and_existing_mode_preserved(self):
        (self.source / 'src/replace.txt').chmod(0o660)
        (self.candidate / 'src/replace.txt').chmod(0o660)
        self.apply()
        self.assertEqual((self.source / 'src/new/deep/new.txt').stat().st_mode & 0o7777, 0o644)
        self.assertEqual((self.source / 'src/replace.txt').stat().st_mode & 0o7777, 0o660)

    def test_existing_file_permission_change_is_rejected(self):
        (self.candidate / 'src/replace.txt').chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'Permission changes are not supported'):
            self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_concurrent_edit_between_delete_read_and_unlink_is_preserved(self):
        original = project_handoff.read_at
        reads = []
        def concurrent_edit(root, relative):
            result = original(root, relative)
            if root == self.source and relative == 'src/delete.txt':
                reads.append(relative)
                if len(reads) == 2:
                    (self.source / relative).write_bytes(b'concurrent user edit must survive\n')
            return result
        with patch.object(project_handoff, 'read_at', side_effect=concurrent_edit):
            with self.assertRaisesRegex(ValueError, 'Integration failed; failed_requires_reconciliation'):
                self.apply()
        self.assertEqual((self.source / 'src/delete.txt').read_bytes(), b'concurrent user edit must survive\n')
        self.assertEqual(self.journal()[1]['state'], 'failed_requires_reconciliation')

    def test_rollback_failure_remains_visible_and_blocks_next_write(self):
        original = project_handoff.replace_at
        calls = []
        def fail_write_and_rollback(*args, **kwargs):
            calls.append(args[1])
            if len(calls) >= 2:
                raise PermissionError('injected write and rollback denial')
            return original(*args, **kwargs)
        with patch.object(project_handoff, 'replace_at', side_effect=fail_write_and_rollback):
            with self.assertRaisesRegex(ValueError, 'failed_requires_reconciliation'):
                self.apply()
        path, journal = self.journal()
        self.assertEqual(journal['state'], 'failed_requires_reconciliation')
        self.assertTrue(journal['rollback_errors'])
        self.assertNotEqual(self.bytes_of(self.source), self.baseline_bytes)
        with self.assertRaisesRegex(ValueError, 'Previous integration is unresolved'):
            self.apply()

    def test_expiry_between_validation_and_write_rolls_back(self):
        original = project_handoff.validate_handoff
        clock = [datetime.datetime.now(datetime.timezone.utc)]
        def validate_and_expire(**options):
            result = original(**options)
            clock[0] = datetime.datetime.fromisoformat(self.assignment_expiry) + datetime.timedelta(seconds=1)
            return result
        with patch.object(project_handoff, 'current_time', side_effect=lambda: clock[0]), patch.object(project_handoff, 'validate_handoff', side_effect=validate_and_expire):
            with self.assertRaisesRegex(ValueError, 'Ownership expired before write'):
                self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)
        self.assertEqual(self.journal()[1]['state'], 'rolled_back')

    def test_registry_revocation_between_writes_rolls_back(self):
        original = project_handoff.replace_at
        calls = []
        def revoke_after_first(*args, **kwargs):
            calls.append(args[1])
            result = original(*args, **kwargs)
            if len(calls) == 1:
                revoked = copy.deepcopy(self.registry)
                revoked['assignments'][0]['state'] = 'revoked'
                self.sign(self.registry_path, revoked, self.private)
            return result
        with patch.object(project_handoff, 'replace_at', side_effect=revoke_after_first):
            with self.assertRaisesRegex(ValueError, 'Registry changed or assignment was revoked'):
                self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)
        self.assertEqual(self.journal()[1]['state'], 'rolled_back')

    def test_contract_replacement_between_writes_rolls_back(self):
        original = project_handoff.replace_at
        calls = []
        def revoke_after_first(*args, **kwargs):
            calls.append(args[1])
            result = original(*args, **kwargs)
            if len(calls) == 1:
                replaced = copy.deepcopy(self.payload)
                replaced['generation'] += 1
                self.sign(self.contract_path, replaced, self.private)
            return result
        with patch.object(project_handoff, 'replace_at', side_effect=revoke_after_first):
            with self.assertRaisesRegex(ValueError, 'Task contract changed before write'):
                self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_external_process_lock_excludes_second_integrator(self):
        lock = self.source.parent / ('.' + self.source.name + '.project-write.lock')
        script = "import fcntl, os, sys\nfd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)\nfcntl.flock(fd, fcntl.LOCK_EX)\nprint('locked', flush=True)\nsys.stdin.read(1)\nos.close(fd)\n"
        process = subprocess.Popen([sys.executable, '-c', script, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.environment)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'locked')
            with self.assertRaisesRegex(ValueError, 'Another integrator holds the source write lock'):
                self.apply()
        finally:
            stdout, stderr = process.communicate('x', timeout=10)
            self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_unresolved_lock_marker_blocks_new_integration(self):
        lock = self.source.parent / ('.' + self.source.name + '.project-write.lock')
        lock.write_text(json.dumps({'state': 'applying', 'journal': str(self.root / 'prior-journal.json')}))
        with self.assertRaisesRegex(ValueError, 'Previous integration is unresolved'):
            self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_terminal_lock_marker_cannot_hide_unfinished_journal(self):
        journal = self.root / 'previous-journal.json'
        journal.write_text(json.dumps({'state': 'applying', 'source_root': str(self.source)}))
        lock = self.source.parent / ('.' + self.source.name + '.project-write.lock')
        lock.write_text(json.dumps({'state': 'rolled_back', 'journal': str(journal)}))
        with self.assertRaisesRegex(ValueError, 'Previous integration lock and journal disagree'):
            self.apply()
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_journal_cannot_be_written_into_source(self):
        with self.assertRaisesRegex(ValueError, 'Journal must be outside source and candidate'):
            project_handoff.apply_handoff(journal_root=self.source / 'journal', **self.options, **self.handoff_options)
        self.assertEqual(self.bytes_of(self.source), self.baseline_bytes)

    def test_journal_cannot_be_written_into_candidate(self):
        with self.assertRaisesRegex(ValueError, 'Journal must be outside source and candidate'):
            project_handoff.apply_handoff(journal_root=self.candidate / 'journal', **self.options, **self.handoff_options)

    def test_repeated_application_cannot_reuse_stale_base(self):
        self.apply()
        after = self.bytes_of(self.source)
        with self.assertRaisesRegex(ValueError, 'Manifest includes missing|Source file digest mismatch|Source files are not accounted'):
            self.apply()
        self.assertEqual(self.bytes_of(self.source), after)
        verify_source_manifest(self.handoff['result_manifest'], self.source)

    def test_contract_generation_cannot_move_backwards(self):
        path, previous = self.transfer()
        previous['contract_generation'] = self.registry['contract_generation'] + 1
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, 'Contract generation cannot move backwards'):
            self.validate()

    def test_contract_bytes_cannot_change_with_same_generation(self):
        path, previous = self.transfer()
        previous['contract_generation'] = self.registry['contract_generation']
        self.registry['previous_registry_sha256'] = self.sign(path, previous, self.private)
        self.seal_registry()
        with self.assertRaisesRegex(ValueError, 'Contract bytes changed without a new contract generation'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
