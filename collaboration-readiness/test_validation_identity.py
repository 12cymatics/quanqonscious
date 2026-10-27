import base64
import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from validation_identity import digest_bytes, read_json, relative_path, validate_ownership, verify_signed_contract, verify_source_manifest


class ValidationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='validation-identity-fixture-')
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'input.txt').write_text('synthetic source fixture\n')
        self.openssl = shutil.which('openssl')
        self.assertIsNotNone(self.openssl)
        self.private_key = self.root / 'fixture-private.pem'
        self.public_key = self.root / 'fixture-public.pem'
        self.command([self.openssl, 'genpkey', '-algorithm', 'ED25519', '-out', str(self.private_key)])
        self.command([self.openssl, 'pkey', '-in', str(self.private_key), '-pubout', '-out', str(self.public_key)])
        self.public_key_sha256 = hashlib.sha256(self.public_key.read_bytes()).hexdigest()
        self.contract_path = self.root / 'fixture-contract.json'
        self.payload = {'fixture': True, 'generation': 1, 'request': 'Synthetic signature verification only'}
        self.sign()
        self.manifest = {'schema_version': 1, 'base_head': None, 'files': [{'path': 'input.txt', 'sha256': self.hash(self.source / 'input.txt')}], 'excluded_paths': [], 'external_assets': []}

    def command(self, argv, **kwargs):
        return subprocess.run(argv, check=True, capture_output=True, text=True, **kwargs)

    def hash(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def canonical(self, value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')

    def sign(self):
        payload_path = self.root / 'fixture-payload.bin'
        signature_path = self.root / 'fixture-signature.bin'
        payload_path.write_bytes(self.canonical(self.payload))
        self.command([self.openssl, 'pkeyutl', '-sign', '-rawin', '-inkey', str(self.private_key), '-in', str(payload_path), '-out', str(signature_path)])
        self.envelope = {'payload': copy.deepcopy(self.payload), 'signature': base64.b64encode(signature_path.read_bytes()).decode('ascii')}
        self.contract_path.write_text(json.dumps(self.envelope, indent=2) + '\n')

    def verify(self):
        return verify_signed_contract(self.contract_path, self.public_key, self.public_key_sha256)

    def test_ed25519_signed_contract(self):
        payload, digest = self.verify()
        self.assertEqual(payload, self.payload)
        self.assertEqual(digest, hashlib.sha256(self.canonical(self.payload)).hexdigest())

    def test_payload_tampering(self):
        self.envelope['payload']['generation'] = 2
        self.contract_path.write_text(json.dumps(self.envelope))
        with self.assertRaises(ValueError):
            self.verify()

    def test_signature_tampering(self):
        signature = bytearray(base64.b64decode(self.envelope['signature']))
        signature[0] ^= 1
        self.envelope['signature'] = base64.b64encode(signature).decode('ascii')
        self.contract_path.write_text(json.dumps(self.envelope))
        with self.assertRaises(ValueError):
            self.verify()

    def test_signature_not_base64(self):
        self.envelope['signature'] = '!invalid!'
        self.contract_path.write_text(json.dumps(self.envelope))
        with self.assertRaises(ValueError):
            self.verify()

    def test_pinned_public_key_mismatch(self):
        with self.assertRaises(ValueError):
            verify_signed_contract(self.contract_path, self.public_key, '0' * 64)

    def test_other_signing_key(self):
        other_private = self.root / 'other-private.pem'
        other_public = self.root / 'other-public.pem'
        self.command([self.openssl, 'genpkey', '-algorithm', 'ED25519', '-out', str(other_private)])
        self.command([self.openssl, 'pkey', '-in', str(other_private), '-pubout', '-out', str(other_public)])
        with self.assertRaises(ValueError):
            verify_signed_contract(self.contract_path, other_public, self.hash(other_public))

    def test_non_ed25519_key(self):
        other_private = self.root / 'ec-private.pem'
        other_public = self.root / 'ec-public.pem'
        self.command([self.openssl, 'genpkey', '-algorithm', 'EC', '-pkeyopt', 'ec_paramgen_curve:prime256v1', '-out', str(other_private)])
        self.command([self.openssl, 'pkey', '-in', str(other_private), '-pubout', '-out', str(other_public)])
        with self.assertRaises(ValueError):
            verify_signed_contract(self.contract_path, other_public, self.hash(other_public))

    def test_unknown_envelope_field(self):
        self.envelope['approval'] = True
        self.contract_path.write_text(json.dumps(self.envelope))
        with self.assertRaises(ValueError):
            self.verify()

    def test_duplicate_json_keys(self):
        for document in ['{"x":1,"x":2}', '{"outer":{"x":1,"x":2}}']:
            with self.subTest(document=document):
                self.contract_path.write_text(document)
                with self.assertRaises(ValueError):
                    read_json(self.contract_path)

    def test_nonfinite_json(self):
        for value in ['NaN', 'Infinity', '-Infinity', '1e999']:
            with self.subTest(value=value):
                self.contract_path.write_text('{"value":' + value + '}')
                with self.assertRaises(ValueError):
                    read_json(self.contract_path)

    def test_literal_relative_path(self):
        self.assertEqual(relative_path('src/equation.py'), pathlib.PurePosixPath('src/equation.py'))
        for path in ['', '.', '..', '/tmp/x', '../x', 'x/../y', 'x/./y', 'x//y', 'x/', 'x\\y', '*.py', 'x?', '[x]', 'x\ny', 'x\x00y', None, True]:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    relative_path(path)

    def test_source_manifest_binds_bytes(self):
        digest = verify_source_manifest(self.manifest, self.source)
        self.assertEqual(digest, hashlib.sha256(self.canonical(self.manifest)).hexdigest())
        self.assertEqual(digest_bytes(b'abc'), hashlib.sha256(b'abc').hexdigest())

    def test_modified_source_bytes(self):
        (self.source / 'input.txt').write_text('modified bytes\n')
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_unmanifested_untracked_file(self):
        (self.source / '.untracked').write_text('not in agreed manifest\n')
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_missing_manifest_file(self):
        (self.source / 'input.txt').unlink()
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_duplicate_manifest_file(self):
        self.manifest['files'].append(copy.deepcopy(self.manifest['files'][0]))
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_manifest_case_alias(self):
        self.manifest['files'].append({'path': 'INPUT.txt', 'sha256': self.hash(self.source / 'input.txt')})
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_manifest_hardlink_alias(self):
        os.link(self.source / 'input.txt', self.source / 'linked.txt')
        self.manifest['files'].append({'path': 'linked.txt', 'sha256': self.hash(self.source / 'linked.txt')})
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_source_symlink(self):
        (self.source / 'alias').symlink_to(self.source / 'input.txt')
        self.manifest['files'].append({'path': 'alias', 'sha256': self.hash(self.source / 'input.txt')})
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_excluded_symlink(self):
        (self.source / 'excluded').mkdir()
        (self.source / 'excluded' / 'alias').symlink_to(self.root / 'fixture-private.pem')
        self.manifest['excluded_paths'] = [{'path': 'excluded', 'reason': 'Explicit synthetic exclusion'}]
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_explicit_exclusion(self):
        (self.source / 'excluded.txt').write_text('explicitly outside fixture scope\n')
        self.manifest['excluded_paths'] = [{'path': 'excluded.txt', 'reason': 'Synthetic declared exclusion'}]
        self.assertEqual(verify_source_manifest(self.manifest, self.source), hashlib.sha256(self.canonical(self.manifest)).hexdigest())

    def test_exclusion_cannot_hide_included_source(self):
        self.manifest['excluded_paths'] = [{'path': 'input.txt', 'reason': 'Conflicting synthetic exclusion'}]
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_overlapping_exclusions(self):
        (self.source / 'excluded').mkdir()
        (self.source / 'excluded' / 'child.txt').write_text('excluded\n')
        self.manifest['excluded_paths'] = [{'path': 'excluded', 'reason': 'Fixture exclusion'}, {'path': 'excluded/child.txt', 'reason': 'Overlapping fixture exclusion'}]
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_external_asset_is_explicit_without_reading_it(self):
        self.manifest['external_assets'] = [{'id': 'external-fixture', 'location': 'fixture://not-a-real-private-store', 'sha256': 'a' * 64, 'reason': 'Not required by this synthetic calculation'}]
        self.assertEqual(verify_source_manifest(self.manifest, self.source), hashlib.sha256(self.canonical(self.manifest)).hexdigest())

    def test_unknown_manifest_field(self):
        self.manifest['ignore_missing'] = True
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_git_head_is_bound(self):
        env = dict(os.environ, GIT_AUTHOR_NAME='Synthetic Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid', GIT_COMMITTER_NAME='Synthetic Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        self.command(['git', 'init', '-q', str(self.source)], env=env)
        self.command(['git', '-C', str(self.source), 'add', 'input.txt'], env=env)
        self.command(['git', '-C', str(self.source), 'commit', '-q', '-m', 'synthetic fixture'], env=env)
        self.manifest['base_head'] = self.command(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], env=env).stdout.strip()
        verify_source_manifest(self.manifest, self.source)
        self.manifest['base_head'] = '0' * 40
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_non_git_source_cannot_invent_head(self):
        self.manifest['base_head'] = '0' * 40
        with self.assertRaises(ValueError):
            verify_source_manifest(self.manifest, self.source)

    def test_disjoint_ownership(self):
        self.assertTrue(validate_ownership([{'owner': 'a', 'write_paths': ['src/a.py']}, {'owner': 'b', 'write_paths': ['src/b.py']}], self.source))

    def test_parent_child_ownership(self):
        with self.assertRaises(ValueError):
            validate_ownership([{'owner': 'a', 'write_paths': ['src']}, {'owner': 'b', 'write_paths': ['src/b.py']}], self.source)

    def test_case_alias_ownership(self):
        (self.source / 'CaseOwned').mkdir()
        with self.assertRaises(ValueError):
            validate_ownership([{'owner': 'a', 'write_paths': ['CaseOwned']}, {'owner': 'b', 'write_paths': ['caseowned']}], self.source)

    def test_case_alias_parent_ownership(self):
        with self.assertRaises(ValueError):
            validate_ownership([{'owner': 'a', 'write_paths': ['Src']}, {'owner': 'b', 'write_paths': ['src/child.py']}], self.source)

    def test_hardlink_ownership(self):
        os.link(self.source / 'input.txt', self.source / 'linked.txt')
        with self.assertRaises(ValueError):
            validate_ownership([{'owner': 'a', 'write_paths': ['input.txt']}, {'owner': 'b', 'write_paths': ['linked.txt']}], self.source)

    def test_unicode_alias_ownership(self):
        with self.assertRaises(ValueError):
            validate_ownership([{'owner': 'a', 'write_paths': ['caf\u00e9/file']}, {'owner': 'b', 'write_paths': ['cafe\u0301/file']}], self.source)

    def test_ownership_symlink_escape(self):
        (self.source / 'outside').symlink_to(self.root)
        with self.assertRaises(ValueError):
            validate_ownership([{'owner': 'a', 'write_paths': ['outside/file']}], self.source)

    def test_invalid_assignment(self):
        for assignments in [[None], [{'owner': '', 'write_paths': ['x']}], [{'owner': 'a', 'write_paths': None}]]:
            with self.subTest(assignments=assignments):
                with self.assertRaises(ValueError):
                    validate_ownership(assignments, self.source)


if __name__ == '__main__':
    unittest.main()
