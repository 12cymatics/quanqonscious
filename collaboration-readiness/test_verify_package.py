import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from verify_package import verify


class PackageManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'active.py').write_text('VALUE = 17\n')
        self.manifest = {'active.py': hashlib.sha256((self.root / 'active.py').read_bytes()).hexdigest()}
        self.save()

    def save(self):
        (self.root / 'SHA256.json').write_text(json.dumps(self.manifest))

    def test_actual_matching_bytes(self):
        result = verify(self.root)
        self.assertEqual(result['package_integrity'], 'passed')
        self.assertEqual(result['files_checked'], 1)

    def test_modified_bytes_rejected(self):
        (self.root / 'active.py').write_text('VALUE = 0\n')
        with self.assertRaisesRegex(ValueError, 'Package bytes differ from recorded hashes'):
            verify(self.root)

    def test_missing_file_rejected(self):
        (self.root / 'active.py').unlink()
        with self.assertRaisesRegex(ValueError, 'Package inventory differs from manifest; missing='):
            verify(self.root)

    def test_unrecorded_file_rejected(self):
        (self.root / 'hidden-bypass.py').write_text('VALUE = 0\n')
        with self.assertRaisesRegex(ValueError, 'Package inventory differs from manifest'):
            verify(self.root)

    def test_duplicate_json_key_rejected(self):
        (self.root / 'SHA256.json').write_text('{"active.py":"' + self.manifest['active.py'] + '","active.py":"' + self.manifest['active.py'] + '"}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            verify(self.root)

    def test_symlink_rejected(self):
        (self.root / 'alias.py').symlink_to(self.root / 'active.py')
        with self.assertRaisesRegex(ValueError, 'Package contains a symlink'):
            verify(self.root)

    def test_hardlink_rejected(self):
        os.link(self.root / 'active.py', self.root / 'alias.py')
        with self.assertRaisesRegex(ValueError, 'Package contains a hard-linked file'):
            verify(self.root)

    def test_path_escape_rejected(self):
        self.manifest['../outside.py'] = 'a' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'Path must be canonical and relative'):
            verify(self.root)

    def test_self_hash_rejected(self):
        self.manifest['SHA256.json'] = 'a' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'Manifest cannot hash itself'):
            verify(self.root)

    def test_git_metadata_separate(self):
        (self.root / '.git').mkdir()
        (self.root / '.git' / 'HEAD').write_text('metadata\n')
        self.assertEqual(verify(self.root)['files_checked'], 1)


if __name__ == '__main__':
    unittest.main()
