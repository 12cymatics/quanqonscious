import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from validation_identity import _key, _read_regular, read_json, relative_path, require


def inventory(root):
    entries = {}
    names = set()
    inodes = set()
    for directory, folders, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in sorted(folders + files):
            path = current / name
            relative = str(path.relative_to(root))
            value = path.lstat()
            require(not stat.S_ISLNK(value.st_mode), 'Package contains a symlink: ' + relative)
            if current == root and name == '.git' and stat.S_ISDIR(value.st_mode):
                folders.remove(name)
                continue
            relative_path(relative)
            key = _key(tuple(relative.split('/')))
            require(key not in names, 'Package contains a portable path collision: ' + relative)
            names.add(key)
            require(stat.S_ISDIR(value.st_mode) or stat.S_ISREG(value.st_mode), 'Package contains an unsupported entry: ' + relative)
            if stat.S_ISREG(value.st_mode):
                identity = (value.st_dev, value.st_ino)
                require(value.st_nlink == 1 and identity not in inodes, 'Package contains a hard-linked file: ' + relative)
                inodes.add(identity)
            entries[relative] = (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    return entries


def verify(root, manifest_path=None):
    root = Path(root).resolve(strict=True)
    require(root.is_dir(), 'Package root must be a directory')
    manifest_path = Path(manifest_path) if manifest_path is not None else root / 'SHA256.json'
    manifest_path = manifest_path.absolute()
    require(manifest_path.is_relative_to(root) and not manifest_path.is_symlink(), 'Package manifest must be a regular file within package root')
    manifest_name = str(manifest_path.relative_to(root))
    relative_path(manifest_name)
    manifest_bytes = _read_regular(manifest_path)
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest, 'Package manifest must contain file hashes')
    keys = set()
    for name, digest in manifest.items():
        path = relative_path(name)
        require(name != manifest_name and path.parts[0] != '.git', 'Manifest cannot hash itself or Git metadata')
        key = _key(path.parts)
        require(key not in keys, 'Manifest contains a portable path collision: ' + name)
        keys.add(key)
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None, 'Invalid recorded SHA-256: ' + name)
    initial = inventory(root)
    actual = {name for name, value in initial.items() if stat.S_ISREG(value[2]) and name != manifest_name}
    require(actual == set(manifest), 'Package inventory differs from manifest; missing=' + repr(sorted(set(manifest) - actual)) + '; unrecorded=' + repr(sorted(actual - set(manifest))))
    mismatches = []
    for name, expected in sorted(manifest.items()):
        observed = hashlib.sha256(_read_regular(root / name)).hexdigest()
        if observed != expected:
            mismatches.append({'path': name, 'recorded': expected, 'observed': observed})
    require(not mismatches, 'Package bytes differ from recorded hashes: ' + json.dumps(mismatches, sort_keys=True))
    require(inventory(root) == initial and _read_regular(manifest_path) == manifest_bytes, 'Package changed during verification')
    return {'package_integrity': 'passed', 'files_checked': len(manifest), 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'source_root': str(root)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
