import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import unicodedata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Nonfinite JSON number: ' + value)


def _float(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite JSON number: ' + value)
    return result


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def _state(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_regular(path):
    path = Path(path)
    require(not path.is_symlink(), 'Symlink input is not supported: ' + str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), 'Input must be a regular file: ' + str(path))
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        require(_state(before) == _state(os.fstat(descriptor)) == _state(path.lstat()), 'Input changed while reading: ' + str(path))
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def read_json(path):
    return json.loads(_read_regular(path).decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)


def relative_path(raw):
    require(isinstance(raw, str) and bool(raw), 'Path must be a nonempty string')
    require(not any(c in raw for c in '\\*?[]') and not any(ord(c) < 32 or ord(c) == 127 for c in raw), 'Paths must be literal POSIX paths')
    parts = raw.split('/')
    require(all(part not in {'', '.', '..'} for part in parts), 'Path must be canonical and relative: ' + raw)
    path = PurePosixPath(raw)
    require(not path.is_absolute() and str(path) == raw, 'Path must be canonical and relative: ' + raw)
    return path


def _text(value, field):
    require(isinstance(value, str) and bool(value.strip()), field + ' must be a nonempty string')
    return value


def _digest(value, field):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, field + ' must be a lowercase SHA-256 digest')
    return value


def _find_openssl():
    candidates = [shutil.which('openssl'), str(Path.home() / 'miniforge3/bin/openssl'), '/opt/homebrew/opt/openssl@3/bin/openssl', '/usr/local/opt/openssl@3/bin/openssl']
    for candidate in dict.fromkeys(candidates):
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            result = subprocess.run([candidate, 'version'], capture_output=True, check=False)
        except OSError:
            continue
        match = re.match(rb'OpenSSL ([0-9]+)\.', result.stdout)
        if result.returncode == 0 and match and int(match.group(1)) >= 3:
            return candidate
    return None


OPENSSL = _find_openssl()


def verify_signed_contract(path, public_key_path, public_key_sha256):
    _digest(public_key_sha256, 'Pinned public-key digest')
    public_key = _read_regular(public_key_path)
    require(digest_bytes(public_key) == public_key_sha256, 'Public-key digest does not match externally pinned identity')
    envelope = read_json(path)
    require(isinstance(envelope, dict) and set(envelope) == {'payload', 'signature'}, 'Signed contract must contain exactly payload and signature')
    payload = envelope['payload']
    require(isinstance(payload, dict), 'Contract payload must be an object')
    encoded = _text(envelope['signature'], 'Contract signature')
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('Contract signature must be valid base64') from error
    require(len(signature) == 64 and base64.b64encode(signature).decode('ascii') == encoded, 'Contract signature must be a canonical 64-byte Ed25519 signature')
    payload_bytes = _canonical(payload)
    require(OPENSSL is not None, 'OpenSSL 3 or later is required for Ed25519 verification')
    with tempfile.TemporaryDirectory(prefix='contract-verification-') as folder:
        folder = Path(folder)
        key_path = folder / 'public-key'
        payload_path = folder / 'payload'
        signature_path = folder / 'signature'
        key_path.write_bytes(public_key)
        payload_path.write_bytes(payload_bytes)
        signature_path.write_bytes(signature)
        key_command = [OPENSSL, 'pkey', '-pubin', '-in', str(key_path), '-outform', 'DER']
        if not public_key.startswith(b'-----BEGIN '):
            key_command.extend(['-inform', 'DER'])
        key_result = subprocess.run(key_command, capture_output=True, check=False)
        require(key_result.returncode == 0, 'Cannot parse the pinned public key')
        require(len(key_result.stdout) == 44 and key_result.stdout[:12] == bytes.fromhex('302a300506032b6570032100'), 'Pinned public key must use Ed25519')
        command = [OPENSSL, 'pkeyutl', '-verify', '-pubin', '-inkey', str(key_path), '-rawin', '-in', str(payload_path), '-sigfile', str(signature_path)]
        if not public_key.startswith(b'-----BEGIN '):
            command.extend(['-keyform', 'DER'])
        result = subprocess.run(command, capture_output=True, check=False)
        require(result.returncode == 0, 'Ed25519 contract signature verification failed')
    require(_read_regular(public_key_path) == public_key, 'Pinned public key changed during verification')
    require(read_json(path) == envelope, 'Contract changed during verification')
    return payload, digest_bytes(payload_bytes)


def _key(parts):
    return tuple(unicodedata.normalize('NFC', part).casefold() for part in parts)


def _overlap(first, second):
    return first[:len(second)] == second or second[:len(first)] == first


def _root(path):
    root = Path(path).resolve(strict=True)
    require(root.is_dir(), 'Root must be an existing directory')
    return root


def _directory(path, parent=None):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def _inventory(root, skip_git=True):
    entries = {}
    aliases = {}
    inodes = {}
    root_descriptor = _directory(root)
    root_state = _state(os.fstat(root_descriptor))

    def visit(descriptor, parts):
        before = _state(os.fstat(descriptor))
        with os.scandir(descriptor) as listing:
            names = sorted(entry.name for entry in listing)
        for name in names:
            child_parts = parts + (name,)
            relative = '/'.join(child_parts)
            relative_path(relative)
            value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            require(not stat.S_ISLNK(value.st_mode), 'Symlink source or ownership path is not supported: ' + relative)
            if skip_git and not parts and name == '.git' and stat.S_ISDIR(value.st_mode):
                continue
            require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode), 'Unsupported filesystem entry: ' + relative)
            key = _key(child_parts)
            require(key not in aliases, 'Case or Unicode path alias: ' + relative)
            aliases[key] = relative
            entries[relative] = _state(value)
            if stat.S_ISDIR(value.st_mode):
                child = _directory(name, descriptor)
                try:
                    require(_state(os.fstat(child)) == _state(value), 'Directory changed during inventory: ' + relative)
                    visit(child, child_parts)
                finally:
                    os.close(child)
            else:
                inode = (value.st_dev, value.st_ino)
                require(inode not in inodes and value.st_nlink == 1, 'Hard-linked source or ownership path is not supported: ' + relative)
                inodes[inode] = relative
        require(before == _state(os.fstat(descriptor)), 'Directory changed during inventory: ' + '/'.join(parts))

    try:
        visit(root_descriptor, ())
        require(root_state == _state(root.lstat()), 'Root changed during inventory')
    finally:
        os.close(root_descriptor)
    return root_state, entries


def _hash_source(root, relative, expected):
    descriptor = _directory(root)
    try:
        for part in relative.parts[:-1]:
            child = _directory(part, descriptor)
            os.close(descriptor)
            descriptor = child
        leaf = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        try:
            before = os.fstat(leaf)
            require(stat.S_ISREG(before.st_mode) and _state(before) == expected, 'Source changed before hashing: ' + str(relative))
            digest = hashlib.sha256()
            while True:
                chunk = os.read(leaf, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            require(expected == _state(os.fstat(leaf)) == _state(os.stat(relative.parts[-1], dir_fd=descriptor, follow_symlinks=False)), 'Source changed while hashing: ' + str(relative))
            return digest.hexdigest()
        finally:
            os.close(leaf)
    finally:
        os.close(descriptor)


def _git_head(root):
    executable = shutil.which('git')
    require(executable is not None, 'Git is required to establish repository identity')
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_OPTIONAL_LOCKS='0', GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    command = [executable, '--no-optional-locks', '-C', str(root), 'rev-parse']
    result = subprocess.run(command + ['--verify', 'HEAD'], capture_output=True, check=False, env=environment)
    if result.returncode == 0:
        head = result.stdout.decode('ascii').strip()
        require(re.fullmatch('(?:[0-9a-f]{40}|[0-9a-f]{64})', head) is not None, 'Git returned an invalid HEAD identity')
        return head
    probe = subprocess.run(command + ['--git-dir'], capture_output=True, check=False, env=environment)
    markers = [(ancestor / '.git') for ancestor in (root, *root.parents)]
    require(probe.returncode != 0 and not any(marker.exists() or marker.is_symlink() for marker in markers), 'Repository HEAD is unavailable or invalid')
    return None


def verify_source_manifest(manifest, source_root):
    require(isinstance(manifest, dict) and set(manifest) == {'schema_version', 'base_head', 'files', 'excluded_paths', 'external_assets'}, 'Source manifest fields differ from schema version 1')
    require(type(manifest['schema_version']) is int and manifest['schema_version'] == 1, 'Expected source manifest schema version 1')
    head = manifest['base_head']
    require(head is None or isinstance(head, str) and re.fullmatch('(?:[0-9a-f]{40}|[0-9a-f]{64})', head) is not None, 'base_head must be null or a full Git object identity')
    for field in ('files', 'excluded_paths', 'external_assets'):
        require(isinstance(manifest[field], list), field + ' must be a list')
    files = {}
    file_keys = []
    for item in manifest['files']:
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'Source file entries require exactly path and sha256')
        relative = relative_path(item['path'])
        key = _key(relative.parts)
        require(all(not _overlap(key, prior) for prior in file_keys), 'Duplicate, aliased or overlapping source file: ' + str(relative))
        files[str(relative)] = _digest(item['sha256'], 'Source file digest')
        file_keys.append(key)
    excluded = []
    for item in manifest['excluded_paths']:
        require(isinstance(item, dict) and set(item) == {'path', 'reason'}, 'Excluded entries require exactly path and reason')
        relative = relative_path(item['path'])
        key = _key(relative.parts)
        _text(item['reason'], 'Exclusion reason')
        require(all(not _overlap(key, prior) for prior in file_keys + excluded), 'Exclusion overlaps an included file or exclusion: ' + str(relative))
        excluded.append(key)
    external_ids = set()
    for item in manifest['external_assets']:
        require(isinstance(item, dict) and set(item) == {'id', 'location', 'sha256', 'reason'}, 'External assets require exactly id, location, sha256 and reason')
        identifier = _text(item['id'], 'External asset ID')
        require(identifier not in external_ids, 'Duplicate external asset ID: ' + identifier)
        external_ids.add(identifier)
        _text(item['location'], 'External asset location')
        _text(item['reason'], 'External asset reason')
        _digest(item['sha256'], 'External asset digest')
    manifest_digest = digest_bytes(_canonical(manifest))
    root = _root(source_root)
    require(_git_head(root) == head, 'Source base_head does not match the actual repository HEAD')
    initial = _inventory(root)
    actual = {name: value for name, value in initial[1].items() if stat.S_ISREG(value[2])}
    require(set(files) <= set(actual), 'Manifest includes missing or non-regular files: ' + ', '.join(sorted(set(files) - set(actual))))
    unaccounted = [name for name in actual if name not in files and not any(_key(PurePosixPath(name).parts)[:len(prefix)] == prefix for prefix in excluded)]
    require(not unaccounted, 'Source files are not accounted for: ' + ', '.join(sorted(unaccounted)))
    for name, expected_digest in files.items():
        require(_hash_source(root, relative_path(name), actual[name]) == expected_digest, 'Source file digest mismatch: ' + name)
    require(_inventory(root) == initial, 'Source inventory changed during verification')
    require(_git_head(root) == head, 'Repository HEAD changed during source verification')
    require(digest_bytes(_canonical(manifest)) == manifest_digest, 'Source manifest changed during verification')
    return manifest_digest


def _resolve_owned(root, relative):
    current = root
    for part in relative.parts:
        if current.exists():
            require(current.is_dir() and not current.is_symlink(), 'Ownership parent is not a regular directory: ' + str(current))
            with os.scandir(current) as listing:
                matches = [entry.name for entry in listing if _key((entry.name,)) == _key((part,))]
            require(len(matches) <= 1 and (not matches or matches[0] == part), 'Case or Unicode ownership alias: ' + str(relative))
        current = current / part
        require(not current.is_symlink(), 'Symlink ownership path is not supported: ' + str(relative))
    resolved = current.resolve(strict=False)
    require(resolved.is_relative_to(root), 'Ownership path escapes root')
    return resolved


def validate_ownership(assignments, ownership_root):
    require(isinstance(assignments, list), 'Assignments must be a list')
    root = _root(ownership_root)
    seen = []
    inode_owners = {}
    for assignment in assignments:
        require(isinstance(assignment, dict), 'Assignment must be an object')
        owner = _text(assignment.get('owner'), 'Assignment owner')
        paths = assignment.get('write_paths')
        require(isinstance(paths, list), 'Assignment write_paths must be a list')
        for raw in paths:
            relative = relative_path(raw)
            key = _key(relative.parts)
            require(all(not _overlap(key, prior) for prior in seen), 'Overlapping, duplicate or aliased ownership: ' + str(relative))
            full = _resolve_owned(root, relative)
            seen.append(key)
            if not full.exists():
                continue
            value = full.lstat()
            require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode), 'Unsupported ownership target: ' + str(relative))
            states = [_state(value)]
            if stat.S_ISDIR(value.st_mode):
                states.extend(_inventory(full, skip_git=False)[1].values())
            for state in states:
                inode = state[:2]
                require(inode not in inode_owners, 'Filesystem alias crosses ownership paths: ' + str(relative))
                if stat.S_ISREG(state[2]):
                    require(state[3] == 1, 'Hard-linked ownership target is not supported: ' + str(relative))
                inode_owners[inode] = (owner, str(relative))
    return True
