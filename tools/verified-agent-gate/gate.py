import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
ROOT = Path(__file__).resolve().parent
GENESIS = '0' * 64
CONTRACT_SHA256 = 'b801c62880a6862d716a6107909494e58cf9a6c0c58921eff89296bf293e5fd2'
POLICY_SHA256 = 'bdcd68b207acca59f428e0a2ef3053bea235bdca64c6a61ce9f8894b60d82a85'
POLICY_KEY_HEX = '0455b330e44af194e8530b2bdcaceafa51558cd12a172d53c51acf5e426ea137'
LINUX_F_ADD_SEALS = 1033
LINUX_F_GET_SEALS = 1034
LINUX_IMMUTABLE_SEALS = 15
def require(condition, message):
    if not condition:
        raise ValueError(message)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result
def parse(raw):
    obj = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('JSON constant')))
    require(raw == canonical(obj) + b'\n', 'noncanonical JSON bytes')
    return obj
def b64(raw):
    return base64.b64encode(raw).decode('ascii')
def unb64(value):
    require(isinstance(value, str), 'base64 must be a string')
    raw = base64.b64decode(value, validate=True)
    require(b64(raw) == value, 'noncanonical base64')
    return raw
def digest_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
def fields(value, names):
    require(isinstance(value, dict) and set(value) == set(names), 'unexpected or missing fields')
def private_key(path):
    path = Path(path)
    require(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, 'private key permissions must exclude group and others')
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes())
def key_bytes(private):
    return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
def signed(private, payload):
    return canonical({'payload': payload, 'signature': b64(private.sign(canonical(payload)))}) + b'\n'
def read_signed(raw, public):
    envelope = parse(raw)
    fields(envelope, ('payload', 'signature'))
    Ed25519PublicKey.from_public_bytes(public).verify(unb64(envelope['signature']), canonical(envelope['payload']))
    return envelope['payload']
def write_once(path, raw, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
def source(name, raw):
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', name) is not None, 'invalid source name')
    return {'name': name, 'length': len(raw), 'sha256': sha(raw), 'bytes_b64': b64(raw)}
def source_bytes(item):
    fields(item, ('name', 'length', 'sha256', 'bytes_b64'))
    require(type(item['length']) is int and digest_string(item['sha256']), 'source length or digest type mismatch')
    raw = unb64(item['bytes_b64'])
    require(item == source(item['name'], raw), 'source byte length or digest mismatch')
    return raw
def original_sources():
    originals = ROOT / 'originals'
    policy = (originals / 'policy.canonical.json').read_bytes()
    require(sha(policy) == POLICY_SHA256, 'original policy changed')
    require((originals / 'policy.ed25519.pub').read_bytes().hex() == POLICY_KEY_HEX, 'original public key changed')
    require((originals / 'policy.sha256').read_text('ascii').strip() == POLICY_SHA256, 'original policy digest changed')
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(POLICY_KEY_HEX)).verify((originals / 'policy.ed25519.sig').read_bytes(), policy)
    obj = json.loads(policy, object_pairs_hook=unique)
    require(canonical(obj) == policy, 'original policy canonical encoding changed')
    instruction = (originals / 'instruction.bin').read_bytes()
    require(instruction == bytes.fromhex(obj['original_instruction_hex']), 'original instruction changed')
    require(sha(instruction) == obj['original_instruction_sha256'], 'original instruction digest changed')
    contract = (originals / 'verified_delegation_instruction.txt').read_bytes()
    require(sha(contract) == CONTRACT_SHA256, 'complete contract changed')
    require(contract.split(b'\n', 1)[0] == instruction, 'contract first line mismatch')
    return [source('instruction.bin', instruction), source('verified_delegation_instruction.txt', contract)]
def init_authority(directory, key_path, checker, extras=(), landrun=None):
    directory, key_path, checker = Path(directory).resolve(), Path(key_path).resolve(), Path(checker).resolve()
    require(not directory.exists() and not key_path.exists(), 'authority or private key already exists')
    require(not key_path.is_relative_to(directory), 'private key must be outside authority directory')
    sources = original_sources() + [source(name, Path(path).read_bytes()) for name, path in extras]
    require(len({item['name'] for item in sources}) == len(sources), 'duplicate source name')
    checker_raw = checker.read_bytes()
    require(os.access(checker, os.X_OK), 'checker is not executable')
    private = Ed25519PrivateKey.generate()
    public = key_bytes(private)
    directory.mkdir(mode=0o700, parents=True)
    write_once(key_path, private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    write_once(directory / 'public.ed25519', public, 0o444)
    write_once(directory / 'bytegate', checker_raw, 0o555)
    landrun_raw = None if landrun is None else Path(landrun).read_bytes()
    if landrun_raw is not None:
        write_once(directory / 'landrun', landrun_raw, 0o555)
    manifest = {'schema': 'bytegate-authority-v3', 'key_id': sha(public), 'sources': sources, 'checker_sha256': sha(checker_raw), 'gate_sha256': sha(Path(__file__).read_bytes()), 'snapshot_helper_sha256': sha((ROOT / 'snapshot.py').read_bytes()), 'landrun_sha256': None if landrun_raw is None else sha(landrun_raw)}
    write_once(directory / 'authority.json', signed(private, manifest), 0o444)
    return sha(public)
class Authority:
    def __init__(self, directory, pinned_key):
        self.directory = Path(directory).resolve()
        self.public = (self.directory / 'public.ed25519').read_bytes()
        require(digest_string(pinned_key) and sha(self.public) == pinned_key, 'pinned public key mismatch')
        self.manifest = read_signed((self.directory / 'authority.json').read_bytes(), self.public)
        fields(self.manifest, ('schema', 'key_id', 'sources', 'checker_sha256', 'gate_sha256', 'snapshot_helper_sha256', 'landrun_sha256'))
        require(self.manifest['schema'] == 'bytegate-authority-v3' and self.manifest['key_id'] == pinned_key, 'authority schema or key mismatch')
        require(self.manifest['gate_sha256'] == sha(Path(__file__).read_bytes()), 'gate code bytes changed')
        require(self.manifest['snapshot_helper_sha256'] == sha((ROOT / 'snapshot.py').read_bytes()), 'snapshot helper bytes changed')
        self.sources = self.manifest['sources']
        require(isinstance(self.sources, list) and self.sources[:2] == original_sources(), 'authority lacks immutable originals')
        for item in self.sources:
            source_bytes(item)
        require(len({item['name'] for item in self.sources}) == len(self.sources), 'duplicate authority source')
        self.source_raw = canonical(self.sources)
        self.key_id = pinned_key
        self.checker = self.directory / 'bytegate'
    def signer(self, path):
        private = private_key(path)
        require(key_bytes(private) == self.public, 'private key differs from pinned authority')
        return private
    def verify(self, raw, expected_digest, expected_parent):
        require(digest_string(expected_digest) and sha(raw) == expected_digest, 'expected envelope digest mismatch')
        value = read_signed(raw, self.public)
        fields(value, ('schema', 'key_id', 'id', 'parent', 'sources', 'operation', 'command', 'artifacts', 'environment', 'cwd', 'timeout_ms', 'snapshot', 'snapshot_mode', 'runtime_roots'))
        require(value['schema'] == 'bytegate-envelope-v3' and value['key_id'] == self.key_id, 'envelope schema or key mismatch')
        require(digest_string(value['id']), 'invalid unique handoff identifier')
        require(expected_parent is None or digest_string(expected_parent), 'invalid expected parent')
        require(value['parent'] == expected_parent, 'parent does not match trusted handoff')
        require(value['sources'] == self.sources, 'authoritative instruction bytes changed or missing')
        operation = source_bytes(value['operation'])
        require(operation and operation.decode('utf-8').strip(), 'operation is empty or not UTF-8')
        command = value['command']
        require(isinstance(command, list) and command and all(isinstance(x, str) and x and '\0' not in x for x in command), 'invalid command')
        require(isinstance(value['environment'], dict) and all(isinstance(k, str) and k and '=' not in k and '\0' not in k and isinstance(v, str) and '\0' not in v for k, v in value['environment'].items()), 'invalid exact environment')
        require(isinstance(value['cwd'], str) and Path(value['cwd']).is_absolute(), 'absolute cwd required')
        require(type(value['timeout_ms']) is int and value['timeout_ms'] > 0, 'positive timeout required')
        artifacts = value['artifacts']
        require(isinstance(artifacts, list) and artifacts, 'bound executable required')
        indexes = []
        for item in artifacts:
            fields(item, ('argv_index', 'path', 'length', 'sha256'))
            index = item['argv_index']
            require(type(index) is int and 0 <= index < len(command), 'artifact index outside command')
            require(item['path'] == command[index] and Path(item['path']).is_absolute(), 'artifact path differs from signed command')
            require(type(item['length']) is int and item['length'] >= 0 and digest_string(item['sha256']), 'invalid artifact descriptor')
            indexes.append(index)
        require(indexes == sorted(set(indexes)) and indexes[0] == 0, 'duplicate, unordered or missing executable artifact')
        require(isinstance(value['runtime_roots'], list) and all(isinstance(p, str) and Path(p).is_absolute() for p in value['runtime_roots']), 'invalid runtime roots')
        require(value['snapshot_mode'] in ('landlock', 'trusted-host-readonly'), 'unknown snapshot isolation mode')
        if value['snapshot'] is not None:
            require(value['snapshot_mode'] != 'landlock' or self.manifest['landrun_sha256'] is not None, 'Landlock snapshot requires signed runner')
            fields(value['snapshot'], ('root', 'files', 'writable'))
            require(Path(value['cwd']).is_relative_to(Path(value['snapshot']['root'])), 'cwd outside signed snapshot')
            require(all(not self.directory.is_relative_to(Path(p)) and not Path(value['snapshot']['root']).is_relative_to(Path(p)) for p in value['runtime_roots']), 'runtime roots include protected authority or snapshot')
        return value
def create_envelope(authority, key_path, operation, command, file_args=(), parent=None, cwd=None, environment=None, timeout_ms=60000, identifier=None, snapshot=None, runtime_roots=(), snapshot_mode='landlock'):
    command = list(command)
    require(bool(command), 'empty command')
    indexes = sorted(set([0, *file_args]))
    artifacts = []
    for index in indexes:
        require(type(index) is int and 0 <= index < len(command), 'file argument outside command')
        path = Path(command[index]).resolve(strict=True)
        require(path.is_file(), 'bound artifact is not a regular file')
        raw = path.read_bytes()
        command[index] = str(path)
        artifacts.append({'argv_index': index, 'path': str(path), 'length': len(raw), 'sha256': sha(raw)})
    require(os.access(command[0], os.X_OK), 'command is not executable')
    if snapshot is not None:
        import snapshot as snapshots
        snapshots.validate(snapshot)
        require(all(not Path(key_path).resolve().is_relative_to(Path(p).resolve()) for p in runtime_roots), 'runtime read root exposes signing key')
        cwd = cwd or snapshot['root']
    value = {'schema': 'bytegate-envelope-v3', 'key_id': authority.key_id, 'id': identifier or uuid.uuid4().hex + uuid.uuid4().hex, 'parent': parent, 'sources': authority.sources, 'operation': source('assigned_operation.txt', operation), 'command': command, 'artifacts': artifacts, 'environment': environment if environment is not None else {'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PATH': '/usr/bin:/bin'}, 'cwd': str(Path(cwd or os.getcwd()).resolve()), 'timeout_ms': timeout_ms, 'snapshot': snapshot, 'snapshot_mode': snapshot_mode, 'runtime_roots': list(runtime_roots)}
    raw = signed(authority.signer(key_path), value)
    authority.verify(raw, sha(raw), parent)
    return raw
def sealed_file(raw, name):
    require(sys.platform == 'linux', 'sealed execution requires Linux')
    fd = os.memfd_create(name, os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with os.fdopen(os.dup(fd), 'wb') as stream:
            stream.write(raw)
        os.lseek(fd, 0, os.SEEK_SET)
        os.fchmod(fd, 0o500)
        fcntl.fcntl(fd, LINUX_F_ADD_SEALS, LINUX_IMMUTABLE_SEALS)
        require(fcntl.fcntl(fd, LINUX_F_GET_SEALS) & LINUX_IMMUTABLE_SEALS == LINUX_IMMUTABLE_SEALS, 'sealed file verification failed')
        return fd
    except BaseException:
        os.close(fd)
        raise
def lean_accept(authority, envelope, used, parents, expected_parent):
    raw = authority.checker.read_bytes()
    require(sha(raw) == authority.manifest['checker_sha256'], 'Lean checker bytes changed')
    fd = sealed_file(raw, 'bytegate-lean')
    claims = {'source': list(canonical(envelope['sources'])), 'expected': list(authority.source_raw), 'claimedLength': len(authority.source_raw), 'signatureOK': True, 'digestOK': True, 'canonicalOK': True, 'filesOK': True, 'id': envelope['id'], 'parent': envelope['parent'], 'expectedParent': expected_parent, 'used': list(used), 'parents': list(parents)}
    try:
        result = subprocess.run(['/proc/self/fd/' + str(fd)], input=canonical(claims), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'LANG': 'C.UTF-8'}, pass_fds=(fd,), check=False, timeout=30)
    finally:
        os.close(fd)
    require(result.returncode == 0 and result.stdout == b'ACCEPTED\n' and not result.stderr, 'Lean acceptance rejected the handoff')
class Journal:
    def __init__(self, directory, authority):
        self.directory, self.authority = Path(directory).resolve(), authority
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.directory / 'envelopes').mkdir(mode=0o700, exist_ok=True)
        self.stream = os.fdopen(os.open(self.directory / 'receipts.jsonl', os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'r+b')
    def close(self):
        self.stream.close()
    def lock(self):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
    def unlock(self):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
    def read(self, expected_head=None):
        self.stream.seek(0)
        head, accepted, completed, receipts = GENESIS, {}, set(), []
        for raw in self.stream:
            p = read_signed(raw, self.authority.public)
            fields(p, ('schema', 'key_id', 'sequence', 'previous', 'phase', 'id', 'envelope_sha256', 'parent', 'source_sha256', 'artifacts', 'exit_code', 'stdout_sha256', 'stderr_sha256', 'error'))
            require(p['schema'] == 'bytegate-receipt-v3' and p['key_id'] == self.authority.key_id, 'receipt schema or key mismatch')
            require(p['sequence'] == len(receipts) and type(p['sequence']) is int and p['previous'] == head, 'receipt chain changed')
            require(p['source_sha256'] == sha(self.authority.source_raw), 'receipt instruction source mismatch')
            identifier = p['id']
            require(digest_string(identifier) and digest_string(p['envelope_sha256']), 'receipt identifier invalid')
            if p['phase'] == 'accepted':
                require(identifier not in accepted, 'duplicate accepted handoff')
                require(p['parent'] is None or p['parent'] in {v['envelope_sha256'] for v in accepted.values()}, 'unauthenticated receipt parent')
                require(p['exit_code'] is None and p['stdout_sha256'] is None and p['stderr_sha256'] is None and p['error'] is None, 'invalid acceptance receipt')
                envelope_raw = (self.directory / 'envelopes' / (p['envelope_sha256'] + '.json')).read_bytes()
                envelope = self.authority.verify(envelope_raw, p['envelope_sha256'], p['parent'])
                require(envelope['id'] == identifier and envelope['artifacts'] == p['artifacts'], 'receipt does not bind stored handoff')
                accepted[identifier] = p
            else:
                require(p['phase'] in ('completed', 'failed') and identifier in accepted and identifier not in completed, 'completion without one acceptance')
                first = accepted[identifier]
                require(all(p[k] == first[k] for k in ('envelope_sha256', 'parent', 'source_sha256', 'artifacts')), 'completion changed handoff')
                require(type(p['exit_code']) is int and digest_string(p['stdout_sha256']) and digest_string(p['stderr_sha256']), 'invalid completion output')
                require((p['phase'] == 'completed') == (p['exit_code'] == 0), 'completion phase disagrees with exit status')
                require(p['error'] is None or isinstance(p['error'], str), 'invalid error description')
                completed.add(identifier)
            head = sha(raw)
            receipts.append(p)
        if expected_head is not None:
            require(digest_string(expected_head) and head == expected_head, 'journal differs from externally pinned head')
        return head, accepted, completed, receipts
    def append(self, private, envelope, envelope_digest, phase, exit_code=None, stdout=None, stderr=None, error=None):
        head, _, _, receipts = self.read()
        p = {'schema': 'bytegate-receipt-v3', 'key_id': self.authority.key_id, 'sequence': len(receipts), 'previous': head, 'phase': phase, 'id': envelope['id'], 'envelope_sha256': envelope_digest, 'parent': envelope['parent'], 'source_sha256': sha(self.authority.source_raw), 'artifacts': envelope['artifacts'], 'exit_code': exit_code, 'stdout_sha256': None if stdout is None else sha(stdout), 'stderr_sha256': None if stderr is None else sha(stderr), 'error': error}
        raw = signed(private, p)
        self.stream.write(raw)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return sha(raw)
def dispatch(authority, key_path, envelope_raw, expected_digest, expected_parent, journal_directory, expected_head):
    require(digest_string(expected_head), 'externally pinned journal head required')
    envelope = authority.verify(envelope_raw, expected_digest, expected_parent)
    private = authority.signer(key_path)
    descriptors, command = [], list(envelope['command'])
    journal = None
    try:
        if envelope['snapshot'] is not None:
            import snapshot as snapshots
            snapshots.validate(envelope['snapshot'])
            require(all(not Path(key_path).resolve().is_relative_to(Path(p).resolve()) for p in envelope['runtime_roots']), 'runtime read root exposes signing key')
        for artifact in envelope['artifacts']:
            raw = Path(artifact['path']).read_bytes()
            require(len(raw) == artifact['length'] and sha(raw) == artifact['sha256'], 'bound code file changed')
            fd = sealed_file(raw, 'bytegate-artifact-' + str(artifact['argv_index']))
            descriptors.append(fd)
            if artifact['argv_index'] != 0:
                command[artifact['argv_index']] = '/proc/self/fd/' + str(fd)
        journal = Journal(journal_directory, authority)
        journal.lock()
        try:
            _, accepted, _, _ = journal.read(expected_head)
            require(envelope['id'] not in accepted, 'replayed handoff')
            known_parents = {value['envelope_sha256'] for value in accepted.values()}
            require(expected_parent is None or expected_parent in known_parents, 'parent has no authenticated acceptance')
            lean_accept(authority, envelope, accepted, known_parents, expected_parent)
            destination = journal.directory / 'envelopes' / (expected_digest + '.json')
            if destination.exists():
                require(destination.read_bytes() == envelope_raw, 'stored envelope changed')
            else:
                write_once(destination, envelope_raw, 0o444)
            accepted_head = journal.append(private, envelope, expected_digest, 'accepted')
        finally:
            journal.unlock()
        error = None
        try:
            executable = '/proc/self/fd/' + str(descriptors[0])
            if envelope['snapshot'] is not None and envelope['snapshot_mode'] == 'landlock':
                runner = (authority.directory / 'landrun').read_bytes()
                require(sha(runner) == authority.manifest['landrun_sha256'], 'Landlock runner bytes changed')
                runner_fd = sealed_file(runner, 'bytegate-landrun')
                descriptors.append(runner_fd)
                sandbox_command = ['/proc/self/fd/' + str(runner_fd), '--ro', envelope['snapshot']['root'], '--rox', '/proc/self/fd', '--rw', '/dev/null']
                for path in envelope['runtime_roots']:
                    sandbox_command += ['--rox', path]
                for path in envelope['snapshot']['writable']:
                    sandbox_command += ['--rw', str(Path(envelope['snapshot']['root']) / path)]
                for name, value in sorted(envelope['environment'].items()):
                    sandbox_command += ['--env', name + '=' + value]
                sandbox_command += ['--', executable, *command[1:]]
                executable, command = sandbox_command[0], sandbox_command
            result = subprocess.run(command, executable=executable, cwd=envelope['cwd'], env=envelope['environment'], pass_fds=tuple(descriptors), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=envelope['timeout_ms'] / 1000)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr, error = 124, exc.stdout or b'', exc.stderr or b'', 'signed execution timeout exceeded'
        except OSError as exc:
            code, stdout, stderr, error = 126, b'', b'', str(exc)
        if envelope['snapshot'] is not None:
            try:
                snapshots.validate(envelope['snapshot'])
            except (ValueError, OSError) as exc:
                code, error = 125, 'source snapshot changed during execution: ' + str(exc)
        journal.lock()
        try:
            _, accepted, _, receipts = journal.read()
            require(envelope['id'] in accepted, 'acceptance receipt disappeared')
            acceptance_index = accepted[envelope['id']]['sequence']
            require(sha(signed(private, receipts[acceptance_index])) == accepted_head, 'acceptance checkpoint changed')
            head = journal.append(private, envelope, expected_digest, 'completed' if code == 0 else 'failed', code, stdout, stderr, error)
        finally:
            journal.unlock()
        return {'exit_code': code, 'stdout': stdout, 'stderr': stderr, 'head': head, 'accepted_head': accepted_head, 'envelope_sha256': expected_digest, 'error': error}
    finally:
        if journal is not None:
            journal.close()
        for fd in descriptors:
            os.close(fd)
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    init = sub.add_parser('init')
    init.add_argument('--authority', required=True)
    init.add_argument('--private', required=True)
    init.add_argument('--checker', required=True)
    init.add_argument('--source', action='append', default=[])
    init.add_argument('--landrun')
    for name in ('create', 'verify', 'dispatch', 'audit'):
        p = sub.add_parser(name)
        p.add_argument('--authority', required=True)
        p.add_argument('--pin', required=True)
        if name in ('create', 'dispatch'):
            p.add_argument('--private', required=True)
        if name in ('verify', 'dispatch'):
            p.add_argument('--envelope', required=True)
            p.add_argument('--digest', required=True)
        if name in ('create', 'verify', 'dispatch'):
            p.add_argument('--parent')
        if name in ('dispatch', 'audit'):
            p.add_argument('--journal', required=True)
            p.add_argument('--head', required=True)
        if name == 'create':
            p.add_argument('--operation-file', required=True)
            p.add_argument('--output', required=True)
            p.add_argument('--file-arg', type=int, action='append', default=[])
            p.add_argument('--cwd')
            p.add_argument('--timeout-ms', type=int, default=60000)
            p.add_argument('--environment-file')
            p.add_argument('--snapshot-manifest')
            p.add_argument('--runtime-root', action='append', default=[])
            p.add_argument('--snapshot-mode', choices=('landlock', 'trusted-host-readonly'), default='landlock')
            p.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'init':
        extras = []
        for item in args.source:
            require('=' in item, 'source must be NAME=PATH')
            extras.append(item.split('=', 1))
        print(init_authority(args.authority, args.private, args.checker, extras, args.landrun))
        return 0
    authority = Authority(args.authority, args.pin)
    if args.action == 'create':
        cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
        environment = json.loads(Path(args.environment_file).read_bytes(), object_pairs_hook=unique) if args.environment_file else None
        snapshot = json.loads(Path(args.snapshot_manifest).read_bytes(), object_pairs_hook=unique) if args.snapshot_manifest else None
        raw = create_envelope(authority, args.private, Path(args.operation_file).read_bytes(), cmd, args.file_arg, args.parent, args.cwd, environment, args.timeout_ms, snapshot=snapshot, runtime_roots=args.runtime_root, snapshot_mode=args.snapshot_mode)
        write_once(args.output, raw, 0o444)
        print(sha(raw))
    elif args.action == 'verify':
        value = authority.verify(Path(args.envelope).read_bytes(), args.digest, args.parent)
        print(json.dumps({'verified_envelope': args.digest, 'id': value['id'], 'parent': value['parent'], 'source_sha256': sha(authority.source_raw)}, sort_keys=True))
    elif args.action == 'audit':
        journal = Journal(args.journal, authority)
        journal.lock()
        try:
            head, accepted, completed, receipts = journal.read(args.head)
            print(json.dumps({'head': head, 'accepted': len(accepted), 'completed': len(completed), 'receipts': len(receipts), 'unfinished': sorted(set(accepted) - completed)}, sort_keys=True))
        finally:
            journal.unlock()
            journal.close()
    else:
        result = dispatch(authority, args.private, Path(args.envelope).read_bytes(), args.digest, args.parent, args.journal, args.head)
        sys.stdout.buffer.write(result.pop('stdout'))
        sys.stderr.buffer.write(result.pop('stderr'))
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return result['exit_code'] if result['exit_code'] >= 0 else 128 - result['exit_code']
    return 0
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('REJECTED:', type(exc).__name__, str(exc), file=sys.stderr)
        sys.exit(1)
