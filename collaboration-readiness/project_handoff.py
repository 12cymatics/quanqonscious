import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
import uuid

from validate_readiness import canonical, exact_keys, load_contract, nonempty
from validation_identity import digest_bytes, read_json, relative_path, require, validate_ownership, verify_signed_contract, verify_source_manifest


def timestamp(value, label):
    require(isinstance(value, str), label + ' must be an ISO timestamp')
    parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, label + ' requires a timezone')
    return parsed


def current_time():
    return datetime.datetime.now(datetime.timezone.utc)


def file_map(manifest):
    return {item['path']: item['sha256'] for item in manifest['files']}


def inside(path, parent):
    parts, prefix = relative_path(path).parts, relative_path(parent).parts
    return parts[:len(prefix)] == prefix


def assigned_paths(paths, assignment):
    require(isinstance(paths, list) and paths, 'At least one requested path is required')
    require(len(paths) == len(set(paths)), 'Duplicate requested path')
    for path in paths:
        relative_path(path)
        require(any(inside(path, owned) for owned in assignment['write_paths']), 'Path is outside assigned ownership: ' + path)


def historical_registry(registry):
    exact_keys(registry, {'schema_version', 'task_id', 'contract_sha256', 'contract_generation', 'source_manifest_sha256', 'generation', 'writer', 'issued_at', 'expires_at', 'previous_registry_sha256', 'reconciliations', 'assignments', 'exclusive_resources', 'authorization'}, 'Previous coordination registry')
    require(type(registry['schema_version']) is int and registry['schema_version'] == 1, 'Unsupported previous coordination registry schema')
    require(type(registry['generation']) is int and registry['generation'] > 0 and type(registry['contract_generation']) is int and registry['contract_generation'] > 0, 'Previous registry generation is invalid')
    require(nonempty(registry['task_id']) and nonempty(registry['writer']), 'Previous registry task or writer missing')
    for field in ('contract_sha256', 'source_manifest_sha256'):
        value = registry[field]
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Previous registry digest invalid: ' + field)
    issued = timestamp(registry['issued_at'], 'Previous registry issue time')
    expires = timestamp(registry['expires_at'], 'Previous registry expiry')
    require(issued < expires, 'Previous registry time interval invalid')
    exact_keys(registry['authorization'], {'action', 'user_request_sha256'}, 'Previous registry authorization')
    require(registry['authorization']['action'] == 'apply_project_changes', 'Previous registry authorization is invalid')
    require(isinstance(registry['assignments'], list) and registry['assignments'], 'Previous registry assignments missing')
    identifiers = set()
    for item in registry['assignments']:
        exact_keys(item, {'id', 'owner', 'write_paths', 'state', 'expires_at', 'pending_operations', 'worker_public_key_sha256'}, 'Previous registry assignment')
        require(nonempty(item['id']) and item['id'] not in identifiers and nonempty(item['owner']), 'Previous registry has duplicate or missing assignment identity')
        identifiers.add(item['id'])
        require(item['state'] == 'active' and isinstance(item['write_paths'], list) and item['write_paths'], 'Previous assignment is invalid')
        for path in item['write_paths']:
            relative_path(path)
        require(isinstance(item['pending_operations'], list), 'Previous pending operations must be a list')
        require(issued < timestamp(item['expires_at'], 'Previous assignment expiry') <= expires, 'Previous assignment time interval invalid')
        value = item['worker_public_key_sha256']
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Previous worker key digest invalid')
    require(isinstance(registry['exclusive_resources'], list) and isinstance(registry['reconciliations'], list), 'Previous registry resource or reconciliation lists invalid')
    resources = set()
    for resource in registry['exclusive_resources']:
        exact_keys(resource, {'kind', 'id', 'assignment_id'}, 'Previous exclusive resource')
        require(resource['kind'] in {'branch', 'live_session'} and nonempty(resource['id']) and resource['assignment_id'] in identifiers, 'Previous exclusive resource identity invalid')
        key = (resource['kind'], resource['id'])
        require(key not in resources, 'Previous registry has duplicate exclusive resource')
        resources.add(key)


def verify_registry(path, trusted, contract, contract_hash, revision, expected_hash, expected_generation, previous_registry=None):
    registry, registry_hash = verify_signed_contract(path, trusted['public_key_path'], trusted['public_key_sha256'])
    require(registry_hash == expected_hash, 'Registry differs from independently pinned registry digest')
    exact_keys(registry, {'schema_version', 'task_id', 'contract_sha256', 'contract_generation', 'source_manifest_sha256', 'generation', 'writer', 'issued_at', 'expires_at', 'previous_registry_sha256', 'reconciliations', 'assignments', 'exclusive_resources', 'authorization'}, 'Coordination registry')
    require(type(registry['schema_version']) is int and registry['schema_version'] == 1, 'Unsupported coordination registry schema')
    require(type(expected_generation) is int and expected_generation > 0 and type(registry['generation']) is int and registry['generation'] == expected_generation, 'Registry generation differs from current trusted generation')
    require(registry['task_id'] == contract['task_id'] and registry['contract_sha256'] == contract_hash and type(registry['contract_generation']) is int and registry['contract_generation'] == contract['generation'], 'Registry contract identity differs from trusted task')
    require(registry['source_manifest_sha256'] == revision.removeprefix('sha256:'), 'Registry source identity differs from actual source')
    require(registry['writer'] == contract['ownership']['integrator'], 'Registry writer is not the selected integrator')
    issued = timestamp(registry['issued_at'], 'Registry issue time')
    expires = timestamp(registry['expires_at'], 'Registry expiry')
    require(issued <= current_time() < expires and issued < expires, 'Registry is future-dated or expired; reconcile ownership')
    authorization = registry['authorization']
    exact_keys(authorization, {'action', 'user_request_sha256'}, 'Registry authorization binding')
    require(authorization['action'] == 'apply_project_changes' and authorization['user_request_sha256'] == contract['user_request_sha256'], 'Registry lacks task-bound project-change authorization')
    require(isinstance(registry['assignments'], list) and registry['assignments'], 'Assigned production ownership is required')
    assignments = {}
    for assignment in registry['assignments']:
        exact_keys(assignment, {'id', 'owner', 'write_paths', 'state', 'expires_at', 'pending_operations', 'worker_public_key_sha256'}, 'Registry assignment')
        require(nonempty(assignment['id']) and assignment['id'] not in assignments and nonempty(assignment['owner']), 'Duplicate or missing assignment identity')
        require(assignment['state'] == 'active', 'Assignment is not active')
        require(isinstance(assignment['write_paths'], list) and assignment['write_paths'], 'Assignment must own literal paths')
        require(assignment['pending_operations'] == [], 'Assignment has unreconciled pending operations')
        expiry = timestamp(assignment['expires_at'], 'Assignment expiry')
        require(current_time() < expiry <= expires, 'Assignment is expired or outlives registry')
        key_hash = assignment['worker_public_key_sha256']
        require(isinstance(key_hash, str) and len(key_hash) == 64 and all(c in '0123456789abcdef' for c in key_hash), 'Worker signing-key digest is required')
        assignments[assignment['id']] = assignment
    actual = sorted((item['owner'], tuple(item['write_paths'])) for item in assignments.values())
    expected = sorted((item['owner'], tuple(item['write_paths'])) for item in contract['ownership']['assignments'])
    require(actual == expected, 'Registry ownership differs from signed task ownership')
    validate_ownership(list(assignments.values()), contract['source_root'])
    require(isinstance(registry['exclusive_resources'], list), 'Exclusive resource assignments must be a list')
    resources = set()
    for resource in registry['exclusive_resources']:
        exact_keys(resource, {'kind', 'id', 'assignment_id'}, 'Exclusive resource')
        require(resource['kind'] in {'branch', 'live_session'} and nonempty(resource['id']), 'Unknown or missing exclusive resource')
        key = (resource['kind'], resource['id'])
        require(key not in resources and resource['assignment_id'] in assignments, 'Exclusive resource has duplicate or unknown ownership')
        resources.add(key)
    require(isinstance(registry['reconciliations'], list), 'Reconciliation records must be a list')
    if registry['previous_registry_sha256'] is None:
        require(registry['generation'] == 1 and not registry['reconciliations'] and previous_registry is None, 'Initial registry cannot omit prior-generation history')
    else:
        require(previous_registry is not None, 'Ownership transition requires the previous signed registry')
        previous, prior_hash = verify_signed_contract(previous_registry, trusted['public_key_path'], trusted['public_key_sha256'])
        require(prior_hash == registry['previous_registry_sha256'], 'Previous registry digest mismatch')
        historical_registry(previous)
        require(previous['contract_generation'] <= registry['contract_generation'], 'Contract generation cannot move backwards across registry transition')
        require(previous['contract_generation'] != registry['contract_generation'] or previous['contract_sha256'] == registry['contract_sha256'], 'Contract bytes changed without a new contract generation')
        require(previous.get('task_id') == registry['task_id'] and previous.get('writer') == registry['writer'] and type(previous.get('generation')) is int and previous['generation'] + 1 == registry['generation'], 'Registry transition is not the next generation of this task')
        require(isinstance(previous.get('assignments'), list), 'Previous registry assignments missing')
        changed = {}
        for old in previous['assignments']:
            require(isinstance(old, dict) and nonempty(old.get('id')), 'Previous assignment identity missing')
            replacement = assignments.get(old['id'])
            if replacement is None or old['pending_operations'] or any(replacement.get(field) != old.get(field) for field in ('owner', 'write_paths', 'worker_public_key_sha256')):
                changed[old['id']] = old
        old_assignments = {item['id']: item for item in previous['assignments']}
        new_resources = {(item['kind'], item['id']): item['assignment_id'] for item in registry['exclusive_resources']}
        for resource in previous['exclusive_resources']:
            if new_resources.get((resource['kind'], resource['id'])) != resource['assignment_id']:
                changed[resource['assignment_id']] = old_assignments[resource['assignment_id']]
        reconciled = set()
        for item in registry['reconciliations']:
            exact_keys(item, {'assignment_id', 'owner', 'write_paths', 'outcome', 'pending_operations', 'final_source_manifest_sha256'}, 'Ownership reconciliation')
            identifier = item['assignment_id']
            require(identifier in changed and identifier not in reconciled, 'Reconciliation does not identify a transferred assignment')
            old = changed[identifier]
            require(item['owner'] == old.get('owner') and item['write_paths'] == old.get('write_paths'), 'Reconciliation differs from prior ownership')
            require(item['outcome'] == 'reconciled' and item['pending_operations'] == [], 'Transfer has unreconciled pending operations')
            require(item['final_source_manifest_sha256'] == registry['source_manifest_sha256'], 'Reconciliation source is stale')
            reconciled.add(identifier)
        require(reconciled == set(changed), 'Ownership transfer lacks signed reconciliation of the old owner')
    return registry, registry_hash, assignments


def admit(*, registry_path, expected_registry_sha256, registry_generation, worker, assignment_id, requested_paths, previous_registry=None, **trusted):
    payload, contract_hash, revision, source, gates, checks = load_contract(trusted['contract_path'], trusted['public_key_path'], trusted['public_key_sha256'], trusted['expected_contract_sha256'], trusted['source_root'])
    registry, registry_hash, assignments = verify_registry(registry_path, trusted, payload, contract_hash, revision, expected_registry_sha256, registry_generation, previous_registry)
    require(assignment_id in assignments, 'Unknown assignment')
    assignment = assignments[assignment_id]
    require(assignment['owner'] == worker, 'Worker does not own this assignment')
    assigned_paths(requested_paths, assignment)
    exclusions = [item['path'] for item in payload['source_manifest']['excluded_paths']]
    for path in requested_paths:
        require(not any(inside(path, excluded) or inside(excluded, path) for excluded in exclusions), 'Requested write overlaps excluded source: ' + path)
    result = {'state': 'admitted', 'task_id': payload['task_id'], 'worker': worker, 'assignment_id': assignment_id, 'contract_sha256': contract_hash, 'contract_generation': payload['generation'], 'registry_sha256': registry_hash, 'registry_generation': registry_generation, 'source_manifest_sha256': revision.removeprefix('sha256:'), 'requested_paths': requested_paths, 'writes_performed': False, 'publication_performed': False}
    return result, payload, registry, assignment, source


def manifest_delta(before, after):
    old, new = file_map(before), file_map(after)
    return [{'path': path, 'operation': 'create' if path not in old else 'delete' if path not in new else 'replace', 'before_sha256': old.get(path), 'after_sha256': new.get(path)} for path in sorted(set(old) | set(new)) if old.get(path) != new.get(path)]


def validate_handoff(*, handoff_path, worker_public_key_path, artifact_root, **options):
    assignment_id = options['assignment_id']
    initial, payload, registry, assignment, source = admit(**options)
    handoff, handoff_hash = verify_signed_contract(handoff_path, worker_public_key_path, assignment['worker_public_key_sha256'])
    exact_keys(handoff, {'schema_version', 'task_id', 'contract_sha256', 'contract_generation', 'registry_sha256', 'registry_generation', 'assignment_id', 'worker', 'base_manifest_sha256', 'result_manifest', 'changes', 'outcome', 'pending_operations'}, 'Worker handoff')
    require(type(handoff['schema_version']) is int and handoff['schema_version'] == 1, 'Unsupported handoff schema')
    for field in ('task_id', 'contract_sha256', 'contract_generation', 'registry_sha256', 'registry_generation', 'assignment_id', 'worker'):
        require(type(handoff[field]) is type(initial[field]) and handoff[field] == initial[field], 'Handoff identity mismatch: ' + field)
    require(handoff['base_manifest_sha256'] == initial['source_manifest_sha256'], 'Handoff base source is stale')
    require(handoff['outcome'] == 'ready_for_integration' and handoff['pending_operations'] == [], 'Handoff has unresolved work or operations')
    candidate = Path(artifact_root).resolve(strict=True)
    require(candidate.is_dir() and not candidate.is_relative_to(source) and not source.is_relative_to(candidate), 'Candidate and live source must be separate trees')
    result_manifest = handoff['result_manifest']
    verify_source_manifest(result_manifest, candidate)
    for field in ('schema_version', 'base_head', 'excluded_paths', 'external_assets'):
        require(result_manifest[field] == payload['source_manifest'][field], 'Handoff changes source scope or repository base: ' + field)
    delta = manifest_delta(payload['source_manifest'], result_manifest)
    require(delta and handoff['changes'] == delta, 'Declared handoff changes differ from actual candidate source delta')
    paths = [item['path'] for item in delta]
    require(set(paths) == set(options['requested_paths']), 'Requested paths do not equal actual handoff changes')
    assigned_paths(paths, assignment)
    verify_source_manifest(payload['source_manifest'], source)
    result = dict(initial, state='handoff_validated', handoff_sha256=handoff_hash, candidate_manifest_sha256=digest_bytes(canonical(result_manifest)), changes=delta)
    return result, payload, registry, assignment, source, candidate, handoff


def read_at(root, relative):
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative_path(relative).parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        leaf = os.open(relative_path(relative).parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        try:
            before = os.fstat(leaf)
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'Handoff input is not an unaliased regular file: ' + relative)
            chunks = []
            while True:
                chunk = os.read(leaf, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(leaf)
            require((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns, before.st_size) == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns, after.st_size), 'Handoff input changed while reading: ' + relative)
            return b''.join(chunks), stat.S_IMODE(before.st_mode)
        finally:
            os.close(leaf)
    finally:
        os.close(descriptor)


def replace_at(root, relative, expected_hash, data, mode, created_directories):
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    anchors = [(Path(root), os.fstat(descriptor))]
    current = Path(root)
    try:
        parts = relative_path(relative).parts
        for part in parts[:-1]:
            current = current / part
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            except FileNotFoundError:
                require(data is not None and expected_hash is None, 'Required source parent disappeared: ' + relative)
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                created_directories.append(current)
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            anchors.append((current, os.fstat(descriptor)))
        name = parts[-1]
        for path, held in anchors:
            present = path.lstat()
            require(stat.S_ISDIR(present.st_mode) and (present.st_dev, present.st_ino) == (held.st_dev, held.st_ino), 'Source parent changed before write: ' + relative)
        try:
            value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            value = None
        if expected_hash is None:
            require(value is None, 'Create target already exists: ' + relative)
        else:
            require(value is not None and stat.S_ISREG(value.st_mode) and value.st_nlink == 1, 'Write target changed type or identity: ' + relative)
            old, old_mode = read_at(root, relative)
            require(digest_bytes(old) == expected_hash, 'Write target changed bytes: ' + relative)
        if data is None:
            require(value is not None, 'Delete target is missing: ' + relative)
            present = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            require((present.st_dev, present.st_ino, present.st_mtime_ns, present.st_ctime_ns, present.st_size) == (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns, value.st_size), 'Write target changed before deletion: ' + relative)
            os.unlink(name, dir_fd=descriptor)
        else:
            temporary = '.handoff-' + uuid.uuid4().hex
            try:
                leaf = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=descriptor)
                try:
                    remaining = memoryview(data)
                    while remaining:
                        written = os.write(leaf, remaining)
                        require(written > 0, 'Incomplete handoff write')
                        remaining = remaining[written:]
                    os.fchmod(leaf, mode)
                    os.fsync(leaf)
                finally:
                    os.close(leaf)
                for anchor, held in anchors:
                    present = anchor.lstat()
                    require(stat.S_ISDIR(present.st_mode) and (present.st_dev, present.st_ino) == (held.st_dev, held.st_ino), 'Source parent changed during write: ' + relative)
                if expected_hash is None:
                    os.link(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
                    os.unlink(temporary, dir_fd=descriptor)
                else:
                    present = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    require((present.st_dev, present.st_ino, present.st_mtime_ns, present.st_ctime_ns, present.st_size) == (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns, value.st_size), 'Write target changed before replacement: ' + relative)
                    os.replace(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
            finally:
                try:
                    os.unlink(temporary, dir_fd=descriptor)
                except FileNotFoundError:
                    pass
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_journal(path, value):
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def apply_handoff(*, journal_root, **options):
    source = Path(options['source_root']).resolve(strict=True)
    lock_path = source.parent / ('.' + source.name + '.project-write.lock')
    lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(lock).st_mode) and os.fstat(lock).st_nlink == 1, 'Invalid source integration lock')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another integrator holds the source write lock') from error
        prior_lock = os.read(lock, 1024 * 1024)
        if prior_lock:
            prior = json.loads(prior_lock)
            require(prior.get('state') in {'applied_requires_acceptance', 'rolled_back'}, 'Previous integration is unresolved; reconcile its journal before writing: ' + str(prior.get('journal')))
            require(isinstance(prior.get('journal'), str), 'Previous integration journal identity is missing')
            previous_journal = read_json(prior['journal'])
            require(previous_journal.get('state') == prior['state'] and previous_journal.get('source_root') == str(source), 'Previous integration lock and journal disagree')
        result, payload, registry, assignment, source, candidate, handoff = validate_handoff(**options)
        journal_parent = Path(journal_root).resolve()
        require(not journal_parent.is_relative_to(source) and not journal_parent.is_relative_to(candidate), 'Journal must be outside source and candidate')
        journal_parent.mkdir(parents=True, exist_ok=True)
        run = journal_parent / ('integration-' + uuid.uuid4().hex)
        run.mkdir(mode=0o700)
        path = run / 'journal.json'
        prepared = []
        for index, change in enumerate(result['changes']):
            old, old_mode = (None, 0o644) if change['operation'] == 'create' else read_at(source, change['path'])
            new, new_mode = (None, old_mode) if change['operation'] == 'delete' else read_at(candidate, change['path'])
            require(old is None or digest_bytes(old) == change['before_sha256'], 'Source changed before transaction preparation')
            require(new is None or digest_bytes(new) == change['after_sha256'], 'Candidate changed before transaction preparation')
            require(not (old_mode & ~0o777) and not (new_mode & ~0o777), 'Special permission bits are not supported by byte-manifest handoffs')
            require(new is None or new_mode == old_mode, 'Permission changes are not supported by byte-manifest handoffs; new files require mode 0644')
            if old is not None:
                (run / ('before-' + str(index))).write_bytes(old)
            prepared.append((change, old, old_mode, new, old_mode if old is not None else new_mode))
        journal = {'schema_version': 1, 'state': 'prepared', 'source_root': str(source), 'handoff_sha256': result['handoff_sha256'], 'contract_sha256': result['contract_sha256'], 'registry_sha256': result['registry_sha256'], 'before_manifest': payload['source_manifest'], 'after_manifest': handoff['result_manifest'], 'changes': result['changes'], 'applied_paths': [], 'error': None, 'rollback_errors': []}
        write_journal(path, journal)
        def lock_state(state):
            data = canonical({'state': state, 'journal': str(path)})
            os.lseek(lock, 0, os.SEEK_SET)
            os.ftruncate(lock, 0)
            os.write(lock, data)
            os.fsync(lock)
        lock_state('applying')
        applied = []
        created = []
        try:
            verify_source_manifest(payload['source_manifest'], source)
            for change, old, old_mode, new, mode in prepared:
                latest, latest_hash = verify_signed_contract(options['registry_path'], options['public_key_path'], options['public_key_sha256'])
                require(latest_hash == result['registry_sha256'] and latest == registry, 'Registry changed or assignment was revoked before write')
                require(current_time() < timestamp(assignment['expires_at'], 'Assignment expiry') and current_time() < timestamp(registry['expires_at'], 'Registry expiry'), 'Ownership expired before write')
                contract_now, contract_hash = verify_signed_contract(options['contract_path'], options['public_key_path'], options['public_key_sha256'])
                require(contract_hash == result['contract_sha256'] and contract_now == payload, 'Task contract changed before write')
                applied.append((change, old, old_mode, new, mode))
                replace_at(source, change['path'], change['before_sha256'], new, mode, created)
                journal['applied_paths'].append(change['path'])
                journal['state'] = 'applying'
                write_journal(path, journal)
            verify_source_manifest(handoff['result_manifest'], source)
        except BaseException as error:
            journal['error'] = type(error).__name__ + ': ' + str(error)
            for change, old, old_mode, new, mode in reversed(applied):
                try:
                    target = source / relative_path(change['path'])
                    if not target.exists() and not target.is_symlink():
                        current_hash = None
                    else:
                        current_bytes, current_mode = read_at(source, change['path'])
                        current_hash = digest_bytes(current_bytes)
                    if current_hash == change['before_sha256']:
                        continue
                    require(current_hash == change['after_sha256'], 'Concurrent change prevents rollback: ' + change['path'])
                    replace_at(source, change['path'], change['after_sha256'], old, old_mode, [])
                except (OSError, ValueError) as rollback_error:
                    journal['rollback_errors'].append(change['path'] + ': ' + str(rollback_error))
            for directory in reversed(created):
                try:
                    directory.rmdir()
                except OSError as cleanup_error:
                    journal['rollback_errors'].append(str(directory) + ': ' + str(cleanup_error))
            if not journal['rollback_errors']:
                try:
                    verify_source_manifest(payload['source_manifest'], source)
                except (OSError, ValueError) as rollback_error:
                    journal['rollback_errors'].append(str(rollback_error))
            journal['state'] = 'failed_requires_reconciliation' if journal['rollback_errors'] else 'rolled_back'
            write_journal(path, journal)
            lock_state(journal['state'])
            raise ValueError('Integration failed; ' + journal['state'] + '; journal=' + str(path) + '; ' + journal['error']) from error
        journal['state'] = 'applied_requires_acceptance'
        write_journal(path, journal)
        lock_state(journal['state'])
        return dict(result, state='applied_requires_acceptance', writes_performed=True, journal=str(path), subject_revision='sha256:' + digest_bytes(canonical(handoff['result_manifest'])))
    finally:
        os.close(lock)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--trusted-public-key', required=True, type=Path)
    parser.add_argument('--trusted-key-sha256', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--registry', required=True, type=Path)
    parser.add_argument('--registry-sha256', required=True)
    parser.add_argument('--registry-generation', required=True, type=int)
    parser.add_argument('--previous-registry', type=Path)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--requested-path', action='append', required=True)
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--worker-public-key', type=Path)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--journal-root', type=Path)
    args = parser.parse_args()
    options = {'contract_path': args.contract, 'public_key_path': args.trusted_public_key, 'public_key_sha256': args.trusted_key_sha256, 'expected_contract_sha256': args.contract_sha256, 'source_root': args.source_root, 'registry_path': args.registry, 'expected_registry_sha256': args.registry_sha256, 'registry_generation': args.registry_generation, 'previous_registry': args.previous_registry, 'worker': args.worker, 'assignment_id': args.assignment, 'requested_paths': args.requested_path}
    if args.handoff:
        require(args.worker_public_key is not None and args.artifact_root is not None, 'Handoff requires worker public key and candidate source root')
        options.update(handoff_path=args.handoff, worker_public_key_path=args.worker_public_key, artifact_root=args.artifact_root)
        if args.apply:
            require(args.journal_root is not None, 'Applying changes requires an external journal root')
            result = apply_handoff(journal_root=args.journal_root, **options)
        else:
            require(args.journal_root is None, 'Journal output is only valid with --apply')
            result = validate_handoff(**options)[0]
    else:
        require(not args.apply and args.worker_public_key is None and args.artifact_root is None and args.journal_root is None, 'Write application requires an actual signed worker handoff')
        result = admit(**options)[0]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
