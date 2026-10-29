import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import zipfile

from validation_identity import _read_regular, read_json, relative_path, require


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_command(name, command, root, logs, expected_exit=0, diagnostic=None, expected_field=None):
    started = now()
    environment = dict(os.environ)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    timed_out = False
    launch_error = None
    try:
        process = subprocess.run(command, cwd=root, env=environment, capture_output=True, timeout=1200)
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b''
        stderr = (error.stderr or b'') + b'\nRequired unpacked-package command timed out.\n'
        returncode = None
        timed_out = True
    except OSError as error:
        stdout = b''
        stderr = str(error).encode()
        returncode = None
        launch_error = str(error)
    (logs / (name + '.stdout')).write_bytes(stdout)
    (logs / (name + '.stderr')).write_bytes(stderr)
    passed = not timed_out and launch_error is None and returncode == expected_exit
    if diagnostic is not None:
        passed = passed and diagnostic.encode() in stderr
    if expected_field is not None and passed:
        try:
            value = json.loads(stdout)
            passed = isinstance(value, dict) and value.get(expected_field[0]) == expected_field[1]
        except (ValueError, UnicodeDecodeError):
            passed = False
    result = {
        'name': name,
        'command': command,
        'working_directory': str(root),
        'started_at': started,
        'finished_at': now(),
        'exit_code': returncode,
        'expected_exit_code': expected_exit,
        'required_diagnostic': diagnostic,
        'expected_field': expected_field,
        'timed_out': timed_out,
        'launch_error': launch_error,
        'passed': passed,
        'stdout_sha256': digest(stdout),
        'stderr_sha256': digest(stderr),
    }
    write_json(logs / (name + '.execution.json'), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def manifest_command(root, logs, name):
    return run_command(name, [sys.executable, '-B', 'verify_package.py'], root, logs, expected_field=('package_integrity', 'passed'))


def create_archive(root, archive):
    manifest_path = root / 'SHA256.json'
    manifest_bytes = _read_regular(manifest_path)
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and manifest, 'Package manifest must be a nonempty object')
    expected = dict(manifest)
    require('SHA256.json' not in expected, 'Package manifest cannot hash itself')
    expected['SHA256.json'] = digest(manifest_bytes)
    names = sorted(expected)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in names:
            path = relative_path(name)
            require(path.parts[0] != '.git', 'Git metadata cannot enter the package')
            source = root / name
            content = _read_regular(source)
            require(digest(content) == expected[name], 'Source changed before archive entry: ' + name)
            mode = source.lstat().st_mode
            require(stat.S_ISREG(mode), 'Archive source is not a regular file: ' + name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | stat.S_IMODE(mode)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    require(_read_regular(manifest_path) == manifest_bytes, 'Manifest changed during archive creation')
    return names, expected


def extract_verified(archive, destination, names, expected):
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive, 'r') as bundle:
        members = bundle.infolist()
        require([member.filename for member in members] == names, 'ZIP member inventory differs from the manifest')
        require(len(set(member.filename for member in members)) == len(members), 'ZIP has duplicate members')
        for member in members:
            relative = relative_path(member.filename)
            require(not member.is_dir() and not member.flag_bits & 1, 'ZIP member must be an unencrypted regular file')
            mode = member.external_attr >> 16
            require(stat.S_ISREG(mode), 'ZIP member is not a regular file: ' + member.filename)
            target = destination / relative
            require(target.resolve(strict=False).is_relative_to(destination), 'ZIP member escapes unpacked directory')
            content = bundle.read(member)
            require(digest(content) == expected[member.filename], 'ZIP member digest differs: ' + member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(content)
            target.chmod(stat.S_IMODE(mode))


def check_packaged(source_root, output_root):
    source = Path(source_root).resolve(strict=True)
    require(source.is_dir(), 'Package source root must be a directory')
    output = Path(output_root).resolve(strict=False)
    require(not output.is_relative_to(source), 'Packaged verification output must be outside the source tree')
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    state = {
        'schema_version': 1,
        'unpacked_acceptance': 'running',
        'source_root': str(source),
        'output_root': str(output),
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'commit': os.environ.get('GITHUB_SHA'),
        'started_at': now(),
        'commands': [],
    }
    record = output / 'packaged-result.json'
    write_json(record, state)
    try:
        state['commands'].append(manifest_command(source, logs, 'source-manifest-before'))
        require(state['commands'][-1]['passed'], 'Source manifest verification failed before packaging')
        archive = output / 'collaboration-readiness.zip'
        names, expected = create_archive(source, archive)
        archive_digest = digest(_read_regular(archive))
        state['archive'] = str(archive)
        state['archive_sha256'] = archive_digest
        state['archive_files'] = len(names)
        write_json(output / 'archive-contents.json', expected)
        state['commands'].append(manifest_command(source, logs, 'source-manifest-after'))
        require(state['commands'][-1]['passed'], 'Source manifest verification failed after packaging')
        unpacked = output / 'unpacked'
        extract_verified(archive, unpacked, names, expected)
        commands = [
            ('unpacked-manifest-before', [sys.executable, '-B', 'verify_package.py'], 0, None, ('package_integrity', 'passed')),
            ('unpacked-draft-contract', [sys.executable, '-B', 'validate_contracts.py', '--self-test'], 0, None, ('contract_consistency', 'passed')),
            ('unpacked-unit-suite', [sys.executable, '-B', '-m', 'unittest', 'discover', '-v'], 0, None, None),
            ('unpacked-template-shape', [sys.executable, '-B', 'validate_readiness.py', '--record', 'readiness.template.json', '--evidence-root', '.'], 0, None, ('readiness_attestation', 'not_evaluated')),
            ('unpacked-template-readiness-rejected', [sys.executable, '-B', 'validate_readiness.py', '--record', 'readiness.template.json', '--evidence-root', '.', '--readiness'], 1, 'Template cannot claim readiness or completion', None),
            ('unpacked-finite-structures', [sys.executable, '-B', 'geometry/verify_structures.py'], 0, None, None),
            ('unpacked-manifest-after', [sys.executable, '-B', 'verify_package.py'], 0, None, ('package_integrity', 'passed')),
        ]
        for name, command, expected_exit, diagnostic, expected_field in commands:
            state['commands'].append(run_command(name, command, unpacked, logs, expected_exit, diagnostic, expected_field))
            write_json(record, state)
        require(digest(_read_regular(archive)) == archive_digest, 'Delivered ZIP changed during unpacked verification')
        require(all(result['passed'] for result in state['commands']), 'One or more required unpacked-package checks failed')
        state['unpacked_acceptance'] = 'passed'
        state['finished_at'] = now()
        write_json(record, state)
        return state
    except Exception as error:
        state['unpacked_acceptance'] = 'failed'
        state['error'] = str(error)
        state['finished_at'] = now()
        write_json(record, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    result = check_packaged(args.source_root, args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, KeyError, OSError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
