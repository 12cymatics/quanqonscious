import os
from pathlib import Path
import stat
import gate
def relative(value):
    path = Path(value)
    gate.require(not path.is_absolute() and '..' not in path.parts and value not in ('', '.'), 'snapshot path must be a nonempty relative path')
    gate.require(str(path) == value, 'snapshot path must use canonical separators')
    return path
def within(value, directory):
    return value == directory or value.startswith(directory + '/')
def create(source_root, selected_paths, destination, writable=()):
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    gate.require(not destination.exists(), 'snapshot destination already exists')
    paths = sorted(set(str(relative(item)) for item in selected_paths))
    writable = sorted(set(str(relative(item)) for item in writable))
    gate.require(not any(within(path, output) or within(output, path) for path in paths for output in writable), 'writable output overlaps immutable source')
    files = []
    destination.mkdir(mode=0o700, parents=True)
    for name in paths:
        original = source_root / name
        gate.require(original.resolve().is_relative_to(source_root) and not original.is_symlink() and original.is_file(), 'source is not a regular file within source root')
        raw = original.read_bytes()
        target = destination / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        executable = bool(original.stat().st_mode & 0o111)
        gate.write_once(target, raw, 0o555 if executable else 0o444)
        files.append({'path': name, 'length': len(raw), 'sha256': gate.sha(raw), 'executable': executable})
    for name in writable:
        (destination / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    for directory, _, _ in os.walk(destination, topdown=False):
        path = Path(directory)
        name = path.relative_to(destination).as_posix()
        if not any(within(name, output) for output in writable):
            path.chmod(0o555)
    manifest = {'root': str(destination), 'files': files, 'writable': writable}
    validate(manifest)
    return manifest
def validate(manifest):
    gate.fields(manifest, ('root', 'files', 'writable'))
    root = Path(manifest['root'])
    gate.require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'snapshot root missing or replaced')
    writable = manifest['writable']
    gate.require(isinstance(writable, list) and writable == sorted(set(writable)), 'invalid writable paths')
    for name in writable:
        relative(name)
        path = root / name
        gate.require(path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(root), 'writable path escapes snapshot')
    expected = set()
    for item in manifest['files']:
        gate.fields(item, ('path', 'length', 'sha256', 'executable'))
        name = str(relative(item['path']))
        gate.require(name not in expected and not any(within(name, output) for output in writable), 'duplicate or writable source file')
        expected.add(name)
        path = root / name
        gate.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'snapshot source missing or replaced')
        raw = path.read_bytes()
        gate.require(type(item['length']) is int and len(raw) == item['length'] and gate.sha(raw) == item['sha256'], 'snapshot source bytes changed')
        gate.require(type(item['executable']) is bool and stat.S_IMODE(path.stat().st_mode) == (0o555 if item['executable'] else 0o444), 'snapshot source permissions changed')
    observed = set()
    for directory, dirs, files in os.walk(root):
        current = Path(directory)
        gate.require(not current.is_symlink(), 'snapshot directory symlink')
        name = current.relative_to(root).as_posix()
        dirs[:] = [child for child in dirs if not any(within((current / child).relative_to(root).as_posix(), output) for output in writable)]
        gate.require(stat.S_IMODE(current.stat().st_mode) == 0o555, 'immutable snapshot directory became writable')
        for child in dirs:
            gate.require(not (current / child).is_symlink(), 'snapshot directory symlink')
        for child in files:
            item = (current / child).relative_to(root).as_posix()
            if not any(within(item, output) for output in writable):
                observed.add(item)
    gate.require(observed == expected, 'snapshot file inventory changed')
    return manifest
