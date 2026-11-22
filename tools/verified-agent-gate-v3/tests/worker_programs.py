BYTE_ECHO = r'''import base64
import hashlib
import json
import sys
from pathlib import Path

original = Path('/job/instruction.bin').read_bytes()
report = {
    'instruction_b64': base64.b64encode(original).decode('ascii'),
    'instruction_sha256': hashlib.sha256(original).hexdigest(),
    'instruction_length': len(original),
    'argv': sys.argv[1:],
    'envelope_present': Path('/job/envelope.json').is_file(),
    'manifest_present': Path('/job/manifest.json').is_file(),
    'global_instruction_b64': base64.b64encode(GATE_INSTRUCTION_BYTES).decode('ascii'),
    'global_instruction_sha256': GATE_INSTRUCTION_SHA256,
    'assigned_operation': GATE_ASSIGNED_OPERATION,
    'handoff_id': GATE_HANDOFF_ID,
    'parent_sha256': GATE_PARENT_SHA256,
}
Path('/output/result.json').write_text(json.dumps(report, sort_keys=True))
print('authoritative bytes received')
'''

BOUNDARY_PROBE = r'''import ctypes
import errno
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

results = {}

def denied(name, action):
    try:
        value = action()
    except OSError as exc:
        results[name] = {'denied': True, 'errno': exc.errno}
    else:
        if hasattr(value, 'close'):
            value.close()
        results[name] = {'denied': False}

denied('host_state_read', lambda: Path(sys.argv[1]).read_bytes())
denied('job_write', lambda: Path('/job/unauthorized-write').write_bytes(b'bad'))
denied('bootstrap_write', lambda: Path('/gate/bootstrap.py').write_bytes(b'bad'))
denied('network_inet', lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
denied('network_unix', lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
denied('subprocess', lambda: subprocess.run(['/usr/bin/true'], check=True))

try:
    child = os.fork()
except OSError as exc:
    results['fork'] = {'denied': True, 'errno': exc.errno}
else:
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    results['fork'] = {'denied': False}

libc = ctypes.CDLL(None, use_errno=True)
rc = libc.system(b'/usr/bin/true')
results['libc_system'] = {'denied': rc != 0, 'return_code': rc}

Path('/output/boundary.json').write_text(json.dumps(results, sort_keys=True))
'''

FAIL_WITH_OUTPUT = r'''import sys
from pathlib import Path
Path('/output/before-failure.txt').write_text('retained output')
print('worker stdout before failure', flush=True)
print('worker stderr before failure', file=sys.stderr, flush=True)
raise SystemExit(7)
'''

TIMEOUT = r'''import time
from pathlib import Path
Path('/output/started.txt').write_text('started')
time.sleep(30)
Path('/output/should-not-exist.txt').write_text('timeout failed')
'''

EXEC_PROBE = r'''import json
import os
from pathlib import Path

try:
    os.execv('/usr/bin/python3', ['/usr/bin/python3', '-I', '-S', '-c',
        "from pathlib import Path; Path('/output/exec-escaped.txt').write_text('bad')"])
except OSError as exc:
    Path('/output/exec-result.json').write_text(json.dumps({'denied': True, 'errno': exc.errno}))
'''
