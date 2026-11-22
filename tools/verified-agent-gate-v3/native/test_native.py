"""Real-process checks for the mandatory Linux boundary, without bypass modes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

RUNNER = Path(__file__).resolve().parent / "gate-sandbox"


def prepare(root, source):
    job, output = root / "job", root / "output"
    job.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    content = {
        "main.py": source.encode(),
        "instruction.bin": b"Preserve every byte: \x00\xff\r\n",
        "task.json": json.dumps({"entrypoint": "main.py", "argv": ["accepted-argument"],
                                  "assigned_operation": "Check the real native execution boundary.",
                                  "handoff_id": "a" * 64, "parent_sha256": None}).encode(),
        "source-manifest.json": b"{}",
        "authorization.txt": b"Native boundary acceptance test; controller authorization is tested separately.",
    }
    entries = []
    for name, raw in sorted(content.items()):
        (job / name).write_bytes(raw)
        entries.append({"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    (job / "manifest.json").write_text(json.dumps({"schema": "job-files-v3", "files": entries}))
    (job / "envelope.json").write_text("{}")
    return job, output


def descendants(pid):
    by_parent = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        by_parent.setdefault(int(fields[1]), []).append(int(entry.name))
    found = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        for child in by_parent.get(parent, []):
            found.append(child)
            pending.append(child)
    return found


def active(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != "Z"


class NativeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gate-native-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def command(self, job, output, timeout=5, *extra):
        return [str(RUNNER), "--controller-pid", str(os.getpid()), "--job", str(job), "--output", str(output), "--timeout", str(timeout), *extra]

    def test_real_worker_kernel_restrictions_and_outputs(self):
        secret = self.root / "controller-secret"
        secret.write_text("must remain inaccessible")
        source = '''import ctypes, errno, json, os, socket, sys
from pathlib import Path
checks = {}
operations = {
    "network": lambda: socket.socket(),
    "socketpair": lambda: socket.socketpair(),
    "fork": os.fork,
    "exec": lambda: os.execv("/usr/bin/true", ["true"]),
    "job_write": lambda: Path("/job/main.py").write_text("changed"),
    "bootstrap_write": lambda: Path("/gate/bootstrap.py").write_text("changed"),
    "host_secret": lambda: Path(HOST_SECRET).read_bytes(),
    "inherited_fd": lambda: os.read(INHERITED_FD, 1),
}
for name, operation in operations.items():
    try:
        operation()
        checks[name] = "UNEXPECTEDLY_ALLOWED"
    except OSError as error:
        checks[name] = error.errno
checks["instruction"] = GATE_INSTRUCTION_BYTES.hex()
checks["argument"] = sys.argv[1]
checks["operation"] = GATE_ASSIGNED_OPERATION
checks["handoff_id"] = GATE_HANDOFF_ID
checks["parent"] = GATE_PARENT_SHA256
checks["pid"] = os.getpid()
checks["environment_keys"] = sorted(os.environ)
Path("/output/result.json").write_text(json.dumps(checks))
'''
        fd = os.open(secret, os.O_RDONLY)
        try:
            source = source.replace("HOST_SECRET", repr(str(secret))).replace("INHERITED_FD", str(fd))
            job, output = prepare(self.root, source)
            process = subprocess.run(self.command(job, output), capture_output=True, text=True, pass_fds=(fd,), timeout=10)
        finally:
            os.close(fd)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads((output / "result.json").read_bytes())
        for key in ["network", "socketpair", "fork", "exec"]:
            self.assertEqual(result[key], 1, key)
        self.assertEqual(result["job_write"], 30)
        self.assertEqual(result["bootstrap_write"], 30)
        self.assertEqual(result["host_secret"], 2)
        self.assertEqual(result["inherited_fd"], 9)
        self.assertEqual(result["instruction"], (job / "instruction.bin").read_bytes().hex())
        self.assertEqual(result["argument"], "accepted-argument")
        self.assertEqual(result["operation"], "Check the real native execution boundary.")
        self.assertEqual(result["handoff_id"], "a" * 64)
        self.assertIsNone(result["parent"])
        self.assertLess(result["pid"], 10)

    def test_snapshot_mutation_rejected_before_entrypoint(self):
        job, output = prepare(self.root, 'from pathlib import Path\nPath("/output/ran").write_text("bad")\n')
        (job / "main.py").write_text('print("modified")\n')
        process = subprocess.run(self.command(job, output), capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 125)
        self.assertFalse((output / "ran").exists())
        self.assertIn("mismatch", process.stderr)

    def test_injected_symlink_rejected_before_entrypoint(self):
        job, output = prepare(self.root, 'from pathlib import Path\nPath("/output/ran").write_text("bad")\n')
        (job / "escape").symlink_to("/usr")
        process = subprocess.run(self.command(job, output), capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 125)
        self.assertFalse((output / "ran").exists())
        self.assertIn("symlink prohibited", process.stderr)

    def test_wall_timeout_terminates_worker(self):
        job, output = prepare(self.root, 'from pathlib import Path\nimport time\nPath("/output/started").write_text("yes")\nwhile True: time.sleep(0.01)\n')
        started = time.monotonic()
        process = subprocess.run(self.command(job, output, 1), capture_output=True, text=True, timeout=6)
        self.assertEqual(process.returncode, 124)
        self.assertTrue((output / "started").exists())
        self.assertLess(time.monotonic() - started, 4)

    def test_supervisor_death_terminates_namespace(self):
        job, output = prepare(self.root, 'from pathlib import Path\nimport time\nPath("/output/started").write_text("yes")\nwhile True: time.sleep(0.01)\n')
        process = subprocess.Popen(self.command(job, output, 30), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not (output / "started").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(process.stderr.read().decode())
                time.sleep(0.01)
            self.assertTrue((output / "started").exists())
            children = descendants(process.pid)
            self.assertGreaterEqual(len(children), 2)
            process.kill()
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while any(active(pid) for pid in children) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(any(active(pid) for pid in children), children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if process.stderr:
                process.stderr.close()

    def test_controller_death_terminates_runner_and_namespace(self):
        job, output = prepare(self.root, 'from pathlib import Path\nimport time\nPath("/output/started").write_text("yes")\nwhile True: time.sleep(0.01)\n')
        controller_source = '''import os, subprocess, sys
child = subprocess.Popen([sys.argv[1], "--controller-pid", str(os.getpid()), "--job", sys.argv[2], "--output", sys.argv[3], "--timeout", "30"])
child.wait()
'''
        process = subprocess.Popen([sys.executable, "-I", "-S", "-c", controller_source,
                                    str(RUNNER), str(job), str(output)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        children = []
        try:
            deadline = time.monotonic() + 5
            while not (output / "started").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(process.stderr.read().decode())
                time.sleep(0.01)
            self.assertTrue((output / "started").exists())
            children = descendants(process.pid)
            self.assertGreaterEqual(len(children), 4)
            process.kill()
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while any(active(pid) for pid in children) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(any(active(pid) for pid in children), children)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for child in children:
                if active(child):
                    os.kill(child, signal.SIGKILL)
            if process.stderr:
                process.stderr.close()

    def test_wrong_controller_rejected_before_entrypoint(self):
        job, output = prepare(self.root, 'from pathlib import Path\nPath("/output/ran").write_text("bad")\n')
        command = self.command(job, output)
        command[2] = str(os.getpid() + 100000)
        process = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(process.returncode, 125)
        self.assertFalse((output / "ran").exists())
        self.assertIn("expected controller", process.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
