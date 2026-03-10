#!/usr/bin/env python3
"""Production launcher for Toroid UI + 29-sutra hybrid simulator.

This runner is designed for full workflow execution:
1. Resolve and serve the toroid HTML artifact.
2. Verify HTTP availability.
3. Execute 29-sutra hybrid simulator in serial, concurrent, and parallel lanes.
4. Persist stdout/stderr logs and structured report paths for auditability.
"""

from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_TOROID_CANDIDATES = [
    "toroid_HTML",
    "toroid HTML",
]


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    command: List[str]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_runtime_dependencies() -> Dict[str, bool]:
    return {
        "numpy": _module_available("numpy"),
        "scipy": _module_available("scipy"),
        "pandas": _module_available("pandas"),
        "sympy": _module_available("sympy"),
        "torch": _module_available("torch"),
        "qiskit": _module_available("qiskit"),
    }


def resolve_toroid_path(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Provided toroid path does not exist: {candidate}")

    for name in DEFAULT_TOROID_CANDIDATES:
        candidate = Path(name)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate toroid file. Tried: " + ", ".join(DEFAULT_TOROID_CANDIDATES)
    )


def run_server(host: str, port: int) -> socketserver.TCPServer:
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def verify_http(url: str, timeout_s: float) -> bool:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def run_command(command: List[str]) -> CommandResult:
    completed = subprocess.run(command, text=True, capture_output=True)
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        command=command,
    )


def build_hybrid_command(
    value: float,
    mode: str,
    output_path: Path,
    max_workers: Optional[int],
) -> List[str]:
    command = [
        sys.executable,
        "hybrid_sutra_platform.py",
        str(value),
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    if max_workers is not None:
        command.extend(["--max-workers", str(max_workers)])
    return command


def write_launcher_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Launch Toroid HTML and the 29-sutra hybrid simulator")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--toroid-path", help="Explicit path to toroid HTML artifact")
    parser.add_argument("--skip-hybrid", action="store_true", help="Only serve Toroid UI")
    parser.add_argument("--value", type=float, default=1.0, help="Input value for hybrid simulator")
    parser.add_argument("--mode", default="hybrid", help="Simulator mode")
    parser.add_argument("--max-workers", type=int, help="Optional max workers for concurrent and parallel lanes")
    parser.add_argument("--duration", type=float, default=10.0, help="Server lifetime in seconds")
    parser.add_argument(
        "--report-path",
        default="runs/toroid_launcher_report.json",
        help="Launcher report output path",
    )
    parser.add_argument(
        "--hybrid-output",
        default="runs/hybrid_sutra_platform_report.json",
        help="Hybrid simulator JSON output path",
    )
    args = parser.parse_args(argv)

    dependencies = check_runtime_dependencies()
    toroid_file = resolve_toroid_path(args.toroid_path)

    server = run_server(args.host, args.port)
    encoded_path = urllib.parse.quote(toroid_file.name)
    url = f"http://{args.host}:{args.port}/{encoded_path}"
    http_ok = verify_http(url, timeout_s=2.0)
    print(f"Toroid HTML URL: {url}")
    print(f"HTTP verification: {'OK' if http_ok else 'FAILED'}")

    hybrid_result: Optional[CommandResult] = None
    if not args.skip_hybrid:
        hybrid_output_path = Path(args.hybrid_output)
        hybrid_output_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_hybrid_command(args.value, args.mode, hybrid_output_path, args.max_workers)
        print("Executing hybrid command:", " ".join(command))
        hybrid_result = run_command(command)
        if hybrid_result.stdout:
            print(hybrid_result.stdout)
        if hybrid_result.stderr:
            print(hybrid_result.stderr, file=sys.stderr)

    report_payload: Dict[str, object] = {
        "toroid_file": str(toroid_file),
        "url": url,
        "http_ok": http_ok,
        "dependencies": dependencies,
        "hybrid_executed": not args.skip_hybrid,
        "hybrid_success": hybrid_result.returncode == 0 if hybrid_result else None,
        "hybrid_command": hybrid_result.command if hybrid_result else None,
        "hybrid_stdout": hybrid_result.stdout if hybrid_result else None,
        "hybrid_stderr": hybrid_result.stderr if hybrid_result else None,
    }
    write_launcher_report(Path(args.report_path), report_payload)
    print(f"Launcher report written to {args.report_path}")

    time.sleep(max(args.duration, 0.0))
    server.shutdown()
    server.server_close()

    if hybrid_result is not None and hybrid_result.returncode != 0:
        return hybrid_result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
