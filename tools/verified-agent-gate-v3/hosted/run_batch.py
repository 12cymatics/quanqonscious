#!/usr/bin/env python3
"""Run one explicitly authorized, ephemeral GitHub-hosted agent batch."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate3 import canonical, init_state, sha
from export_audit import export_public, regular_bytes

SPEC_SHA256 = "b801c62880a6862d716a6107909494e58cf9a6c0c58921eff89296bf293e5fd2"
SPEC_LENGTH = 3494


class HostedRejected(ValueError):
    def __init__(self, reason: str, settings: tuple[str, ...] = ()):
        super().__init__(reason)
        self.reason = reason
        self.settings = settings


def private_write(path: Path, raw: bytes, mode: int = 0o400) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def github_identity() -> dict:
    names = {
        "repository": "GITHUB_REPOSITORY", "repository_id": "GITHUB_REPOSITORY_ID",
        "actor": "GITHUB_ACTOR", "actor_id": "GITHUB_ACTOR_ID", "ref": "GITHUB_REF",
        "commit_sha": "GITHUB_SHA", "workflow_ref": "GITHUB_WORKFLOW_REF",
        "workflow_sha": "GITHUB_WORKFLOW_SHA", "run_id": "GITHUB_RUN_ID",
        "run_attempt": "GITHUB_RUN_ATTEMPT", "event": "GITHUB_EVENT_NAME",
    }
    data = {key: os.environ.get(value, "") for key, value in names.items()}
    if os.environ.get("GITHUB_ACTIONS") != "true" or data["event"] != "workflow_dispatch" or not all(data.values()):
        raise HostedRejected("manual_workflow_context_required", tuple(value for key, value in names.items() if not data[key]))
    if not data["run_id"].isdigit() or not data["run_attempt"].isdigit():
        raise HostedRejected("invalid_workflow_run_identity")
    for name in ("commit_sha", "workflow_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", data[name]):
            raise HostedRejected("invalid_workflow_commit_identity")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--model", default=os.environ.get("GATE_MODEL"))
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    state = authority_path = None
    private_root = None
    failure_type = None
    runner_returncode = 2
    export_result = None
    failure_stage = "validate_workflow_context"
    seed = b""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    encoded_key = os.environ.pop("GATE_AUTHORITY_KEY_B64", "")
    fingerprint = os.environ.get("GATE_AUTHORITY_FINGERPRINT", "")
    try:
        identity = github_identity()
        if not isinstance(args.model, str) or not args.model.strip() or len(args.model) > 200 or "\0" in args.model:
            raise HostedRejected("model_selection_required", ("model",))
        if not 1 <= args.max_turns <= 32:
            raise HostedRejected("turn_limit_out_of_range", ("max_turns",))
        failure_stage = "validate_environment_credentials"
        missing = tuple(name for name, value in (("OPENAI_API_KEY", api_key), ("GATE_AUTHORITY_KEY_B64", encoded_key), ("GATE_AUTHORITY_FINGERPRINT", fingerprint)) if not value)
        if missing:
            raise HostedRejected("missing_verified_gate_environment_settings", missing)
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise HostedRejected("authority_fingerprint_must_be_64_lowercase_hex", ("GATE_AUTHORITY_FINGERPRINT",))
        try:
            seed = base64.b64decode(encoded_key, validate=True)
        except (ValueError, UnicodeError):
            raise HostedRejected("authority_seed_must_be_canonical_base64", ("GATE_AUTHORITY_KEY_B64",)) from None
        if len(seed) != 32 or base64.b64encode(seed).decode("ascii") != encoded_key:
            raise HostedRejected("authority_seed_must_encode_exactly_32_bytes", ("GATE_AUTHORITY_KEY_B64",))
        authority = Ed25519PrivateKey.from_private_bytes(seed)
        public = authority.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        if sha(public) != fingerprint:
            raise HostedRejected("authority_key_does_not_match_enrolled_fingerprint", ("GATE_AUTHORITY_KEY_B64", "GATE_AUTHORITY_FINGERPRINT"))
        failure_stage = "verify_immutable_contract"
        source_path = ROOT / "source" / "verified_delegation_instruction.txt"
        source = regular_bytes(source_path)
        if len(source) != SPEC_LENGTH or sha(source) != SPEC_SHA256:
            raise HostedRejected("immutable_delegation_contract_changed")
        failure_stage = "bind_exact_operator_task"
        task = regular_bytes(args.task_file) if args.task_file else os.environ.pop("GATE_TASK", "").encode("utf-8", "strict")
        if not task or len(task) > 65536 or not task.decode("utf-8", "strict").strip():
            raise HostedRejected("task_must_be_nonempty_utf8_at_most_65536_bytes", ("task",))
        event = json.loads(regular_bytes(Path(os.environ["GITHUB_EVENT_PATH"])))
        request_inputs = event.get("inputs", {})
        if request_inputs.get("task", "").encode("utf-8", "strict") != task or request_inputs.get("model") != args.model:
            raise HostedRejected("task_or_model_differs_from_workflow_request", ("task", "model"))
        if int(request_inputs.get("max_turns", 8)) != args.max_turns:
            raise HostedRejected("turn_limit_differs_from_workflow_request", ("max_turns",))
        failure_stage = "prepare_private_batch_directories"
        workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
        temp_root = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
        results = args.results.resolve()
        artifacts = args.artifacts.resolve()
        for target in (results, artifacts):
            if not target.is_relative_to(temp_root) or target == temp_root or target.is_relative_to(workspace):
                raise HostedRejected("public_paths_must_be_in_runner_temporary_storage")
        if results.exists() or artifacts.exists() or results.is_relative_to(artifacts) or artifacts.is_relative_to(results):
            raise HostedRejected("public_results_directories_must_be_new_and_disjoint")
        private_root = Path(tempfile.mkdtemp(prefix="verified-agent-gate-private-", dir=temp_root))
        private_root.chmod(0o700)
        state = private_root / "state"
        authority_path = private_root / "authority.key"
        task_path = private_root / "operator-task.bin"
        authorization_path = private_root / "authorization.json"
        private_write(authority_path, seed)
        private_write(task_path, task)
        failure_stage = "bind_controller_source_identity"
        controller_sources = {
            name: sha(regular_bytes(ROOT / name)) for name in (
                "hosted/agent_runner.py", "hosted/run_batch.py", "hosted/export_audit.py", "hosted/bootstrap.sh",
            )
        }
        workflow_path = workspace / ".github" / "workflows" / "verified-agent-gate.yml"
        controller_sources[".github/workflows/verified-agent-gate.yml"] = sha(regular_bytes(workflow_path))
        authorization = {
            "schema": "verified-agent-gate-github-authorization-v1",
            "authorization": "The authenticated manual workflow request authorizes this exact task and selected model for this batch.",
            "github": identity, "task_b64": base64.b64encode(task).decode("ascii"),
            "batch_nonce": secrets.token_hex(32),
            "task_sha256": sha(task), "task_length": len(task),
            "model": args.model, "max_turns": args.max_turns,
            "immutable_contract_sha256": SPEC_SHA256, "immutable_contract_length": SPEC_LENGTH,
            "controller_sources_sha256": controller_sources,
            "state_scope": "fresh private state; parent and replay history are limited to this workflow run",
        }
        private_write(authorization_path, canonical(authorization))
        failure_stage = "initialize_authenticated_controller"
        init_state(state, source_path, authority_path, fingerprint, authorization_path)
        diagnostics_stdout = private_root / "controller-stdout.bin"
        diagnostics_stderr = private_root / "controller-stderr.bin"
        command = [sys.executable, str(ROOT / "hosted" / "agent_runner.py"),
                   "--state", str(state), "--task-file", str(task_path), "--model", args.model,
                   "--max-turns", str(args.max_turns), "--results", str(results)]
        child_env = {key: value for key, value in os.environ.items() if key not in {"GATE_AUTHORITY_KEY_B64", "GATE_TASK"}}
        failure_stage = "run_selected_api_model"
        with diagnostics_stdout.open("xb") as stdout, diagnostics_stderr.open("xb") as stderr:
            process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     env=child_env, check=False)
        runner_returncode = process.returncode
        os.environ.pop("OPENAI_API_KEY", None)
        audit_seed = regular_bytes(state / "audit.key")
        sensitive = (seed, encoded_key.encode("ascii"), seed.hex().encode("ascii"), api_key.encode("utf-8"),
                     audit_seed, base64.b64encode(audit_seed), audit_seed.hex().encode("ascii"))
        results.mkdir(mode=0o700, parents=False, exist_ok=True)
        for name, path in (("hosted-controller-stdout.txt", diagnostics_stdout), ("hosted-controller-stderr.txt", diagnostics_stderr)):
            raw = regular_bytes(path)
            for secret in sensitive:
                if secret:
                    raw = raw.replace(secret, b"[CREDENTIAL REDACTED]")
            private_write(results / name, raw, 0o600)
        metadata = {
            "schema": "verified-agent-gate-hosted-batch-v1", "github": identity,
            "model": args.model, "max_turns": args.max_turns,
            "batch_nonce": authorization["batch_nonce"], "authorization_sha256": sha(canonical(authorization)),
            "task_sha256": sha(task), "task_length": len(task),
            "authority_fingerprint": fingerprint, "runner_exit_code": runner_returncode,
            "objective_status": "not_independently_verified",
            "batch_scope": "one-workflow-run", "resumable_controller_state": False,
        }
        failure_stage = "verify_and_export_public_archive"
        export_result = export_public(state, results, artifacts / "verified-agent-gate-run.tar.gz", metadata,
                                      sensitive, Path(os.environ["GATE_BUILD_PROVENANCE"]))
        private_write(artifacts / "verified-agent-gate-run.sha256",
                      (export_result["sha256"] + "  verified-agent-gate-run.tar.gz\n").encode("ascii"), 0o600)
        private_write(artifacts / "export-complete.json", canonical({
            "verified": True, "archive_sha256": export_result["sha256"],
            "authority_fingerprint": fingerprint, "github": identity,
        }), 0o600)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
                output.write("exported=true\n")
        print(json.dumps({"status": "completed" if runner_returncode == 0 else "failed",
                          "runner_exit_code": runner_returncode, "archive": export_result["archive"],
                          "archive_sha256": export_result["sha256"], "authority_fingerprint": fingerprint,
                          "objective_status": "not_independently_verified"}, sort_keys=True))
        return 0 if runner_returncode == 0 else 1
    except Exception as error:
        failure_type = type(error).__name__
        print(json.dumps({"status": "REJECTED", "type": failure_type,
                          "stage": failure_stage,
                          "reason": error.reason if isinstance(error, HostedRejected) else "stage_failed_without_publishing_unverified_data",
                          "settings_to_check": list(error.settings) if isinstance(error, HostedRejected) else [],
                          "public_archive_created": export_result is not None}), file=sys.stderr)
        return 1
    finally:
        os.environ.pop("GATE_AUTHORITY_KEY_B64", None)
        os.environ.pop("OPENAI_API_KEY", None)
        for key_path in ([authority_path] if authority_path else []) + ([state / "authority.key", state / "audit.key"] if state else []):
            if key_path.exists() and not key_path.is_symlink():
                key_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
