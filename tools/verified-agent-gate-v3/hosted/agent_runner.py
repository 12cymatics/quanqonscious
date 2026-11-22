#!/usr/bin/env python3
"""Responses API controller exposing only authenticated isolated Python jobs.

Protocol references:
https://developers.openai.com/api/docs/guides/function-calling
https://developers.openai.com/api/docs/guides/reasoning
https://developers.openai.com/api/reference/resources/responses/methods/create
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate3 import Gate, Rejected, RESERVED, atomic, canonical, sha, validate_argv

ENDPOINT = 'https://api.openai.com/v1/responses'
MAX_TASK_BYTES = 256 * 1024
MAX_ARGUMENT_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_FILES = 64
MAX_FILE_BYTES = 512 * 1024
MAX_OUTPUT_ITEMS = 64
MAX_TURNS = 32
MODEL_JOB_LIMITS = {'timeout_seconds': 60, 'memory_mib': 512, 'file_mib': 16, 'cpu_seconds': 30}
OPERATOR_TASK = 'operator-task.bin'
RESERVED_INPUTS = RESERVED | {OPERATOR_TASK}
IDENTIFIER = re.compile(r'[A-Za-z0-9_-]{1,256}\Z')
MODEL_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z')

TOOL = {
    'type': 'function', 'name': 'run_verified_python', 'strict': True,
    'description': (
        'Execute a Python job through the signed instruction gate and enforced isolated runner. '
        'Supply the complete source files. The controller preserves the authoritative instruction '
        'and exact user task at /job/instruction.bin and /job/operator-task.bin. '
        'Use /output for deliverables. Python standard library is available. Network, shell, '
        'subprocesses, forks, external tools, and access to controller state are unavailable. '
        'The controller chooses the signed parent and fixed execution limits.'
    ),
    'parameters': {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'operation': {'type': 'string', 'description': 'Immediate work allocation preserving every original requirement.'},
            'entrypoint': {'type': 'string', 'description': 'Relative .py path present in files.'},
            'files': {
                'type': 'array', 'minItems': 1, 'maxItems': MAX_FILES,
                'items': {
                    'type': 'object', 'additionalProperties': False,
                    'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                    'required': ['path', 'content'],
                },
            },
            'argv': {'type': 'array', 'maxItems': 128, 'items': {'type': 'string'}},
        },
        'required': ['operation', 'entrypoint', 'files', 'argv'],
    },
}


class RunnerRejected(RuntimeError):
    pass


class ApiFailure(RuntimeError):
    pass


def decode_json(raw: str | bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RunnerRejected('duplicate JSON field')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(RunnerRejected('nonfinite JSON number')))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise RunnerRejected('invalid JSON response or tool arguments') from error


def limited_regular(path: Path, maximum: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise RunnerRejected('input file is not regular or exceeds its supported size')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise RunnerRejected('input file changed beyond its supported size')
        return raw
    finally:
        os.close(fd)


def job_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode('utf-8', 'strict')) > 1024:
        raise RunnerRejected('invalid job path')
    if '\\' in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise RunnerRejected('job path contains prohibited characters')
    parts = value.split('/')
    if PurePosixPath(value).is_absolute() or any(part in {'', '.', '..'} for part in parts):
        raise RunnerRejected('job path must remain inside its signed job')
    if any(len(part.encode('utf-8')) > 255 for part in parts):
        raise RunnerRejected('job path component exceeds filesystem limits')
    if parts[0] in RESERVED_INPUTS:
        raise RunnerRejected('model cannot set a controller-owned input')
    return value


def validate_arguments(raw: str) -> dict:
    if not isinstance(raw, str) or len(raw.encode('utf-8', 'strict')) > MAX_ARGUMENT_BYTES:
        raise RunnerRejected('tool argument size is invalid')
    arguments = decode_json(raw)
    if not isinstance(arguments, dict) or set(arguments) != {'operation', 'entrypoint', 'files', 'argv'}:
        raise RunnerRejected('unexpected tool argument schema')
    operation = arguments['operation']
    if not isinstance(operation, str) or not operation.strip() or '\0' in operation or len(operation.encode('utf-8', 'strict')) > 16384:
        raise RunnerRejected('invalid assigned operation')
    entrypoint = job_path(arguments['entrypoint'])
    if not entrypoint.endswith('.py'):
        raise RunnerRejected('entrypoint must be a Python source file')
    validate_argv(arguments['argv'])
    files = arguments['files']
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise RunnerRejected('job must contain one through 64 source files')
    names, total = set(), 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {'path', 'content'}:
            raise RunnerRejected('unexpected source file schema')
        name = job_path(item['path'])
        if name in names:
            raise RunnerRejected('duplicate source file path')
        names.add(name)
        if not isinstance(item['content'], str):
            raise RunnerRejected('source file content must be UTF-8 text')
        size = len(item['content'].encode('utf-8', 'strict'))
        if size > MAX_FILE_BYTES:
            raise RunnerRejected('source file exceeds 512 KiB')
        total += size
    if total > MAX_ARGUMENT_BYTES or entrypoint not in names:
        raise RunnerRejected('entrypoint is missing or total source size is excessive')
    for name in names:
        if any(name.startswith(other + '/') for other in names if other != name):
            raise RunnerRejected('source file and directory paths conflict')
    return arguments


def validate_response(response: dict, seen_calls: set[str]) -> tuple[list[dict], list[str], bool]:
    if not isinstance(response, dict) or response.get('object') != 'response':
        raise RunnerRejected('unexpected API response schema')
    if not isinstance(response.get('id'), str) or not IDENTIFIER.fullmatch(response['id']):
        raise RunnerRejected('invalid API response identifier')
    if response.get('status') != 'completed' or response.get('error') is not None:
        raise RunnerRejected('API response is incomplete or failed; no tool is dispatched')
    output = response.get('output')
    if not isinstance(output, list) or len(output) > MAX_OUTPUT_ITEMS:
        raise RunnerRejected('invalid API output list')
    calls, texts, refused = [], [], False
    for item in output:
        if not isinstance(item, dict):
            raise RunnerRejected('invalid API output item')
        kind = item.get('type')
        if kind == 'function_call':
            allowed = {'type', 'id', 'name', 'call_id', 'arguments', 'status', 'async', 'caller', 'namespace'}
            if set(item) - allowed or not {'type', 'name', 'call_id', 'arguments'}.issubset(item):
                raise RunnerRejected('unexpected function-call schema')
            if item['name'] != TOOL['name']:
                raise RunnerRejected('model requested a tool outside the enforced route')
            if item.get('status', 'completed') != 'completed' or item.get('async', False) is not False:
                raise RunnerRejected('asynchronous or incomplete calls are not permitted')
            if item.get('caller') not in (None, {'type': 'direct'}) or item.get('namespace') not in (None, ''):
                raise RunnerRejected('indirect or namespaced calls are not permitted')
            call_id = item['call_id']
            if not isinstance(call_id, str) or not IDENTIFIER.fullmatch(call_id) or call_id in seen_calls:
                raise RunnerRejected('duplicate or invalid model call identifier')
            validate_arguments(item['arguments'])
            calls.append(item)
        elif kind == 'message':
            if set(item) - {'type', 'id', 'role', 'status', 'content', 'phase'} or item.get('role') != 'assistant':
                raise RunnerRejected('unexpected assistant-message schema')
            if item.get('status', 'completed') != 'completed' or not isinstance(item.get('content'), list):
                raise RunnerRejected('assistant message is incomplete')
            for content in item['content']:
                if not isinstance(content, dict):
                    raise RunnerRejected('invalid assistant content')
                if content.get('type') == 'output_text':
                    if set(content) - {'type', 'text', 'annotations', 'logprobs'} or not isinstance(content.get('text'), str):
                        raise RunnerRejected('unexpected output-text schema')
                    if not isinstance(content.get('annotations', []), list):
                        raise RunnerRejected('invalid output-text annotations')
                    texts.append(content['text'])
                elif content.get('type') == 'refusal':
                    if set(content) != {'type', 'refusal'} or not isinstance(content['refusal'], str):
                        raise RunnerRejected('invalid refusal schema')
                    refused = True
                    texts.append(content['refusal'])
                else:
                    raise RunnerRejected('unsupported assistant content type')
        elif kind == 'reasoning':
            if set(item) - {'type', 'id', 'status', 'summary', 'content', 'encrypted_content'}:
                raise RunnerRejected('unexpected reasoning-item schema')
            if item.get('status', 'completed') not in ('completed', None):
                raise RunnerRejected('incomplete reasoning item')
            if not isinstance(item.get('summary', []), list) or not isinstance(item.get('content', []), list):
                raise RunnerRejected('invalid reasoning item')
            if item.get('encrypted_content') is not None and not isinstance(item['encrypted_content'], str):
                raise RunnerRejected('invalid encrypted reasoning state')
            for content in item.get('summary', []):
                if not isinstance(content, dict) or set(content) != {'type', 'text'} or content['type'] != 'summary_text' or not isinstance(content['text'], str):
                    raise RunnerRejected('invalid reasoning summary schema')
            for content in item.get('content', []):
                if not isinstance(content, dict) or set(content) != {'type', 'text'} or content['type'] != 'reasoning_text' or not isinstance(content['text'], str):
                    raise RunnerRejected('invalid reasoning content schema')
        else:
            raise RunnerRejected('unsupported model output type; no alternative tool is available')
    if len(calls) > 1:
        raise RunnerRejected('multiple calls in one turn are prohibited')
    if refused and calls:
        raise RunnerRejected('refusal and execution cannot be combined')
    return calls, texts, refused


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ApiFailure('API redirect refused; credentials were not forwarded')


class ResponsesHTTPS:
    """One HTTPS request per invocation. No automatic retries or endpoint overrides."""
    mode = 'live_api'

    def __init__(self, api_key: str, timeout: int = 120):
        if not isinstance(api_key, str) or not api_key.strip() or any(char in api_key for char in '\r\n'):
            raise ApiFailure('OPENAI_API_KEY must be configured in the trusted controller')
        self._key = api_key
        self._timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                                  urllib.request.HTTPSHandler(context=ssl.create_default_context()))

    def create(self, request: dict) -> dict:
        raw = canonical(request)
        if len(raw) > MAX_REQUEST_BYTES:
            raise RunnerRejected('complete API history exceeds its supported size')
        call = urllib.request.Request(ENDPOINT, data=raw, method='POST', headers={
            'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json',
            'Accept': 'application/json', 'User-Agent': 'verified-agent-gate/3.0.0',
        })
        try:
            with self._opener.open(call, timeout=self._timeout) as response:
                if response.status != 200 or response.headers.get_content_type() != 'application/json':
                    raise ApiFailure('API returned an unexpected status or content type')
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise ApiFailure(f'API returned HTTP {error.code}; no retry was attempted') from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ApiFailure('API transport failed; outcome is unknown and no retry was attempted') from None
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ApiFailure('API response exceeded the supported size; no retry was attempted')
        return decode_json(payload)


class HostedRunner:
    def __init__(self, *, state: Path, task: bytes, model: str, max_turns: int, results: Path, transport):
        if not isinstance(task, bytes) or not task.strip() or len(task) > MAX_TASK_BYTES:
            raise RunnerRejected('task must contain one through 262144 UTF-8 bytes')
        task.decode('utf-8', 'strict')
        if not isinstance(model, str) or not MODEL_NAME.fullmatch(model):
            raise RunnerRejected('invalid model identifier')
        if type(max_turns) is not int or not 1 <= max_turns <= MAX_TURNS:
            raise RunnerRejected('max-turns must be between 1 and 32')
        self.gate = Gate(state)
        self.gate.audit_verify()
        self.results = results.resolve()
        state_path = self.gate.state.resolve()
        if self.results == state_path or self.results.is_relative_to(state_path) or state_path.is_relative_to(self.results):
            raise RunnerRejected('public results and private controller state must be disjoint')
        if self.results.exists():
            raise RunnerRejected('results directory must be new; a prior run is never silently resumed')
        self.results.mkdir(mode=0o700, parents=False)
        self.task, self.model, self.max_turns, self.transport = task, model, max_turns, transport
        self.seen_calls: set[str] = set()
        self.parent = None
        self.calls = 0
        self.attempts = 0
        self.turns = 0
        self.receipts = []
        self.response_ids = []
        atomic(self.results / OPERATOR_TASK, task)
        atomic(self.results / 'authoritative-instruction.bin', self.gate.source)
        self.instructions = (
            'You are operating under an authenticated execution contract. Carry out the user task using '
            'only run_verified_python. No shell, built-in tool, network, alternate executor, or delegation '
            'interface is available. Do not claim that signing a command proves semantic obedience or '
            'mathematical correctness. Report what the actual outputs establish and what remains unproved. '
            'The controller already holds the user authorization; do not invent or replace it. '
            'Every job receives the full original instruction and exact operator task, independent of your '
            'assigned operation. The operation may narrow the immediate work but may not weaken requirements. '
            'A failed, unsafe, malformed, or unauthenticated job stops this run. Produce deliverables in /output. '
            'Your final prose is recorded as a model claim, not an independent verification of task completion.\n'
            'AUTHORITATIVE_INSTRUCTION_BYTE_LENGTH=' + str(len(self.gate.source)) + '\n'
            'AUTHORITATIVE_INSTRUCTION_SHA256=' + sha(self.gate.source) + '\n'
            'AUTHORITATIVE_INSTRUCTION_BASE64=' + base64.b64encode(self.gate.source).decode('ascii') + '\n'
            'OPERATOR_TASK_BYTE_LENGTH=' + str(len(task)) + '\n'
            'OPERATOR_TASK_SHA256=' + sha(task) + '\n'
            'OPERATOR_TASK_BASE64=' + base64.b64encode(task).decode('ascii') + '\n'
            'AUTHORITATIVE_INSTRUCTION_UTF8:\n' + self.gate.source.decode('utf-8', 'strict')
        )

    def summary(self, status: str, code: int, reason: str | None = None) -> dict:
        value = {'schema': 'hosted-agent-run-v1', 'status': status, 'exit_code': code,
                 'objective_status': 'not_independently_verified', 'model': self.model,
                 'transport': getattr(self.transport, 'mode', 'injected_test_transport'),
                 'api_turns': self.turns, 'gated_dispatches': self.calls,
                 'dispatch_attempts': self.attempts,
                 'dispatch_outcome_unknown': self.attempts != self.calls,
                 'instruction_sha256': sha(self.gate.source), 'task_sha256': sha(self.task),
                 'response_ids': self.response_ids, 'receipt_files': self.receipts,
                 'last_successful_parent_sha256': self.parent}
        if reason is not None:
            value['reason'] = reason
        atomic(self.results / 'run-summary.json', canonical(value), replace=True)
        return value

    def _copy_file(self, source: Path, destination: Path):
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RunnerRejected('only verified regular output artifacts may be exported')
            with os.fdopen(descriptor, 'rb', closefd=False) as origin, destination.open('xb') as target:
                os.chmod(destination, 0o600)
                shutil.copyfileobj(origin, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        finally:
            os.close(descriptor)

    def route(self, call: dict) -> tuple[dict, bool]:
        validate_response({'id': 'resp_router', 'object': 'response', 'status': 'completed',
                           'error': None, 'output': [call]}, self.seen_calls)
        arguments = validate_arguments(call['arguments'])
        self.seen_calls.add(call['call_id'])
        with tempfile.TemporaryDirectory(prefix='verified-model-input-') as temporary:
            job = Path(temporary)
            for item in arguments['files']:
                path = job / item['path']
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic(path, item['content'].encode('utf-8'))
            atomic(job / OPERATOR_TASK, self.task)
            envelope = self.gate.create(job, arguments['entrypoint'], arguments['operation'],
                                        argv=arguments['argv'], parent=self.parent, limits=MODEL_JOB_LIMITS)
            verified = self.gate.verify(envelope)
            self.attempts += 1
            atomic(self.results / f'dispatch-attempt-{self.attempts:03d}.json', canonical({
                'call_id': call['call_id'], 'handoff_id': verified['handoff_id'],
                'envelope_sha256': verified['envelope_sha256'], 'parent_sha256': self.parent,
                'status': 'verified_before_dispatch',
            }))
            receipt = self.gate.dispatch(envelope)
        self.calls += 1
        data = receipt['payload']['data']
        folder = self.results / f'call-{self.calls:03d}'
        folder.mkdir(mode=0o700)
        atomic(folder / 'receipt.json', canonical(receipt))
        self._copy_file(envelope, folder / 'envelope.json')
        relative_receipt = str((folder / 'receipt.json').relative_to(self.results))
        self.receipts.append(relative_receipt)
        self.gate.audit_verify()
        retained = self.gate.state / 'results' / data['handoff_id']
        self._copy_file(retained / 'stdout.bin', folder / 'stdout.bin')
        self._copy_file(retained / 'stderr.bin', folder / 'stderr.bin')
        artifact_feedback = []
        inline_budget = 256 * 1024
        for item in data['output_files']:
            name = item['path']
            if PurePosixPath(name).is_absolute() or any(part in {'', '.', '..'} for part in name.split('/')):
                raise RunnerRejected('verified output path is invalid')
            destination = folder / 'output' / name
            self._copy_file(retained / 'output' / name, destination)
            detail = dict(item)
            if item['size'] <= 65536 and item['size'] <= inline_budget:
                raw = limited_regular(destination, 65536)
                detail['content_b64'] = base64.b64encode(raw).decode('ascii')
                inline_budget -= len(raw)
            else:
                detail['content_b64'] = None
                detail['content_location'] = 'complete artifact retained in run results'
            artifact_feedback.append(detail)
        feedback = {'handoff_id': data['handoff_id'], 'envelope_sha256': data['envelope_sha256'],
                    'receipt_sha256': sha(canonical(receipt)), 'exit_code': data['exit_code'],
                    'unsafe_output': data['unsafe_output'], 'output_files': artifact_feedback,
                    'objective_status': 'not_independently_verified'}
        for stream in ('stdout', 'stderr'):
            path = folder / (stream + '.bin')
            detail = {'sha256': data[stream + '_sha256'], 'size': path.stat().st_size}
            if path.stat().st_size <= 131072:
                raw = limited_regular(path, 131072)
                detail['content_b64'] = base64.b64encode(raw).decode('ascii')
                try:
                    detail['utf8'] = raw.decode('utf-8', 'strict')
                except UnicodeError:
                    detail['utf8'] = None
            else:
                detail['content_b64'] = None
                detail['content_location'] = 'complete stream retained in run results'
            feedback[stream] = detail
        success = data['exit_code'] == 0 and data['unsafe_output'] is False
        if success:
            self.parent = verified['envelope_sha256']
        return feedback, success

    def run(self) -> dict:
        history = [{'role': 'user', 'content': [{'type': 'input_text', 'text': self.task.decode('utf-8')}]}]
        self.summary('running', 3)
        try:
            for turn in range(1, self.max_turns + 1):
                self.turns = turn
                request = {'model': self.model, 'instructions': self.instructions,
                           'input': history, 'tools': [copy.deepcopy(TOOL)], 'tool_choice': 'auto',
                           'parallel_tool_calls': False, 'store': False, 'max_output_tokens': 16384,
                           'include': ['reasoning.encrypted_content']}
                raw_request = canonical(request)
                if len(raw_request) > MAX_REQUEST_BYTES:
                    raise RunnerRejected('complete API history exceeds its supported size')
                atomic(self.results / f'api-request-{turn:03d}.json', raw_request)
                response = self.transport.create(request)
                if len(canonical(response)) > MAX_RESPONSE_BYTES:
                    raise RunnerRejected('API response exceeds its supported size')
                atomic(self.results / f'api-response-{turn:03d}.json', canonical(response))
                calls, texts, refused = validate_response(response, self.seen_calls)
                if response['id'] in self.response_ids:
                    raise RunnerRejected('duplicate API response identifier')
                self.response_ids.append(response['id'])
                history.extend(response['output'])
                if not calls:
                    if not texts:
                        raise RunnerRejected('model stopped without a tool call or final text')
                    atomic(self.results / 'model-final.txt', '\n\n'.join(texts).encode('utf-8'))
                    if refused:
                        return self.summary('model_refused', 3)
                    if not self.calls:
                        return self.summary('model_responded_without_execution', 3)
                    self.gate.audit_verify()
                    return self.summary('model_response_complete', 0)
                feedback, success = self.route(calls[0])
                atomic(self.results / f'function-output-{turn:03d}.json', canonical(feedback))
                history.append({'type': 'function_call_output', 'call_id': calls[0]['call_id'],
                                'output': canonical(feedback).decode('ascii')})
                if not success:
                    return self.summary('worker_failed', 4, 'gated worker returned a nonzero status or unsafe output')
                self.summary('running', 3)
            return self.summary('turn_limit_reached', 3)
        except (RunnerRejected, Rejected, ApiFailure) as error:
            return self.summary('rejected', 2, str(error))
        except Exception as error:
            return self.summary('controller_error', 2, type(error).__name__ + '; no automatic retry was attempted')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--task-file', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-turns', type=int, default=8)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    try:
        task = limited_regular(args.task_file, MAX_TASK_BYTES)
        transport = ResponsesHTTPS(os.environ.get('OPENAI_API_KEY', ''))
        runner = HostedRunner(state=args.state, task=task, model=args.model, max_turns=args.max_turns,
                              results=args.results, transport=transport)
        result = runner.run()
        print(canonical(result).decode('ascii'))
        return result['exit_code']
    except (RunnerRejected, Rejected, ApiFailure, OSError, UnicodeError) as error:
        print(canonical({'status': 'rejected', 'reason': str(error),
                         'objective_status': 'not_independently_verified'}).decode('ascii'), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
