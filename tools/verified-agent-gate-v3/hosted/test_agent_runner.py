"""Offline Responses wire fixtures with real cryptographic and isolated dispatch.

These tests do not call the live OpenAI API. Deterministic response fixtures test
the controller protocol; every dispatched worker uses the production Gate.
"""
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from agent_runner import ApiFailure, ENDPOINT, HostedRunner, MODEL_JOB_LIMITS, OPERATOR_TASK, ResponsesHTTPS, RunnerRejected, TOOL, validate_arguments, validate_response
from gate3 import Gate, init_state, keygen, sha

WORKER = '''import base64
import hashlib
import json
import os
from pathlib import Path
task = Path('/job/operator-task.bin').read_bytes()
data = {
    'sum_of_squares_1_through_12': sum(i*i for i in range(1, 13)),
    'task_b64': base64.b64encode(task).decode('ascii'),
    'instruction_b64': base64.b64encode(GATE_INSTRUCTION_BYTES).decode('ascii'),
    'operation': GATE_ASSIGNED_OPERATION,
    'handoff_id': GATE_HANDOFF_ID,
    'parent': GATE_PARENT_SHA256,
    'api_key_available': 'OPENAI_API_KEY' in os.environ,
    'signing_key_available': 'GATE_AUTHORITY_KEY_B64' in os.environ,
}
Path('/output/answer.json').write_text(json.dumps(data, sort_keys=True))
print('Computed exact integer sum of squares:', data['sum_of_squares_1_through_12'])
'''


def arguments(source=WORKER):
    return {'operation': 'Compute the requested exact integer result and preserve the source bytes',
            'entrypoint': 'work.py', 'files': [{'path': 'work.py', 'content': source}], 'argv': []}


def function_call(identifier='call_test_1', *, source=WORKER, name='run_verified_python', args=None):
    return {'id': 'fc_' + identifier, 'type': 'function_call', 'status': 'completed',
            'call_id': identifier, 'name': name,
            'arguments': json.dumps(arguments(source) if args is None else args)}


def response(identifier, output):
    return {'id': identifier, 'object': 'response', 'status': 'completed', 'error': None, 'output': output}


def final_message(text='The artifacts contain the exact calculated integer result.'):
    return {'id': 'msg_final', 'type': 'message', 'status': 'completed', 'role': 'assistant',
            'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}


class OfflineResponses:
    mode = 'deterministic_offline_test_no_live_api'

    def __init__(self, items):
        self.items = list(items)
        self.requests = []

    def create(self, request):
        self.requests.append(copy.deepcopy(request))
        if not self.items:
            raise AssertionError('unexpected additional model request')
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)


class StrictRouterTests(unittest.TestCase):
    def test_one_custom_function_and_strict_schema(self):
        self.assertEqual(TOOL['type'], 'function')
        self.assertEqual(TOOL['name'], 'run_verified_python')
        self.assertIs(TOOL['strict'], True)
        self.assertIs(TOOL['parameters']['additionalProperties'], False)
        self.assertEqual(set(TOOL['parameters']['required']), set(TOOL['parameters']['properties']))

    def test_traversal_absolute_and_reserved_paths_rejected(self):
        for path in ('../work.py', '/tmp/work.py', 'a/../../work.py', 'a//work.py',
                     'a/./work.py', 'a\\work.py', 'operator-task.bin', 'instruction.bin', 'task.json'):
            with self.subTest(path=path):
                value = arguments()
                value['files'].append({'path': path, 'content': 'not controller-owned content'})
                with self.assertRaises(RunnerRejected):
                    validate_arguments(json.dumps(value))

    def test_parent_limits_runtime_and_keys_cannot_be_supplied(self):
        for field in ('parent', 'limits', 'state', 'authority_key', 'environment', 'command'):
            with self.subTest(field=field):
                value = arguments()
                value[field] = 'untrusted'
                with self.assertRaises(RunnerRejected):
                    validate_arguments(json.dumps(value))

    def test_duplicate_json_fields_and_file_paths_rejected(self):
        with self.assertRaises(RunnerRejected):
            validate_arguments('{"operation":"x","operation":"y"}')
        value = arguments()
        value['files'].append(dict(value['files'][0]))
        with self.assertRaises(RunnerRejected):
            validate_arguments(json.dumps(value))

    def test_file_directory_collision_rejected(self):
        value = arguments()
        value['files'].extend([{'path': 'data', 'content': ''}, {'path': 'data/input.txt', 'content': ''}])
        with self.assertRaises(RunnerRejected):
            validate_arguments(json.dumps(value))

    def test_unknown_tool_and_builtin_output_rejected(self):
        with self.assertRaises(RunnerRejected):
            validate_response(response('resp_unknown', [function_call(name='shell')]), set())
        with self.assertRaises(RunnerRejected):
            validate_response(response('resp_builtin', [{'type': 'code_interpreter_call'}]), set())

    def test_indirect_async_and_namespaced_calls_rejected(self):
        for extra in ({'caller': {'type': 'program', 'caller_id': 'other'}}, {'async': True}, {'namespace': 'shell'}):
            with self.subTest(extra=extra):
                call = function_call()
                call.update(extra)
                with self.assertRaises(RunnerRejected):
                    validate_response(response('resp_indirect', [call]), set())

    def test_multiple_calls_and_incomplete_response_rejected(self):
        with self.assertRaises(RunnerRejected):
            validate_response(response('resp_many', [function_call('a'), function_call('b')]), set())
        incomplete = response('resp_incomplete', [function_call()])
        incomplete['status'] = 'incomplete'
        with self.assertRaises(RunnerRejected):
            validate_response(incomplete, set())

    def test_missing_key_does_not_create_a_fake_transport(self):
        with self.assertRaises(ApiFailure):
            ResponsesHTTPS('')


class HostedDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hosted-gate-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        private, public = self.base / 'authority.key', self.base / 'authority.pub'
        fingerprint = keygen(private, public)
        self.state = self.base / 'state'
        self.spec = ROOT / 'source' / 'verified_delegation_instruction.txt'
        self.source = self.spec.read_bytes()
        init_state(self.state, self.spec, private, fingerprint, ROOT / 'tests' / 'authorization.txt')
        self.task = 'Compute the exact sum of squares from 1 through 12.\r\nPreserve these bytes: café.\n'.encode('utf-8')

    def runner(self, items, max_turns=8):
        transport = OfflineResponses(items)
        runner = HostedRunner(state=self.state, task=self.task, model='offline-test-model', max_turns=max_turns,
                              results=self.base / 'public-results', transport=transport)
        return runner, transport

    def test_real_two_job_chain_and_stateless_wire_preserve_instructions(self):
        reasoning = {'id': 'rs_test', 'type': 'reasoning', 'summary': [], 'content': [],
                     'encrypted_content': 'opaque-offline-wire-fixture'}
        runner, transport = self.runner([
            response('resp_first', [reasoning, function_call('call_first')]),
            response('resp_second', [function_call('call_second')]),
            response('resp_final', [final_message()]),
        ])
        secret = 'offline-secret-sentinel-must-not-enter-model-or-job'
        with patch.dict(os.environ, {'OPENAI_API_KEY': secret, 'GATE_AUTHORITY_KEY_B64': secret}):
            summary = runner.run()
        self.assertEqual(summary['status'], 'model_response_complete', summary)
        self.assertEqual(summary['exit_code'], 0)
        self.assertEqual(summary['gated_dispatches'], 2)
        self.assertEqual(summary['dispatch_attempts'], 2)
        self.assertFalse(summary['dispatch_outcome_unknown'])
        self.assertEqual(summary['objective_status'], 'not_independently_verified')
        self.assertEqual(summary['transport'], 'deterministic_offline_test_no_live_api')
        self.assertEqual(len(transport.requests), 3)
        for request in transport.requests:
            self.assertEqual([tool['name'] for tool in request['tools']], ['run_verified_python'])
            self.assertIs(request['parallel_tool_calls'], False)
            self.assertIs(request['store'], False)
            self.assertIn(base64.b64encode(self.source).decode('ascii'), request['instructions'])
            self.assertIn(base64.b64encode(self.task).decode('ascii'), request['instructions'])
            self.assertEqual(request['input'][0]['content'][0]['text'].encode('utf-8'), self.task)
            self.assertNotIn(secret, json.dumps(request))
        self.assertIn(reasoning, transport.requests[1]['input'])
        self.assertEqual(transport.requests[1]['input'][-1]['type'], 'function_call_output')
        self.assertEqual(transport.requests[1]['input'][-1]['call_id'], 'call_first')
        receipts = []
        for index in (1, 2):
            folder = runner.results / f'call-{index:03d}'
            artifact = json.loads((folder / 'output' / 'answer.json').read_bytes())
            receipt = json.loads((folder / 'receipt.json').read_bytes())
            receipts.append(receipt)
            self.assertEqual(artifact['sum_of_squares_1_through_12'], 650)
            self.assertEqual(base64.b64decode(artifact['task_b64']), self.task)
            self.assertEqual(base64.b64decode(artifact['instruction_b64']), self.source)
            self.assertFalse(artifact['api_key_available'])
            self.assertFalse(artifact['signing_key_available'])
            job = self.state / 'jobs' / receipt['payload']['data']['handoff_id'] / 'code'
            self.assertEqual((job / OPERATOR_TASK).read_bytes(), self.task)
            for path in job.rglob('*'):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes())
        self.assertIsNone(receipts[0]['payload']['data']['parent_sha256'])
        self.assertEqual(receipts[1]['payload']['data']['parent_sha256'], receipts[0]['payload']['data']['envelope_sha256'])
        audit = Gate(self.state).audit_verify()
        self.assertEqual(audit['events'].count('completed'), 2)
        self.assertEqual((runner.results / OPERATOR_TASK).read_bytes(), self.task)

    def test_unknown_output_prevents_even_an_otherwise_valid_call(self):
        runner, transport = self.runner([response('resp_mixed', [function_call(), {'type': 'shell_call'}])])
        summary = runner.run()
        self.assertEqual(summary['status'], 'rejected')
        self.assertEqual(summary['gated_dispatches'], 0)
        self.assertNotIn('started', Gate(self.state).audit_verify()['events'])
        self.assertEqual(len(transport.requests), 1)

    def test_duplicate_call_identifier_never_dispatches_twice(self):
        runner, _ = self.runner([response('resp_one', [function_call('same_call')]),
                                 response('resp_two', [function_call('same_call')])])
        summary = runner.run()
        self.assertEqual(summary['status'], 'rejected')
        self.assertEqual(summary['gated_dispatches'], 1)
        self.assertEqual(Gate(self.state).audit_verify()['events'].count('started'), 1)

    def test_worker_failure_stops_before_another_model_call(self):
        runner, transport = self.runner([response('resp_fail', [function_call(source='raise SystemExit(7)\n')]),
                                         response('resp_not_called', [final_message()])])
        summary = runner.run()
        self.assertEqual(summary['status'], 'worker_failed', summary)
        self.assertEqual(summary['exit_code'], 4)
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(json.loads((runner.results / 'call-001' / 'receipt.json').read_bytes())['payload']['data']['exit_code'], 7)

    def test_transport_unknown_outcome_is_not_retried(self):
        runner, transport = self.runner([ApiFailure('API outcome unknown; no retry'),
                                         response('resp_not_called', [function_call()])])
        summary = runner.run()
        self.assertEqual(summary['status'], 'rejected')
        self.assertEqual(summary['gated_dispatches'], 0)
        self.assertEqual(len(transport.requests), 1)

    def test_lost_dispatch_ack_preserves_reference_and_never_retries(self):
        runner, transport = self.runner([response('resp_execute', [function_call()]),
                                         response('resp_not_called', [final_message()])])
        dispatch = runner.gate.dispatch
        def execute_then_lose_ack(envelope):
            dispatch(envelope)
            raise ConnectionError('offline fault injection after actual completed dispatch')
        with patch.object(runner.gate, 'dispatch', side_effect=execute_then_lose_ack):
            summary = runner.run()
        self.assertEqual(summary['status'], 'controller_error')
        self.assertEqual(summary['dispatch_attempts'], 1)
        self.assertEqual(summary['gated_dispatches'], 0)
        self.assertTrue(summary['dispatch_outcome_unknown'])
        self.assertEqual(len(transport.requests), 1)
        reference = json.loads((runner.results / 'dispatch-attempt-001.json').read_bytes())
        stored = self.state / 'jobs' / reference['handoff_id'] / 'envelope.json'
        self.assertEqual(sha(stored.read_bytes()), reference['envelope_sha256'])
        self.assertEqual(Gate(self.state).audit_verify()['events'].count('completed'), 1)

    def test_turn_limit_bounds_execution_and_is_not_success(self):
        runner, transport = self.runner([response('resp_limit', [function_call()])], max_turns=1)
        summary = runner.run()
        self.assertEqual(summary['status'], 'turn_limit_reached', summary)
        self.assertEqual(summary['exit_code'], 3)
        self.assertEqual(summary['gated_dispatches'], 1)
        self.assertEqual(len(transport.requests), 1)

    def test_final_prose_without_execution_does_not_claim_completion(self):
        runner, _ = self.runner([response('resp_claim', [final_message('Everything is complete.')])])
        summary = runner.run()
        self.assertEqual(summary['status'], 'model_responded_without_execution')
        self.assertEqual(summary['objective_status'], 'not_independently_verified')
        self.assertEqual(summary['exit_code'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
