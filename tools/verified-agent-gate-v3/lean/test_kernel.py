import hashlib
import json
from pathlib import Path
import struct
import subprocess
import unittest

KERNEL = Path(__file__).with_name("gate-kernel")


def invoke(action, fields):
    encoded = b"".join(field.hex().encode("ascii") + b"\n" for field in fields)
    return subprocess.run([str(KERNEL), action], input=encoded, capture_output=True, timeout=15)


def reference_encode(fields):
    return b"VGK3\0" + b"".join(struct.pack(">Q", len(field)) + field for field in fields)


class CompiledKernelTests(unittest.TestCase):
    def setUp(self):
        self.instruction = bytes(range(256)) * 17 + "Exact source: वेद\n".encode("utf-8")
        self.policy = hashlib.sha256(b"trusted policy").digest()
        self.digest = hashlib.sha256(self.instruction).digest()
        self.fields = [self.instruction, self.policy, self.digest,
                       b"Implement the specified production operation", b"",
                       bytes(range(32)), bytes(range(32, 64)),
                       b'{"entrypoint":"agent.py","source_length":' +
                       str(len(self.instruction)).encode("ascii") + b'}']
        self.expected = [self.instruction, self.policy, self.digest, b""]

    def test_native_encoding_matches_independent_wire_format(self):
        result = invoke("encode", self.fields)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(bytes.fromhex(result.stdout.decode().strip()), reference_encode(self.fields))

    def test_verification_returns_every_exact_binary_field(self):
        result = invoke("verify", [reference_encode(self.fields)] + self.expected)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([bytes.fromhex(line.decode()) for line in result.stdout.splitlines()], self.fields)

    def test_instruction_cannot_be_replaced_by_operation_text(self):
        fields = list(self.fields)
        fields[3] = fields[0]
        fields[0] = b"Build this exactly"
        result = invoke("verify", [reference_encode(fields)] + self.expected)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")

    def test_every_trusted_value_is_enforced(self):
        raw = reference_encode(self.fields)
        for i in range(4):
            with self.subTest(trusted_value=i):
                expected = list(self.expected)
                expected[i] = expected[i] + b"x"
                result = invoke("verify", [raw] + expected)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")

    def test_parent_inheritance_uses_exact_registered_parent_digest(self):
        parent_raw = reference_encode(self.fields)
        fields = list(self.fields)
        fields[4] = hashlib.sha256(parent_raw).digest()
        fields[5] = bytes(reversed(range(32)))
        fields[3] = b"Child production operation"
        result = invoke("verify", [reference_encode(fields)] + self.expected[:3] + [fields[4]])
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = [bytes.fromhex(line.decode()) for line in result.stdout.splitlines()]
        self.assertEqual(actual[0], self.instruction)
        self.assertEqual(actual[4], hashlib.sha256(parent_raw).digest())
        self.assertEqual(invoke("verify", [reference_encode(fields)] + self.expected).returncode, 1)

    def test_malformed_or_ambiguous_framing_is_rejected(self):
        raw = reference_encode(self.fields)
        attacks = [raw[:-1], raw + b"\0", b"X" + raw[1:],
                   b"VGK3\0" + b"\xff" * 8 + raw[13:],
                   b"VGK3\0", raw + struct.pack(">Q", 0)]
        for value in attacks:
            with self.subTest(length=len(value)):
                result = invoke("verify", [value] + self.expected)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")

    def test_invalid_fixed_field_lengths_or_empty_fields_are_rejected(self):
        for field_id, bad in [(0, b""), (1, b"x" * 31), (2, b"x" * 33),
                              (3, b""), (4, b"x"), (5, b"x" * 16),
                              (6, b"x" * 31), (7, b"")]:
            with self.subTest(field=field_id):
                fields = list(self.fields)
                fields[field_id] = bad
                self.assertEqual(invoke("encode", fields).returncode, 1)
                result = invoke("verify", [reference_encode(fields)] + self.expected)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")

    def test_noncanonical_transport_is_rejected(self):
        good = b"".join(field.hex().encode() + b"\n" for field in self.fields)
        attacks = [good.upper(), good + b"\n", good.rstrip(b"\n"), b"g" + good[1:]]
        for encoded in attacks:
            result = subprocess.run([str(KERNEL), "encode"], input=encoded,
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")

    def test_serialize_and_decode_large_exact_instruction(self):
        fields = list(self.fields)
        fields[0] = bytes(range(256)) * 256
        fields[2] = hashlib.sha256(fields[0]).digest()
        result = invoke("encode", fields)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = bytes.fromhex(result.stdout.decode().strip())
        result = invoke("verify", [payload, fields[0], fields[1], fields[2], b""])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(bytes.fromhex(result.stdout.splitlines()[0].decode()), fields[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
