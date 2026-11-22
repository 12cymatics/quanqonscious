"""Real dispatch, signed export, and publication-boundary integrity checks."""
import base64
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hosted"))

from gate3 import Gate, canonical, init_state, keygen
import export_audit


def archive_members(path):
    with tarfile.open(path, "r:gz") as archive:
        return [(member, archive.extractfile(member).read()) for member in archive]


def change_member(source, destination, selected_name, replacement):
    members = archive_members(source)
    found = False
    with tarfile.open(destination, "w:gz") as archive:
        for member, raw in members:
            if member.name == selected_name:
                raw = replacement(raw)
                member.size = len(raw)
                found = True
            archive.addfile(member, io.BytesIO(raw))
    if not found:
        raise AssertionError("The selected real archive member was absent")


class SignedPublicExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gate-public-export-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.private = self.base / "authority.key"
        self.public = self.base / "authority.pub"
        self.fingerprint = keygen(self.private, self.public)
        self.authorization = self.base / "authorization.txt"
        self.authorization.write_bytes((ROOT / "tests" / "authorization.txt").read_bytes())
        self.state = self.base / "state"
        self.source = (ROOT / "source" / "verified_delegation_instruction.txt").read_bytes()
        init_state(self.state, ROOT / "source" / "verified_delegation_instruction.txt",
                   self.private, self.fingerprint, self.authorization)
        project = self.base / "project"
        project.mkdir(mode=0o700)
        (project / "preserve_instruction.py").write_text(
            'from pathlib import Path\n'
            'Path("/output/authoritative-instruction.bin").write_bytes(GATE_INSTRUCTION_BYTES)\n'
            'Path("/output/assignment.txt").write_text(GATE_ASSIGNED_OPERATION, encoding="utf-8")\n',
            encoding="utf-8",
        )
        self.gate = Gate(self.state)
        self.envelope = self.gate.create(project, "preserve_instruction.py",
                                         "Retain the exact authoritative bytes and authenticated assignment.")
        self.receipt = self.gate.dispatch(self.envelope)
        self.assertEqual(self.receipt["payload"]["data"]["exit_code"], 0)
        self.handoff = self.receipt["payload"]["data"]["handoff_id"]
        self.results = self.base / "agent-results"
        self.results.mkdir(mode=0o700)
        (self.results / "model-final.txt").write_text(
            "Export integration check; the test did not call an API model.", encoding="utf-8")
        self.destination = self.base / "artifacts" / "verified-agent-gate-run.tar.gz"
        self.metadata = {"github": {"run_id": "local-export-test", "run_attempt": "1"},
                         "objective_status": "not_independently_verified"}

    def export(self, destination=None, sensitive_values=()):
        return export_audit.export_public(
            self.state, self.results, destination or self.destination,
            self.metadata, sensitive_values=sensitive_values,
        )

    def test_real_dispatch_exports_exact_instruction_and_verifiable_signed_history(self):
        result = self.export()
        verified = export_audit.verify_public(self.destination, self.fingerprint)
        self.assertTrue(verified["verified"])
        self.assertFalse(verified["resumable_controller_state"])
        self.assertEqual(verified["github"], self.metadata["github"])
        self.assertEqual(result["audit"]["unfinished_handoffs"], [])
        self.assertIn("completed", result["audit"]["events"])
        members = {member.name: raw for member, raw in archive_members(self.destination)}
        self.assertEqual(members["controller/instruction.bin"], self.source)
        output = "controller/results/" + self.handoff + "/output/authoritative-instruction.bin"
        self.assertEqual(members[output], self.source)
        self.assertEqual(members["controller/receipts/" + self.handoff + ".json"], canonical(self.receipt))

    def test_wrong_independently_enrolled_fingerprint_rejects_real_archive(self):
        self.export()
        with self.assertRaisesRegex(ValueError, "independently enrolled fingerprint"):
            export_audit.verify_public(self.destination, "0" * 64)

    def test_modified_member_rejected_by_signed_inventory(self):
        self.export()
        changed = self.base / "modified.tar.gz"
        change_member(self.destination, changed, "agent-results/model-final.txt", lambda _: b"Changed after signing")
        with self.assertRaisesRegex(ValueError, "contents changed"):
            export_audit.verify_public(changed, self.fingerprint)

    def test_replaced_inventory_signature_rejected_cryptographically(self):
        self.export()
        changed = self.base / "invalid-signature.tar.gz"

        def alter_signature(raw):
            value = json.loads(raw)
            signature = bytearray(base64.b64decode(value["signature_b64"]))
            signature[0] ^= 1
            value["signature_b64"] = base64.b64encode(signature).decode("ascii")
            return canonical(value)

        change_member(self.destination, changed, "export-manifest.json", alter_signature)
        from cryptography.exceptions import InvalidSignature
        with self.assertRaises(InvalidSignature):
            export_audit.verify_public(changed, self.fingerprint)

    def test_private_keys_and_encodings_are_absent_from_all_public_members(self):
        self.export()
        private_values = [self.private.read_bytes(), (self.state / "audit.key").read_bytes()]
        forbidden = [representation for key in private_values
                     for representation in (key, base64.b64encode(key), key.hex().encode("ascii"))]
        for member, raw in archive_members(self.destination):
            self.assertFalse(member.name.endswith(("authority.key", "audit.key")), member.name)
            for secret in forbidden:
                self.assertNotIn(secret, raw, member.name)

    def test_leaked_authority_audit_or_api_credential_blocks_archive_creation(self):
        authority_seed = self.private.read_bytes()
        audit_seed = (self.state / "audit.key").read_bytes()
        api_credential = b"synthetic-api-credential-for-publication-boundary-check"
        cases = [authority_seed, base64.b64encode(authority_seed), authority_seed.hex().encode("ascii"),
                 audit_seed, base64.b64encode(audit_seed), audit_seed.hex().encode("ascii"), api_credential]
        for index, leaked in enumerate(cases):
            with self.subTest(credential_representation=index):
                (self.results / "model-final.txt").write_bytes(b"Sensitive content: " + leaked)
                destination = self.base / "artifacts" / ("blocked-" + str(index) + ".tar.gz")
                with self.assertRaisesRegex(ValueError, "credential value"):
                    self.export(destination, (api_credential,))
                self.assertFalse(destination.exists())
                self.assertFalse(destination.with_name(destination.name + ".partial").exists())

    def test_failed_staging_integrity_check_leaves_no_publishable_archive(self):
        actual_verifier = export_audit.verify_public
        inspected_staging = []

        def corrupt_then_verify(path, fingerprint):
            self.assertEqual(path.name, "verified-agent-gate-run.tar.gz.partial")
            self.assertTrue(path.is_file())
            self.assertFalse(self.destination.exists())
            inspected_staging.append(path)
            change_member(path, path, "agent-results/model-final.txt", lambda _: b"Corrupted before promotion")
            return actual_verifier(path, fingerprint)

        with mock.patch.object(export_audit, "verify_public", side_effect=corrupt_then_verify):
            with self.assertRaisesRegex(ValueError, "contents changed"):
                self.export()
        self.assertEqual(len(inspected_staging), 1)
        self.assertFalse(self.destination.exists())
        self.assertFalse(inspected_staging[0].exists())
        self.assertEqual(list(self.destination.parent.glob("verified-agent-gate-run.*")), [])

    def test_archive_symlink_member_rejected_without_extraction(self):
        self.export()
        changed = self.base / "symlink.tar.gz"
        with tarfile.open(changed, "w:gz") as archive:
            for member, raw in archive_members(self.destination):
                archive.addfile(member, io.BytesIO(raw))
            link = tarfile.TarInfo("agent-results/forbidden-link")
            link.type = tarfile.SYMTYPE
            link.linkname = str(self.private)
            archive.addfile(link)
        with self.assertRaisesRegex(ValueError, "unsafe member"):
            export_audit.verify_public(changed, self.fingerprint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
