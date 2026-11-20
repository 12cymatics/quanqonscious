from pathlib import Path
import hashlib,json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
p=Path(__file__).parent
payload=(p/"policy.canonical.json").read_bytes()
obj=json.loads(payload)
assert payload==json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
raw=(p/"instruction.bin").read_bytes()
assert raw==bytes.fromhex(obj["original_instruction_hex"])
assert hashlib.sha256(raw).hexdigest()==obj["original_instruction_sha256"]
assert hashlib.sha256(payload).hexdigest()==(p/"policy.sha256").read_text().strip()
Ed25519PublicKey.from_public_bytes((p/"policy.ed25519.pub").read_bytes()).verify((p/"policy.ed25519.sig").read_bytes(),payload)
print("PASS: canonical bytes, SHA-256, Ed25519 signature")
