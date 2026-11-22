#!/usr/bin/env python3
"""Check the exact enrollment-page crypto and UI without retaining generated keys."""

import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


PAGE = Path(__file__).with_name("enroll.html")


def main():
    html = PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
    styles = re.findall(r"<style>([\s\S]*?)</style>", html)
    assert len(scripts) == len(styles) == 1
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    for block, kind in ((scripts[0], "script"), (styles[0], "style")):
        digest = base64.b64encode(hashlib.sha256(block.encode()).digest()).decode()
        assert f"{kind}-src 'sha256-{digest}'" in csp, f"{kind} CSP hash mismatch"
    assert "default-src 'none'" in csp and "connect-src 'none'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert not re.search(r"<(?:script|iframe|img|link)\b[^>]+(?:src|href)=", html, re.I)
    assert not re.search(r"\s+on\w+=", html, re.I)
    assert not re.search(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|localStorage|sessionStorage|indexedDB)\b", scripts[0])

    node = shutil.which("node")
    assert node, "Node.js with native WebCrypto is required for this cross-runtime check"
    harness = r'''
const fs = require("node:fs");
const vm = require("node:vm");
const {webcrypto} = require("node:crypto");
const script = JSON.parse(fs.readFileSync(0, "utf8"));
const assert = require("node:assert/strict");
function makePage(supported = true, providedCrypto = webcrypto) {
  const elements = new Map();
  const listeners = new Map();
  const copied = [];
  const doc = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      value: "", type: id === "private-key" ? "password" : "text",
      checked: false, hidden: id === "key-section", disabled: ["generate", "clear"].includes(id),
      textContent: "", events: new Map(), classList: {toggle() {}},
      addEventListener(event, handler) { this.events.set(event, handler); }
    });
    return elements.get(id);
  }};
  const win = {isSecureContext: supported, crypto: providedCrypto,
    addEventListener(event, handler) { listeners.set(event, handler); }};
  win.top = win; win.self = win;
  const context = vm.createContext({window: win, document: doc,
    navigator: {clipboard: {writeText: async text => copied.push(text)}},
    Uint8Array, TextEncoder, btoa, atob});
  vm.runInContext(script, context);
  return {context, elements, listeners, copied};
}
(async () => {
  const page = makePage();
  const el = id => page.elements.get(id);
  assert.equal(vm.runInContext("authority", page.context), null);
  assert.equal(el("private-key").value, "");
  assert.equal(el("key-section").hidden, true);
  assert.equal(page.copied.length, 0);
  await el("generate").events.get("click")();
  assert.equal(page.copied.length, 0, "generation must not copy anything");
  assert.equal(el("private-key").value, "", "private key must start hidden and absent from the field");
  assert.equal(el("key-section").hidden, false);
  assert.equal(el("generate").disabled, true, "clear before generating a replacement identity");
  const created = vm.runInContext("authority", page.context);
  el("reveal").checked = true;
  el("reveal").events.get("change")();
  assert.equal(el("private-key").value, created.privateB64);
  el("reveal").checked = false;
  el("reveal").events.get("change")();
  assert.equal(el("private-key").value, "");
  await el("copy-private").events.get("click")();
  assert.equal(page.copied.length, 1);
  assert.equal(page.copied[0], created.privateB64);
  page.copied.fill("");

  const records = [];
  for (let i = 0; i < 8; i++) {
    const record = await vm.runInContext("createAuthority(window.crypto)", page.context);
    const d = Buffer.from(record.privateB64, "base64").toString("base64url");
    const x = Buffer.from(record.publicB64, "base64").toString("base64url");
    const privateKey = await webcrypto.subtle.importKey("jwk", {kty:"OKP", crv:"Ed25519", d, x}, {name:"Ed25519"}, false, ["sign"]);
    const message = Buffer.from("VERIFIED-AGENT-GATE-V3\0enrollment-cross-runtime-check", "utf8");
    const signature = await webcrypto.subtle.sign({name:"Ed25519"}, privateKey, message);
    records.push({privateB64: record.privateB64, publicB64: record.publicB64,
      fingerprint: record.fingerprint, signatureB64: Buffer.from(signature).toString("base64")});
    record.seed.fill(0); record.privateB64 = "";
  }
  const seedReference = created.seed;
  page.listeners.get("pagehide")();
  assert.equal(vm.runInContext("authority", page.context), null);
  assert(seedReference.every(byte => byte === 0));
  assert.equal(created.privateB64, "");
  assert.equal(el("private-key").value, "");
  assert.equal(el("fingerprint").value, "");
  assert.equal(el("public-key").value, "");
  assert.equal(el("key-section").hidden, true);

  const insecure = makePage(false);
  assert.equal(insecure.elements.get("generate").disabled, true);
  await insecure.elements.get("generate").events.get("click")();
  assert.equal(vm.runInContext("authority", insecure.context), null);
  const unsupported = makePage(true, {subtle: {generateKey: async () => {
    const error = new Error("unsupported"); error.name = "NotSupportedError"; throw error;
  }}});
  await unsupported.elements.get("generate").events.get("click")();
  assert.equal(vm.runInContext("authority", unsupported.context), null);
  assert(unsupported.elements.get("status").textContent.includes("does not support native Ed25519"));
  // Generated test seeds go only through this subprocess pipe, never a file or test log.
  process.stdout.write(JSON.stringify(records));
})().catch(() => { process.stderr.write("enrollment JavaScript check failed\n"); process.exitCode = 1; });
'''
    result = subprocess.run(
        [node, "-e", harness], input=json.dumps(scripts[0]), text=True,
        capture_output=True, timeout=30, check=False,
    )
    if result.returncode:
        raise AssertionError("Exact enrollment JavaScript failed its native crypto or UI checks")
    records = json.loads(result.stdout)
    assert len(records) == 8
    assert len({record["fingerprint"] for record in records}) == 8
    message = b"VERIFIED-AGENT-GATE-V3\0enrollment-cross-runtime-check"
    for record in records:
        seed = base64.b64decode(record["privateB64"], validate=True)
        assert len(seed) == 32
        assert base64.b64encode(seed).decode() == record["privateB64"]
        private = Ed25519PrivateKey.from_private_bytes(seed)
        public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        assert public == base64.b64decode(record["publicB64"], validate=True)
        assert hashlib.sha256(public).hexdigest() == record["fingerprint"]
        signature = base64.b64decode(record["signatureB64"], validate=True)
        private.public_key().verify(signature, message)
        assert private.sign(message) == signature
        record.clear()
    records.clear()
    print("PASS: exact page JavaScript; 8 WebCrypto/Python key and signature checks; explicit-copy and clear lifecycle; unsupported crypto rejection; CSP hashes. No generated test keys retained in files.")


if __name__ == "__main__":
    main()
