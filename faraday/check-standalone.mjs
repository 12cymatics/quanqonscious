// The single-file build must be self-contained, and must be the same program.
//
// Checks a built copy rather than a committed one, so there is nothing here to
// drift: the build runs from the current cymatic.html and kernel sources every
// time this executes.
//
//     node faraday/check-standalone.mjs

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildStandalone, SCRIPTS, WASM_TAG, WASM_PATH } from './build-standalone.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const read = p => readFileSync(resolve(REPO, p), 'utf8');

let passed = 0, failed = 0;
const check = (name, fn) => {
  try { fn(); console.log(`  ok   ${name}`); passed++; }
  catch (e) { console.log(`  FAIL ${name}\n         ${e.message}`); failed++; }
};
const eq = (got, want, what) => {
  if (got !== want) throw new Error(`${what}: got ${got}, want ${want}`);
};

let built;
try {
  built = buildStandalone();
} catch (e) {
  console.log(`single-file build\n\n  FAIL the build refused to run\n         ${e.message}`);
  process.exit(1);
}
const html = read('cymatic.html');
const sources = SCRIPTS.map(({ path }) => read(path));

console.log('single-file build\n');

check('nothing is loaded from outside the file', () => {
  // The whole point. A surviving src= is a page that breaks when moved.
  const srcs = built.match(/<script[^>]*\bsrc=/gi) ?? [];
  eq(srcs.length, 0, `external <script src> count (${srcs.join(', ')})`);
});

check('every <script src> tag is gone from the output', () => {
  for (const { tag, path } of SCRIPTS)
    eq(built.includes(tag), false, `the original tag for ${path} is still present`);
});

check('the inlined blocks keep their ids', () => {
  /* The Navier-Stokes panel reads its own solver source back out of these
     elements to build its worker. An id dropped by the builder leaves the
     built page unable to run the check, with nothing else to notice. */
  for (const { open } of SCRIPTS)
    eq(built.includes(open), true, `the inlined block opens with ${open}`);
});

check('every source is carried in full, byte for byte', () => {
  // Substring, not a length check: a truncated inline would still be "present"
  // by any looser test, and would fail at runtime rather than here.
  SCRIPTS.forEach(({ path }, i) =>
    eq(built.includes(sources[i]), true, `${path} body embedded verbatim`));
});

check('the page around the scripts is untouched', () => {
  // Everything except the swapped tags must survive: the build must not be a
  // rewrite of the page, only a substitution of how its code arrives.
  const [beforeBuilt] = built.split('<script>');
  const [beforeHtml] = html.split(SCRIPTS[0].tag);
  eq(beforeBuilt, beforeHtml, 'markup preceding the scripts');
});

const wasmBytes = readFileSync(resolve(REPO, WASM_PATH));
const wasmB64 = Buffer.from(wasmBytes).toString('base64');

check('the output grew by exactly the inlined bodies', () => {
  // Pins that nothing else was added or dropped: each src tag is replaced by
  // its open tag, a newline, the source, a newline and the close tag, and the
  // empty wasm tag by the same tag carrying the base64 assignment.
  let want = html.length;
  SCRIPTS.forEach(({ tag, open }, i) => {
    want += open.length + 1 + sources[i].length + 1 + '</script>'.length - tag.length;
  });
  const wasmBlock = `<script id="dnsWasmBase64">\nglobalThis.FARADAY_DISC_WASM_BASE64 = "${wasmB64}";\n</script>`;
  want += wasmBlock.length - WASM_TAG.length;
  eq(built.length, want, 'built length');
});

check('the compiled period map is carried, and decodes to the committed bytes', () => {
  // Not "a base64 string is present": the string is decoded and compared with
  // dns/faraday_disc.wasm byte for byte. A truncated or stale inline would still
  // look like base64, and would fail in the page instead of here.
  eq(built.includes(`globalThis.FARADAY_DISC_WASM_BASE64 = "${wasmB64}";`), true,
     'the assignment is present with the exact encoding');
  const m = built.match(/globalThis\.FARADAY_DISC_WASM_BASE64 = "([A-Za-z0-9+/=]+)";/);
  eq(m !== null, true, 'the assignment parses');
  const back = Buffer.from(m[1], 'base64');
  eq(back.length, wasmBytes.length, 'decoded byte length');
  eq(Buffer.compare(back, wasmBytes), 0, 'decoded bytes equal the module on disk');
  // And the decoded bytes are a module WebAssembly will take, which a corrupted
  // encoding would not be even at the right length.
  eq(WebAssembly.validate(back), true, 'the decoded module validates');
});

check('a page without the wasm tag is refused', () => {
  // The one failure that produces a page which LOADS and then refuses at the
  // panel, which is the hardest kind to notice.
  let threw = null;
  try { buildStandalone({ 'cymatic.html': html.replace(WASM_TAG, '') }); }
  catch (e) { threw = e; }
  if (threw === null)
    throw new Error('the build accepted a page with no place to put the compiled '
                    + 'period map, and emitted one without it');
  eq(/dnsWasmBase64/.test(threw.message), true, 'the refusal names the tag');
});

check('a literal </script> in a source is refused, not silently emitted', () => {
  // The one input that produces a broken PAGE rather than a build error, so
  // the guard is exercised against a source that really contains one. An
  // earlier version asserted a regex against a string it defined itself,
  // which would have passed with the guard deleted from the builder.
  let threw = null;
  try {
    buildStandalone({ 'faraday/kernel.js': 'const s = "</script>";' });
  } catch (e) { threw = e; }
  if (threw === null)
    throw new Error('the build accepted a source containing </script>; the '
                    + 'emitted page would end the block early and render the '
                    + 'rest of the kernel as visible text');
  eq(/<\/script/.test(threw.message), true,
     `the refusal names the offending tag (got: ${threw.message})`);
});

check('a clean source is accepted, so the guard is not refusing everything', () => {
  // Without this, a guard that threw unconditionally would pass the check
  // above and still break every build.
  buildStandalone({ 'faraday/kernel.js': 'const s = 1;' });
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
