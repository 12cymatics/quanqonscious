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
import { buildStandalone, PAIR } from './build-standalone.mjs';

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
const kernel = read('faraday/kernel.js');
const bench = read('faraday/benchmark.js');

console.log('single-file build\n');

check('nothing is loaded from outside the file', () => {
  // The whole point. A surviving src= is a page that breaks when moved.
  const srcs = built.match(/<script[^>]*\bsrc=/gi) ?? [];
  eq(srcs.length, 0, `external <script src> count (${srcs.join(', ')})`);
});

check('the <script src> pair is gone from the output', () => {
  eq(built.includes(PAIR), false, 'the original tag pair still present');
});

check('both kernels are carried in full, byte for byte', () => {
  // Substring, not a length check: a truncated inline would still be "present"
  // by any looser test, and would fail at runtime rather than here.
  eq(built.includes(kernel), true, 'kernel.js body embedded verbatim');
  eq(built.includes(bench), true, 'benchmark.js body embedded verbatim');
});

check('the page around the scripts is untouched', () => {
  // Everything except the swapped tags must survive: the build must not be a
  // rewrite of the page, only a substitution of how its code arrives.
  const [beforeBuilt] = built.split('<script>');
  const [beforeHtml] = html.split(PAIR);
  eq(beforeBuilt, beforeHtml, 'markup preceding the scripts');
});

check('the output grew by exactly the two inlined bodies', () => {
  // Pins that nothing else was added or dropped. The delta is the two sources
  // plus the four tags that replace the two src tags, minus the pair removed.
  const tags = '<script>\n\n</script>'.length * 2 + '\n'.length;
  eq(built.length, html.length - PAIR.length + kernel.length + bench.length + tags,
     'built length');
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
