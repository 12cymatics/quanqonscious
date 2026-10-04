/* The committed dns/faraday_disc.wasm comes from the committed
   dns/faraday_disc.cpp, and produces the same numbers as dns/faraday-disc.js.
   Both halves are checked here by building the source again and holding the
   three -- fresh module, committed module, JavaScript solver -- to the same
   values on the same state.

   Why the gate is behavioural rather than a byte comparison. Byte identity is
   the stronger statement and it does hold: measured with Ubuntu clang 18.1.3,
   rebuilding gives the committed 25254 bytes exactly. But it is a property of
   one compiler build, not of this repository, and a runner image that ships a
   different clang would fail a byte gate while the module was perfectly correct.
   Equality of the NUMBERS is the property that matters and it is compiler
   independent, so it is what is asserted; the byte comparison is reported
   alongside it, because when it does hold it is worth knowing.

   There is no skip. If clang cannot build the module this fails, because a
   checkout that cannot rebuild its own binary has no way to know the binary
   matches its source.

       node dns/check-wasm-build.mjs
*/

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const require = createRequire(import.meta.url);
const D = require(join(REPO, 'dns', 'faraday-disc.js'));
const F = require(join(REPO, 'dns', 'faraday-floquet.js'));
const W = require(join(REPO, 'dns', 'faraday-disc-wasm.js'));
const K = require(join(REPO, 'faraday', 'kernel.js'));

let pass = 0;
const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}

const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837,
               nu: 1.0035510043427237e-6, gamma: 0.07273614042160757,
               g: 9.80665 };

const work = mkdtempSync(join(tmpdir(), 'wasm-build-'));
const fresh = join(work, 'faraday_disc.wasm');
let buildLog = '';
try {
  buildLog = execFileSync('sh', [join(HERE, 'build-wasm.sh'), fresh],
                          { encoding: 'utf8' });
} catch (e){
  rmSync(work, { recursive: true, force: true });
  console.log('\nThe module could not be rebuilt from dns/faraday_disc.cpp:\n');
  console.log(String(e.stderr || e.message));
  console.log(
    'clang++ with the wasm32 target and wasm-ld are required. On Ubuntu that is\n'
    + 'the clang and lld packages. This is a failure rather than a skip: without\n'
    + 'a rebuild there is no way to know the committed binary is the committed\n'
    + 'source, and the binary is what the page runs.');
  process.exit(1);
}
console.log(buildLog.trim());

const freshBytes = new Uint8Array(readFileSync(fresh));
const committed = new Uint8Array(readFileSync(join(HERE, 'faraday_disc.wasm')));

console.log('\nrebuilt against committed');
ok(freshBytes.length > 1000, 'the rebuild produced a module of plausible size',
   `${freshBytes.length} bytes`);
ok(WebAssembly.validate(freshBytes), 'which WebAssembly accepts');
let sameBytes = freshBytes.length === committed.length;
if (sameBytes) for (let i = 0; i < freshBytes.length; i++)
  if (freshBytes[i] !== committed[i]){ sameBytes = false; break; }
console.log(`       byte comparison: fresh ${freshBytes.length} bytes, committed `
  + `${committed.length}; ${sameBytes ? 'identical' : 'DIFFERENT'} `
  + `(reported, not gated -- see the header)`);

/* The state the three are held to. Every block non-zero, so a term dropped
   anywhere in the transcription shows up, and an elevation that is a real mode
   so the surface terms are exercised on something they are built for. */
const m = 12, nr = 20, nz = 10;
const jp = K.jpZerosNearN(m, m + 1.9, 1).sort((a, b) => a - b)[0];
const k = jp/CELL.R;
const n = 2*(nr - 1)*nz + nr*nz + nr;
const mk = () => new FaradayDiscOf();
function FaradayDiscOf(){
  return new D.FaradayDisc({ m, nr, nz, ...CELL, contact: 'free',
                             accel: 7.060788, omegaD: 2*Math.PI*111 });
}
const S = mk();
const dt = S.stableStep(0.4);
const href = 1e-9, uref = href*Math.PI*111;
const v = new Float64Array(n);
for (let i = 0; i < n; i++) v[i] = Math.sin(0.7*i + 0.3) + 0.25*Math.cos(1.9*i);
for (let i = 0; i < nr; i++) v[2*(nr-1)*nz + nr*nz + i] = K.besselJ(m, k*S.rc[i]);

const steps = Math.max(1, Math.round((2*Math.PI/(2*Math.PI*111))/dt));
const runWasm = bytes => {
  const out = new Float64Array(n);
  W.createDiscEngine({ bytes }).load(mk(), steps, dt, 7.060788, 2*Math.PI*111, S.g)
   .applyPeriodMap(v, out, dt, href, uref);
  return out;
};
const jsOut = new Float64Array(n);
F.applyDiscPeriodMap(mk(), v, steps, dt, jsOut, href, uref);
const freshOut = runWasm(freshBytes);
const commOut = runWasm(committed);

const identical = (a, b) => {
  let same = 0;
  for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++;
  return same;
};
console.log('\none drive period of ' + steps + ' steps on ' + n + ' unknowns');
ok(identical(freshOut, commOut) === n,
   'the freshly built module and the committed one give identical values',
   `${identical(freshOut, commOut)} of ${n} identical`);
ok(identical(freshOut, jsOut) === n,
   'and both give the JavaScript solver\'s values, to the last bit',
   `${identical(freshOut, jsOut)} of ${n} identical`);
let moved = 0;
for (let i = 0; i < n; i++) if (freshOut[i] !== v[i]) moved++;
ok(moved > 0.9*n,
   'on a state the period map actually moved, so the equalities are not three '
   + 'copies of the input', `${moved} of ${n} moved`);

rmSync(work, { recursive: true, force: true });

console.log('\n' + '-'.repeat(66));
if (failures.length){
  console.log(`${pass} passed, ${failures.length} FAILED\n`);
  for (const f of failures) console.log('  FAIL  ' + f);
  process.exit(1);
}
console.log(`${pass} checks passed, 0 failed`);
