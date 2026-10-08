#!/usr/bin/env node
/* The committed dns/faraday_cell3d.wasm comes from the committed
   dns/faraday_cell3d.cpp, and produces the same numbers as dns/faraday-cell3d.js.
   All three are checked here by building the source again and holding the fresh
   module, the committed module and the JavaScript solver to identical values on
   identical state -- operator by operator, and then over a full drive period.

   BIT FOR BIT, NOT APPROXIMATELY. `Object.is` on every element of every array:
   the C++ is a transcription, not a reimplementation, and the only property worth
   asserting about a transcription is that it changed nothing. An agreement to
   twelve digits would mean the two had drifted and nobody could say where.

   Why the gate is behavioural rather than a byte comparison of the module. Byte
   identity is the stronger statement and it does hold on one compiler, but it is a
   property of that compiler build and not of this repository: a runner image with
   a different clang would fail a byte gate while the module was perfectly correct.
   Equality of the NUMBERS is compiler independent, so it is what is asserted; the
   byte comparison is reported alongside, because when it does hold it is worth
   knowing.

   There is no skip. If clang cannot build the module this FAILS, because a
   checkout that cannot rebuild its own binary has no way to know the binary
   matches its source -- and the binary is what the page runs.

       node dns/check-cell3d-wasm.mjs
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
const K = require(join(REPO, 'faraday', 'kernel.js'));
const C = require(join(HERE, 'faraday-cell3d.js'));
const { FaradayCell3D } = C;
const WM = require(join(HERE, 'faraday-cell3d-wasm.js'));
const { FaradayCell3DWasm, CELL3D_ARRAYS, CELL3D_JS_NAME, CELL3D_WASM_EXPORTS } = WM;

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(`${label}${detail ? '  -- ' + detail : ''}`);
         console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
function throws(label, fn, fragment){
  try { fn(); ok(false, label, 'did not throw'); }
  catch (e){ ok(!fragment || String(e.message).includes(fragment), label,
                `threw "${String(e.message).slice(0, 140)}"`); }
}
const section = t => console.log('\n' + t);

/* The apparatus the renderer draws: a 24.25 mm quartz cell, 3 mm of water. */
const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837,
               nu: 1.0035510043427237e-6, gamma: 0.07273614042160757,
               g: 9.80665 };

/* A deterministic sequence, so a failure is reproducible. Its quality is
   irrelevant: it is only used to excite every degree of freedom, which is what a
   bit-for-bit comparison needs -- a smooth field leaves most of the branches in
   famLaplacian's boundary handling unvisited. */
function rnd(seed){
  let s = seed >>> 0;
  return () => { s = (s*1664525 + 1013904223) >>> 0; return s/4294967296; };
}

/* Every element, by Object.is, so that a NaN against a NaN counts as equal and a
   -0 against a +0 does not. Returns null when identical. */
function diff(name, a, b){
  if (!a || !b) return `${name}: missing (${!!a} vs ${!!b})`;
  if (a.length !== b.length) return `${name}: length ${a.length} vs ${b.length}`;
  let n = 0, first = -1, worst = 0;
  for (let i = 0; i < a.length; i++)
    if (!Object.is(a[i], b[i])){
      n++;
      if (first < 0) first = i;
      const d = Math.abs(a[i] - b[i])/Math.max(Number.MIN_VALUE, Math.abs(a[i]));
      /* `!(d <= worst)` and not `d > worst`, so a NaN on either side STICKS in the report
         instead of being skipped: an element read as NaN used to print "worst relative
         0.000e+0" beside the count that failed it, which says the opposite of the truth. */
      if (!(d <= worst)) worst = d;
    }
  return n ? `${name}: ${n}/${a.length} elements differ, first at ${first} `
           + `(${a[first]} vs ${b[first]}), worst relative ${worst.toExponential(3)}`
         : null;
}

const kOf = m => K.jpZerosNearN(m, m + 1.9, 1).sort((a, b) => a - b)[0]/CELL.R;
const omegaOf = (k, h) => Math.sqrt((CELL.g*k + CELL.gamma*k*k*k/CELL.rho)
                                    *Math.tanh(k*h));

/* The state both sides start from: a mode plus noise on every array, so that the
   axis rows, the rim rows, the floor and the surface half cell are all non-zero
   and every branch is taken. */
function fillState(S, seed, m, rel){
  const k = kOf(m), r = rnd(seed);
  for (let i = 0; i < S.nr; i++)
    for (let kk = 0; kk < S.nth; kk++)
      S.eta[S.ie(i, kk)] = rel*S.h*K.besselJ(m, k*S.rc[i])*Math.cos(m*(kk + 0.5)*S.dth)
                         + 0.02*rel*S.h*(2*r() - 1);
  for (let i = 0; i < S.NU; i++) S.u[i] = 1e-3*(2*r() - 1);
  for (let i = 0; i < S.NV; i++) S.v[i] = 1e-3*(2*r() - 1);
  for (let i = 0; i < S.NW; i++) S.w[i] = 1e-3*(2*r() - 1);
  for (let i = 0; i < S.NP; i++) S.p[i] = 1e-2*(2*r() - 1);
  for (let i = 0; i < S.NE; i++) S.Ht[i] = 1e-2*(2*r() - 1);
  S.refreshMetric();
  return S;
}

/* A matched pair: a JavaScript solver and a C++ one on identical state. The
   JavaScript one is a SEPARATE object from the module's own `js`, which is the
   grid source and the clock -- stepping that one would advance the time the
   module's drive reads. That mistake cost a measurement: forty driven steps
   agreed bit for bit undriven and diverged to 1e-1 relative driven, entirely
   from the shared counter. */
function pair(opts, seed, m, rel, bytes){
  const S = fillState(new FaradayCell3D(opts), seed, m, rel);
  const W = new FaradayCell3DWasm(opts, bytes ? { bytes } : undefined);
  for (const nm of ['u', 'v', 'w', 'om', 'p', 'eta', 'Ht']) W.views[nm].set(S[nm]);
  W.refreshMetric();
  return { S, W };
}

/* ---- 1. the module rebuilds from its source --------------------------- */
section('1. the module rebuilds from its source');
const work = mkdtempSync(join(tmpdir(), 'cell3d-wasm-'));
let freshBytes = null;
try {
  const freshPath = join(work, 'faraday_cell3d.wasm');
  let log = '';
  try {
    log = execFileSync('sh', [join(HERE, 'build-wasm-cell3d.sh'), freshPath],
                       { encoding: 'utf8' });
  } catch (e){
    console.log('\nThe module could not be rebuilt from dns/faraday_cell3d.cpp:\n');
    console.log(String(e.stderr || e.message));
    console.log('\nThis is a FAILURE and not a skip: a checkout that cannot rebuild '
      + 'its own binary has no way to know the binary matches its source, and the '
      + 'binary is what the page runs. clang with the wasm32 target and lld are '
      + 'what it needs -- no Emscripten, no sysroot.');
    rmSync(work, { recursive: true, force: true });
    process.exit(1);
  }
  ok(/wrote/.test(log), 'clang built it', log.trim());
  freshBytes = new Uint8Array(readFileSync(freshPath));
  const committed = new Uint8Array(readFileSync(join(HERE, 'faraday_cell3d.wasm')));
  ok(freshBytes.length > 20000, 'the fresh module is a module',
     `${freshBytes.length} bytes`);
  const byteSame = freshBytes.length === committed.length
    && freshBytes.every((b, i) => b === committed[i]);
  console.log(`  note  byte identity with the committed module: ${byteSame ? 'yes' : 'no'}`
    + ` (fresh ${freshBytes.length}, committed ${committed.length}). Reported, not `
    + `asserted: it is a property of one compiler build.`);

  /* No imports at all. An import that reappeared would be a libc call, and a libm
     call is exactly what cannot be bit-identical to JavaScript -- which is why the
     cos in the drive is evaluated on the JavaScript side and passed in. memset and
     memcpy are supplied inside the module for the same reason. */
  for (const [nm, bytes] of [['committed', committed], ['fresh', freshBytes]]){
    const imports = WebAssembly.Module.imports(new WebAssembly.Module(bytes));
    ok(imports.length === 0, `the ${nm} module imports nothing`,
       JSON.stringify(imports));
  }
  const exps = WebAssembly.Module.exports(new WebAssembly.Module(committed))
    .map(e => e.name);
  const missing = CELL3D_WASM_EXPORTS.filter(n => !exps.includes(n));
  ok(missing.length === 0, `it exports all ${CELL3D_WASM_EXPORTS.length} entry `
     + `points the loader names`, missing.join(', '));

/* ---- 2. the array table is the module's own --------------------------- */
section('2. the array table is the module\'s own');
{
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact: 'free' };
  const W = new FaradayCell3DWasm(opts);
  const X = W.inst.exports;
  let badPtr = [], badLen = [];
  for (let key = 0; key < CELL3D_ARRAYS.length; key++){
    if (!(X.cell3d_ptr(key) > 0)) badPtr.push(CELL3D_ARRAYS[key]);
    if (!(X.cell3d_len(key) > 0)) badLen.push(CELL3D_ARRAYS[key]);
  }
  ok(badPtr.length === 0, `all ${CELL3D_ARRAYS.length} arrays have a non-null pointer`,
     badPtr.join(', '));
  ok(badLen.length === 0, 'all of them have a positive length', badLen.join(', '));
  /* One past the end must refuse rather than answer. A null pointer is offset
     zero, which is `rf`, and a view built there would have looked like a grid
     that had gone wrong. */
  ok(X.cell3d_ptr(CELL3D_ARRAYS.length) === 0,
     'a key past the table returns null rather than the first array',
     String(X.cell3d_ptr(CELL3D_ARRAYS.length)));
  ok(X.cell3d_len(CELL3D_ARRAYS.length) === -1,
     'and a length of -1', String(X.cell3d_len(CELL3D_ARRAYS.length)));

  /* Every length matches the JavaScript array it stands for, so a view cannot be
     one element short and read a neighbour's first value as its own last. */
  const wrong = [];
  for (const name of CELL3D_ARRAYS){
    const js = CELL3D_JS_NAME[name] === undefined ? name : CELL3D_JS_NAME[name];
    if (js === null) continue;
    const a = W.js[js];
    if (!a){ wrong.push(`${name}: no JavaScript field ${js}`); continue; }
    if (a.length !== W.views[name].length)
      wrong.push(`${name}: ${W.views[name].length} against ${js}'s ${a.length}`);
  }
  ok(wrong.length === 0, 'every array is the same length as its JavaScript field',
     wrong.join('; '));
  W.release();
}

/* ---- 3. the metric ---------------------------------------------------- */
section('3. the metric, every array, both contact lines');
const METRIC = ['H', 'Hr', 'Hth', 'Hdr', 'Hdth', 'Hx', 'Hxr', 'Hxt',
                'Ex', 'Er', 'Eth', 'Edr', 'Edth', 'Exr', 'Ext',
                'Tx', 'Txr', 'Txt', 'Tr', 'Tth', 'Tdr', 'Tdth'];
for (const contact of ['free', 'pinned']){
  const opts = { nr: 10, nth: 12, nz: 6, ...CELL, contact };
  const { S, W } = pair(opts, 7, 3, 0.1);
  const bad = METRIC.map(n => diff(n, S[n], W.views[n])).filter(Boolean);
  ok(bad.length === 0, `refreshMetric: all ${METRIC.length} arrays, ${contact}`,
     bad.join('; '));
  const HCOL = ['hcolP', 'hcolU', 'hcolV', 'hcolW'];
  const badH = HCOL.map(n => diff(n, S[CELL3D_JS_NAME[n]], W.views[n])).filter(Boolean);
  ok(badH.length === 0, `and H at every node of all four families, ${contact}`,
     badH.join('; '));
  W.release();
}

/* ---- 4. the projection ----------------------------------------------- */
section('4. the projection, operator by operator');
for (const contact of ['free', 'pinned']){
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact };
  const { S, W } = pair(opts, 11, 2, 0.12);
  const bad = [];

  S.omegaFromW(); W.omegaFromW();
  bad.push(diff('omegaFromW', S.om, W.views.om));

  S.divergence(S.u, S.v, S.om, S._div); W.divergence('u', 'v', 'om', 'div');
  bad.push(diff('divergence', S._div, W.views.div));

  S.gradient(S.p, S._gu, S._gv, S._gw); W.gradient('p');
  bad.push(diff('gradient r', S._gu, W.views.gu));
  bad.push(diff('gradient theta', S._gv, W.views.gv));
  bad.push(diff('gradient sigma', S._gw, W.views.gw));

  S.omegaOf(S._gu, S._gv, S._gw, S._gom); W.omegaOf();
  bad.push(diff('omegaOf', S._gom, W.views.gom));

  S.applyL(S.p, S._div); W.applyL('p', 'div');
  bad.push(diff('applyL', S._div, W.views.div));

  /* The preconditioner: one application of it to the divergence above, in each engine,
     and the factor it applied -- built by this JavaScript solver on first use, and by the
     C++ engine's own JavaScript twin and copied in. Two builds from two cells, so the
     factors agreeing is not the copy agreeing with itself. */
  S.applyPreconditioner(S._div, S._z); W.applyPreconditioner('div', 'cgz');
  bad.push(diff('preconditioner factor', S._pcBand, W.views.pcband));
  bad.push(diff('preconditioner applied', S._z, W.views.cgz));

  S.wFromOmega(); W.wFromOmega();
  bad.push(diff('wFromOmega', S.w, W.views.w));

  ok(bad.filter(Boolean).length === 0, `six operators and the preconditioner, ${contact}`,
     bad.filter(Boolean).join('; '));

  /* The conjugate gradient, which is where a transcription is most likely to
     drift: every iteration a full matvec and a full preconditioner solve, and the
     stopping test a comparison of one of those sums against a tolerance. The
     ITERATION COUNT is asserted as well as the field, because two solvers that agree
     on the answer and disagree on the count took different paths to it.

     The count must also be more than a few, or "the same count" says nothing about
     the recurrence. That guard used to read `> 20`, which was the diagonal
     preconditioner's territory -- it took over a hundred here. The flat-cell
     preconditioner takes this deformed cell to 1e-12 in 8 iterations free and 10
     pinned, measured, which is the point of it; `> 4` still requires the recurrence
     to have run, not one preconditioner application to have been compared. */
  S.divergence(S.u, S.v, S.om, S._div); W.divergence('u', 'v', 'om', 'div');
  S.p.fill(0); W.views.p.fill(0);
  const rJ = S.solveP(S._div, 1e-12, 2000);
  const rW = W.solveP('div', 1e-12, 2000);
  ok(S.cgIters === W.cgIters && S.cgIters > 4,
     `the conjugate gradient takes the same ${S.cgIters} iterations, ${contact}`,
     `${S.cgIters} against ${W.cgIters}`);
  ok(Object.is(rJ, rW), `and reaches the same residual ${rJ.toExponential(6)}`,
     `${rJ} against ${rW}`);
  ok(diff('p', S.p, W.views.p) === null, 'and the same pressure field',
     diff('p', S.p, W.views.p));
  ok(Object.is(S.maxDivergence(), W.maxDivergence()),
     `and the same reported divergence ${S.maxDivergence().toExponential(3)}`,
     `${S.maxDivergence()} against ${W.maxDivergence()}`);
  W.release();
}

/* ---- 5. the viscous operator ----------------------------------------- */
section('5. the viscous operator and the surface traction');
for (const contact of ['free', 'pinned']){
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact };
  const { S, W } = pair(opts, 23, 3, 0.15);
  const bad = [];
  S.axisU(); W.axisU();
  bad.push(diff('axisU', S.u, W.views.u));
  S.refreshSurfaceFluxes(); W.refreshSurfaceFluxes();
  bad.push(diff('surface flux r', S._fsr, W.views.fsr));
  bad.push(diff('surface flux theta', S._fst, W.views.fst));
  bad.push(diff('surface flux z', S._fsz, W.views.fsz));
  S.viscous(S._lu, S._lv, S._lw, S._bcU, S._bcV, S._bcW); W.viscous(1);
  bad.push(diff('grad^2 u', S._lu, W.views.lapU));
  bad.push(diff('grad^2 v', S._lv, W.views.lapV));
  bad.push(diff('grad^2 w', S._lw, W.views.lapW));
  ok(bad.filter(Boolean).length === 0,
     `axisU, the three surface fluxes and the vector Laplacian, ${contact}`,
     bad.filter(Boolean).join('; '));
  W.release();
}

/* ---- 6. advection, curvature and the surface pressure ---------------- */
section('6. advection, curvature and the surface pressure');
for (const contact of ['free', 'pinned']){
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact, accel: 9,
                 omegaD: 2*omegaOf(kOf(3), CELL.h) };
  const { S, W } = pair(opts, 31, 3, 0.15);
  const bad = [];
  S.omegaFromW(); W.omegaFromW();
  S.advectTransport(S._au, S._av, S._aw); W.advectTransport();
  bad.push(diff('transport r', S._au, W.views.advU));
  bad.push(diff('transport theta', S._av, W.views.advV));
  bad.push(diff('transport z', S._aw, W.views.advW));
  S.advectCurvature(S._au, S._av); W.advectCurvature();
  bad.push(diff('basis r', S._au, W.views.advU));
  bad.push(diff('basis theta', S._av, W.views.advV));
  S.curvature(S._kap); W.curvature();
  bad.push(diff('mean curvature', S._kap, W.views.kap));
  ok(Object.is(S.surfaceArea(), W.surfaceArea()),
     `the surface area, ${contact}: ${S.surfaceArea().toExponential(9)}`,
     `${S.surfaceArea()} against ${W.surfaceArea()}`);
  ok(Object.is(S.surfaceExcessArea(), W.surfaceExcessArea()),
     `and the excess over flat: ${S.surfaceExcessArea().toExponential(9)}`,
     `${S.surfaceExcessArea()} against ${W.surfaceExcessArea()}`);
  /* The surface pressure carries the drive's cos, which the module does not
     compute: the loader evaluates it and passes the number in. Agreement here is
     what says that arrangement is faithful. */
  S.surfacePressure(S._ps); W.surfacePressure();
  bad.push(diff('surface pressure', S._ps, W.views.psurf));
  ok(bad.filter(Boolean).length === 0,
     `six arrays and the surface pressure, ${contact}`,
     bad.filter(Boolean).join('; '));
  W.release();
}

/* ---- 7. the step ----------------------------------------------------- */
section('7. the whole step, driven and undriven, both contact lines');
for (const [contact, accel, label] of [['free', 0, 'undriven'],
                                       ['free', 12, 'driven'],
                                       ['pinned', 8, 'driven']]){
  const m = 3, k = kOf(m), omega = omegaOf(k, CELL.h);
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact, accel, omegaD: 2*omega };
  const { S, W } = pair(opts, 41, m, 0.08);
  const dt = S.stableStep();
  const N = 40;
  for (let n = 0; n < N; n++){ S.step(dt); W.step(dt); }
  const names = ['u', 'v', 'w', 'om', 'p', 'eta', 'Ht', ...METRIC];
  const bad = names.map(n => diff(n, S[n], W.views[n])).filter(Boolean);
  ok(bad.length === 0,
     `${N} steps, ${contact} ${label}: all ${names.length} arrays`, bad.join('; '));
  ok(Object.is(S.t, W.t), `and the same clock, ${S.t.toExponential(6)} s`,
     `${S.t} against ${W.t}`);
  ok(Math.abs(S.eta[0]) > 0, 'on a surface that is not flat',
     `eta[0] = ${S.eta[0]}`);
  W.release();
}

/* ---- 8. a full drive period at a renderer-sized grid ----------------- */
section('8. a full drive period at a renderer-sized grid');
{
  const m = 4, k = kOf(m), omega = omegaOf(k, CELL.h);
  const opts = { nr: 12, nth: 24, nz: 8, ...CELL, contact: 'free', accel: 11,
                 omegaD: 2*omega };
  const { S, W } = pair(opts, 53, m, 0.1);
  const dt = S.stableStep();
  const period = 2*Math.PI/opts.omegaD;
  const N = Math.round(period/dt);
  for (let n = 0; n < N; n++){ S.step(dt); W.step(dt); }
  const names = ['u', 'v', 'w', 'om', 'p', 'eta', 'Ht', ...METRIC];
  const bad = names.map(n => diff(n, S[n], W.views[n])).filter(Boolean);
  ok(bad.length === 0,
     `${N} steps, one whole drive period, 12x24x8: all ${names.length} arrays`,
     bad.join('; '));
  const eJ = S.energy();
  W.pullState();
  const eW = W.js.energy();
  ok(Object.is(eJ.total, eW.total) && Object.is(eJ.kinetic, eW.kinetic)
     && Object.is(eJ.capillary, eW.capillary),
     `and the same energy split after pullState: total ${eJ.total.toExponential(9)} J`,
     `${JSON.stringify(eJ)} against ${JSON.stringify(eW)}`);
  ok(Object.is(S.t, W.js.t), 'pullState carries the clock across too',
     `${S.t} against ${W.js.t}`);
  /* The resampler is deliberately NOT ported -- the renderer is JavaScript and it
     runs once per pixel per frame, not once per step -- so after a pull the
     JavaScript surface accessors must read the C++ field. */
  let worst = 0;
  for (let i = 0; i < 40; i++){
    const r = (i + 0.37)/40*opts.R, th = (i*0.613) % (2*Math.PI);
    const a = S.etaAt(r, th), b = W.js.etaAt(r, th);
    if (!Object.is(a, b)) worst = Math.max(worst, Math.abs(a - b));
  }
  ok(worst === 0, 'and etaAt reads the pulled field identically at 40 positions',
     `worst difference ${worst}`);
  W.release();
}

/* ---- 9. the fresh module against the committed one ------------------- */
section('9. the freshly built module against the committed one');
{
  const m = 3, k = kOf(m), omega = omegaOf(k, CELL.h);
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact: 'free', accel: 10,
                 omegaD: 2*omega };
  const run = bytes => {
    const S = fillState(new FaradayCell3D(opts), 67, m, 0.1);
    const W = new FaradayCell3DWasm(opts, bytes ? { bytes } : undefined);
    for (const nm of ['u', 'v', 'w', 'om', 'p', 'eta', 'Ht']) W.views[nm].set(S[nm]);
    W.refreshMetric();
    const dt = S.stableStep();
    for (let n = 0; n < 25; n++) W.step(dt);
    const out = {};
    for (const nm of ['u', 'v', 'w', 'om', 'p', 'eta']) out[nm] = W.views[nm].slice();
    out.t = W.t;
    W.release();
    return out;
  };
  const a = run(null), b = run(freshBytes);
  const bad = ['u', 'v', 'w', 'om', 'p', 'eta'].map(n => diff(n, a[n], b[n]))
    .filter(Boolean);
  ok(bad.length === 0, '25 steps: the two modules produce identical fields',
     bad.join('; '));
  ok(Object.is(a.t, b.t), 'and the same clock');
}

/* ---- 10. the refusals ------------------------------------------------ */
section('10. the refusals');
{
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL, contact: 'free' };
  const W = new FaradayCell3DWasm(opts);

  throws('a second live cell is refused rather than sharing the first\'s arena',
    () => new FaradayCell3DWasm(opts), 'already live');

  throws('a non-positive step is refused', () => W.step(0), 'finite positive');
  throws('a non-finite step is refused', () => W.step(NaN), 'finite positive');

  /* The module clears the gravity after every step, so a step taken through the
     raw export without setting it refuses instead of silently reusing the previous
     instant of the drive. That is the whole reason the cos is passed in rather
     than computed, so the refusal matters more than it looks.
     .
     BOTH HALVES ARE CHECKED, and the second one only because an injection showed
     the first was not enough: this block used to assert the refusal on a cell that
     had never stepped, where the gravity is unset from cell3d_init. Deleting the
     clearing at the end of cell3d_step left the gate entirely GREEN. A successful
     step is taken first here, so what is asserted is that the gravity it consumed
     was given back. */
  W.views.eta.fill(0);
  W.refreshMetric();
  W.inst.exports.cell3d_clearError();
  const codeUnset = W.inst.exports.cell3d_step(1e-6);
  ok(codeUnset === 6, 'a cell that has never stepped has no gravity, code 6',
     `code ${codeUnset}`);
  W.inst.exports.cell3d_clearError();
  W.step(1e-6);
  const codeAfter = W.inst.exports.cell3d_step(1e-6);
  ok(codeAfter === 6,
     'and a step CLEARS it, so the next one cannot reuse that instant of the drive',
     `code ${codeAfter}`);
  ok(/one transcendental/.test(W.errorMessage(6)),
     'the loader\'s message says why the cos is not computed in C++');
  W.inst.exports.cell3d_clearError();

  /* The surface reaching the floor. Both implementations refuse, and the C++ one
     names the same cell: z = sigma*(h + eta) requires a positive depth, and a
     non-positive one means the layer has broken and there is no single-valued
     surface to follow. */
  const S = new FaradayCell3D(opts);
  S.eta[S.ie(3, 5)] = -S.h;
  W.views.eta.set(S.eta);
  let jsMsg = '', waMsg = '';
  try { S.refreshMetric(); } catch (e){ jsMsg = e.message; }
  try { W.refreshMetric(); } catch (e){ waMsg = e.message; }
  ok(/reached the floor/.test(jsMsg) && /reached the floor/.test(waMsg),
     'both refuse when the layer breaks',
     `js "${jsMsg.slice(0, 50)}" wasm "${waMsg.slice(0, 50)}"`);
  const rOf = s => (s.match(/r = ([0-9.e+-]+) m/) || [])[1];
  ok(rOf(jsMsg) !== undefined && rOf(jsMsg) === rOf(waMsg),
     `and name the same radius, ${rOf(jsMsg)} m`,
     `${rOf(jsMsg)} against ${rOf(waMsg)}`);
  W.inst.exports.cell3d_clearError();
  W.release();
}

/* ---- 11. what it bought ---------------------------------------------- */
section('11. what it bought, measured');
{
  const m = 4, k = kOf(m), omega = omegaOf(k, CELL.h);
  for (const [nr, nth, nz, steps] of [[10, 24, 8, 6], [16, 24, 10, 4]]){
    const opts = { nr, nth, nz, ...CELL, contact: 'free', accel: 10,
                   omegaD: 2*omega };
    const { S, W } = pair(opts, 71, m, 0.1);
    const dt = S.stableStep();
    let t0 = process.hrtime.bigint();
    for (let n = 0; n < steps; n++) S.step(dt);
    const js = Number(process.hrtime.bigint() - t0)/1e6/steps;
    W.step(dt);
    t0 = process.hrtime.bigint();
    for (let n = 0; n < steps; n++) W.step(dt);
    const wa = Number(process.hrtime.bigint() - t0)/1e6/steps;
    console.log(`  note  ${nr}x${nth}x${nz}: JavaScript ${js.toFixed(1)} ms a step, `
      + `C++ ${wa.toFixed(1)} ms, ${(js/wa).toFixed(2)}x. Reported and not asserted: `
      + `a timing is a property of the machine, and the numbers above are the claim.`);
    ok(wa > 0 && js > 0, `${nr}x${nth}x${nz}: both engines ran`);
    W.release();
  }
}

} finally { rmSync(work, { recursive: true, force: true }); }

console.log(`\n${pass} checks passed, ${failures.length} failed`);
if (failures.length){ for (const f of failures) console.log('  - ' + f); process.exit(1); }
