#!/usr/bin/env node
/* Gate for S11, dns/cell3d-pool.js: the three-dimensional step's explicit terms spread over
   worker threads.

   The claim is that the answer does not depend on how many threads form it -- not to a
   tolerance, to the bit -- so this gate compares states with Object.is and never with a
   bound. It checks the rows first, in one thread: every partition of the radial rows forms
   exactly the whole. Then the pool itself, over real worker_threads, against plain step(), on
   one to five workers and in both engines. Then what the pool refuses. Then what it buys,
   which is measured and reported but not asserted, because a timing is a property of the
   machine.

       node dns/check-cell3d-pool.mjs
*/

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const { FaradayCell3D } = require(join(here, 'faraday-cell3d.js'));
const { FaradayCell3DWasm } = require(join(here, 'faraday-cell3d-wasm.js'));
const POOL = require(join(here, 'cell3d-pool.js'));
const { spawnNodePool } = require(join(here, 'cell3d-pool-node.cjs'));
const { Cell3DPool, cell3dPoolView, cell3dBands, CELL3D_POOL_STATE, CELL3D_POOL_OUT } = POOL;

let pass = 0; const failures = [];
function ok(cond, label, detail){
  if (cond){ pass++; console.log('  ok   ' + label); }
  else { failures.push(label + (detail ? '  -- ' + detail : '')); console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
async function rejects(label, fn, fragment){
  let msg = null;
  try { await fn(); } catch (e){ msg = String(e && e.message || e); }
  ok(msg !== null && msg.includes(fragment), label, msg === null ? 'it did not refuse' : msg);
}
const section = t => console.log('\n' + t);

const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837, nu: 1.0035510043427237e-6,
               gamma: 0.07273614042160757, g: 9.80665 };

/* A deformed, moving, driven cell: m = 3 and m = 1 on the surface, every velocity component
   excited, so no row is trivially zero and no band can agree with another by accident. */
function seedState(S){
  const h = S.h;
  for (let i = 0; i < S.nr; i++)
    for (let k = 0; k < S.nth; k++){
      const x = S.rc[i]/S.R, th = (k + 0.5)*S.dth;
      S.eta[S.ie(i, k)] = 0.2*h*x*x*x*(1 - 0.6*x*x)*Math.cos(3*th)
                        + 0.05*h*x*(1 - 0.5*x*x)*Math.sin(th + 0.3);
    }
  let s = 2027;
  const r = () => (s = (s*16807) % 2147483647)/2147483647 - 0.5;
  for (const a of [S.u, S.v, S.w]) for (let c = 0; c < a.length; c++) a[c] = 1e-3*r();
  S.axisU();
  for (let k = 0; k < S.nth; k++) for (let j = 0; j < S.nz; j++) S.u[S.iu(S.nr, k, j)] = 0;
  for (let i = 0; i < S.nr; i++) for (let k = 0; k < S.nth; k++) S.w[S.iw(i, k, 0)] = 0;
  S.refreshMetric();
}
function make(engine, opts){
  if (engine === 'cpp'){
    const W = new FaradayCell3DWasm(opts);
    seedState(W.js); W.pushState(); W.refreshMetric();
    return W;
  }
  const S = new FaradayCell3D(opts);
  seedState(S);
  return S;
}
const done = E => { if (E.release) E.release(); };
/* Compared after the steps: what travels, and everything the step derives from it -- the
   surface, its whole metric, the pressure -- so a difference anywhere is found. */
const ARRAYS = [...new Set([...CELL3D_POOL_STATE, 'p', 'eta',
  'Hdr', 'Hdth', 'Ex', 'Er', 'Eth', 'Edr', 'Edth', 'Exr', 'Ext',
  'Tx', 'Txr', 'Txt', 'Tr', 'Tth', 'Tdr', 'Tdth', '_hcolP'])];
function differing(VA, VB){
  const out = [];
  for (const name of ARRAYS){
    const a = VA.arr(name), b = VB.arr(name);
    let n = 0;
    for (let c = 0; c < a.length; c++) if (!Object.is(a[c], b[c])) n++;
    if (n) out.push(`${name}: ${n} of ${a.length}`);
  }
  return out;
}
function snapshot(V){
  const o = {};
  for (const name of ARRAYS) o[name] = Float64Array.from(V.arr(name));
  return { arr: name => o[name] };
}

/* ---- 1. the rows ---------------------------------------------------------- */
section('1. every partition of the rows forms exactly the whole, in one thread');
for (const engine of ['js', 'cpp'])
  for (const contact of ['free', 'pinned']){
    const opts = { nr: 11, nth: 16, nz: 7, ...CELL, accel: 5, omegaD: 140, contact };
    const E = make(engine, opts), V = cell3dPoolView(E);
    E.stepBegin(4e-5);
    E.stepExplicit(0, opts.nr - 1);
    const whole = CELL3D_POOL_OUT.map(([name]) => Float64Array.from(V.arr(name)));
    const bad = [];
    for (const cuts of [[[0, 10]], [[0, 3], [4, 10]], [[0, 0], [1, 1], [2, 5], [6, 10]],
                        [[0, 5], [6, 6], [7, 10]], [[7, 10], [0, 2], [3, 6]]]){
      for (const [name] of CELL3D_POOL_OUT) V.arr(name).fill(NaN);
      for (const [a0, a1] of cuts) E.stepExplicit(a0, a1);
      CELL3D_POOL_OUT.forEach(([name, fam], j) => {
        const F = V.js.FAM[fam];
        let n = 0;
        for (let a = F.rLo; a <= F.rHi; a++)
          for (let k = 0; k < opts.nth; k++)
            for (let b = F.sLo; b <= F.sHi; b++){
              const c = F.idx(a, k, b);
              if (!Object.is(V.arr(name)[c], whole[j][c])) n++;
            }
        if (n) bad.push(`${JSON.stringify(cuts)} ${name}: ${n}`);
      });
    }
    ok(bad.length === 0,
       `${engine === 'cpp' ? 'C++' : 'JavaScript'}, ${contact}: five partitions of 11 rows, `
       + 'single rows and out of order included, each the whole to the bit in all six arrays',
       bad.join('; '));
    done(E);
  }

/* ---- 2. the pool, over worker threads -------------------------------------- */
section('2. the pool against step(), over real worker threads, both engines');
const STEPS = 6;
for (const engine of ['js', 'cpp'])
  for (const contact of ['free', 'pinned']){
    const opts = { nr: 10, nth: 16, nz: 6, ...CELL, accel: 9, omegaD: 160, contact };
    const A = make(engine, opts), VA = cell3dPoolView(A);
    const dt = VA.js.stableStep();
    for (let n = 0; n < STEPS; n++) A.step(dt);
    const plain = snapshot(VA);
    done(A);
    const rows = [];
    for (const workers of [1, 2, 3, 4, 5]){
      const B = make(engine, opts), VB = cell3dPoolView(B);
      const pool = await spawnNodePool(B, opts, workers, engine);
      try { for (let n = 0; n < STEPS; n++) await pool.step(dt); }
      finally { await pool.close(); }
      rows.push({ workers, bad: differing(plain, VB),
                  bands: [pool.ownBand, ...pool.bands].map(b => `${b[0]}-${b[1]}`).join(' ') });
      done(B);
    }
    console.log(`       ${engine} ${contact}: bands ` + rows.map(r => `[${r.bands}]`).join(', '));
    ok(rows.every(r => r.bad.length === 0),
       `${engine === 'cpp' ? 'C++' : 'JavaScript'}, ${contact}: ${STEPS} driven steps on 1, 2, `
       + `3, 4 and 5 workers leave every one of ${ARRAYS.length} arrays exactly where `
       + `${STEPS} calls of step() leave them`,
       rows.filter(r => r.bad.length).map(r => `${r.workers} workers: ${r.bad.join(', ')}`).join('; '));
  }

/* ---- 3. what it refuses ------------------------------------------------------ */
section('3. what the pool refuses rather than answering');
{
  const opts = { nr: 8, nth: 12, nz: 6, ...CELL };
  const S = make('js', opts);
  /* a worker whose cell is not the owner's: one built for a different radius */
  const w = new Worker(join(here, 'cell3d-pool-node.cjs'),
                       { workerData: { cell3dPool: true, engine: 'js' } });
  const ch = { post: (m, t) => w.postMessage(m, t), onMessage: fn => w.on('message', fn) };
  await rejects('a worker whose cell is a different cell is refused at start, naming the array',
    () => new Cell3DPool(S, [ch]).start({ ...opts, R: opts.R*1.01 }), 'gave a different grid');
  await w.terminate();
  /* more workers than rows */
  await rejects('more bands than radial rows is refused rather than leaving a worker idle',
    async () => cell3dBands(8, 9), 'at most as many workers as rows');
  /* the engines must match */
  await rejects('a JavaScript owner with C++ workers is refused',
    () => spawnNodePool(S, opts, 1, 'cpp'), 'both ends of a pool run one engine');
  /* a worker that fails mid-step: its cell is built, then handed a state of the wrong size */
  const pool = await spawnNodePool(S, opts, 1, 'js');
  await rejects('a worker that cannot form its rows fails the step with its own reason, '
    + 'and the owner does not form them in its place',
    () => pool.request(pool.channels[0], { kind: 'explicit', state: new Float64Array(5),
                                           a0: 4, a1: 7 }),
    'were built for different grids');
  await pool.close();
}

/* ---- 4. what it buys --------------------------------------------------------- */
section('4. what it buys, measured');
{
  const cores = availableParallelism();
  const opts = { nr: 16, nth: 40, nz: 10, ...CELL, accel: 9, omegaD: 160 };
  for (const engine of ['js', 'cpp']){
    const A = make(engine, opts), VA = cell3dPoolView(A);
    const dt = VA.js.stableStep();
    for (let n = 0; n < 4; n++) A.step(dt);
    let t0 = performance.now();
    for (let n = 0; n < 12; n++) A.step(dt);
    const plain = (performance.now() - t0)/12;
    done(A);
    const line = [`plain ${plain.toFixed(1)} ms`];
    for (let workers = 1; workers < cores && workers <= 7; workers++){
      const B = make(engine, opts);
      const pool = await spawnNodePool(B, opts, workers, engine);
      try {
        for (let n = 0; n < 4; n++) await pool.step(dt);
        t0 = performance.now();
        for (let n = 0; n < 12; n++) await pool.step(dt);
      } finally { await pool.close(); }
      const ms = (performance.now() - t0)/12;
      line.push(`${workers + 1} threads ${ms.toFixed(1)} ms (${(plain/ms).toFixed(2)}x)`);
      done(B);
    }
    console.log(`  note  ${engine === 'cpp' ? 'C++' : 'JavaScript'} at 16x40x10 on ${cores} `
      + `cores, a step: ${line.join(', ')}. Reported and not asserted: a timing is a property `
      + 'of the machine.');
  }
}

console.log('\n' + '-'.repeat(66));
if (failures.length){
  console.log(`${pass} passed, ${failures.length} FAILED\n`);
  for (const f of failures) console.log('  FAIL  ' + f);
  process.exit(1);
}
console.log(`${pass} checks passed, 0 failed`);
