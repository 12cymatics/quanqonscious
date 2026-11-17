import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const K = require(join(here, '..', 'faraday', 'kernel.js'));
const D = require(join(here, 'faraday-disc.js'));
const F = require(join(here, 'faraday-floquet.js'));

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}

const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837, nu: 1.0035510043427237e-6,
               gamma: 0.07273614042160757, g: 9.80665 };
const m = 12, omegaD = 2*Math.PI*111;
const k = K.jpZeroNear(m, 14)/CELL.R;
const sized = D.suggestGrid({ m, k, ...CELL, contact: 'free', omegaResponse: omegaD/2, omegaDrive: omegaD,
                              profile: r => K.besselJ(m, k*r) });
if (sized.unreachable) throw new Error('check-floquet-mode: no grid carries the test mode.');
const grid = { nr: sized.nr, nz: sized.nz, rStretch: sized.rStretch, zStretch: sized.zStretch };
const seedSolver = new D.FaradayDisc({ m, ...grid, ...CELL, contact: 'free' });
const eta0 = Float64Array.from(seedSolver.rc, r => K.besselJ(m, k*r));
const common = { ...grid, ...CELL, m, contact: 'free', eta0, k, omegaD, engine: 'wasm' };

console.log('1. the growing mode of a strongly driven layer comes back with its shape');
const strong = F.floquetDisc({ ...common, accel: 60, vector: true, samples: 16 });
const mode = strong.mode;
ok(mode && strong.muMax > 1, `the drive is above onset (|mu| = ${strong.muMax.toFixed(6)})`);
ok(Math.abs(Math.abs(mode.multiplier) - strong.muMax) < 1e-12*strong.muMax && mode.multiplier < 0,
   `the shape belongs to the leading multiplier, which is real and negative: a subharmonic response (${mode.multiplier})`);

console.log('2. the shape is an eigenvector of the period map, checked by applying the map again here');
{
  const n = F.discStateSize(grid.nr, grid.nz, m);
  const S = new D.FaradayDisc({ ...grid, ...CELL, m, contact: 'free', accel: 60, omegaD });
  const out = new Float64Array(n);
  F.applyDiscPeriodMap(S, mode.vector, strong.steps, strong.dt, out, mode.href, mode.uref);
  let r2 = 0, x2 = 0;
  for (let q = 0; q < n; q++){ r2 += (out[q] - mode.multiplier*mode.vector[q])**2; x2 += mode.vector[q]**2; }
  const residual = Math.sqrt(r2/x2)/Math.abs(mode.multiplier);
  ok(residual < 2e-2, `|P x - mu x| / |mu x| = ${residual.toExponential(3)}`);
  ok(Math.abs(residual - mode.ritzResidual) < 1e-9, `and it is the residual the routine reports (${mode.ritzResidual.toExponential(3)})`);
}

console.log('3. the recorded period starts at the shape and ends at the multiplier times it');
{
  const f = mode.frames;
  ok(f.length === 17 && f[0].t === 0 && Math.abs(f[16].t - strong.Td) < 1e-15*strong.Td + strong.dt*0.5,
     `${f.length} frames from t = 0 to the drive period`);
  ok(f.every((x, i) => i === 0 || x.t > f[i - 1].t), 'frame times increase');
  const base = 2*(grid.nr - 1)*grid.nz + grid.nr*grid.nz;
  let e0 = 0;
  for (let i = 0; i < grid.nr; i++) e0 = Math.max(e0, Math.abs(f[0].eta[i] - mode.vector[base + i]*mode.href));
  ok(e0 === 0, 'the first frame is the eigenvector itself');
  let num = 0, den = 0;
  for (let i = 0; i < grid.nr; i++){ num += (f[16].eta[i] - mode.multiplier*f[0].eta[i])**2; den += (mode.multiplier*f[0].eta[i])**2; }
  ok(Math.sqrt(num/den) < 3e-2, `the surface after one period is mu times the start to ${Math.sqrt(num/den).toExponential(3)}`);
}

console.log('4. a leading multiplier that is not real is refused, not given a shape');
{
  const R = 0.015, h = 0.001, w = { rho: 1000, nu: 1e-6, gamma: 0.0728, g: 9.80665 };
  const mm = 7, kk = K.jZeroNear(mm, 18.29)/R, od = 2*Math.PI*111;
  const gz = D.suggestGrid({ m: mm, k: kk, R, h, ...w, contact: 'pinned', omegaResponse: od/2, omegaDrive: od,
                            profile: r => K.besselJ(mm, kk*r) });
  const S = new D.FaradayDisc({ m: mm, nr: gz.nr, nz: gz.nz, rStretch: gz.rStretch, zStretch: gz.zStretch, R, h, ...w, contact: 'pinned' });
  const seed = Float64Array.from(S.rc, r => K.besselJ(mm, kk*r));
  const opts = { nr: gz.nr, nz: gz.nz, rStretch: gz.rStretch, zStretch: gz.zStretch, R, h, ...w, m: mm,
                 contact: 'pinned', eta0: seed, k: kk, omegaD: od, accel: 20, engine: 'wasm' };
  const plain = F.floquetDisc(opts);
  ok(Math.abs(plain.eigs[0].im) > 1e-3, `the 7-fold mode of a 1 mm layer at 111 Hz and 20 m/s^2 leads with ${plain.eigs[0].re.toFixed(4)}${plain.eigs[0].im >= 0 ? '+' : ''}${plain.eigs[0].im.toFixed(4)}i`);
  let msg = '';
  try { F.floquetDisc({ ...opts, vector: true, samples: 4 }); }
  catch (e){ msg = String(e.message); }
  ok(/is complex/.test(msg), 'and asking for its shape is refused: it is not locked to the drive', msg.slice(0, 160));
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
