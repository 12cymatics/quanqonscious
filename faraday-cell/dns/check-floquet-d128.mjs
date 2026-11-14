import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const DY = require(join(HERE, 'dyadic128.js'));
const FQD = require(join(HERE, 'faraday-floquet-d128.js'));
const E = require(join(HERE, 'faraday-disc-d128.js'));
const F = require(join(HERE, 'faraday-floquet.js'));
const D = require(join(HERE, 'faraday-disc.js'));
const K = require(join(HERE, '..', 'faraday', 'kernel.js'));

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}
const N = DY.toNumberForDisplay;
const toD = x => { if (x === 0) return DY.ZERO; let mm = Math.abs(x), e2 = 0; while (!Number.isInteger(mm)){ mm *= 2; e2++; } return DY.store(BigInt(Math.sign(x)*mm), -e2, 1n); };
const log2abs = x => DY.isZero(x) ? -Infinity : x.e + 128;
const C = FQD.complex;

console.log('1. the Hessenberg eigenvalues are exact to 128 bits where the answer is known exactly');
{
  const roots = [7n, -5n, 3n, 2n, -1n, 9n, -11n, 4n], n = roots.length;
  let coef = [1n];
  for (const r of roots){ const nc = new Array(coef.length + 1).fill(0n); for (let i = 0; i < coef.length; i++){ nc[i] += coef[i]; nc[i + 1] -= r*coef[i]; } coef = nc; }
  const H = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === 0 ? DY.fromBigInt(-coef[j + 1]) : (i === j + 1 ? DY.ONE : DY.ZERO)));
  const eigs = FQD.hessenbergEigs(H, n);
  let worst = DY.ZERO;
  for (const z of eigs){
    let best = null;
    for (const r of roots){ const d = C.cabs(C.csub(z, C.cx(DY.fromBigInt(r)))); if (!best || DY.cmp(d, best) < 0) best = d; }
    if (DY.cmp(best, worst) > 0) worst = best;
  }
  const sorted = eigs.every((z, i) => i === 0 || DY.cmp(C.cnorm2(z), C.cnorm2(eigs[i - 1])) <= 0);
  ok(eigs.length === n && DY.cmp(worst, DY.store(1n, -100, 1n)) <= 0 && sorted,
     `the companion matrix of (x-7)(x+5)(x-3)(x-2)(x+1)(x-9)(x+11)(x-4) gives all eight roots within 2^${log2abs(worst)}, largest modulus first`);
  const three = DY.fromBigInt(3n), four = DY.fromBigInt(4n);
  const B = [[DY.fromBigInt(-2n), DY.fromBigInt(5n), DY.ONE, DY.fromBigInt(2n), DY.fromBigInt(-3n)],
             [DY.ZERO, three, DY.neg(four), DY.fromBigInt(6n), DY.ONE],
             [DY.ZERO, four, three, DY.fromBigInt(-2n), DY.fromBigInt(5n)],
             [DY.ZERO, DY.ZERO, DY.ZERO, DY.fromRational(1n, 2n), DY.fromBigInt(9n)],
             [DY.ZERO, DY.ZERO, DY.ZERO, DY.ZERO, DY.fromBigInt(7n)]];
  const want = [C.cx(three, four), C.cx(three, DY.neg(four)), C.cx(DY.fromBigInt(7n)), C.cx(DY.fromBigInt(-2n)), C.cx(DY.fromRational(1n, 2n))];
  const got = FQD.hessenbergEigs(B, 5);
  let worstB = DY.ZERO;
  for (const w of want){
    let best = null;
    for (const z of got){ const d = C.cabs(C.csub(z, w)); if (!best || DY.cmp(d, best) < 0) best = d; }
    if (DY.cmp(best, worstB) > 0) worstB = best;
  }
  ok(DY.cmp(worstB, DY.store(1n, -100, 1n)) <= 0, `a block with the pair 3 +- 4i beside 7, -2 and 1/2 gives all five within 2^${log2abs(worstB)}`);
  {
    const g = (a, b) => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
    const R = (n, d = 1n) => { if (d < 0n){ n = -n; d = -d; } const k = g(n, d) || 1n; return { n: n/k, d: d/k }; };
    const rof = x => x.e >= 0 ? R(x.m << BigInt(x.e)) : R(x.m, 1n << BigInt(-x.e));
    const ra = (p, q) => R(p.n*q.d + q.n*p.d, p.d*q.d), rs = (p, q) => R(p.n*q.d - q.n*p.d, p.d*q.d), rm = (p, q) => R(p.n*q.n, p.d*q.d);
    const a = DY.neg(DY.ONE), b = DY.ONE, d = DY.neg(DY.fromRational(1n, 3n)), c = DY.mul(DY.mul(a, d), DY.sub(DY.ONE, DY.store(1n, -60, 1n)));
    const e2 = FQD.hessenbergEigs([[a, b], [c, d]], 2);
    const tr = ra(rof(a), rof(d)), det = rs(rm(rof(a), rof(d)), rm(rof(b), rof(c)));
    const steps = e2.map(z => {
      const l = rof(z.re), p = ra(rs(rm(l, l), rm(tr, l)), det), dp = rs(rm(R(2n), l), tr);
      return DY.isZero(z.im) ? (p.n === 0n ? -Infinity : ((p.n < 0n ? -p.n : p.n)*dp.d).toString(2).length - (p.d*(dp.n < 0n ? -dp.n : dp.n)).toString(2).length) : Infinity;
    });
    ok(steps.every(x => x <= -100), `a 2x2 block whose eigenvalues are -4/3 and -2e-19 gives both real and within 2^${Math.max(...steps)} of its exact roots, `
      + `by an exact Newton step on the characteristic polynomial, so the quadratic formula takes the branch that does not cancel`);
  }
  const y = FQD.realEigenvector(H, n, DY.fromBigInt(9n));
  let res = DY.ZERO, nrm = DY.ZERO;
  for (let i = 0; i < n; i++){
    let s = DY.mul(DY.fromBigInt(-9n), y[i]);
    for (let j = 0; j < n; j++) s = DY.add(s, DY.mul(H[i][j], y[j]));
    res = DY.add(res, DY.mul(s, s)); nrm = DY.add(nrm, DY.mul(y[i], y[i]));
  }
  ok(DY.cmp(DY.sqrt(res), DY.store(1n, -100, 1n)) <= 0 && DY.cmp(DY.abs(DY.sub(nrm, DY.ONE)), DY.store(1n, -120, 1n)) <= 0,
     `inverse iteration returns a unit eigenvector for the root 9 with |Hy - 9y| = 2^${log2abs(DY.sqrt(res))}`);
}

const CELL = { R: 12.125e-3, h: 3e-3, rho: 998.2041322005837, nu: 1.0035510043427237e-6, gamma: 0.07273614042160757, g: 9.80665 };
const asStrings = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, String(v)]));
function seeded(m, near, f, contact){
  const k = K.jpZeroNear(m, near)/CELL.R, omegaD = 2*Math.PI*f;
  const sized = D.suggestGrid({ m, k, ...CELL, contact, omegaResponse: omegaD/2, omegaDrive: omegaD, profile: r => K.besselJ(m, k*r) });
  if (sized.unreachable) throw new Error('check-floquet-d128: no grid carries the test mode.');
  const grid = { nr: sized.nr, nz: sized.nz, rStretch: sized.rStretch, zStretch: sized.zStretch };
  const S = new D.FaradayDisc({ m, ...grid, ...CELL, contact });
  const eta0 = Float64Array.from(S.rc, r => K.besselJ(m, k*r));
  return { k, omegaD, grid, eta0 };
}

console.log('2. the float-free Floquet solve gives the float solve\'s multiplier, to the float solve\'s accuracy');
const m = 12, accel = 60;
const sd = seeded(m, 14, 111, 'free');
const fl = F.floquetDisc({ ...sd.grid, ...CELL, m, contact: 'free', eta0: sd.eta0, k: sd.k, omegaD: sd.omegaD, accel, engine: 'wasm', vector: true, samples: 16 });
const dq = FQD.floquetDisc({ m, nr: sd.grid.nr, nz: sd.grid.nz, rStretch: String(sd.grid.rStretch), zStretch: String(sd.grid.zStretch), ...asStrings(CELL),
  accel: String(accel), omegaD: String(sd.omegaD), contact: 'free', eta0: Array.from(sd.eta0, toD), k: toD(sd.k), vector: true, samples: 16, steps: fl.steps });
{
  const rel = Math.abs(N(dq.muMax) - fl.muMax)/fl.muMax;
  ok(dq.krylov === fl.krylov && rel <= 1e-11, `|mu| = ${N(dq.muMax).toPrecision(16)} against the float ${fl.muMax.toPrecision(16)}, ${rel.toExponential(2)} apart, both at Krylov ${dq.krylov}`);
  ok(DY.cmp(dq.orthogonality, DY.store(1n, -110, 1n)) <= 0, `the ${dq.krylov + 1} Arnoldi basis vectors are orthonormal to 2^${log2abs(dq.orthogonality)}`);
  const imRel = DY.div(DY.abs(dq.eigs[0].im), C.cabs(dq.eigs[0]));
  ok(log2abs(imRel) <= -100, `the leading multiplier is real to 2^${log2abs(imRel)}, where the float solve reads an imaginary part of ${Math.abs(fl.eigs[0].im).toExponential(2)}`);
  const gr = Math.abs(N(dq.growth) - fl.growth)/Math.abs(fl.growth);
  ok(gr <= 1e-11, `growth ${N(dq.growth).toPrecision(14)} per second against ${fl.growth.toPrecision(14)}, ${gr.toExponential(2)} apart`);
  const mu = dq.mode.multiplier;
  ok(DY.sign(mu) < 0 && DY.eq(DY.abs(mu), dq.muMax), `the shape belongs to the leading multiplier, real and negative: a subharmonic response (${N(mu).toPrecision(12)})`);
  let dot = 0;
  for (let q = 0; q < dq.stateSize; q++) dot += N(dq.mode.vector[q])*fl.mode.vector[q];
  const s = Math.sign(dot);
  let worst = 0;
  for (let q = 0; q < dq.stateSize; q++) worst = Math.max(worst, Math.abs(N(dq.mode.vector[q]) - s*fl.mode.vector[q]));
  ok(worst <= 1e-8, `the unit eigenvector matches the float one in every component to ${worst.toExponential(2)}`);
  const rr = Math.abs(N(dq.mode.ritzResidual) - fl.mode.ritzResidual)/fl.mode.ritzResidual;
  ok(rr <= 1e-8, `Ritz residual ${N(dq.mode.ritzResidual).toExponential(10)} against the float ${fl.mode.ritzResidual.toExponential(10)}`);
}

console.log('3. the shape is an eigenvector of the period map, checked by applying the map again here');
{
  const eng = E.createDiscD128({ m, nr: sd.grid.nr, nz: sd.grid.nz, rStretch: String(sd.grid.rStretch), zStretch: String(sd.grid.zStretch), ...asStrings(CELL),
    accel: String(accel), omegaD: String(sd.omegaD), contact: 'free', steps: fl.steps, kMax: 2 });
  const x = dq.mode.vector, out = eng.applyPeriodMap(x), lam = dq.mode.multiplier;
  let r2 = DY.ZERO;
  for (let q = 0; q < x.length; q++){ const e = DY.sub(out[q], DY.mul(lam, x[q])); r2 = DY.add(r2, DY.mul(e, e)); }
  const residual = DY.div(DY.sqrt(r2), DY.abs(lam));
  ok(N(residual) < 2e-2, `|P x - mu x| / |mu x| = ${N(residual).toExponential(3)}`);
  ok(DY.eq(residual, dq.mode.ritzResidual), 'and it is, bit for bit, the residual the routine reports');
}

console.log('4. the recorded period starts at the shape and ends at the multiplier times it');
{
  const f = dq.mode.frames, nr = sd.grid.nr, nz = sd.grid.nz;
  ok(f.length === 17 && DY.isZero(f[0].t) && DY.eq(f[16].t, DY.mul(dq.dt, DY.fromBigInt(BigInt(dq.steps)))), `${f.length} frames from t = 0 to exactly the drive period`);
  ok(f.every((x, i) => i === 0 || DY.cmp(x.t, f[i - 1].t) > 0), 'frame times increase');
  const base = 2*(nr - 1)*nz + nr*nz;
  ok(f[0].eta.every((e, i) => DY.eq(e, DY.mul(dq.mode.vector[base + i], dq.mode.href))), 'the first frame is the eigenvector itself, bit for bit');
  let num = DY.ZERO, den = DY.ZERO;
  for (let i = 0; i < nr; i++){
    const want = DY.mul(dq.mode.multiplier, f[0].eta[i]), e = DY.sub(f[16].eta[i], want);
    num = DY.add(num, DY.mul(e, e)); den = DY.add(den, DY.mul(want, want));
  }
  const rel = N(DY.sqrt(DY.div(num, den)));
  ok(rel < 3e-2, `the surface after one period is mu times the start to ${rel.toExponential(3)}`);
}

console.log('5. a leading multiplier that is not real is refused, not given a shape');
{
  const c = seeded(4, 5.3, 70, 'free');
  let msg = '';
  try {
    FQD.floquetDisc({ m: 4, nr: c.grid.nr, nz: c.grid.nz, rStretch: String(c.grid.rStretch), zStretch: String(c.grid.zStretch), ...asStrings(CELL),
      accel: '0', omegaD: String(c.omegaD), contact: 'free', eta0: Array.from(c.eta0, toD), k: toD(c.k), vector: true, samples: 4 });
  } catch (e){ msg = String(e.message); }
  const parts = /complex \(([-0-9.e+]+) ([+-]) ([-0-9.e+]+)i\)/.exec(msg);
  ok(parts && Number(parts[3]) > 0.5, `the undriven 4-fold mode at 70 Hz leads with ${parts ? parts[1] + ' ' + parts[2] + ' ' + parts[3] + 'i' : 'no multiplier'}, and asking for its shape is refused`, msg.slice(0, 200));
}

console.log('6. refusals');
{
  let refused = false;
  try { FQD.floquetDisc({ m, nr: sd.grid.nr, nz: sd.grid.nz, rStretch: '2.2', zStretch: '3', ...asStrings(CELL), accel: '60', omegaD: String(sd.omegaD), contact: 'free', krylovTol: 0.02 }); }
  catch (e){ refused = /float is not accepted/.test(e.message); }
  ok(refused, 'a tolerance passed as a float is refused');
  let gateRefused = false;
  try { FQD.floquetDisc({ m: 12, nr: 12, nz: 6, rStretch: '2.2', zStretch: '2.2', ...asStrings(CELL), accel: '60', omegaD: String(sd.omegaD), contact: 'free',
                          eta0: Array.from({ length: 12 }, (_, i) => toD(K.besselJ(12, sd.k*CELL.R*(i + 0.5)/12))), k: toD(sd.k) }); }
  catch (e){ gateRefused = /cannot carry the mode/.test(e.message); }
  ok(gateRefused, 'a 12x6 grid that cannot carry the 12-fold mode is refused before any step is taken');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
