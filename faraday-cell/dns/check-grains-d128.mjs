import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const DY = require(join(HERE, 'dyadic128.js'));
const GD = require(join(HERE, 'faraday-grains-d128.js'));
const FG = require(join(HERE, 'faraday-grains.js'));
const SCAN = require(join(HERE, 'wasm-float-scan.js'));

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}
const N = DY.toNumberForDisplay;
const toD = x => { if (x === 0) return DY.ZERO; let mm = Math.abs(x), e2 = 0; while (!Number.isInteger(mm)){ mm *= 2; e2++; } return DY.store(BigInt(Math.sign(x)*mm), -e2, 1n); };

console.log('1. the committed module is the committed source, and holds no floating point');
const work = mkdtempSync(join(tmpdir(), 'grains-d128-'));
const freshPath = join(work, 'faraday_grains_d128.wasm');
execFileSync('sh', [join(HERE, 'build-wasm-grains-d128.sh'), freshPath], { encoding: 'utf8' });
const fresh = new Uint8Array(readFileSync(freshPath)), committed = new Uint8Array(readFileSync(join(HERE, 'faraday_grains_d128.wasm')));
rmSync(work, { recursive: true, force: true });
for (const [label, bytes] of [['committed', committed], ['fresh', fresh]]){
  const s = SCAN.scanWasm(bytes);
  ok(WebAssembly.Module.imports(new WebAssembly.Module(bytes)).length === 0 && s.instructions > 5000 && s.floatInstructions === 0 && s.floatTypes === 0,
     `the ${label} module imports nothing and has no f32 or f64 instruction, local, global or signature in ${s.instructions} instructions`);
}
ok(GD.O_NEILL_WALL_SHEAR_FACTOR === '1.7009', `the wall drag factor is O'Neill's 1.7009, held as a decimal string (${GD.O_NEILL_WALL_SHEAR_FACTOR})`);

const Rf = 0.015, nr = 24, nz = 6, hf = 12e-4, TdF = 0.01, omegaF = 2*Math.PI/TdF;
const rfF = Array.from({ length: nr + 1 }, (_, i) => Rf*i/nr), zfF = Array.from({ length: nz + 1 }, (_, j) => hf*j/nz);
const zcF = zfF.slice(0, nz).map((z, j) => 0.5*(z + zfF[j + 1]));
const R = DY.fromDecimal('0.015'), Td = DY.fromDecimal('0.01'), twoPi = DY.scale2(DY.pi(), 1), omega = DY.div(twoPi, Td);
const rf = Array.from({ length: nr + 1 }, (_, i) => DY.mulRat(R, BigInt(i), BigInt(nr)));
const zf = Array.from({ length: nz + 1 }, (_, j) => DY.mulRat(DY.fromDecimal('12e-4'), BigInt(j), BigInt(nz)));
const zc = zf.slice(0, nz).map((z, j) => DY.scale2(DY.add(z, zf[j + 1]), -1));
function syntheticMode({ m, multiplier, perDrive = 256, U, V }){
  const frames = [];
  for (let k = 0; k <= perDrive; k++){
    const t = DY.mulRat(Td, BigInt(k), BigInt(perDrive)), u = [], v = [];
    for (let i = 0; i <= nr; i++){ const a = U(rf[i], t), b = V(rf[i], t); for (let j = 0; j < nz; j++){ u.push(a); v.push(b); } }
    frames.push({ step: k, t, eta: new Array(nr).fill(DY.ONE), u, v });
  }
  return { mode: { multiplier: DY.fromBigInt(BigInt(multiplier)), frames, grid: { rf, zc, zf } }, steps: perDrive, m };
}
const fluid = { rho: '1000', mu: '1e-3' }, grain = { d: '2e-4', rhoP: '2650' };
const table = (s, extra = {}) => GD.floorFieldTable({ mode: s.mode, m: s.m, Td, steps: s.steps, amplitude: '1', zGrain: '1e-4', ...extra });
const grainsOf = (T, o) => GD.createGrains({ R: '0.015', ...grain, ...fluid, g: '9.80665', omegaD: omega, table: T, ...o });

console.log('2. a frictionless grain in a uniform oscillation follows the analytic steady response');
{
  const U0 = DY.fromDecimal('0.02');
  const s = syntheticMode({ m: 1, multiplier: 1, U: (r, t) => DY.mul(U0, DY.cos(DY.mul(omega, t))), V: (r, t) => DY.neg(DY.mul(U0, DY.cos(DY.mul(omega, t)))) });
  const T = table(s);
  const G = grainsOf(T, { count: 1, muS: '0', muK: '0', accel: '0', stepsPerPeriod: 2048, positions: { x: [DY.ZERO], y: [DY.ZERO] } });
  G.advance(40*2048);
  let xmin = Infinity, xmax = -Infinity;
  for (let k = 0; k < 2*512; k++){ G.advance(4); const x = N(G.positions().x[0]); xmin = Math.min(xmin, x); xmax = Math.max(xmax, x); }
  const p = GD.grainParameters({ R: '0.015', ...grain, ...fluid, muS: '0', muK: '0', g: '9.80665', accel: '0', omegaD: omega });
  const Cn = N(p.C), Mn = N(p.M), mfn = N(p.mf);
  const gain = Math.hypot(Cn, 1.5*mfn*omegaF)/Math.hypot(Cn, Mn*omegaF), expected = gain*0.02/omegaF, measured = (xmax - xmin)/2;
  ok(Math.abs(measured/expected - 1) < 5e-3, `displacement amplitude ${measured.toExponential(5)} m against analytic ${expected.toExponential(5)} m`);
}

console.log('3. static friction holds a grain exactly until the peak force exceeds mu_s N');
{
  const U0 = DY.fromDecimal('0.02');
  const s = syntheticMode({ m: 1, multiplier: 1, U: (r, t) => DY.mul(U0, DY.cos(DY.mul(omega, t))), V: (r, t) => DY.neg(DY.mul(U0, DY.cos(DY.mul(omega, t)))) });
  const T = table(s);
  const p = GD.grainParameters({ R: '0.015', ...grain, ...fluid, muS: '0', muK: '0', g: '9.80665', accel: '0', omegaD: omega });
  const Fmax = DY.mul(U0, DY.sqrt(DY.add(DY.mul(p.C, p.C), DY.mul(DY.mul(p.mfAdded, omega), DY.mul(p.mfAdded, omega)))));
  const muStar = DY.div(Fmax, DY.mul(DY.sub(p.mp, p.mf), DY.fromDecimal('9.80665')));
  for (const [factor, moves] of [['1.01', false], ['0.99', true]]){
    const muS = DY.mul(DY.fromDecimal(factor), muStar), muK = DY.mul(DY.fromDecimal('0.8'), muS);
    const G = grainsOf(T, { count: 64, muS, muK, accel: '0', stepsPerPeriod: 1024, seed: 5 });
    const before = G.positions();
    G.advance(6*1024);
    const after = G.positions();
    let moved = 0;
    for (let n = 0; n < 64; n++) if (!DY.eq(before.x[n], after.x[n]) || !DY.eq(before.y[n], after.y[n])) moved++;
    ok(moves ? moved === 64 : moved === 0, `mu_s = ${factor} x threshold: ${moved} of 64 grains moved, ${moves ? 'all' : 'none'} expected`);
  }
}

console.log('4. without flow nothing moves, to the bit');
{
  const s = syntheticMode({ m: 3, multiplier: -1, U: () => DY.ZERO, V: () => DY.ZERO });
  const T = table(s);
  const G = grainsOf(T, { count: 200, muS: '0', muK: '0', accel: '5', stepsPerPeriod: 512, seed: 7 });
  const before = G.positions();
  G.advance(8*512);
  const after = G.positions();
  ok(before.x.every((x, n) => DY.eq(x, after.x[n])) && before.y.every((y, n) => DY.eq(y, after.y[n])), 'positions are unchanged after four drive periods');
}

console.log('5. a subharmonic table is periodic over two drive periods with the second half exactly reversed');
{
  const piD = DY.pi();
  const shape = r => DY.sin(DY.div(DY.mul(piD, r), R));
  const s = syntheticMode({ m: 2, multiplier: -1, U: (r, t) => DY.mul(shape(r), DY.cos(DY.scale2(DY.mul(omega, t), -1))),
                           V: (r, t) => DY.mul(DY.scale2(shape(r), -1), DY.cos(DY.scale2(DY.mul(omega, t), -1))) });
  const T = table(s);
  ok(DY.eq(T.period, DY.scale2(Td, 1)) && T.samples === 512, `response period ${N(T.period)} s over ${T.samples} samples`);
  let reversed = true;
  for (let k = 0; k < T.samples/2; k++) for (let i = 0; i < T.faces; i++) if (!DY.eq(T.U[k + T.samples/2][i], DY.neg(T.U[k][i]))) reversed = false;
  ok(reversed, 'the second drive period is the first negated, bit for bit');
  let err = 0;
  for (let k = 0; k < T.samples; k++){
    const t = k*N(T.period)/T.samples, exact = -(omegaF/2)*Math.sin(Math.PI*rfF[12]/Rf)*Math.sin(omegaF*t/2);
    err = Math.max(err, Math.abs(N(T.dU[k][12]) - exact));
  }
  ok(err < 2e-4*(omegaF/2), `the tabulated fluid acceleration matches the exact derivative (worst ${err.toExponential(3)})`);
}

console.log('6. grains stay inside the cell when the flow drives them at the wall');
{
  const U0 = DY.fromDecimal('0.5');
  const s = syntheticMode({ m: 0, multiplier: 1, U: (r, t) => DY.div(DY.mul(DY.mul(U0, DY.add(DY.ONE, DY.cos(DY.mul(omega, t)))), r), R), V: () => DY.ZERO });
  const T = table(s);
  const G = grainsOf(T, { count: 300, muS: '0.3', muK: '0.2', accel: '0', stepsPerPeriod: 1024, seed: 11 });
  G.advance(10*1024);
  const pos = G.positions(), rMax = G.params.rMax, r2 = DY.mul(rMax, rMax);
  let outside = 0, atWall = 0;
  const near = DY.mul(DY.mul(rMax, rMax), DY.fromDecimal('0.998'));
  for (let n = 0; n < 300; n++){
    const q = DY.add(DY.mul(pos.x[n], pos.x[n]), DY.mul(pos.y[n], pos.y[n]));
    if (DY.cmp(q, DY.mul(r2, DY.add(DY.ONE, DY.store(1n, -100, 1n)))) > 0) outside++;
    if (DY.cmp(q, near) > 0) atWall++;
  }
  ok(outside === 0, 'no grain beyond R - d/2, to 2^-100');
  ok(atWall > 150, `the outward flow carried ${atWall} of 300 grains to the wall`);
}

console.log('7. the same grains in the same flow and streaming, against the float integrator');
{
  const faces = 9, samples = 8, period = 1/40, m = 3, Rg = 0.012;
  const rfx = Array.from({ length: faces }, (_, i) => Rg*i/(faces - 1));
  const Ufl = [], Vfl = [];
  for (let k = 0; k < samples; k++){ const ph = 2*Math.PI*k/samples; Ufl.push(rfx.map(r => 0.004*Math.sin(Math.PI*r/Rg)*Math.cos(ph))); Vfl.push(rfx.map(r => 0.003*Math.sin(2*Math.PI*r/Rg)*Math.cos(ph))); }
  const dts = period/samples, dUfl = [], dVfl = [];
  for (let k = 0; k < samples; k++){ const kp = (k + 1) % samples, km = (k - 1 + samples) % samples; dUfl.push(rfx.map((_, i) => (Ufl[kp][i] - Ufl[km][i])/(2*dts))); dVfl.push(rfx.map((_, i) => (Vfl[kp][i] - Vfl[km][i])/(2*dts))); }
  const stream = { PrCC: rfx.map(r => 2e-5*Math.sin(Math.PI*r/Rg)), QrCC: rfx.map(r => 1e-5*r/Rg), PrSS: rfx.map(r => -1.5e-5*Math.sin(Math.PI*r/Rg)), PtSC: rfx.map(r => 1.2e-5*r/Rg), QtSC: rfx.map(() => 3e-6) };
  const Tf = { m, rf: Float64Array.from(rfx), R: Rg, faces, samples, period, U: Float64Array.from(Ufl.flat()), V: Float64Array.from(Vfl.flat()), dU: Float64Array.from(dUfl.flat()), dV: Float64Array.from(dVfl.flat()),
               streaming: { m, faces, rf: Float64Array.from(rfx), ...Object.fromEntries(Object.entries(stream).map(([k, v]) => [k, Float64Array.from(v)])) } };
  const Tq = { m, rf: rfx.map(toD), R: toD(Rg), faces, samples, period: toD(period), multiplier: DY.ONE, U: Ufl.map(r => r.map(toD)), V: Vfl.map(r => r.map(toD)), dU: dUfl.map(r => r.map(toD)), dV: dVfl.map(r => r.map(toD)) };
  const Sq = { m, faces, rf: Tq.rf, ...Object.fromEntries(Object.entries(stream).map(([k, v]) => [k, v.map(toD)])) };
  const gF = { R: Rg, d: 2e-4, rhoP: 1050, rho: 998, mu: 1e-3, muS: 0.3, muK: 0.2, g: 9.80665, accel: 5, omegaD: 2*Math.PI*80 };
  const GF = FG.createGrains({ ...gF, count: 200, seed: 7 });
  const GQ = GD.createGrains({ ...Object.fromEntries(Object.entries(gF).map(([k, v]) => [k, String(v)])), count: 200, table: Tq, streaming: Sq, stepsPerPeriod: 128,
                               positions: { x: Array.from(GF.x, toD), y: Array.from(GF.y, toD) } });
  const x0 = Array.from(GF.x), y0 = Array.from(GF.y);
  GQ.advance(128*20);
  FG.advanceGrains(GF, Tf, 20*period, period/128);
  const pos = GQ.positions();
  let worst = 0, travel = 0;
  for (let n = 0; n < 200; n++){
    worst = Math.max(worst, Math.abs(N(pos.x[n]) - GF.x[n]), Math.abs(N(pos.y[n]) - GF.y[n]));
    travel = Math.max(travel, Math.hypot(GF.x[n] - x0[n], GF.y[n] - y0[n]));
  }
  ok(travel > 1e-6 && worst <= 1e-9*travel, `200 grains over 20 periods with streaming: positions within ${worst.toExponential(2)} m of the float integrator's, against travel up to ${travel.toExponential(2)} m`);
  console.log(`       sliding steps ${GQ.slidSteps()} here, ${GF.slidSteps} in the float integrator: a stick decision differs only where the force sits on mu_s N within float rounding`);
}

console.log('8. the streaming profile reaches its classical limits and agrees with the float profile');
{
  const far = GD.streamingProfile(DY.fromBigInt(200n));
  const q = DY.store(1n, -2, 1n);
  ok(DY.eq(far.a, DY.neg(q)) || DY.cmp(DY.abs(DY.add(far.a, q)), DY.store(1n, -120, 1n)) <= 0, `a -> -1/4 far from the floor (${N(far.a)})`);
  ok(DY.cmp(DY.abs(DY.add(far.bRe, DY.fromRational(1n, 2n))), DY.store(1n, -120, 1n)) <= 0 && DY.cmp(DY.abs(DY.sub(far.bIm, DY.fromRational(3n, 4n))), DY.store(1n, -120, 1n)) <= 0,
     `b -> -1/2 + 3i/4 far from the floor (Schlichting's -3/4 and Longuet-Higgins' +3/4)`);
  let worst = 0;
  for (const z of ['0.1', '0.5', '1.121', '2', '4']){
    const a = GD.streamingProfile(DY.fromDecimal(z)), b = FG.streamingProfile(Number(z));
    worst = Math.max(worst, Math.abs(N(a.a) - b.a), Math.abs(N(a.bRe) - b.bRe), Math.abs(N(a.bIm) - b.bIm));
  }
  ok(worst < 1e-15, `at five depths the 128-bit profile agrees with the float one to ${worst.toExponential(2)}`);
}

console.log('9. the drift field is read back at its own probes, and respects the mode\'s rotation and mirror');
{
  const piD = DY.pi();
  const shape = r => DY.sin(DY.div(DY.mul(piD, r), R));
  const s = syntheticMode({ m: 2, multiplier: -1, U: (r, t) => DY.mul(DY.mul(DY.fromDecimal('0.03'), shape(r)), DY.cos(DY.scale2(DY.mul(omega, t), -1))),
                           V: (r, t) => DY.mul(DY.mul(DY.fromDecimal('0.02'), shape(r)), DY.cos(DY.scale2(DY.mul(omega, t), -1))) });
  const T = table(s);
  const gp = { R: '0.015', d: '2e-4', rhoP: '1050', rho: '1000', mu: '1e-3', muS: '0.3', muK: '0.2', g: '9.80665', accel: '4', omegaD: omega };
  const Dm = GD.driftMap({ ...gp, table: T, nr: 10, nth: 6, settle: 4, measure: 4, stepsPerPeriod: 128 });
  ok(Dm.M === 4 && Dm.nth === 6, `a subharmonic 2-fold mode gives a drift field of period pi/2 in angle (M = ${Dm.M})`);
  const pts = [], expect = [];
  for (const [i, j] of [[3, 2], [5, 3], [8, 1], [9, 4]]){
    const n = i*Dm.nth + j, r = DY.mulRat(Dm.rMax, BigInt(i), BigInt(Dm.nr - 1)), th = DY.mulRat(piD, BigInt(j), BigInt((Dm.nth - 1)*Dm.M));
    for (const [rot, mir] of [[0, 1], [1, 1], [3, 1], [0, -1], [2, -1]]){
      const ang = DY.add(DY.mul(th, DY.fromBigInt(BigInt(mir))), DY.mulRat(DY.scale2(piD, 1), BigInt(rot), BigInt(Dm.M)));
      const ca = DY.cos(ang), sa = DY.sin(ang);
      pts.push({ x: DY.mul(r, ca), y: DY.mul(r, sa) });
      const ur = Dm.UR[n], ut = DY.mul(DY.fromBigInt(BigInt(mir)), Dm.UT[n]);
      expect.push({ vx: DY.sub(DY.mul(ur, ca), DY.mul(ut, sa)), vy: DY.add(DY.mul(ur, sa), DY.mul(ut, ca)) });
    }
  }
  const sand = GD.createSand({ drift: Dm, count: pts.length, positions: { x: pts.map(p => p.x), y: pts.map(p => p.y) } });
  let worst = 0, scale = 0;
  for (let n = 0; n < Dm.UR.length; n++) scale = Math.max(scale, Math.hypot(N(Dm.UR[n]), N(Dm.UT[n])));
  for (let k = 0; k < pts.length; k++){ const v = sand.driftAt(k); worst = Math.max(worst, Math.abs(N(v.vx) - N(expect[k].vx)), Math.abs(N(v.vy) - N(expect[k].vy))); }
  ok(scale > 0 && worst <= 1e-12*scale, `at four probes, each turned by multiples of 2 pi/M and mirrored, the evaluated drift is the probe's to ${(worst/scale).toExponential(2)} of the largest drift`);
}

console.log('10. packing leaves at most one grain per pixel inside the disc and loses none');
{
  const zeroDrift = { m: 2, M: 4, nr: 3, nth: 4, rMax: DY.fromDecimal('0.0149'), dr: DY.fromDecimal('0.00745'),
                      UR: new Array(12).fill(DY.ZERO), G: new Array(12).fill(DY.ZERO), C: [DY.ONE, DY.fromRational(1n, 2n), DY.fromRational(-1n, 2n), DY.neg(DY.ONE)] };
  const count = 1200, xs = [], ys = [];
  for (let n = 0; n < count; n++){ xs.push(DY.mulRat(DY.fromDecimal('0.004'), BigInt(n % 7), 7n)); ys.push(DY.mulRat(DY.fromDecimal('0.004'), BigInt(n % 5), 5n)); }
  const sand = GD.createSand({ drift: zeroDrift, count, positions: { x: xs, y: ys } });
  sand.raster({ n: 121, center: 60, radius: 57, cellRadius: '0.015', seed: 9 });
  const before = Array.from(sand.bins());
  let passes = 0, moved;
  do { moved = sand.pack(count); passes++; } while (moved > 0 && passes < 50);
  const bins = Array.from(sand.bins());
  const total = bins.reduce((a, b) => a + b, 0), maxBin = Math.max(...bins);
  let outsideCover = 0;
  for (let y = 0; y < 121; y++) for (let x = 0; x < 121; x++){
    const dx = 2*(x - 60) + 1, dy = 2*(y - 60) + 1;
    if (bins[y*121 + x] && dx*dx + dy*dy > (2*57 + 1)**2) outsideCover++;
  }
  ok(Math.max(...before) > 30 && maxBin === 1 && total === count && outsideCover === 0,
     `${count} grains piled ${Math.max(...before)} deep spread in ${passes} passes to one per pixel, all ${total} of them inside the disc`);
  const rim = GD.createSand({ drift: zeroDrift, count: 400, positions: { x: new Array(400).fill(DY.fromDecimal('0.0146')), y: new Array(400).fill(DY.ZERO) } });
  rim.raster({ n: 121, center: 60, radius: 57, cellRadius: '0.015', seed: 4 });
  rim.pack(400);
  const rb = Array.from(rim.bins());
  let outside = 0;
  for (let y = 0; y < 121; y++) for (let x = 0; x < 121; x++){
    const dx = 2*(x - 60) + 1, dy = 2*(y - 60) + 1;
    if (rb[y*121 + x] && dx*dx + dy*dy > (2*57 + 1)**2) outside += rb[y*121 + x];
  }
  ok(Math.max(...rb) === 1 && rb.reduce((a, b) => a + b, 0) === 400 && outside === 0,
     'a pile of 400 grains against the rim spreads to one per pixel without a grain leaving the disc');
}

console.log('11. refusals');
{
  const refuses = (fn, re) => { try { fn(); return false; } catch (e){ return re.test(e.message); } };
  const s = syntheticMode({ m: 1, multiplier: 1, U: () => DY.ONE, V: () => DY.ZERO });
  const T = table(s);
  ok(refuses(() => grainsOf(T, { count: 1, muS: 0.3, muK: '0.2', accel: '0' }), /float is not accepted/), 'a friction coefficient passed as a float is refused');
  ok(refuses(() => grainsOf(T, { count: 1, muS: '0.3', muK: '0.2', accel: '9.80665' }), /leaves the floor/), 'a drive reaching gravity is refused: flight is not modelled');
  ok(refuses(() => grainsOf(T, { count: 1, muS: '0.2', muK: '0.3', accel: '0' }), /muK exceeds muS/), 'sliding friction above static friction is refused');
  ok(refuses(() => GD.createGrains({ R: '0.015', d: '2e-4', rhoP: '900', ...fluid, g: '9.80665', omegaD: omega, table: T, count: 1, muS: '0.3', muK: '0.2', accel: '0' }), /not denser/), 'a grain lighter than the liquid is refused');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
