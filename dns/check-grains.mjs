import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const G = require('./faraday-grains.js');

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}
function throwsLike(fn, re, name){
  try { fn(); ok(false, name, 'did not throw'); }
  catch (e){ ok(re.test(String(e.message)), name, String(e.message)); }
}

const R = 0.015, nr = 24, nz = 6, h = 12*1e-4;
const rf = Float64Array.from({ length: nr + 1 }, (_, i) => R*i/nr);
const zf = Float64Array.from({ length: nz + 1 }, (_, j) => h*j/nz);
const zc = Float64Array.from({ length: nz }, (_, j) => 0.5*(zf[j] + zf[j + 1]));
const Td = 0.01, omega = 2*Math.PI/Td;

function syntheticMode({ m, multiplier, perDrive = 256, U, V }){
  const frames = [];
  for (let k = 0; k <= perDrive; k++){
    const t = k*Td/perDrive;
    const u = new Float64Array((nr + 1)*nz), v = new Float64Array((nr + 1)*nz);
    for (let i = 0; i <= nr; i++) for (let j = 0; j < nz; j++){
      u[i*nz + j] = U(rf[i], t); v[i*nz + j] = V(rf[i], t);
    }
    frames.push({ t, eta: new Float64Array(nr).fill(1), u, v });
  }
  return { m, mode: { multiplier, frames, grid: { rf, zc, zf } } };
}

const fluid = { rho: 1000, mu: 1e-3 };
const grain = { d: 2e-4, rhoP: 2650 };
const vol = Math.PI*grain.d**3/6, mp = grain.rhoP*vol, mf = fluid.rho*vol;
const O_NEILL_1968 = 1.7009;
const C = O_NEILL_1968*3*Math.PI*fluid.mu*grain.d, M = mp + mf/2;
ok(G.O_NEILL_WALL_SHEAR_FACTOR === O_NEILL_1968, `the wall drag factor is O'Neill's 1.7009 (${G.O_NEILL_WALL_SHEAR_FACTOR})`);

console.log('1. a frictionless grain in a uniform oscillation follows the analytic steady response');
{
  const U0 = 0.02;
  const s = syntheticMode({ m: 1, multiplier: 1, U: (r, t) => U0*Math.cos(omega*t), V: (r, t) => -U0*Math.cos(omega*t) });
  const T = G.floorFieldTable({ mode: s.mode, m: 1, Td, amplitude: 1, zGrain: grain.d/2 });
  const S = G.createGrains({ count: 1, R, ...grain, ...fluid, muS: 0, muK: 0, g: 9.80665, accel: 0, omegaD: omega, seed: 3 });
  S.x[0] = 0; S.y[0] = 0;
  G.advanceGrains(S, T, 40*Td, Td/2048);
  let xmin = Infinity, xmax = -Infinity;
  while (S.t < 42*Td - 1e-12){ G.advanceGrains(S, T, S.t + Td/512, Td/2048); xmin = Math.min(xmin, S.x[0]); xmax = Math.max(xmax, S.x[0]); }
  const num = { re: C, im: 1.5*mf*omega }, den = { re: C, im: M*omega };
  const gain = Math.hypot(num.re, num.im)/Math.hypot(den.re, den.im);
  const expected = gain*U0/omega;
  const measured = (xmax - xmin)/2;
  ok(Math.abs(measured/expected - 1) < 5e-3, `displacement amplitude ${measured.toExponential(5)} m against analytic ${expected.toExponential(5)} m`,
     `ratio ${measured/expected}`);
}

console.log('2. static friction holds a grain exactly until the peak force exceeds mu_s N');
{
  const U0 = 0.02;
  const s = syntheticMode({ m: 1, multiplier: 1, U: (r, t) => U0*Math.cos(omega*t), V: (r, t) => -U0*Math.cos(omega*t) });
  const T = G.floorFieldTable({ mode: s.mode, m: 1, Td, amplitude: 1, zGrain: grain.d/2 });
  const Fmax = U0*Math.hypot(C, 1.5*mf*omega), N = (mp - mf)*9.80665, muStar = Fmax/N;
  for (const [factor, moves] of [[1.01, false], [0.99, true]]){
    const S = G.createGrains({ count: 64, R, ...grain, ...fluid, muS: factor*muStar, muK: 0.8*factor*muStar, g: 9.80665, accel: 0, omegaD: omega, seed: 5 });
    const x0 = Float64Array.from(S.x), y0 = Float64Array.from(S.y);
    G.advanceGrains(S, T, 6*Td, Td/1024);
    let moved = 0;
    for (let n = 0; n < S.count; n++) if (S.x[n] !== x0[n] || S.y[n] !== y0[n]) moved++;
    ok(moves ? moved === S.count : moved === 0,
       `mu_s = ${factor} x threshold: ${moved} of ${S.count} grains moved, ${moves ? 'all' : 'none'} expected`);
  }
}

console.log('3. without flow nothing moves, to the bit');
{
  const s = syntheticMode({ m: 3, multiplier: -1, U: () => 0, V: () => 0 });
  const T = G.floorFieldTable({ mode: s.mode, m: 3, Td, amplitude: 1, zGrain: grain.d/2 });
  const S = G.createGrains({ count: 200, R, ...grain, ...fluid, muS: 0, muK: 0, g: 9.80665, accel: 5, omegaD: omega, seed: 7 });
  const x0 = Float64Array.from(S.x), y0 = Float64Array.from(S.y);
  G.advanceGrains(S, T, 4*Td, Td/512);
  ok(S.x.every((v, n) => v === x0[n]) && S.y.every((v, n) => v === y0[n]), 'positions are unchanged after four drive periods');
}

console.log('4. a subharmonic mode is periodic over two drive periods with the second half reversed');
{
  const s = syntheticMode({ m: 2, multiplier: -1, U: (r, t) => Math.sin(Math.PI*r/R)*Math.cos(omega*t/2), V: (r, t) => 0.5*Math.sin(Math.PI*r/R)*Math.cos(omega*t/2) });
  const T = G.floorFieldTable({ mode: s.mode, m: 2, Td, amplitude: 1, zGrain: grain.d/2 });
  ok(T.period === 2*Td && T.samples === 512, `response period ${T.period} s over ${T.samples} samples`);
  let worst = 0;
  for (let k = 0; k < T.samples/2; k++) for (let i = 0; i < T.faces; i++)
    worst = Math.max(worst, Math.abs(T.U[(k + T.samples/2)*T.faces + i] + T.U[k*T.faces + i]));
  ok(worst < 1e-12, `the second drive period is the first reversed (worst ${worst.toExponential(2)})`);
  let err = 0;
  for (let k = 0; k < T.samples; k++){
    const t = k*T.period/T.samples, exact = -(omega/2)*Math.sin(Math.PI*rf[12]/R)*Math.sin(omega*t/2);
    err = Math.max(err, Math.abs(T.dU[k*T.faces + 12] - exact));
  }
  ok(err < 2e-4*(omega/2), `the tabulated fluid acceleration matches the exact derivative (worst ${err.toExponential(3)})`);
}

console.log('5. grains stay inside the cell when the flow drives them at the wall');
{
  const U0 = 0.5;
  const s = syntheticMode({ m: 0, multiplier: 1, U: (r, t) => U0*(1 + Math.cos(omega*t))*r/R, V: () => 0 });
  const T = G.floorFieldTable({ mode: s.mode, m: 0, Td, amplitude: 1, zGrain: grain.d/2 });
  const S = G.createGrains({ count: 300, R, ...grain, ...fluid, muS: 0.3, muK: 0.2, g: 9.80665, accel: 0, omegaD: omega, seed: 11 });
  G.advanceGrains(S, T, 10*Td, Td/1024);
  let worst = 0, atWall = 0;
  for (let n = 0; n < S.count; n++){ const r = Math.hypot(S.x[n], S.y[n]); worst = Math.max(worst, r); if (r > S.rMax*0.999) atWall++; }
  ok(worst <= S.rMax*(1 + 1e-12), `no grain beyond R - d/2 (largest radius ${worst.toExponential(6)} against ${S.rMax.toExponential(6)})`);
  ok(atWall > S.count/2, `the outward flow carried ${atWall} of ${S.count} grains to the wall`);
}

console.log('6. a uniform oscillation with friction drives no net drift once the start-up has passed');
{
  const U0 = 0.05;
  const s = syntheticMode({ m: 1, multiplier: 1, U: (r, t) => U0*Math.cos(omega*t), V: (r, t) => -U0*Math.cos(omega*t) });
  const T = G.floorFieldTable({ mode: s.mode, m: 1, Td, amplitude: 1, zGrain: grain.d/2 });
  const S = G.createGrains({ count: 50, R: 0.5, ...grain, ...fluid, muS: 0.1, muK: 0.08, g: 9.80665, accel: 0, omegaD: omega, seed: 17 });
  const meanX = () => { let a = 0; const n = 256; for (let k = 0; k < n; k++){ G.advanceGrains(S, T, S.t + Td/n, Td/1024); a += S.x[0]; } return a/n; };
  G.advanceGrains(S, T, 20*Td, Td/1024);
  const a = meanX();
  G.advanceGrains(S, T, S.t + 50*Td, Td/1024);
  const b = meanX();
  const amplitude = U0/omega;
  ok(S.slidSteps > 0 && Math.abs(b - a) < 1e-6*amplitude, `the period-mean position moved ${Math.abs(b - a).toExponential(2)} m over 50 periods, against a ${amplitude.toExponential(2)} m stroke`);
}

console.log('7. refusals');
{
  const base = { count: 1, R, ...grain, ...fluid, muS: 0.5, muK: 0.4, g: 9.80665, accel: 0, omegaD: omega };
  throwsLike(() => G.createGrains({ ...base, accel: 9.80665 }), /leaves the floor/, 'a drive reaching gravity is refused: flight is not modelled');
  throwsLike(() => G.createGrains({ ...base, rhoP: 900 }), /does not rest on the floor/, 'a grain lighter than the liquid is refused');
  throwsLike(() => G.createGrains({ ...base, muK: 0.6 }), /sliding friction above static/, 'mu_k above mu_s is refused');
  throwsLike(() => G.floorFieldTable({ mode: syntheticMode({ m: 1, multiplier: 1, U: () => 0, V: () => 0 }).mode, m: 1, Td, amplitude: 1, zGrain: 1 }),
             /not inside/, 'a grain centre above the layer is refused');
}

console.log('8. the floor-layer streaming profile solves its equation, independently integrated here');
{
  const N = 200000, Z = 40, hz = Z/N;
  const fa = new Float64Array(N + 1), fbr = new Float64Array(N + 1), fbi = new Float64Array(N + 1);
  for (let q = 0; q <= N; q++){
    const z = q*hz, er = Math.exp(-z)*Math.cos(z), ei = -Math.exp(-z)*Math.sin(z);
    const Fr = 1 - er, Fi = -ei, pr = er - ei, pi = er + ei;
    const gr = z - (Fr + Fi)/2, gi = -(Fi - Fr)/2;
    fa[q] = Fr*Fr + Fi*Fi - 1; fbr[q] = -(gr*pr + gi*pi); fbi[q] = -(gi*pr - gr*pi);
  }
  const twice = f => { const d = new Float64Array(N + 1); let tail = 0;
    for (let q = N; q > 0; q--){ tail += 0.5*(f[q] + f[q - 1])*hz; d[q - 1] = -tail; }
    const v = new Float64Array(N + 1); for (let q = 1; q <= N; q++) v[q] = v[q - 1] + 0.5*(d[q] + d[q - 1])*hz; return v; };
  const A = twice(fa), BR = twice(fbr), BI = twice(fbi);
  let worst = 0;
  for (const z of [0.25, 0.8, 1.6, 3, 6, 12]){
    const q = Math.round(z/hz), p = G.streamingProfile(q*hz);
    worst = Math.max(worst, Math.abs(A[q] - p.a), Math.abs(BR[q] - p.bRe), Math.abs(BI[q] - p.bIm));
  }
  ok(worst < 1e-7, `closed-form a(zeta), b(zeta) against the integrated equation: worst ${worst.toExponential(2)}`);
  const e = G.streamingProfile(60);
  ok(Math.abs(e.a + e.bRe + 0.75) < 1e-12, `a standing wave streams -(3/4) U U'/omega at the layer edge (Schlichting): ${(e.a + e.bRe).toFixed(12)}`);
  ok(Math.abs(e.bIm - 0.75) < 1e-12, `a progressive wave streams (3/4) U^2/c at the layer edge (Longuet-Higgins 1953): ${e.bIm.toFixed(12)}`);
}

console.log('9. the polar streaming algebra agrees with a Cartesian evaluation of (U.grad)U* and (div U)U*');
{
  const m = 3, nrf = 4000, Rr = 0.015, om = 2*Math.PI*40;
  const rfx = Float64Array.from({ length: nrf + 1 }, (_, i) => Rr*i/nrf);
  const Af = r => ({ re: Math.sin(5*r/Rr)*r/Rr, im: 0.4*Math.sin(3*r/Rr)*(r/Rr)**2 });
  const Bf = r => ({ re: -0.7*Math.sin(4*r/Rr)*r/Rr, im: 0.2*Math.cos(2*r/Rr)*(r/Rr)**2 });
  const A = { re: Float64Array.from(rfx, r => Af(r).re), im: Float64Array.from(rfx, r => Af(r).im) };
  const B = { re: Float64Array.from(rfx, r => Bf(r).re), im: Float64Array.from(rfx, r => Bf(r).im) };
  const prof = G.streamingProfile(1.3);
  const S = { m, rf: rfx, faces: nrf + 1, ...G.streamingCoefficients({ rf: rfx, m, A, B, omega: om, profile: prof }) };
  const field = (x, y) => { const r = Math.hypot(x, y), th = Math.atan2(y, x), a = Af(r), b = Bf(r), c = Math.cos(m*th), s2 = Math.sin(m*th);
    const ur = { re: a.re*c, im: a.im*c }, ut = { re: b.re*s2, im: b.im*s2 }, ct = Math.cos(th), st = Math.sin(th);
    return { x: { re: ur.re*ct - ut.re*st, im: ur.im*ct - ut.im*st }, y: { re: ur.re*st + ut.re*ct, im: ur.im*st + ut.im*ct } }; };
  let worst = 0, scale = 0;
  for (const [r, th] of [[0.004, 0.3], [0.007, 1.1], [0.011, 2.0], [0.013, 4.4]]){
    const x = r*Math.cos(th), y = r*Math.sin(th), hh = 1e-7;
    const U = field(x, y), Ux = [field(x + hh, y), field(x - hh, y)], Uy = [field(x, y + hh), field(x, y - hh)];
    const d = (p, q, comp) => ({ re: (p[comp].re - q[comp].re)/(2*hh), im: (p[comp].im - q[comp].im)/(2*hh) });
    const dxUx = d(Ux[0], Ux[1], 'x'), dyUx = d(Uy[0], Uy[1], 'x'), dxUy = d(Ux[0], Ux[1], 'y'), dyUy = d(Uy[0], Uy[1], 'y');
    const mulc = (p, q) => ({ re: p.re*q.re + p.im*q.im, im: p.im*q.re - p.re*q.im });
    const Px = { re: mulc(U.x, dxUx).re + mulc(U.y, dyUx).re }, Py = { re: mulc(U.x, dxUy).re + mulc(U.y, dyUy).re };
    const div = { re: dxUx.re + dyUy.re, im: dxUx.im + dyUy.im };
    const Qx = mulc(div, U.x), Qy = mulc(div, U.y);
    const ux = (prof.a*Px.re + prof.bRe*Qx.re - prof.bIm*Qx.im)/om, uy = (prof.a*Py.re + prof.bRe*Qy.re - prof.bIm*Qy.im)/om;
    const v = G.streamingAt(S, r, th), ct = Math.cos(th), st = Math.sin(th);
    const mx = v.ur*ct - v.ut*st, my = v.ur*st + v.ut*ct;
    worst = Math.max(worst, Math.hypot(mx - ux, my - uy)); scale = Math.max(scale, Math.hypot(ux, uy));
  }
  ok(worst < 1e-3*scale, `polar against Cartesian at four points: worst ${worst.toExponential(2)} against a scale of ${scale.toExponential(2)}`);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
