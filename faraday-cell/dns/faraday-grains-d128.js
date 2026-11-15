'use strict';

const DG = (function(){
  if (typeof require === 'function') return require('./dyadic128.js');
  if (typeof globalThis !== 'undefined' && globalThis.DYADIC128) return globalThis.DYADIC128;
  throw new Error('faraday-grains-d128: dns/dyadic128.js is not loaded.');
})();

const GRAIN_EXPORTS = ['setupGrains', 'prepareGrains', 'advanceGrains', 'ptrScal', 'ptrRf', 'ptrUt', 'ptrVt', 'ptrDUt', 'ptrDVt', 'ptrNormal',
  'ptrStreamR', 'ptrStreamS', 'ptrStreamT', 'ptrX', 'ptrY', 'ptrVX', 'ptrVY', 'ptrStuck', 'getSlidSteps', 'getLastError', 'memory',
  'setupSand', 'prepareSand', 'advectSand', 'driftAt', 'ptrSandX', 'ptrSandY', 'ptrDriftUR', 'ptrDriftG', 'ptrDriftC', 'ptrDriftDt', 'ptrDriftOut',
  'setupRaster', 'binSand', 'packSand', 'ptrOcc'];

const O_NEILL_WALL_SHEAR_FACTOR = '1.7009';

function grainBytes(explicit){
  if (explicit) return explicit;
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_GRAINS_D128_WASM_BASE64){
    const b64 = globalThis.FARADAY_GRAINS_D128_WASM_BASE64;
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  if (typeof require === 'function'){
    const { readFileSync } = require('fs');
    const { join } = require('path');
    return new Uint8Array(readFileSync(join(__dirname, 'faraday_grains_d128.wasm')));
  }
  throw new Error('faraday-grains-d128: no module bytes.');
}

function val(v, name){
  if (typeof v === 'string') return DG.fromDecimal(v);
  if (typeof v === 'bigint') return DG.fromBigInt(v);
  if (v && typeof v.m === 'bigint' && Number.isSafeInteger(v.e)) return v;
  throw new TypeError(`${name}: a decimal string, a BigInt or a 128-bit value is required; a float is not accepted.`);
}
function pos(v, name){ const x = val(v, name); if (DG.sign(x) <= 0) throw new RangeError(`${name}: a positive value is required.`); return x; }
function nonneg(v, name){ const x = val(v, name); if (DG.sign(x) < 0) throw new RangeError(`${name}: a non-negative value is required.`); return x; }
const maxD = arr => arr.reduce((a, b) => DG.cmp(a, b) >= 0 ? a : b);

function growthFree(mode, steps){
  const mag = DG.abs(mode.multiplier), lnMag = DG.ln(mag);
  return frame => DG.exp(DG.neg(DG.mulRat(lnMag, BigInt(frame.step), BigInt(steps))));
}

function etaPeakOf(mode, damp){
  const perDrive = mode.frames.length - 1;
  let peak = DG.ZERO;
  for (let q = 0; q < perDrive; q++){
    const d = damp(mode.frames[q]);
    for (const e of mode.frames[q].eta){ const a = DG.mul(DG.abs(e), d); if (DG.cmp(a, peak) > 0) peak = a; }
  }
  if (DG.isZero(peak)) throw new RangeError('the mode has no surface displacement.');
  return peak;
}

function floorFieldTable(o){
  const { mode, steps } = o;
  if (!mode || !mode.frames || !mode.grid) throw new TypeError('floorFieldTable: a float-free Floquet mode with its frames is required.');
  const amplitude = pos(o.amplitude, 'amplitude'), zGrain = pos(o.zGrain, 'zGrain'), Td = pos(o.Td, 'Td');
  const { rf, zc, zf } = mode.grid, nz = zc.length, nr = rf.length - 1, faces = nr + 1;
  const frames = mode.frames, perDrive = frames.length - 1;
  const sign = DG.sign(mode.multiplier);
  if (DG.cmp(zGrain, zf[nz]) >= 0) throw new RangeError('floorFieldTable: the grain centre is not inside the layer.');
  let j1 = 0;
  while (j1 < nz && DG.cmp(zc[j1], zGrain) < 0) j1++;
  if (j1 >= nz) throw new RangeError('floorFieldTable: the grain centre is above the top cell centre.');
  const zLo = j1 === 0 ? DG.ZERO : zc[j1 - 1], zHi = zc[j1];
  const wHi = DG.div(DG.sub(zGrain, zLo), DG.sub(zHi, zLo)), wLo = DG.sub(DG.ONE, wHi);
  const at = (arr, i, j) => j < 0 ? DG.ZERO : arr[i*nz + j];
  const damp = growthFree(mode, steps);
  const scale = DG.div(amplitude, etaPeakOf(mode, damp));
  const reps = sign < 0 ? 2 : 1, responseSamples = reps*perDrive;
  const period = sign < 0 ? DG.scale2(Td, 1) : Td;
  const knots = [];
  for (let rep = 0; rep < reps; rep++)
    for (let k = 0; k < perDrive; k++){
      const f = frames[k];
      let s = DG.mul(scale, damp(f));
      if (rep && sign < 0) s = DG.neg(s);
      const u = [], v = [];
      for (let i = 0; i < faces; i++){
        u.push(DG.mul(s, DG.add(DG.mul(wLo, at(f.u, i, j1 - 1)), DG.mul(wHi, at(f.u, i, j1)))));
        v.push(DG.mul(s, DG.add(DG.mul(wLo, at(f.v, i, j1 - 1)), DG.mul(wHi, at(f.v, i, j1)))));
      }
      knots.push({ step: BigInt(f.step + rep*steps), u, v });
    }
  const U = [], V = [], per = BigInt(perDrive), S = BigInt(steps);
  for (let k = 0, q = 0; k < responseSamples; k++){
    const tau = BigInt(k)*S;
    while (q + 1 < knots.length && knots[q + 1].step*per <= tau) q++;
    const a = knots[q], b = q + 1 < knots.length ? knots[q + 1] : { step: BigInt(reps)*S, u: knots[0].u, v: knots[0].v };
    const w = DG.fromRational(tau - a.step*per, (b.step - a.step)*per), w0 = DG.sub(DG.ONE, w);
    U.push(a.u.map((x, i) => DG.add(DG.mul(w0, x), DG.mul(w, b.u[i]))));
    V.push(a.v.map((x, i) => DG.add(DG.mul(w0, x), DG.mul(w, b.v[i]))));
  }
  const twoDts = DG.scale2(DG.div(period, DG.fromBigInt(BigInt(responseSamples))), 1);
  const dU = [], dV = [];
  for (let k = 0; k < responseSamples; k++){
    const kp = (k + 1) % responseSamples, km = (k - 1 + responseSamples) % responseSamples;
    dU.push(U[kp].map((x, i) => DG.div(DG.sub(x, U[km][i]), twoDts)));
    dV.push(V[kp].map((x, i) => DG.div(DG.sub(x, V[km][i]), twoDts)));
  }
  return { m: o.m, rf, R: rf[nr], faces, samples: responseSamples, period, U, V, dU, dV, amplitude, zGrain, multiplier: mode.multiplier, scale };
}

function streamingProfile(zeta){
  const z = nonneg(zeta, 'zeta');
  const e1 = DG.exp(DG.neg(z)), e2 = DG.exp(DG.neg(DG.scale2(z, 1))), c = DG.cos(z), s = DG.sin(z);
  const one = DG.ONE, quarter = x => DG.scale2(x, -2), half = x => DG.scale2(x, -1);
  const oneMinusE2 = DG.sub(one, e2);
  const a = DG.sub(DG.mul(e1, s), quarter(oneMinusE2));
  const eRe = DG.mul(e1, c), eIm = DG.mul(e1, s), zp1 = DG.add(z, one);
  const pRe = DG.sub(DG.mul(eRe, z), DG.mul(eIm, zp1)), pIm = DG.add(DG.mul(eRe, zp1), DG.mul(eIm, z));
  const bRe = DG.add(DG.neg(half(pRe)), half(DG.sub(eIm, DG.sub(one, eRe))));
  const bIm = DG.sub(DG.add(DG.sub(DG.fromRational(1n, 2n), half(pIm)), half(DG.add(DG.sub(one, eRe), eIm))), quarter(oneMinusE2));
  return { a, bRe, bIm };
}

function streamingCoefficients(o){
  const { rf, m, omega } = o, prof = o.profile, faces = rf.length, nr = faces - 1;
  const Ar = o.A.re, Ai = o.A.im, Br = o.B.re, Bi = o.B.im;
  const mm = DG.fromBigInt(BigInt(m));
  const PrCC = new Array(faces).fill(DG.ZERO), PrSS = PrCC.slice(), PtSC = PrCC.slice(), QrCC = PrCC.slice(), QtSC = PrCC.slice();
  const coefQ = (qRe, qIm) => DG.div(DG.sub(DG.mul(prof.bRe, qRe), DG.mul(prof.bIm, qIm)), omega);
  const coefP = pRe => DG.div(DG.mul(prof.a, pRe), omega);
  for (let i = 1; i < faces; i++){
    const r = rf[i], il = Math.max(0, i - 1), ih = Math.min(nr, i + 1);
    const dr = DG.sub(rf[ih], rf[il]);
    const dAr = DG.div(DG.sub(Ar[ih], Ar[il]), dr), dAi = DG.div(DG.sub(Ai[ih], Ai[il]), dr);
    const dBr = DG.div(DG.sub(Br[ih], Br[il]), dr), dBi = DG.div(DG.sub(Bi[ih], Bi[il]), dr);
    const Dr = DG.add(dAr, DG.div(DG.add(Ar[i], DG.mul(mm, Br[i])), r)), Di = DG.add(dAi, DG.div(DG.add(Ai[i], DG.mul(mm, Bi[i])), r));
    const AdA = DG.add(DG.mul(Ar[i], dAr), DG.mul(Ai[i], dAi));
    const BA = DG.add(DG.mul(Br[i], Ar[i]), DG.mul(Bi[i], Ai[i])), BB = DG.add(DG.mul(Br[i], Br[i]), DG.mul(Bi[i], Bi[i]));
    const AdB = DG.add(DG.mul(Ar[i], dBr), DG.mul(Ai[i], dBi));
    const DAre = DG.add(DG.mul(Dr, Ar[i]), DG.mul(Di, Ai[i])), DAim = DG.sub(DG.mul(Di, Ar[i]), DG.mul(Dr, Ai[i]));
    const DBre = DG.add(DG.mul(Dr, Br[i]), DG.mul(Di, Bi[i])), DBim = DG.sub(DG.mul(Di, Br[i]), DG.mul(Dr, Bi[i]));
    PrCC[i] = coefP(AdA);
    PrSS[i] = coefP(DG.neg(DG.div(DG.add(DG.mul(mm, BA), BB), r)));
    PtSC[i] = coefP(DG.add(AdB, DG.div(DG.add(DG.mul(mm, BB), BA), r)));
    QrCC[i] = coefQ(DAre, DAim);
    QtSC[i] = coefQ(DBre, DBim);
  }
  for (const arr of [PrCC, PrSS, PtSC, QrCC, QtSC]) arr[0] = m === 1 ? arr[1] : DG.ZERO;
  return { PrCC, PrSS, PtSC, QrCC, QtSC };
}

function floorStreaming(o){
  const { mode, m, steps } = o;
  const amplitude = pos(o.amplitude, 'amplitude'), nu = pos(o.nu, 'nu'), k = pos(o.k, 'k'), zGrain = pos(o.zGrain, 'zGrain'), Td = pos(o.Td, 'Td');
  const { rf, zc } = mode.grid, nz = zc.length, faces = rf.length;
  const frames = mode.frames, perDrive = frames.length - 1;
  const sign = DG.sign(mode.multiplier);
  const period = sign < 0 ? DG.scale2(Td, 1) : Td, omega = DG.div(DG.scale2(DG.pi(), 1), period);
  const delta = DG.sqrt(DG.div(DG.scale2(nu, 1), omega)), zTarget = DG.scale2(delta, 2);
  let jo = 0;
  for (let j = 1; j < nz; j++) if (DG.cmp(DG.abs(DG.sub(zc[j], zTarget)), DG.abs(DG.sub(zc[jo], zTarget))) < 0) jo = j;
  const zOuter = zc[jo];
  if (DG.cmp(zOuter, DG.scale2(delta, 1)) < 0) throw new RangeError('floorStreaming: the cell centre nearest four Stokes depths is inside the layer.');
  const damp = growthFree(mode, steps);
  const kz = DG.mul(k, zOuter), cosh = DG.scale2(DG.add(DG.exp(kz), DG.exp(DG.neg(kz))), -1);
  const scale = DG.div(DG.div(amplitude, etaPeakOf(mode, damp)), cosh);
  const Ar = new Array(faces).fill(DG.ZERO), Ai = Ar.slice(), Br = Ar.slice(), Bi = Ar.slice();
  const reps = sign < 0 ? 2 : 1;
  for (let rep = 0; rep < reps; rep++)
    for (let q = 0; q < perDrive; q++){
      const f0 = frames[q], f1 = frames[q + 1], halfSpan = DG.scale2(DG.sub(f1.t, f0.t), -1);
      for (const f of [f0, f1]){
        const t = rep ? DG.add(f.t, Td) : f.t;
        let s = DG.mul(DG.mul(scale, damp(f)), DG.div(DG.scale2(halfSpan, 1), period));
        if (rep && sign < 0) s = DG.neg(s);
        const wt = DG.mul(omega, t), cr = DG.mul(DG.cos(wt), s), ci = DG.neg(DG.mul(DG.sin(wt), s));
        for (let i = 0; i < faces; i++){
          const u = f.u[i*nz + jo], v = f.v[i*nz + jo];
          Ar[i] = DG.add(Ar[i], DG.mul(u, cr)); Ai[i] = DG.add(Ai[i], DG.mul(u, ci));
          Br[i] = DG.add(Br[i], DG.mul(v, cr)); Bi[i] = DG.add(Bi[i], DG.mul(v, ci));
        }
      }
    }
  const zeta = DG.div(zGrain, delta);
  const profile = streamingProfile(zeta);
  const coefficients = streamingCoefficients({ rf, m, A: { re: Ar, im: Ai }, B: { re: Br, im: Bi }, omega, profile });
  return { m, rf, faces, omega, delta, zOuter, zetaGrain: zeta, profile, A: { re: Ar, im: Ai }, B: { re: Br, im: Bi }, ...coefficients };
}

function xorshift32(seed){
  let a = (seed >>> 0) || 0x9E3779B9;
  return () => { a ^= a << 13; a >>>= 0; a ^= a >>> 17; a ^= a << 5; a >>>= 0; return BigInt(a); };
}

function grainParameters(o){
  const R = pos(o.R, 'R'), d = pos(o.d, 'd'), rhoP = pos(o.rhoP, 'rhoP'), rho = pos(o.rho, 'rho'), mu = pos(o.mu, 'mu');
  const muS = nonneg(o.muS, 'muS'), muK = nonneg(o.muK, 'muK'), g = pos(o.g, 'g'), accel = nonneg(o.accel, 'accel'), omegaD = pos(o.omegaD, 'omegaD');
  if (DG.cmp(muK, muS) > 0) throw new RangeError('muK exceeds muS: sliding friction above static friction.');
  if (DG.cmp(rhoP, rho) <= 0) throw new RangeError("the grain is not denser than the liquid, so it does not rest on the floor.");
  if (DG.cmp(accel, g) >= 0) throw new RangeError('the peak acceleration reaches gravity, so a grain leaves the floor once a cycle. Flight off the floor is not modelled; refusing rather than holding the grain down.');
  if (DG.cmp(DG.scale2(d, -1), R) >= 0) throw new RangeError('the grain is wider than the cell.');
  const vol = DG.mulRat(DG.mul(DG.pi(), DG.mul(d, DG.mul(d, d))), 1n, 6n);
  const mp = DG.mul(rhoP, vol), mf = DG.mul(rho, vol);
  return { R, d, rhoP, rho, mu, muS, muK, g, accel, omegaD, vol, mp, mf, M: DG.add(mp, DG.scale2(mf, -1)),
           C: DG.mul(DG.mul(DG.fromDecimal(O_NEILL_WALL_SHEAR_FACTOR), DG.mulRat(DG.pi(), 3n, 1n)), DG.mul(mu, d)),
           mfAdded: DG.mulRat(mf, 3n, 2n), rMax: DG.sub(R, DG.scale2(d, -1)) };
}

function phaseTables(T, P, gp){
  const S = T.samples, faces = T.faces, twoPi = DG.scale2(DG.pi(), 1);
  const perTd = DG.cmp(T.period, DG.div(twoPi, gp.omegaD)) > 0 ? 2n : 1n;
  const buoyant = DG.sub(gp.mp, gp.mf);
  const Ut = [], Vt = [], dUt = [], dVt = [], normal = [];
  for (let k = 0; k < P; k++){
    const num = BigInt(2*k + 1)*BigInt(S), den = 2n*BigInt(P);
    const k0 = Number((num/den) % BigInt(S)), k1 = (k0 + 1) % S;
    const w1 = DG.fromRational(num % den, den), w0 = DG.sub(DG.ONE, w1);
    const mix = (A, i) => DG.add(DG.mul(w0, A[k0][i]), DG.mul(w1, A[k1][i]));
    for (let i = 0; i < faces; i++){ Ut.push(mix(T.U, i)); Vt.push(mix(T.V, i)); dUt.push(mix(T.dU, i)); dVt.push(mix(T.dV, i)); }
    const phase = DG.mulRat(twoPi, perTd*BigInt(2*k + 1), 2n*BigInt(P));
    normal.push(DG.mul(buoyant, DG.add(gp.g, DG.mul(gp.accel, DG.cos(phase)))));
  }
  return { Ut, Vt, dUt, dVt, normal };
}

function putD(dv, ptr, k, a){
  const base = ptr + 24*k, mag = a.m < 0n ? -a.m : a.m;
  dv.setBigUint64(base, mag >> 64n, true); dv.setBigUint64(base + 8, mag & ((1n << 64n) - 1n), true);
  dv.setInt32(base + 16, a.m === 0n ? 0 : a.e, true); dv.setInt32(base + 20, a.m < 0n ? 1 : 0, true);
}
function getD(dv, ptr, k){
  const base = ptr + 24*k;
  const mag = (dv.getBigUint64(base, true) << 64n) | dv.getBigUint64(base + 8, true);
  if (mag === 0n) return DG.ZERO;
  return Object.freeze({ m: dv.getInt32(base + 20, true) ? -mag : mag, e: dv.getInt32(base + 16, true) });
}

function createGrains(o){
  const count = o.count;
  if (!Number.isSafeInteger(count) || count < 1) throw new RangeError(`count = ${count}: a positive whole number of grains.`);
  const gp = grainParameters(o), T = o.table, S = o.streaming;
  const P = o.stepsPerPeriod === undefined ? 256 : o.stepsPerPeriod;
  if (!Number.isSafeInteger(P) || P < 1) throw new RangeError('stepsPerPeriod: a positive whole number.');
  if (!T) throw new TypeError('createGrains: the floor field table is required.');
  if (S && (S.faces !== T.faces || S.m !== T.m || S.rf.some((x, i) => !DG.eq(x, T.rf[i])))) throw new RangeError(
    'createGrains: the streaming field was built on a different grid or mode from the oscillatory table.');
  const h = DG.div(T.period, DG.fromBigInt(BigInt(P)));
  const X = new WebAssembly.Instance(new WebAssembly.Module(grainBytes(o.bytes)), {}).exports;
  for (const name of GRAIN_EXPORTS) if (!(name in X)) throw new Error(`faraday-grains-d128: the module does not export ${name}.`);
  const fail = (code, what) => { if (code) throw new Error(`faraday-grains-d128: ${what} failed with code ${code}.`); };
  fail(X.setupGrains(count, T.faces, P, T.m), 'setupGrains');
  const dv = () => new DataView(X.memory.buffer);
  const write = (ptr, arr) => { const d = dv(); for (let k = 0; k < arr.length; k++) putD(d, ptr, k, arr[k]); };
  const read = (ptr, n) => { const d = dv(), out = new Array(n); for (let k = 0; k < n; k++) out[k] = getD(d, ptr, k); return out; };
  write(X.ptrScal(), [gp.C, gp.M, gp.mfAdded, gp.muS, gp.muK, h, gp.rMax]);
  write(X.ptrRf(), T.rf);
  const pt = phaseTables(T, P, gp);
  write(X.ptrUt(), pt.Ut); write(X.ptrVt(), pt.Vt); write(X.ptrDUt(), pt.dUt); write(X.ptrDVt(), pt.dVt);
  write(X.ptrNormal(), pt.normal);
  if (S){
    write(X.ptrStreamR(), S.PrCC.map((x, i) => DG.add(x, S.QrCC[i])));
    write(X.ptrStreamS(), S.PrSS);
    write(X.ptrStreamT(), S.PtSC.map((x, i) => DG.add(x, S.QtSC[i])));
  }
  let xs, ys;
  if (o.positions){
    if (o.positions.x.length !== count || o.positions.y.length !== count) throw new RangeError('positions: one x and one y per grain.');
    xs = o.positions.x.map((v, i) => val(v, `positions.x[${i}]`)); ys = o.positions.y.map((v, i) => val(v, `positions.y[${i}]`));
  } else {
    const rand = xorshift32(o.seed === undefined ? 1 : o.seed), r2max = DG.mul(gp.rMax, gp.rMax);
    xs = []; ys = [];
    while (xs.length < count){
      const x = DG.mul(gp.rMax, DG.store((rand() << 1n) - (1n << 32n), -32, 1n)), y = DG.mul(gp.rMax, DG.store((rand() << 1n) - (1n << 32n), -32, 1n));
      if (DG.cmp(DG.add(DG.mul(x, x), DG.mul(y, y)), r2max) <= 0){ xs.push(x); ys.push(y); }
    }
  }
  write(X.ptrX(), xs); write(X.ptrY(), ys);
  fail(X.prepareGrains(), 'prepareGrains');
  let phase = 0, stepsTaken = 0;
  return {
    count, period: T.period, h, stepsPerPeriod: P, params: gp, exports: X,
    advance(nSteps){ fail(X.advanceGrains(nSteps, phase), 'advanceGrains'); phase = (phase + nSteps) % P; stepsTaken += nSteps; },
    time: () => DG.mul(h, DG.fromBigInt(BigInt(stepsTaken))),
    positions: () => ({ x: read(X.ptrX(), count), y: read(X.ptrY(), count) }),
    velocities: () => ({ vx: read(X.ptrVX(), count), vy: read(X.ptrVY(), count) }),
    stuck: () => Array.from(new Int32Array(X.memory.buffer, X.ptrStuck(), count)),
    slidSteps: () => X.getSlidSteps()
  };
}

function driftMap(o){
  const T = o.table, m = T.m;
  const nr = o.nr === undefined ? 64 : o.nr, nthIn = o.nth === undefined ? 16 : o.nth;
  const settle = o.settle === undefined ? 12 : o.settle, measure = o.measure === undefined ? 12 : o.measure;
  for (const [k, v, lo] of [['nr', nr, 2], ['nth', nthIn, 4], ['settle', settle, 0], ['measure', measure, 1]])
    if (!Number.isSafeInteger(v) || v < lo) throw new RangeError(`driftMap: ${k} = ${v}: a whole number of at least ${lo} is required.`);
  const sign = DG.sign(T.multiplier);
  const M = m === 0 ? 0 : (sign < 0 ? 2*m : m), nth = m === 0 ? 1 : nthIn;
  const gp = grainParameters(o), rMax = gp.rMax, pi = DG.pi();
  const xs = [], ys = [], dirs = [];
  for (let i = 0; i < nr; i++){
    const r = DG.mulRat(rMax, BigInt(i), BigInt(nr - 1));
    for (let j = 0; j < nth; j++){
      const th = nth === 1 ? DG.ZERO : DG.mulRat(pi, BigInt(j), BigInt((nth - 1)*M));
      const c = DG.cos(th), s = DG.sin(th);
      xs.push(DG.mul(r, c)); ys.push(DG.mul(r, s)); dirs.push({ c, s });
    }
  }
  const probe = createGrains({ ...o, count: nr*nth, positions: { x: xs, y: ys } });
  const P = probe.stepsPerPeriod;
  probe.advance(settle*P);
  const before = probe.positions();
  probe.advance(measure*P);
  const after = probe.positions();
  const span = DG.mul(probe.period, DG.fromBigInt(BigInt(measure)));
  const UR = new Array(nr*nth).fill(DG.ZERO), UT = UR.slice(), G = UR.slice();
  const C = Array.from({ length: nth }, (_, j) => nth === 1 ? DG.ONE : DG.cos(DG.mulRat(pi, BigInt(j), BigInt(nth - 1))));
  const S = Array.from({ length: nth }, (_, j) => nth === 1 ? DG.ZERO : DG.sin(DG.mulRat(pi, BigInt(j), BigInt(nth - 1))));
  for (let i = 1; i < nr; i++) for (let j = 0; j < nth; j++){
    const n = i*nth + j, { c, s } = dirs[n];
    const vx = DG.div(DG.sub(after.x[n], before.x[n]), span), vy = DG.div(DG.sub(after.y[n], before.y[n]), span);
    UR[n] = DG.add(DG.mul(vx, c), DG.mul(vy, s));
    UT[n] = DG.sub(DG.mul(vy, c), DG.mul(vx, s));
    if (nth > 1 && j > 0 && j < nth - 1) G[n] = DG.div(UT[n], S[j]);
  }
  if (nth > 1) for (let i = 1; i < nr; i++){
    for (const [edge, a, b] of [[0, 1, 2], [nth - 1, nth - 2, nth - 3]]){
      const ga = G[i*nth + a], gb = G[i*nth + b];
      G[i*nth + edge] = DG.add(ga, DG.mul(DG.sub(ga, gb), DG.div(DG.sub(C[edge], C[a]), DG.sub(C[a], C[b]))));
    }
  }
  return { m, M, nr, nth, rMax, dr: DG.div(rMax, DG.fromBigInt(BigInt(nr - 1))), UR, UT, G, C, period: probe.period, settle, measure,
           stepsPerPeriod: P, grain: gp, probes: { before, after } };
}

function createSand(o){
  const Dm = o.drift, count = o.count;
  if (!Dm || !Dm.UR) throw new TypeError('createSand: a drift map is required.');
  if (!Number.isSafeInteger(count) || count < 1) throw new RangeError(`count = ${count}: a positive whole number of grains.`);
  const X = new WebAssembly.Instance(new WebAssembly.Module(grainBytes(o.bytes)), {}).exports;
  for (const name of GRAIN_EXPORTS) if (!(name in X)) throw new Error(`faraday-grains-d128: the module does not export ${name}.`);
  const fail = (code, what) => { if (code) throw new Error(`faraday-grains-d128: ${what} failed with code ${code}.`); };
  fail(X.setupSand(count, Dm.nr, Dm.nth, Dm.M), 'setupSand');
  const dv = () => new DataView(X.memory.buffer);
  const write = (ptr, arr) => { const d = dv(); for (let k = 0; k < arr.length; k++) putD(d, ptr, k, arr[k]); };
  const read = (ptr, n) => { const d = dv(), out = new Array(n); for (let k = 0; k < n; k++) out[k] = getD(d, ptr, k); return out; };
  write(X.ptrDriftUR(), Dm.UR); write(X.ptrDriftG(), Dm.G); write(X.ptrDriftC(), Dm.C);
  const scal = new Array(10).fill(DG.ZERO);
  scal[8] = Dm.dr; scal[9] = Dm.rMax;
  write(X.ptrScal(), scal);
  let xs, ys;
  if (o.positions){ xs = o.positions.x.map((v, i) => val(v, `positions.x[${i}]`)); ys = o.positions.y.map((v, i) => val(v, `positions.y[${i}]`)); }
  else {
    const rand = xorshift32(o.seed === undefined ? 1 : o.seed), r2max = DG.mul(Dm.rMax, Dm.rMax);
    xs = []; ys = [];
    while (xs.length < count){
      const x = DG.mul(Dm.rMax, DG.store((rand() << 1n) - (1n << 32n), -32, 1n)), y = DG.mul(Dm.rMax, DG.store((rand() << 1n) - (1n << 32n), -32, 1n));
      if (DG.cmp(DG.add(DG.mul(x, x), DG.mul(y, y)), r2max) <= 0){ xs.push(x); ys.push(y); }
    }
  }
  if (xs.length !== count || ys.length !== count) throw new RangeError('positions: one x and one y per grain.');
  write(X.ptrSandX(), xs); write(X.ptrSandY(), ys);
  fail(X.prepareSand(), 'prepareSand');
  return {
    count, exports: X,
    advect(nSteps, dt){ write(X.ptrDriftDt(), [val(dt, 'dt')]); fail(X.advectSand(nSteps), 'advectSand'); },
    positions: () => ({ x: read(X.ptrSandX(), count), y: read(X.ptrSandY(), count) }),
    driftAt(n){ fail(X.driftAt(n), 'driftAt'); return { vx: getD(dv(), X.ptrDriftOut(), 0), vy: getD(dv(), X.ptrDriftOut(), 1) }; },
    raster(r){
      for (const k of ['n', 'center', 'radius']) if (!Number.isSafeInteger(r[k]) || r[k] < 1) throw new RangeError(`raster.${k}: a positive whole number.`);
      const scaleArr = new Array(11).fill(DG.ZERO);
      scaleArr[8] = Dm.dr; scaleArr[9] = Dm.rMax; scaleArr[10] = DG.div(DG.fromBigInt(BigInt(r.radius)), pos(r.cellRadius, 'raster.cellRadius'));
      write(X.ptrScal(), scaleArr);
      fail(X.setupRaster(r.n, r.center, r.radius, (r.seed === undefined ? 1 : r.seed) >>> 0), 'setupRaster');
      this.rasterSize = r.n;
    },
    pack(budget){ const moved = X.packSand(budget); if (moved < 0) fail(-moved, 'packSand'); return moved; },
    bins(){ fail(X.binSand(), 'binSand'); return new Uint16Array(X.memory.buffer, X.ptrOcc(), this.rasterSize*this.rasterSize); }
  };
}

const FARADAY_GRAINS_D128 = Object.freeze({ floorFieldTable, floorStreaming, streamingProfile, streamingCoefficients, grainParameters, phaseTables,
                                            createGrains, driftMap, createSand, xorshift32, O_NEILL_WALL_SHEAR_FACTOR });
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_GRAINS_D128;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_GRAINS_D128 = FARADAY_GRAINS_D128;
