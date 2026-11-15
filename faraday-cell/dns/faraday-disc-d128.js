'use strict';

const DY = (function(){
  if (typeof require === 'function') return require('./dyadic128.js');
  if (typeof globalThis !== 'undefined' && globalThis.DYADIC128) return globalThis.DYADIC128;
  throw new Error('faraday-disc-d128: dns/dyadic128.js is not loaded.');
})();

const DISC_D128_EXPORTS = ['setup', 'prepare', 'ptrScal', 'ptrRf', 'ptrZf', 'ptrRc', 'ptrZc', 'ptrDrc', 'ptrDzc', 'ptrDrf', 'ptrDzf',
  'ptrDrive', 'ptrVecIn', 'ptrVecOut', 'ptrU', 'ptrV', 'ptrW', 'ptrP', 'ptrEta', 'ptrBasis', 'ptrHess', 'ptrCoef', 'ptrProbe',
  'ptrProbeOut', 'ptrPressureAsymmetry', 'getNVec', 'getLastError', 'getArenaUsed', 'applyLtoProbe', 'loadVecIn', 'storeVecOut',
  'runSteps', 'applyPeriodMap', 'arnoldiStart', 'arnoldiExtend', 'combineBasis', 'basisOrthogonality', 'ptrOrthogonality', 'memory'];

const DISC_D128_ERRORS = {
  1: 'the module arena is exhausted for this grid and Krylov dimension',
  3: 'the pressure operator is not positive definite: a pivot of its factorisation is not positive',
  4: 'the engine has not been prepared',
  5: 'an index or size is out of range',
  6: 'a vector has zero norm'
};

function discD128Bytes(explicit){
  if (explicit) return explicit;
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DISC_D128_WASM_BASE64){
    const b64 = globalThis.FARADAY_DISC_D128_WASM_BASE64;
    if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  if (typeof require === 'function'){
    const { readFileSync } = require('fs');
    const { join } = require('path');
    return new Uint8Array(readFileSync(join(__dirname, 'faraday_disc_d128.wasm')));
  }
  throw new Error('faraday-disc-d128: no module bytes. Pass them, inline FARADAY_DISC_D128_WASM_BASE64, or run under node beside dns/faraday_disc_d128.wasm.');
}

function requireInt(v, lo, name){
  if (!Number.isSafeInteger(v) || v < lo) throw new RangeError(`${name} = ${v}: a whole number of at least ${lo} is required.`);
  return v;
}

function dyadic(v, name){
  if (typeof v === 'string') return DY.fromDecimal(v);
  if (typeof v === 'bigint') return DY.fromBigInt(v);
  if (v && typeof v.m === 'bigint' && Number.isSafeInteger(v.e)) return v;
  throw new TypeError(`${name}: a decimal string, a BigInt or a 128-bit dyadic value is required; a float is not accepted.`);
}

function positive(v, name){
  const x = dyadic(v, name);
  if (DY.sign(x) <= 0) throw new RangeError(`${name}: a positive value is required.`);
  return x;
}

function gradeToEnd(n, L, s){
  const f = new Array(n + 1);
  if (DY.isZero(s)){ for (let i = 0; i <= n; i++) f[i] = DY.mulRat(L, BigInt(i), BigInt(n)); return f; }
  const th = DY.tanh(s);
  for (let i = 0; i <= n; i++) f[i] = DY.div(DY.mul(L, DY.tanh(DY.mulRat(s, BigInt(i), BigInt(n)))), th);
  f[0] = DY.ZERO; f[n] = L;
  return f;
}

function gradeBothEnds(n, L, s){
  const f = new Array(n + 1);
  if (DY.isZero(s)){ for (let j = 0; j <= n; j++) f[j] = DY.mulRat(L, BigInt(j), BigInt(n)); return f; }
  const th = DY.tanh(s);
  for (let j = 0; j <= n; j++){
    const t = DY.div(DY.tanh(DY.mulRat(s, BigInt(2*j - n), BigInt(n))), th);
    f[j] = DY.scale2(DY.mul(L, DY.add(DY.ONE, t)), -1);
  }
  f[0] = DY.ZERO; f[n] = L;
  return f;
}

function buildGrid(nr, nz, R, h, rStretch, zStretch){
  const rf = gradeToEnd(nr, R, rStretch), zf = gradeBothEnds(nz, h, zStretch);
  const rc = [], zc = [], drc = [], dzc = [], drf = [], dzf = [];
  for (let i = 0; i < nr; i++){ rc.push(DY.scale2(DY.add(rf[i], rf[i + 1]), -1)); drc.push(DY.sub(rf[i + 1], rf[i])); }
  for (let j = 0; j < nz; j++){ zc.push(DY.scale2(DY.add(zf[j], zf[j + 1]), -1)); dzc.push(DY.sub(zf[j + 1], zf[j])); }
  drf.push(rc[0]);
  for (let i = 1; i < nr; i++) drf.push(DY.sub(rc[i], rc[i - 1]));
  drf.push(DY.sub(R, rc[nr - 1]));
  dzf.push(zc[0]);
  for (let j = 1; j < nz; j++) dzf.push(DY.sub(zc[j], zc[j - 1]));
  dzf.push(DY.sub(h, zc[nz - 1]));
  for (const [name, arr] of [['drc', drc], ['dzc', dzc], ['drf', drf], ['dzf', dzf]])
    arr.forEach((x, k) => { if (DY.sign(x) <= 0) throw new RangeError(`buildGrid: ${name}[${k}] is not positive; the grading folds the grid.`); });
  return { rf, zf, rc, zc, drc, dzc, drf, dzf };
}

function minOf(arr){ return arr.reduce((a, b) => DY.cmp(a, b) <= 0 ? a : b); }

function stepLimits(grid, o){
  const four = DY.fromBigInt(4n), two = DY.fromBigInt(2n);
  const drMin = minOf(grid.drc), dzMin = minOf(grid.dzc), rMin = grid.rc[0];
  const kr2 = DY.div(four, DY.mul(drMin, drMin)), kz2 = DY.div(four, DY.mul(dzMin, dzMin));
  const mm = DY.fromBigInt(BigInt(o.m)*BigInt(o.m));
  const ka2 = DY.div(mm, DY.mul(rMin, rMin));
  const kSurf2 = DY.add(kr2, ka2);
  const top = DY.add(DY.add(o.g, DY.abs(o.accel)), DY.mul(DY.div(o.gamma, o.rho), kSurf2));
  const omegaSurf = DY.sqrt(DY.div(top, grid.dzf[grid.dzf.length - 1]));
  return { capillary: DY.div(two, omegaSurf), viscous: DY.div(two, DY.mul(o.nu, DY.add(DY.add(kr2, kz2), ka2))) };
}

function roundDyadicToInteger(x){
  const y = DY.add(x, DY.fromRational(1n, 2n));
  return y.e >= 0 ? y.m << BigInt(y.e) : y.m >> BigInt(-y.e);
}

function putD(dv, ptr, k, a){
  const base = ptr + 24*k, mag = a.m < 0n ? -a.m : a.m;
  dv.setBigUint64(base, mag >> 64n, true); dv.setBigUint64(base + 8, mag & ((1n << 64n) - 1n), true);
  dv.setInt32(base + 16, a.m === 0n ? 0 : a.e, true); dv.setInt32(base + 20, a.m < 0n ? 1 : 0, true);
}
function getD(dv, ptr, k){
  const base = ptr + 24*k;
  const mag = (dv.getBigUint64(base, true) << 64n) | dv.getBigUint64(base + 8, true);
  if (mag === 0n) return DY.ZERO;
  const e = dv.getInt32(base + 16, true), s = dv.getInt32(base + 20, true);
  return Object.freeze({ m: s ? -mag : mag, e });
}

function createDiscD128(o){
  const m = requireInt(o.m, 0, 'm'), nr = requireInt(o.nr, 6, 'nr'), nz = requireInt(o.nz, 4, 'nz');
  const kMax = o.kMax === undefined ? 40 : requireInt(o.kMax, 2, 'kMax');
  const R = positive(o.R, 'R'), h = positive(o.h, 'h'), rho = positive(o.rho, 'rho');
  const nu = positive(o.nu, 'nu'), gamma = positive(o.gamma, 'gamma'), g = positive(o.g, 'g');
  const accel = dyadic(o.accel, 'accel'), omegaD = positive(o.omegaD, 'omegaD');
  const contact = o.contact;
  if (contact !== 'free' && contact !== 'pinned') throw new TypeError(`contact = ${JSON.stringify(contact)}: 'free' or 'pinned'.`);
  const rStretch = dyadic(o.rStretch, 'rStretch'), zStretch = dyadic(o.zStretch, 'zStretch');
  if (DY.sign(rStretch) < 0 || DY.sign(zStretch) < 0) throw new RangeError('a negative stretch inverts the grading.');
  const grid = buildGrid(nr, nz, R, h, rStretch, zStretch);
  const phys = { m, rho, nu, gamma, g, accel };
  const limits = stepLimits(grid, phys);
  const safety = DY.fromRational(2n, 5n);
  const stable = DY.mul(safety, DY.cmp(limits.capillary, limits.viscous) <= 0 ? limits.capillary : limits.viscous);
  const twoPi = DY.scale2(DY.pi(), 1), Td = DY.div(twoPi, omegaD);
  let steps;
  if (o.steps === undefined){
    const s = roundDyadicToInteger(DY.div(Td, stable));
    steps = Number(s < 1n ? 1n : s);
  } else {
    steps = requireInt(o.steps, 1, 'steps');
    const dtAsked = DY.div(Td, DY.fromBigInt(BigInt(steps)));
    if (DY.cmp(dtAsked, DY.scale2(DY.div(stable, safety), -1)) > 0) throw new RangeError(
      `steps = ${steps} gives a step above half this grid's explicit stability limit; refusing rather than integrating past it.`);
  }
  const dt = DY.div(Td, DY.fromBigInt(BigInt(steps)));
  const href = o.href === undefined ? DY.fromDecimal('1e-9') : positive(o.href, 'href');
  const uref = DY.scale2(DY.mul(href, omegaD), -1);

  const bytes = discD128Bytes(o.bytes);
  const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
  const X = inst.exports;
  for (const name of DISC_D128_EXPORTS) if (!(name in X)) throw new Error(`faraday-disc-d128: the module does not export ${name}.`);
  const fail = (code, what) => { if (code) throw new Error(`faraday-disc-d128: ${what}: ${DISC_D128_ERRORS[code] || 'error ' + code}.`); };
  fail(X.setup(nr, nz, steps, m, contact === 'pinned' ? 1 : 0, kMax), 'setup');
  const dv = () => new DataView(X.memory.buffer);
  const writeArray = (ptr, arr) => { const d = dv(); for (let k = 0; k < arr.length; k++) putD(d, ptr, k, arr[k]); };
  const readArray = (ptr, n) => { const d = dv(), out = new Array(n); for (let k = 0; k < n; k++) out[k] = getD(d, ptr, k); return out; };
  writeArray(X.ptrScal(), [rho, nu, gamma, dt, href, uref]);
  writeArray(X.ptrRf(), grid.rf); writeArray(X.ptrZf(), grid.zf);
  writeArray(X.ptrRc(), grid.rc); writeArray(X.ptrZc(), grid.zc);
  writeArray(X.ptrDrc(), grid.drc); writeArray(X.ptrDzc(), grid.dzc);
  writeArray(X.ptrDrf(), grid.drf); writeArray(X.ptrDzf(), grid.dzf);
  const drive = new Array(steps);
  if (DY.isZero(accel)) drive.fill(g);
  else for (let n = 0; n < steps; n++)
    drive[n] = DY.add(g, DY.mul(accel, DY.cos(DY.mulRat(twoPi, BigInt(n), BigInt(steps)))));
  writeArray(X.ptrDrive(), drive);
  fail(X.prepare(), 'prepare');
  const nVec = X.getNVec();

  return {
    m, nr, nz, steps, dt, Td, href, uref, grid, nVec, kMax, contact, limits, exports: X,
    physics: Object.freeze({ rho, nu, gamma, g, accel, omegaD }),
    drive: () => readArray(X.ptrDrive(), steps),
    pressureAsymmetry: () => getD(dv(), X.ptrPressureAsymmetry(), 0),
    writeVec: vec => { if (vec.length !== nVec) throw new RangeError(`a state vector of ${nVec} values is required.`); writeArray(X.ptrVecIn(), vec); },
    readVecOut: () => readArray(X.ptrVecOut(), nVec),
    readVecIn: () => readArray(X.ptrVecIn(), nVec),
    applyPeriodMap(vec){ this.writeVec(vec); fail(X.applyPeriodMap(), 'applyPeriodMap'); return this.readVecOut(); },
    load(vec){ this.writeVec(vec); fail(X.loadVecIn(), 'loadVecIn'); },
    run(from, to){ fail(X.runSteps(from, to), 'runSteps'); },
    fields(){
      return { u: readArray(X.ptrU(), (nr + 1)*nz), v: readArray(X.ptrV(), (nr + 1)*nz),
               w: readArray(X.ptrW(), nr*(nz + 1)), p: readArray(X.ptrP(), nr*nz), eta: readArray(X.ptrEta(), nr) };
    },
    applyL(q){ writeArray(X.ptrProbe(), q); X.applyLtoProbe(); return readArray(X.ptrProbeOut(), nr*nz); },
    store(){ fail(X.storeVecOut(), 'storeVecOut'); return this.readVecOut(); },
    arnoldiStart(vec){ this.writeVec(vec); fail(X.arnoldiStart(), 'arnoldiStart'); },
    arnoldiExtend(j){ fail(X.arnoldiExtend(j), 'arnoldiExtend'); },
    hessColumn(j, rows){ const d = dv(), out = []; for (let i = 0; i < rows; i++) out.push(getD(d, X.ptrHess(), i*kMax + j)); return out; },
    orthogonality(count){ fail(X.basisOrthogonality(count), 'basisOrthogonality'); return getD(dv(), X.ptrOrthogonality(), 0); },
    combine(coefs){
      writeArray(X.ptrCoef(), coefs);
      fail(X.combineBasis(coefs.length), 'combineBasis');
      return this.readVecIn();
    }
  };
}

const SUGGEST_NR = [24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512];
const SUGGEST_NZ = [12, 16, 20, 24, 32, 40, 48, 64, 80, 96];
const SUGGEST_ZSTRETCH = ['2.2', '3.0', '3.8'];

function radialPart(nr, R, rStretch){
  const rf = gradeToEnd(nr, R, rStretch), rc = [], drc = [], drf = [rf[0]];
  for (let i = 0; i < nr; i++){ rc.push(DY.scale2(DY.add(rf[i], rf[i + 1]), -1)); drc.push(DY.sub(rf[i + 1], rf[i])); }
  drf[0] = rc[0];
  for (let i = 1; i < nr; i++) drf.push(DY.sub(rc[i], rc[i - 1]));
  drf.push(DY.sub(R, rc[nr - 1]));
  return { rf, rc, drc, drf };
}
function verticalPart(nz, h, zStretch){
  const zf = gradeBothEnds(nz, h, zStretch), zc = [], dzc = [], dzf = [];
  for (let j = 0; j < nz; j++){ zc.push(DY.scale2(DY.add(zf[j], zf[j + 1]), -1)); dzc.push(DY.sub(zf[j + 1], zf[j])); }
  dzf.push(zc[0]);
  for (let j = 1; j < nz; j++) dzf.push(DY.sub(zc[j], zc[j - 1]));
  dzf.push(DY.sub(h, zc[nz - 1]));
  return { zf, zc, dzc, dzf };
}

function surfaceOperatorError(rad, m, contact, eta, k){
  const nr = rad.rc.length, mm = DY.fromBigInt(BigInt(m)*BigInt(m)), k2 = DY.mul(k, k);
  let num = DY.ZERO, den = DY.ZERO;
  for (let i = 0; i < nr; i++){
    const outer = i === nr - 1 ? (contact === 'free' ? DY.ZERO : DY.div(DY.mul(rad.rf[nr], DY.neg(eta[nr - 1])), rad.drf[nr]))
                               : DY.div(DY.mul(rad.rf[i + 1], DY.sub(eta[i + 1], eta[i])), rad.drf[i + 1]);
    const inner = i === 0 ? DY.ZERO : DY.div(DY.mul(rad.rf[i], DY.sub(eta[i], eta[i - 1])), rad.drf[i]);
    const got = DY.sub(DY.div(DY.sub(outer, inner), DY.mul(rad.rc[i], rad.drc[i])), DY.div(DY.mul(mm, eta[i]), DY.mul(rad.rc[i], rad.rc[i])));
    const want = DY.neg(DY.mul(k2, eta[i])), w = DY.mul(rad.rc[i], rad.drc[i]), e = DY.sub(got, want);
    num = DY.add(num, DY.mul(DY.mul(e, e), w)); den = DY.add(den, DY.mul(DY.mul(want, want), w));
  }
  return DY.isZero(den) ? null : DY.sqrt(DY.div(num, den));
}

function suggestGrid(o){
  const m = requireInt(o.m, 0, 'm'), k = positive(o.k, 'k'), R = positive(o.R, 'R'), h = positive(o.h, 'h');
  const rho = positive(o.rho, 'rho'), nu = positive(o.nu, 'nu'), gamma = positive(o.gamma, 'gamma');
  const g = o.g === undefined ? DY.fromDecimal('9.80665') : positive(o.g, 'g');
  const omegaResponse = positive(o.omegaResponse, 'omegaResponse'), omegaDrive = positive(o.omegaDrive, 'omegaDrive');
  const contact = o.contact;
  if (contact !== 'free' && contact !== 'pinned') throw new TypeError(`contact = ${JSON.stringify(contact)}: 'free' or 'pinned'.`);
  if (typeof o.profile !== 'function') throw new TypeError('suggestGrid: profile must be a function of a 128-bit radius returning a 128-bit elevation.');
  const errTol = o.errTol === undefined ? DY.fromRational(1n, 100n) : positive(o.errTol, 'errTol');
  const layerCells = o.layerCells === undefined ? DY.fromBigInt(2n) : positive(o.layerCells, 'layerCells');
  const nrCap = o.nrCap === undefined ? 512 : o.nrCap, nzCap = o.nzCap === undefined ? 96 : o.nzCap;
  const nrMin = o.nrMin === undefined ? 24 : o.nrMin, nzMin = o.nzMin === undefined ? 12 : o.nzMin;
  const twoPi = DY.scale2(DY.pi(), 1), lambda = DY.div(twoPi, k), Td = DY.div(twoPi, omegaDrive);
  const delta = DY.sqrt(DY.div(DY.scale2(nu, 1), omegaResponse));
  const safety = DY.fromRational(2n, 5n), sixteen = DY.fromBigInt(16n), eight = DY.fromBigInt(8n);
  const verticals = new Map();
  const tried = [];
  let best = null;
  for (const nr of SUGGEST_NR){
    if (nr < nrMin || nr > nrCap) continue;
    const perLambda = DY.div(lambda, DY.div(R, DY.fromBigInt(BigInt(nr))));
    const rStretch = DY.fromDecimal(DY.cmp(perLambda, sixteen) > 0 ? '2.2' : DY.cmp(perLambda, eight) > 0 ? '1.4' : '0.8');
    const rad = radialPart(nr, R, rStretch);
    const eta = rad.rc.map(r => o.profile(r));
    const err = surfaceOperatorError(rad, m, contact, eta, k);
    if (err === null){ tried.push({ nr, rStretch, refused: 'the profile is identically zero on this radial grid' }); continue; }
    const rimCells = DY.div(delta, rad.drc[nr - 1]);
    const drMin = rad.drc.reduce((a, b) => DY.cmp(a, b) <= 0 ? a : b);
    for (const nz of SUGGEST_NZ){
      if (nz < nzMin || nz > nzCap) continue;
      for (const zs of SUGGEST_ZSTRETCH){
        const key = `${nz}|${zs}`;
        if (!verticals.has(key)) verticals.set(key, verticalPart(nz, h, DY.fromDecimal(zs)));
        const ver = verticals.get(key);
        const floorCells = DY.div(delta, ver.dzc[0]), surfaceCells = DY.div(delta, ver.dzc[nz - 1]);
        const layers = [floorCells, surfaceCells, rimCells].reduce((a, b) => DY.cmp(a, b) <= 0 ? a : b);
        const grid = { drc: [drMin], dzc: [ver.dzc.reduce((a, b) => DY.cmp(a, b) <= 0 ? a : b)], rc: rad.rc, dzf: ver.dzf };
        const lim = stepLimits(grid, { m, rho, nu, gamma, g, accel: DY.ZERO });
        const stable = DY.mul(safety, DY.cmp(lim.capillary, lim.viscous) <= 0 ? lim.capillary : lim.viscous);
        const s = roundDyadicToInteger(DY.div(Td, stable)), stepsPerPeriod = s < 1n ? 1n : s;
        const stateSize = 2*(nr - 1)*nz + nr*nz + nr + (m === 1 ? nz : 0);
        const S = DY.fromBigInt(BigInt(stateSize));
        const work = DY.mul(DY.mul(S, DY.sqrt(S)), DY.fromBigInt(stepsPerPeriod));
        const rec = { nr, nz, rStretch, zStretch: DY.fromDecimal(zs), zStretchText: zs, err, layers, layersOk: DY.cmp(layers, layerCells) >= 0,
                      stokes: { delta, cellsInFloorLayer: floorCells, cellsInSurfaceLayer: surfaceCells, cellsInRimLayer: rimCells },
                      stateSize, stepsPerPeriod: Number(stepsPerPeriod), work };
        tried.push(rec);
        if (DY.cmp(err, errTol) <= 0 && rec.layersOk && (!best || DY.cmp(work, best.work) < 0)) best = rec;
      }
    }
  }
  if (best) return Object.assign({}, best, { tried });
  return { unreachable: true, errTol, layerCells, nrCap, nzCap, tried };
}

const FARADAY_DISC_D128 = Object.freeze({ createDiscD128, buildGrid, gradeToEnd, gradeBothEnds, stepLimits, discD128Bytes, DISC_D128_ERRORS,
                                          suggestGrid, surfaceOperatorError, radialPart, verticalPart });
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_DISC_D128;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_DISC_D128 = FARADAY_DISC_D128;
