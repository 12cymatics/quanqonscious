'use strict';

const DQ = (function(){
  if (typeof require === 'function') return require('./dyadic128.js');
  if (typeof globalThis !== 'undefined' && globalThis.DYADIC128) return globalThis.DYADIC128;
  throw new Error('faraday-floquet-d128: dns/dyadic128.js is not loaded.');
})();
const DISC128 = (function(){
  if (typeof require === 'function') return require('./faraday-disc-d128.js');
  if (typeof globalThis !== 'undefined' && globalThis.FARADAY_DISC_D128) return globalThis.FARADAY_DISC_D128;
  throw new Error('faraday-floquet-d128: dns/faraday-disc-d128.js is not loaded.');
})();

const ZC = Object.freeze({ re: DQ.ZERO, im: DQ.ZERO });
const cx = (re, im = DQ.ZERO) => ({ re, im });
const cadd = (a, b) => cx(DQ.add(a.re, b.re), DQ.add(a.im, b.im));
const csub = (a, b) => cx(DQ.sub(a.re, b.re), DQ.sub(a.im, b.im));
const cmul = (a, b) => cx(DQ.sub(DQ.mul(a.re, b.re), DQ.mul(a.im, b.im)), DQ.add(DQ.mul(a.re, b.im), DQ.mul(a.im, b.re)));
const cnorm2 = a => DQ.add(DQ.mul(a.re, a.re), DQ.mul(a.im, a.im));
const cabs = a => DQ.sqrt(cnorm2(a));
const cconj = a => cx(a.re, DQ.neg(a.im));
const cneg = a => cx(DQ.neg(a.re), DQ.neg(a.im));
const cscale = (a, s) => cx(DQ.mul(a.re, s), DQ.mul(a.im, s));
function cdiv(a, b){
  const d = cnorm2(b);
  if (DQ.isZero(d)) throw new RangeError('complex division by zero.');
  return cx(DQ.div(DQ.add(DQ.mul(a.re, b.re), DQ.mul(a.im, b.im)), d), DQ.div(DQ.sub(DQ.mul(a.im, b.re), DQ.mul(a.re, b.im)), d));
}
function csqrt(a){
  const r = cabs(a);
  if (DQ.isZero(r)) return ZC;
  const re = DQ.sqrt(DQ.scale2(DQ.add(r, a.re), -1));
  let im = DQ.sqrt(DQ.scale2(DQ.sub(r, a.re), -1));
  if (DQ.sign(a.im) < 0) im = DQ.neg(im);
  return cx(re, im);
}

function eig2x2(a, b, c, d){
  const tr = cadd(a, d), det = csub(cmul(a, d), cmul(b, c));
  const disc = csqrt(csub(cmul(tr, tr), cscale(det, DQ.fromBigInt(4n))));
  const s = DQ.sign(DQ.add(DQ.mul(tr.re, disc.re), DQ.mul(tr.im, disc.im))) >= 0 ? disc : cneg(disc);
  const half = DQ.fromRational(1n, 2n);
  const r1 = cscale(cadd(tr, s), half);
  const r2 = DQ.isZero(cnorm2(r1)) ? cscale(csub(tr, s), half) : cdiv(det, r1);
  return [r1, r2];
}

const DEFLATE = DQ.store(1n, -120, 1n);

function hessenbergEigs(Hin, n){
  const H = Hin.slice(0, n).map(row => row.slice(0, n).map(x => cx(x)));
  const eigs = [];
  let hi = n - 1;
  while (hi >= 0){
    if (hi === 0){ eigs.push(H[0][0]); break; }
    let it = 0, deflated = false;
    for (; it < 500 && !deflated; it++){
      let small = -1;
      for (let i = hi; i > 0; i--){
        let t = DQ.add(cabs(H[i - 1][i - 1]), cabs(H[i][i]));
        if (DQ.isZero(t)) t = DQ.ONE;
        if (DQ.cmp(cabs(H[i][i - 1]), DQ.mul(DEFLATE, t)) <= 0){ small = i; break; }
      }
      if (small === hi){ eigs.push(H[hi][hi]); H[hi][hi - 1] = ZC; hi--; deflated = true; break; }
      const a = H[hi - 1][hi - 1], b = H[hi - 1][hi], c = H[hi][hi - 1], d = H[hi][hi];
      const [r1, r2] = eig2x2(a, b, c, d);
      if (small === hi - 1 || (small < 0 && hi === 1)){
        eigs.push(r1, r2);
        if (hi - 1 > 0) H[hi - 1][hi - 2] = ZC;
        hi -= 2; deflated = true; break;
      }
      const mu = DQ.cmp(cnorm2(csub(d, r1)), cnorm2(csub(d, r2))) < 0 ? r1 : r2;
      for (let i = 0; i <= hi; i++) H[i][i] = csub(H[i][i], mu);
      const cs = [], sn = [];
      for (let i = 0; i < hi; i++){
        const x = H[i][i], y = H[i + 1][i];
        const r = DQ.sqrt(DQ.add(cnorm2(x), cnorm2(y)));
        if (DQ.isZero(r)){ cs.push(cx(DQ.ONE)); sn.push(ZC); continue; }
        const cc = cx(DQ.div(x.re, r), DQ.div(x.im, r)), ss = cx(DQ.div(y.re, r), DQ.div(y.im, r));
        cs.push(cc); sn.push(ss);
        for (let j = i; j <= hi; j++){
          const u = H[i][j], v = H[i + 1][j];
          H[i][j] = cadd(cmul(cconj(cc), u), cmul(cconj(ss), v));
          H[i + 1][j] = cadd(cmul(cneg(ss), u), cmul(cc, v));
        }
      }
      for (let i = 0; i < hi; i++){
        const cc = cs[i], ss = sn[i];
        for (let j = 0; j <= Math.min(hi, i + 2); j++){
          const u = H[j][i], v = H[j][i + 1];
          H[j][i] = cadd(cmul(u, cc), cmul(v, ss));
          H[j][i + 1] = cadd(cmul(u, cx(DQ.neg(ss.re), ss.im)), cmul(v, cconj(cc)));
        }
      }
      for (let i = 0; i <= hi; i++) H[i][i] = cadd(H[i][i], mu);
    }
    if (!deflated) throw new Error(`hessenbergEigs: QR did not converge on the block ending at ${hi} in 500 iterations.`);
  }
  return eigs.sort((p, q) => DQ.cmp(cnorm2(q), cnorm2(p)));
}

const SHIFT_OFFSET = DQ.store(1n, -80, 1n);

function realEigenvector(Hm, used, lam){
  const big = DQ.cmp(DQ.abs(lam), DQ.ONE) > 0 ? DQ.abs(lam) : DQ.ONE;
  const shift = DQ.add(lam, DQ.mul(SHIFT_OFFSET, big));
  let y = new Array(used).fill(DQ.div(DQ.ONE, DQ.sqrt(DQ.fromBigInt(BigInt(used)))));
  for (let pass = 0; pass < 4; pass++){
    const A = Array.from({ length: used }, (_, i) =>
      Array.from({ length: used + 1 }, (_, j) => j === used ? y[i] : (i === j ? DQ.sub(Hm[i][j], shift) : Hm[i][j])));
    for (let c = 0; c < used; c++){
      let p = c;
      for (let r = c + 1; r < used; r++) if (DQ.cmp(DQ.abs(A[r][c]), DQ.abs(A[p][c])) > 0) p = r;
      if (DQ.isZero(A[p][c])) throw new Error(`realEigenvector: the shifted Hessenberg matrix is singular in column ${c}.`);
      if (p !== c){ const t = A[p]; A[p] = A[c]; A[c] = t; }
      for (let r = c + 1; r < used; r++){
        if (DQ.isZero(A[r][c])) continue;
        const f = DQ.div(A[r][c], A[c][c]);
        for (let j = c; j <= used; j++) A[r][j] = DQ.sub(A[r][j], DQ.mul(f, A[c][j]));
      }
    }
    const x = new Array(used);
    for (let r = used - 1; r >= 0; r--){
      let s = A[r][used];
      for (let j = r + 1; j < used; j++) s = DQ.sub(s, DQ.mul(A[r][j], x[j]));
      x[r] = DQ.div(s, A[r][r]);
    }
    let nrm = DQ.ZERO;
    for (const v of x) nrm = DQ.add(nrm, DQ.mul(v, v));
    nrm = DQ.sqrt(nrm);
    if (DQ.isZero(nrm)) throw new Error('realEigenvector: inverse iteration produced a zero vector.');
    y = x.map(v => DQ.div(v, nrm));
  }
  return y;
}

function surfaceOperator(eng, eta){
  const { nr, m, grid: G, contact } = eng, out = [];
  const mm = DQ.fromBigInt(BigInt(m)*BigInt(m));
  for (let i = 0; i < nr; i++){
    const outer = i === nr - 1 ? (contact === 'free' ? DQ.ZERO : DQ.div(DQ.mul(G.rf[nr], DQ.neg(eta[nr - 1])), G.drf[nr]))
                               : DQ.div(DQ.mul(G.rf[i + 1], DQ.sub(eta[i + 1], eta[i])), G.drf[i + 1]);
    const inner = i === 0 ? DQ.ZERO : DQ.div(DQ.mul(G.rf[i], DQ.sub(eta[i], eta[i - 1])), G.drf[i]);
    out.push(DQ.sub(DQ.div(DQ.sub(outer, inner), DQ.mul(G.rc[i], G.drc[i])), DQ.div(DQ.mul(mm, eta[i]), DQ.mul(G.rc[i], G.rc[i]))));
  }
  return out;
}

function gridGates(eng, eta, kGiven, omegaResponse){
  const { nr, grid: G } = eng;
  const d = surfaceOperator(eng, eta);
  let num = DQ.ZERO, den = DQ.ZERO;
  for (let i = 0; i < nr; i++){ const w = DQ.mul(G.rc[i], G.drc[i]); num = DQ.add(num, DQ.mul(DQ.mul(d[i], eta[i]), w)); den = DQ.add(den, DQ.mul(DQ.mul(eta[i], eta[i]), w)); }
  if (DQ.isZero(den)) throw new Error('the seed elevation is identically zero, so there is no mode to measure the grid against.');
  const lambda = DQ.div(num, den);
  if (DQ.sign(lambda) >= 0) throw new Error('the best-fitting eigenvalue of the surface operator is not negative: the seed has no radial structure this grid can see.');
  const residual = target => {
    let rn = DQ.ZERO, rd = DQ.ZERO;
    for (let i = 0; i < nr; i++){
      const w = DQ.mul(G.rc[i], G.drc[i]), want = DQ.mul(target, eta[i]), e = DQ.sub(d[i], want);
      rn = DQ.add(rn, DQ.mul(DQ.mul(e, e), w)); rd = DQ.add(rd, DQ.mul(DQ.mul(want, want), w));
    }
    return DQ.sqrt(DQ.div(rn, rd));
  };
  const kFit = DQ.sqrt(DQ.neg(lambda));
  const k = kGiven === undefined ? kFit : kGiven;
  const opErr = kGiven === undefined ? residual(lambda) : residual(DQ.neg(DQ.mul(k, k)));
  const delta = DQ.sqrt(DQ.div(DQ.scale2(eng.physics.nu, 1), omegaResponse));
  const layers = [DQ.div(delta, G.dzc[0]), DQ.div(delta, G.dzc[G.dzc.length - 1]), DQ.div(delta, G.drc[nr - 1])];
  const minLayer = layers.reduce((a, b) => DQ.cmp(a, b) <= 0 ? a : b);
  const lam = DQ.div(DQ.scale2(DQ.pi(), 1), k), tenth = DQ.fromRational(1n, 10n);
  let worst = null;
  for (let i = 0; i < nr; i++) if (DQ.cmp(DQ.abs(eta[i]), tenth) > 0){ const c = DQ.div(lam, G.drc[i]); if (!worst || DQ.cmp(c, worst) < 0) worst = c; }
  if (!worst) worst = DQ.div(lam, G.drc.reduce((a, b) => DQ.cmp(a, b) >= 0 ? a : b));
  return { k, kFit, opErr, delta, cellsInFloorLayer: layers[0], cellsInSurfaceLayer: layers[1], cellsInRimLayer: layers[2], minLayer, cellsPerWavelength: worst };
}

function requireCount(v, lo, name){
  if (!Number.isSafeInteger(v) || v < lo) throw new RangeError(`${name} = ${v}: a whole number of at least ${lo} is required.`);
  return v;
}
function rationalTol(v, fallback, name){
  if (v === undefined) return fallback;
  if (typeof v === 'string') return DQ.fromDecimal(v);
  if (v && typeof v.m === 'bigint') return v;
  throw new TypeError(`${name}: a decimal string or a 128-bit value is required; a float is not accepted.`);
}

function floquetDisc(o){
  const krylov = o.krylov === undefined ? 16 : requireCount(o.krylov, 2, 'krylov');
  const krylovCap = o.krylovCap === undefined ? 40 : requireCount(o.krylovCap, krylov, 'krylovCap');
  const krylovStep = o.krylovStep === undefined ? 4 : requireCount(o.krylovStep, 1, 'krylovStep');
  const krylovTol = rationalTol(o.krylovTol, DQ.fromRational(1n, 50n), 'krylovTol');
  const errTol = rationalTol(o.errTol, DQ.fromRational(1n, 100n), 'errTol');
  const layerCells = rationalTol(o.layerCells, DQ.fromBigInt(2n), 'layerCells');
  const eng = DISC128.createDiscD128({ ...o, kMax: krylovCap });
  const { nr, nz, m, nVec } = eng;
  const base = 2*(nr - 1)*nz + nr*nz;
  const v0 = new Array(nVec).fill(DQ.ZERO);
  let eta0;
  if (o.eta0){
    if (o.eta0.length !== nr) throw new RangeError(`eta0 has ${o.eta0.length} values for a ${nr}-cell radius.`);
    eta0 = o.eta0.map((x, i) => { if (x && typeof x.m === 'bigint') return x; if (typeof x === 'string') return DQ.fromDecimal(x); throw new TypeError(`eta0[${i}]: a 128-bit value or decimal string is required.`); });
  } else {
    const pi = DQ.pi();
    eta0 = Array.from({ length: nr }, (_, i) => DQ.cos(DQ.mulRat(pi, BigInt(2*i + 1), BigInt(2*nr))));
  }
  for (let i = 0; i < nr; i++) v0[base + i] = eta0[i];
  const kGiven = o.k === undefined ? undefined : rationalTol(o.k, undefined, 'k');
  const omegaD = eng.physics.omegaD;
  const gates = gridGates(eng, eta0, kGiven, DQ.scale2(omegaD, -1));
  if (DQ.cmp(gates.opErr, errTol) > 0 || DQ.cmp(gates.minLayer, layerCells) < 0) throw new Error(
    `this grid cannot carry the mode being asked about. At nr = ${nr}, nz = ${nz}, m = ${m}: the discrete surface operator misses `
    + `-k^2 eta by ${DQ.toNumberForDisplay(gates.opErr).toExponential(3)} against ${DQ.toNumberForDisplay(errTol)}, and the Stokes layer gets `
    + `${DQ.toNumberForDisplay(gates.minLayer).toFixed(2)} cells against ${DQ.toNumberForDisplay(layerCells)} required. Refusing rather than `
    + `returning a multiplier whose damping comes out of the truncation error.`);

  const H = Array.from({ length: krylovCap + 1 }, () => new Array(krylovCap).fill(DQ.ZERO));
  const breakdownTol = DQ.store(1n, -110, 1n);
  let done = 0, breakdown = false;
  eng.arnoldiStart(v0);
  const extend = target => {
    for (let j = done; j < target; j++){
      eng.arnoldiExtend(j);
      const col = eng.hessColumn(j, j + 2);
      for (let i = 0; i < j + 2; i++) H[i][j] = col[i];
      const hjj = DQ.abs(H[j][j]), big = DQ.cmp(hjj, DQ.ONE) > 0 ? hjj : DQ.ONE;
      done = j + 1;
      if (DQ.cmp(H[j + 1][j], DQ.mul(breakdownTol, big)) <= 0){ breakdown = true; break; }
    }
  };
  const ladderOf = used => {
    const l = [];
    for (const kk of [used - 8, used - 4, used]){
      if (kk < 2) continue;
      const lead = hessenbergEigs(H, kk)[0];
      l.push({ krylov: kk, muMax: cabs(lead) });
    }
    return l;
  };
  const driftOf = l => l.length > 1 ? DQ.abs(DQ.sub(DQ.div(l[l.length - 1].muMax, l[l.length - 2].muMax), DQ.ONE)) : null;
  extend(krylov);
  let ladder = ladderOf(done), drift = driftOf(ladder);
  const escalation = [{ krylov: done, drift }];
  while (!(drift && DQ.cmp(drift, krylovTol) <= 0) && !breakdown && done < krylovCap){
    extend(Math.min(krylovCap, done + krylovStep));
    ladder = ladderOf(done); drift = driftOf(ladder);
    escalation.push({ krylov: done, drift });
  }
  const used = done;
  if (!(drift && DQ.cmp(drift, krylovTol) <= 0) && !breakdown) throw new Error(
    `disc Floquet has not converged in the Krylov dimension: the leading modulus reads `
    + ladder.map(l => `${DQ.toNumberForDisplay(l.muMax).toFixed(8)} at k=${l.krylov}`).join(', ')
    + ` at the cap of ${krylovCap}. Raise krylovCap.`);
  const orthogonality = eng.orthogonality(breakdown ? used : used + 1);
  const Hm = H.slice(0, used).map(row => row.slice(0, used));
  const eigs = hessenbergEigs(H, used);
  const muMax = cabs(eigs[0]);
  const moduli = ladder.map(l => l.muMax);
  const spread = moduli.length > 1
    ? DQ.scale2(DQ.sub(moduli.reduce((a, b) => DQ.cmp(a, b) >= 0 ? a : b), moduli.reduce((a, b) => DQ.cmp(a, b) <= 0 ? a : b)), -1)
    : DQ.ZERO;
  const growth = DQ.div(DQ.ln(muMax), eng.Td);
  const growthWidth = DQ.isZero(spread) ? DQ.ZERO : DQ.abs(DQ.div(DQ.ln(DQ.add(DQ.ONE, DQ.div(spread, muMax))), eng.Td));
  const unstable = DQ.cmp(DQ.sub(muMax, spread), DQ.ONE) > 0 ? true : (DQ.cmp(DQ.add(muMax, spread), DQ.ONE) < 0 ? false : null);

  let mode = null;
  if (o.vector){
    const lead = eigs[0];
    if (DQ.cmp(DQ.abs(lead.im), DQ.mul(DQ.store(1n, -60, 1n), cabs(lead))) > 0) throw new Error(
      `the leading Floquet multiplier at m = ${m} is complex (${DQ.toNumberForDisplay(lead.re).toExponential(6)} `
      + `${DQ.sign(lead.im) < 0 ? '-' : '+'} ${DQ.toNumberForDisplay(DQ.abs(lead.im)).toExponential(6)}i), so its mode is not `
      + `locked to the drive and has no single periodic shape to return.`);
    const lam = lead.re;
    const y = realEigenvector(Hm, used, lam);
    const x = eng.combine(y);
    const samples = o.samples === undefined ? 32 : requireCount(o.samples, 2, 'samples');
    eng.load(x);
    const steps = eng.steps, frames = [];
    const take = n => { const f = eng.fields(); frames.push({ step: n, t: DQ.mul(eng.dt, DQ.fromBigInt(BigInt(n))), eta: f.eta, u: f.u, v: f.v, w: f.w }); };
    take(0);
    let at = 0;
    for (let k = 1; k <= samples; k++){
      const target = Number((2n*BigInt(k)*BigInt(steps) + BigInt(samples))/(2n*BigInt(samples)));
      if (target > at){ eng.run(at, target); at = target; }
      take(target);
    }
    const end = eng.store();
    let r2 = DQ.ZERO;
    for (let q = 0; q < nVec; q++){ const e = DQ.sub(end[q], DQ.mul(lam, x[q])); r2 = DQ.add(r2, DQ.mul(e, e)); }
    mode = { multiplier: lam, vector: x, ritzResidual: DQ.div(DQ.sqrt(r2), DQ.abs(lam)), frames, href: eng.href, uref: eng.uref, samples,
             grid: { rc: eng.grid.rc, rf: eng.grid.rf, zc: eng.grid.zc, zf: eng.grid.zf } };
  }
  return { eigs, muMax, growth, growthWidth, muSpread: spread, unstable, Td: eng.Td, dt: eng.dt, steps: eng.steps, m, mode,
           krylov: used, krylovCap, breakdown, ladder, drift, escalation, stateSize: nVec, gates, Hm, orthogonality };
}

const FARADAY_FLOQUET_D128 = Object.freeze({ floquetDisc, hessenbergEigs, realEigenvector, gridGates, surfaceOperator,
                                             complex: Object.freeze({ cx, cadd, csub, cmul, cdiv, cabs, csqrt, cnorm2 }) });
if (typeof module !== 'undefined' && module.exports) module.exports = FARADAY_FLOQUET_D128;
if (typeof globalThis !== 'undefined') globalThis.FARADAY_FLOQUET_D128 = FARADAY_FLOQUET_D128;
