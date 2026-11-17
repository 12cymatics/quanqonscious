import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const DY = require(join(HERE, 'dyadic128.js'));
const E = require(join(HERE, 'faraday-disc-d128.js'));
const FD = require(join(HERE, 'faraday-disc.js'));
const FQ = require(join(HERE, 'faraday-floquet.js'));
const SCAN = require(join(HERE, 'wasm-float-scan.js'));

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}

const CELL = { R: '12.125e-3', h: '3e-3', rho: '998.2041322005837', nu: '1.0035510043427237e-6',
               gamma: '0.07273614042160757', g: '9.80665', accel: '7.0608', omegaD: '697.4335690969883' };
const STRETCH = { rStretch: '2.2', zStretch: '2.2' };

console.log('1. the committed module is the committed source, and holds no floating point');
const work = mkdtempSync(join(tmpdir(), 'disc-d128-'));
const freshPath = join(work, 'faraday_disc_d128.wasm');
execFileSync('sh', [join(HERE, 'build-wasm-d128.sh'), freshPath], { encoding: 'utf8' });
const fresh = new Uint8Array(readFileSync(freshPath)), committed = new Uint8Array(readFileSync(join(HERE, 'faraday_disc_d128.wasm')));
rmSync(work, { recursive: true, force: true });
for (const [label, bytes] of [['committed', committed], ['fresh', fresh]]){
  ok(WebAssembly.Module.imports(new WebAssembly.Module(bytes)).length === 0, `the ${label} module imports nothing`);
  const s = SCAN.scanWasm(bytes);
  ok(s.instructions > 10000 && s.floatInstructions === 0 && s.floatTypes === 0,
    `the ${label} module has no f32 or f64 instruction, local, global or signature in ${s.instructions} instructions`, JSON.stringify(s.floatOpcodes));
}
console.log(`       byte-identical rebuild: ${fresh.length === committed.length && fresh.every((b, i) => b === committed[i])}`);

const toD = x => { if (x === 0) return DY.ZERO; let mm = Math.abs(x), e2 = 0; while (!Number.isInteger(mm)){ mm *= 2; e2++; } return DY.store(BigInt(Math.sign(x)*mm), -e2, 1n); };
function seedVector(nr, nz, n){
  const v = new Float64Array(n), base = 2*(nr - 1)*nz + nr*nz;
  for (let i = 0; i < nr; i++) v[base + i] = Math.cos(Math.PI*(i + 0.5)/nr);
  for (let q = 0; q < base; q++) v[q] = 1e-3*Math.sin(0.37*q);
  return v;
}

console.log('2. one drive period agrees with the float solver to that solver\'s own accuracy');
{
  const cases = [[28, 14, 12, 'pinned'], [24, 12, 1, 'free'], [24, 12, 0, 'pinned'], [24, 12, 3, 'free'], [20, 10, 2, 'pinned']];
  for (const [nr, nz, m, contact] of cases){
    const S = new FD.FaradayDisc({ m, nr, nz, R: +CELL.R, h: +CELL.h, rho: +CELL.rho, nu: +CELL.nu, gamma: +CELL.gamma, g: +CELL.g,
                                   accel: +CELL.accel, omegaD: +CELL.omegaD, contact });
    const Td = 2*Math.PI/(+CELL.omegaD), steps = Math.max(1, Math.round(Td/S.stableStep(0.4)));
    const eng = E.createDiscD128({ m, nr, nz, ...CELL, contact, ...STRETCH, steps, kMax: 2 });
    const own = E.createDiscD128({ m, nr, nz, ...CELL, contact, ...STRETCH, kMax: 2 });
    ok(own.steps === steps, `${nr}x${nz}, m = ${m}: the stability limit computed without floats gives ${own.steps} steps per period, the float solver's ${steps}`);
    const vecF = seedVector(nr, nz, eng.nVec), outF = new Float64Array(eng.nVec);
    FQ.applyDiscPeriodMap(S, vecF, steps, Td/steps, outF, 1e-9, 1e-9*(+CELL.omegaD)/2);
    const outD = eng.applyPeriodMap(Array.from(vecF, toD));
    let scale = 0, worst = 0;
    for (let q = 0; q < eng.nVec; q++) scale = Math.max(scale, Math.abs(outF[q]));
    for (let q = 0; q < eng.nVec; q++) worst = Math.max(worst, Math.abs(DY.toNumberForDisplay(outD[q]) - outF[q])/scale);
    ok(worst <= 1e-11, `${nr}x${nz}, m = ${m}, ${contact} contact line, ${steps} steps: every one of ${eng.nVec} unknowns within ${worst.toExponential(2)} of the float solver, whose conjugate gradient stops at 1e-11`);
    const engFresh = E.createDiscD128({ m, nr, nz, ...CELL, contact, ...STRETCH, steps, kMax: 2, bytes: fresh });
    const outFresh = engFresh.applyPeriodMap(Array.from(vecF, toD));
    ok(outFresh.every((x, q) => DY.eq(x, outD[q])), `the module rebuilt from source and the committed module agree bit for bit on all ${eng.nVec} unknowns of that period`);
  }
}

const gcd = (a, b) => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
const reducedField = {
  mk(n, d = 1n){ if (d < 0n){ n = -n; d = -d; } const g = gcd(n, d) || 1n; return { n: n/g, d: d/g }; }
};
const lazyField = {
  mk(n, d = 1n){ return d < 0n ? { n: -n, d: -d } : { n, d }; }
};
function field(base){
  const mk = base.mk;
  const F = {
    mk,
    add: (a, b) => a.d === b.d ? mk(a.n + b.n, a.d) : mk(a.n*b.d + b.n*a.d, a.d*b.d),
    sub: (a, b) => a.d === b.d ? mk(a.n - b.n, a.d) : mk(a.n*b.d - b.n*a.d, a.d*b.d),
    mul: (a, b) => mk(a.n*b.n, a.d*b.d),
    div: (a, b) => mk(a.n*b.d, a.d*b.n),
    neg: a => ({ n: -a.n, d: a.d }),
    of: x => x.e >= 0 ? mk(x.m << BigInt(x.e)) : mk(x.m, 1n << BigInt(-x.e))
  };
  F.Z = mk(0n); F.ONE = mk(1n); F.TWO = mk(2n); F.HALF = mk(1n, 2n);
  return F;
}
const QR = field(reducedField), QL = field(lazyField);
const qAbs = a => a.n < 0n ? { n: -a.n, d: a.d } : a;
const qCmp = (a, b) => { const z = a.n*b.d - b.n*a.d; return z < 0n ? -1 : z > 0n ? 1 : 0; };
const log2 = q => q.n === 0n ? -Infinity : (q.n.toString(2).length - q.d.toString(2).length);

function discOperators(eng, K){
  const { nr, nz, m } = eng, G = {};
  for (const k of ['rf', 'zf', 'rc', 'zc', 'drc', 'dzc', 'drf', 'dzf']) G[k] = eng.grid[k].map(K.of);
  const iu = (i, j) => i*nz + j, iw = (i, j) => i*(nz + 1) + j, ip = (i, j) => i*nz + j;
  const mq = K.mk(BigInt(m));
  const { add: qa, sub: qs, mul: qm, div: qd, neg: qneg, Z, HALF } = K;
  const divergence = (U, V, W) => {
    const out = [];
    for (let i = 0; i < nr; i++) for (let j = 0; j < nz; j++){
      const vbar = qm(HALF, qa(V[iu(i, j)], V[iu(i + 1, j)]));
      out.push(qa(qa(qm(G.dzc[j], qs(qm(G.rf[i + 1], U[iu(i + 1, j)]), qm(G.rf[i], U[iu(i, j)]))),
                     qm(qm(G.rc[i], G.drc[i]), qs(W[iw(i, j + 1)], W[iw(i, j)]))), qm(qm(qm(mq, G.drc[i]), G.dzc[j]), vbar)));
    }
    return out;
  };
  const gradient = q => {
    const gu = new Array((nr + 1)*nz).fill(Z), gv = new Array((nr + 1)*nz).fill(Z), gw = new Array(nr*(nz + 1)).fill(Z);
    for (let i = 0; i < nr; i++) for (let j = 0; j < nz; j++){
      const qc = q[ip(i, j)];
      if (qc.n === 0n) continue;
      gu[iu(i + 1, j)] = qa(gu[iu(i + 1, j)], qm(qc, G.dzc[j])); gu[iu(i, j)] = qs(gu[iu(i, j)], qm(qc, G.dzc[j]));
      gw[iw(i, j + 1)] = qa(gw[iw(i, j + 1)], qm(qc, qm(G.rc[i], G.drc[i]))); gw[iw(i, j)] = qs(gw[iw(i, j)], qm(qc, qm(G.rc[i], G.drc[i])));
      const t = qm(qm(qm(qc, mq), qm(G.drc[i], G.dzc[j])), HALF);
      gv[iu(i, j)] = qa(gv[iu(i, j)], t); gv[iu(i + 1, j)] = qa(gv[iu(i + 1, j)], t);
    }
    for (let i = 1; i < nr; i++){
      const half = qm(HALF, qa(G.drc[i - 1], G.drc[i]));
      for (let j = 0; j < nz; j++){
        gu[iu(i, j)] = qd(gu[iu(i, j)], qneg(qm(G.dzc[j], G.drf[i])));
        gv[iu(i, j)] = qd(gv[iu(i, j)], qneg(qm(qm(G.rf[i], G.dzc[j]), half)));
      }
    }
    for (let j = 0; j < nz; j++){
      if (m === 1){ const ga = qd(gv[iu(0, j)], qm(qm(G.rc[0], G.rc[0]), G.dzc[j])); gu[iu(0, j)] = ga; gv[iu(0, j)] = qneg(ga); }
      else { gu[iu(0, j)] = Z; gv[iu(0, j)] = Z; }
      gu[iu(nr, j)] = Z; gv[iu(nr, j)] = Z;
    }
    for (let i = 0; i < nr; i++){ gw[iw(i, 0)] = Z; for (let j = 1; j <= nz; j++) gw[iw(i, j)] = qd(gw[iw(i, j)], qneg(qm(qm(G.rc[i], G.drc[i]), G.dzf[j]))); }
    return { gu, gv, gw };
  };
  return { G, iu, iw, ip, divergence, gradient };
}

function bareissSolve(A, rhs){
  const N = rhs.length, M = [];
  for (let r = 0; r < N; r++){
    let l = 1n;
    for (const x of A[r].concat([rhs[r]])) if (x.n !== 0n) l = l/gcd(l, x.d)*x.d;
    M.push(A[r].concat([rhs[r]]).map(x => x.n*(l/x.d)));
  }
  let prev = 1n;
  for (let k = 0; k < N - 1; k++){
    const akk = M[k][k];
    if (akk === 0n) throw new Error(`bareissSolve: zero leading minor at ${k}; the operator is definite, so this cannot happen.`);
    for (let i = k + 1; i < N; i++){
      const aik = M[i][k], row = M[i], top = M[k];
      for (let j = k + 1; j <= N; j++){
        const aij = row[j], akj = top[j];
        if (aik === 0n){ if (aij !== 0n) row[j] = akk*aij/prev; }
        else row[j] = (akk*aij - aik*akj)/prev;
      }
      row[k] = 0n;
    }
    prev = akk;
  }
  const det = M[N - 1][N - 1], X = new Array(N);
  for (let k = N - 1; k >= 0; k--){
    let s = det*M[k][N];
    for (let j = k + 1; j < N; j++) if (M[k][j] !== 0n) s -= M[k][j]*X[j];
    X[k] = s/M[k][k];
  }
  return X.map(x => QL.mk(x, det));
}

function exactStep(eng, before, gNow){
  const { nr, nz, m } = eng, K = QR;
  const { G, iu, iw, ip, divergence, gradient } = discOperators(eng, K);
  const { add: qa, sub: qs, mul: qm, div: qd, neg: qneg, Z, ONE, TWO, HALF } = K;
  const rho = K.of(eng.physics.rho), nu = K.of(eng.physics.nu), gam = K.of(eng.physics.gamma), dt = K.of(eng.dt), g = K.of(gNow);
  const u = before.u.map(K.of), v = before.v.map(K.of), w = before.w.map(K.of), eta = before.eta.map(K.of);
  const mq = K.mk(BigInt(m)), pinned = eng.contact === 'pinned';
  const wzSurface = (W, i) => {
    const w0 = W[iw(i, nz)], w1 = W[iw(i, nz - 1)], w2 = W[iw(i, nz - 2)];
    const a = qs(G.zf[nz], G.zf[nz - 1]), b = qs(G.zf[nz], G.zf[nz - 2]);
    return qa(qs(qd(qm(w0, qa(a, b)), qm(a, b)), qd(qm(w1, b), qm(a, qs(b, a)))), qd(qm(w2, a), qm(b, qs(b, a))));
  };
  const surfaceLaplacian = i => {
    const outer = i === nr - 1 ? (pinned ? qd(qm(G.rf[nr], qs(Z, eta[nr - 1])), G.drf[nr]) : Z)
                               : qd(qm(G.rf[i + 1], qs(eta[i + 1], eta[i])), G.drf[i + 1]);
    const inner = i === 0 ? Z : qd(qm(G.rf[i], qs(eta[i], eta[i - 1])), G.drf[i]);
    return qs(qd(qs(outer, inner), qm(G.rc[i], G.drc[i])), qd(qm(qm(mq, mq), eta[i]), qm(G.rc[i], G.rc[i])));
  };
  const dudz = [], dvdz = [];
  for (let i = 0; i <= nr; i++){
    const wS = k => w[iw(k, nz)];
    if (i === 0 && m === 1){ dudz[0] = qneg(qd(wS(0), G.drf[0])); dvdz[0] = qneg(dudz[0]); continue; }
    if (i === 0 || i === nr){ dudz[i] = Z; dvdz[i] = Z; continue; }
    dudz[i] = qneg(qd(qs(wS(i), wS(i - 1)), G.drf[i]));
    const wAt = qd(qa(qm(G.drc[i - 1], wS(i)), qm(G.drc[i], wS(i - 1))), qa(G.drc[i - 1], G.drc[i]));
    dvdz[i] = qm(qd(mq, G.rf[i]), wAt);
  }
  const lapUV = (f, other, top) => {
    const out = new Array(f.length).fill(Z);
    for (let i = 1; i < nr; i++){
      const r = G.rf[i], wIn = qd(G.rc[i - 1], G.drc[i - 1]), wOut = qd(G.rc[i], G.drc[i]);
      const vol = qm(qm(r, HALF), qa(G.drc[i - 1], G.drc[i]));
      for (let j = 0; j < nz; j++){
        const c = iu(i, j), fc = f[c];
        const radial = qd(qs(qm(wOut, qs(f[iu(i + 1, j)], fc)), qm(wIn, qs(fc, f[iu(i - 1, j)]))), vol);
        const below = j === 0 ? qd(fc, G.dzf[0]) : qd(qs(fc, f[iu(i, j - 1)]), G.dzf[j]);
        const above = j === nz - 1 ? top[i] : qd(qs(f[iu(i, j + 1)], fc), G.dzf[j + 1]);
        const coupling = qneg(qd(qa(qm(qa(qm(mq, mq), ONE), fc), qm(qm(TWO, mq), other[c])), qm(r, r)));
        out[c] = qa(qa(radial, qd(qs(above, below), G.dzc[j])), coupling);
      }
    }
    return out;
  };
  const lu = lapUV(u, v, dudz), lv = lapUV(v, u, dvdz);
  if (m === 1) for (let j = 0; j < nz; j++){
    const c = iu(0, j), b = iu(1, j), a = u[c];
    const radial = qd(qs(qs(u[b], v[b]), qm(TWO, a)), qm(G.drc[0], G.rc[0]));
    const below = j === 0 ? qd(a, G.dzf[0]) : qd(qs(a, u[iu(0, j - 1)]), G.dzf[j]);
    const above = j === nz - 1 ? dudz[0] : qd(qs(u[iu(0, j + 1)], a), G.dzf[j + 1]);
    lu[c] = qa(radial, qd(qs(above, below), G.dzc[j])); lv[c] = qneg(lu[c]);
  }
  const lw = new Array(w.length).fill(Z);
  for (let i = 0; i < nr; i++){
    const r = G.rc[i], vol = qm(r, G.drc[i]);
    for (let j = 1; j <= nz; j++){
      const c = iw(i, j), wc = w[c];
      const fluxIn = i === 0 ? Z : qd(qm(G.rf[i], qs(wc, w[iw(i - 1, j)])), G.drf[i]);
      const fluxOut = i === nr - 1 ? qd(qm(G.rf[nr], qs(Z, wc)), G.drf[nr]) : qd(qm(G.rf[i + 1], qs(w[iw(i + 1, j)], wc)), G.drf[i + 1]);
      const below = j === 1 ? qd(wc, G.dzc[0]) : qd(qs(wc, w[iw(i, j - 1)]), G.dzc[j - 1]);
      const above = j === nz ? qa(below, qm(TWO, qs(wzSurface(w, i), below))) : qd(qs(w[iw(i, j + 1)], wc), G.dzc[j]);
      lw[c] = qs(qa(qd(qs(fluxOut, fluxIn), vol), qd(qs(above, below), G.dzf[j])), qd(qm(qm(mq, mq), wc), qm(r, r)));
    }
  }
  const i0 = m === 1 ? 0 : 1, dtnu = qm(dt, nu);
  const us = u.slice(), vs = v.slice(), ws = w.slice();
  for (let i = i0; i < nr; i++) for (let j = 0; j < nz; j++){ const c = iu(i, j); us[c] = qa(u[c], qm(dtnu, lu[c])); vs[c] = qa(v[c], qm(dtnu, lv[c])); }
  for (let i = 0; i < nr; i++) for (let j = 1; j <= nz; j++){ const c = iw(i, j); ws[c] = qa(w[c], qm(dtnu, lw[c])); }
  const ps = [];
  for (let i = 0; i < nr; i++) ps.push(qa(qs(qm(qm(rho, g), eta[i]), qm(gam, surfaceLaplacian(i))), qm(qm(qm(TWO, rho), nu), wzSurface(w, i))));
  const rhs = divergence(us, vs, ws).map(x => qd(qm(x, rho), dt));
  for (let i = 0; i < nr; i++) rhs[ip(i, nz - 1)] = qs(rhs[ip(i, nz - 1)], qd(qm(qm(G.rc[i], G.drc[i]), ps[i]), G.dzf[nz]));
  const N = nr*nz, A = Array.from({ length: N }, () => new Array(N).fill(Z));
  for (let c = 0; c < N; c++){
    const e = new Array(N).fill(Z); e[c] = ONE;
    const gr = gradient(e);
    const col = divergence(gr.gu, gr.gv, gr.gw);
    for (let r = 0; r < N; r++) A[r][c] = col[r];
  }
  const p = bareissSolve(A, rhs);
  const L = QL, post = discOperators(eng, L), toL = x => L.mk(x.n, x.d);
  const gr = post.gradient(p);
  for (let i = 0; i < nr; i++) gr.gw[iw(i, nz)] = L.add(gr.gw[iw(i, nz)], L.div(toL(ps[i]), post.G.dzf[nz]));
  const dtr = L.div(toL(dt), toL(rho));
  const un = u.map(toL), vn = v.map(toL), wn = w.map(toL);
  for (let i = i0; i < nr; i++) for (let j = 0; j < nz; j++){ const c = iu(i, j); un[c] = L.sub(toL(us[c]), L.mul(dtr, gr.gu[c])); vn[c] = L.sub(toL(vs[c]), L.mul(dtr, gr.gv[c])); }
  for (let i = 0; i < nr; i++) for (let j = 1; j <= nz; j++){ const c = iw(i, j); wn[c] = L.sub(toL(ws[c]), L.mul(dtr, gr.gw[c])); }
  for (let i = 0; i < nr; i++) wn[iw(i, 0)] = L.Z;
  const en = eta.map((e, i) => L.add(toL(e), L.mul(toL(dt), wn[iw(i, nz)])));
  const div = post.divergence(un, vn, wn);
  return { u: un, v: vn, w: wn, eta: en, p, div };
}

function worstRelative(gotD, exact){
  let scale = QL.Z;
  for (const x of exact){ const a = qAbs(x); if (qCmp(a, scale) > 0) scale = a; }
  let worst = QL.Z;
  for (let k = 0; k < exact.length; k++){ const d = qAbs(QL.sub(QL.of(gotD[k]), exact[k])); if (qCmp(d, worst) > 0) worst = d; }
  return scale.n === 0n ? QL.Z : QL.div(worst, scale);
}

console.log('3. one step equals the exact rational step of the same discrete equations, to the last bits');
{
  for (const [nr, nz, m, contact] of [[6, 4, 2, 'pinned'], [6, 4, 1, 'free'], [7, 4, 0, 'free'], [6, 5, 3, 'pinned']]){
    const eng = E.createDiscD128({ m, nr, nz, ...CELL, contact, ...STRETCH, kMax: 2 });
    const vec = Array.from(seedVector(nr, nz, eng.nVec), toD);
    eng.load(vec);
    const before = eng.fields();
    const gNow = eng.drive()[0];
    const t0 = Date.now();
    const exact = exactStep(eng, before, gNow);
    const tExact = Date.now() - t0;
    eng.run(0, 1);
    const after = eng.fields();
    const errs = ['u', 'v', 'w', 'eta', 'p'].map(k => log2(worstRelative(after[k], exact[k])));
    const worst = Math.max(...errs);
    ok(worst <= -110, `${nr}x${nz}, m = ${m}, ${contact}: u, v, w, eta and p within 2^${worst} of the exact step (largest of 2^${errs.join(', 2^')}; the exact step took ${tExact} ms)`);
    const divScale = exact.u.concat(exact.v, exact.w).reduce((acc, x) => qCmp(qAbs(x), acc) > 0 ? qAbs(x) : acc, QL.Z);
    if (nr === 6 && nz === 4 && m === 2){
      ok(exact.div.every(x => x.n === 0n), 'the exact step leaves the discrete divergence exactly zero, so the projection the solver approximates is the right one');
      const ops = discOperators(eng, QL);
      const dD = ops.divergence(after.u.map(QL.of), after.v.map(QL.of), after.w.map(QL.of));
      let worstDiv = QL.Z;
      for (let i = 0; i < nr; i++) for (let j = 0; j < nz; j++){
        const vol = QL.mul(ops.G.rc[i], QL.mul(ops.G.drc[i], ops.G.dzc[j]));
        const rel = qAbs(QL.div(dD[ops.ip(i, j)], QL.mul(divScale, vol)));
        if (qCmp(rel, worstDiv) > 0) worstDiv = rel;
      }
      ok(log2(worstDiv) <= -110, `the solver's own velocity has discrete divergence 2^${log2(worstDiv)} relative, where the float solver's tolerance is 1e-11`);
    }
  }
}

console.log('4. the pressure operator is read symmetric and factored without a non-positive pivot');
{
  for (const [nr, nz, m] of [[28, 14, 12], [24, 12, 1], [24, 12, 0], [20, 10, 2]]){
    const eng = E.createDiscD128({ m, nr, nz, ...CELL, contact: 'pinned', ...STRETCH, kMax: 2 });
    const a = eng.pressureAsymmetry();
    const l2 = DY.isZero(a) ? -Infinity : a.e + 128;
    ok(l2 <= -115, `${nr}x${nz}, m = ${m}: the largest relative difference between an entry and its mirror is ${DY.isZero(a) ? 'zero' : '2^' + l2}`);
  }
}

console.log('5. without a drive the energy of the perturbation falls');
{
  const nr = 20, nz = 10, m = 2;
  const eng = E.createDiscD128({ m, nr, nz, ...CELL, accel: '0', contact: 'pinned', ...STRETCH, kMax: 2 });
  const energy = f => {
    const G = eng.grid, rho = eng.physics.rho, g = eng.physics.g, gam = eng.physics.gamma;
    let ke = DY.ZERO, pe = DY.ZERO;
    for (let i = 1; i < nr; i++){
      const vol = DY.mul(G.rf[i], DY.scale2(DY.add(G.drc[i - 1], G.drc[i]), -1));
      for (let j = 0; j < nz; j++){ const a = f.u[i*nz + j], b = f.v[i*nz + j]; ke = DY.add(ke, DY.mul(DY.add(DY.mul(a, a), DY.mul(b, b)), DY.mul(vol, G.dzc[j]))); }
    }
    for (let i = 0; i < nr; i++) for (let j = 1; j <= nz; j++){ const c = f.w[i*(nz + 1) + j]; ke = DY.add(ke, DY.mul(DY.mul(c, c), DY.mul(DY.mul(G.rc[i], G.drc[i]), G.dzf[j]))); }
    ke = DY.scale2(DY.mul(rho, ke), -1);
    for (let i = 0; i < nr; i++){
      const e = f.eta[i], eOut = i === nr - 1 ? DY.ZERO : f.eta[i + 1];
      const slope = DY.div(DY.sub(eOut, e), G.drf[i + 1]);
      const area = DY.mul(G.rc[i], G.drc[i]);
      pe = DY.add(pe, DY.mul(DY.mul(DY.mul(rho, g), DY.mul(e, e)), area));
      pe = DY.add(pe, DY.mul(gam, DY.mul(DY.add(DY.mul(slope, slope), DY.div(DY.mul(DY.fromBigInt(BigInt(m*m)), DY.mul(e, e)), DY.mul(G.rc[i], G.rc[i]))), area)));
    }
    return DY.add(ke, DY.scale2(pe, -1));
  };
  eng.load(Array.from(seedVector(nr, nz, eng.nVec), toD));
  const series = [energy(eng.fields())];
  const chunk = Math.floor(eng.steps/8);
  for (let k = 0; k < 8; k++){ eng.run(k*chunk, (k + 1)*chunk); series.push(energy(eng.fields())); }
  const falls = series.slice(1).every((x, k) => DY.cmp(x, series[k]) < 0);
  ok(falls, `the energy falls at each of eight sampled points over one undriven period, from ${DY.toNumberForDisplay(series[0]).toExponential(3)} to ${DY.toNumberForDisplay(series[8]).toExponential(3)} J`);
}

console.log('6. refusals');
{
  const base = { m: 2, nr: 20, nz: 10, ...CELL, contact: 'pinned', ...STRETCH, kMax: 2 };
  const refuses = (o, re) => { try { E.createDiscD128(o); return false; } catch (e){ return re.test(e.message); } };
  ok(refuses({ ...base, rho: 998.2 }, /float is not accepted/), 'a float parameter is refused');
  ok(refuses({ ...base, contact: 'slip' }, /'free' or 'pinned'/), 'an unknown contact line is refused');
  ok(refuses({ ...base, steps: 20 }, /stability limit/), 'a step above half the stability limit is refused');
  ok(refuses({ ...base, rStretch: '-1' }, /negative stretch/), 'a negative stretch is refused');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
