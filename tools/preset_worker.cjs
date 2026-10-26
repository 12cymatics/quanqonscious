const { parentPort, workerData } = require('node:worker_threads');
const K = new Function(workerData.src + '\nreturn KERNEL_FACTORY();')();
const { Q, A4 } = K, X = A4;
const N = workerData.steps || 600;
const bl = x => x === 0n ? 0 : (x < 0n ? -x : x).toString(2).length;
const qNum = q => { let n = q.n; if (n === 0n) return 0; const neg = n < 0n; if (neg) n = -n; const sh = BigInt(Math.max(0, 64 + bl(q.d) - bl(n))); const v = Number((n << sh) / q.d) / 2 ** Number(sh); return neg ? -v : v; };
const PHI = (1 + Math.sqrt(5)) / 2;
const probs = psi => { let e = null;
  for (const q of psi) if (q.n !== 0n) { const x = bl(q.n) - bl(q.d); if (e === null || x > e) e = x; }
  const sc = q => q.n === 0n ? q : (e >= 0 ? Q.mk(q.n, q.d << BigInt(e)) : Q.mk(q.n << BigInt(-e), q.d));
  const p = psi.map(q => { const r = qNum(sc(q)); return r * r; }); const t = p.reduce((a, b) => a + b, 0); return p.map(v => v / t); };
const params = x => ({ dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
  gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
  tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.fromQ(Q.mk(BigInt(x.drive), 4n)),
  geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
  polarisation: Q.mk(BigInt(x.chi), 10n), vortex: Q.mk(BigInt(x.omega), 10n) });
const activeOf = x => Object.entries(x.set).map(([k, v]) => { const sid = Number(k), st = Q.mk(BigInt(v), 100n);
  return x.args && x.args[k] ? { id: sid, strength: st, args: [st, ...x.args[k].map(t => Q.parse(t))] } : { id: sid, strength: st }; });
const observe = (eng, p) => {
  const psi = eng.psi, f = psi.map(qNum), E = f.reduce((a, b) => a + b * b, 0);
  const md = K.magData(eng.magA, eng.magChi), T = md.torus;
  const hole = T.hole.reduce((s, q) => s + Math.abs(qNum(q)), 0), tube = T.tube.reduce((s, q) => s + Math.abs(qNum(q)), 0);
  const sym = E === 0 ? 0 : [0, 1, 2, 3, 4, 5, 6, 7].reduce((a, i) => a + (f[i] + f[i ^ 15]) ** 2, 0) / (2 * E);
  let gold = 0, ng = 0;
  for (let a = 0; a < 4; a++) for (let v = 0; v < 16; v++) if (!(v & (1 << a)) && f[v] !== 0) { gold += Math.abs(Math.abs(f[v | (1 << a)] / f[v]) - PHI); ng++; }
  let dr = 0, m7 = 0, nz7 = 0, intS = 0, absM = 0;
  for (const q of psi) {
    const v = K.absFloor(q), d = v === 0n ? 0n : 1n + (v - 1n) % 9n, s = q.n < 0n ? -1 : 1;
    dr += Math.abs(qNum(q) - s * Number(d)) / 16; absM += Math.abs(qNum(q)) / 16;
    const r = Number(v % 7n); m7 += Math.min(r, 7 - r) / 16; if (v > 0n && r === 0) nz7++;
    const fr = Math.abs(qNum(q) - Math.round(qNum(q))); if (fr <= 1e-3) intS += 1 / 16;
  }
  const rho = K.laplacianV(eng.dielectric).reduce((s, q) => s + Math.abs(qNum(q)), 0);
  const tot = f.reduce((x, y) => x + y, 0), spread = E === 0 ? 0 : 1 - tot * tot / (16 * E);
  const e = K.energy(psi);
  return { E, Eb: bl(e.n) - bl(e.d), spread, hole, tube, flux: hole + tube, H: qNum(md.helicity), rec: md.reconnections.length,
           sym, gold: ng ? gold / ng : NaN, dr, absM, m7, nz7, int: intS, rho, prob: probs(psi) };
};
const out = {};
for (const [id, x] of workerData.presets) {
  const p = params(x), act = activeOf(x); let eng = K.initEngine(); const S = []; let halted = null; const t0 = Date.now();
  const E0 = qNum(K.energy(eng.psi)), n = x.steps || N;
  for (let s = 0; s < n; s++) {
    try { eng = K.step(eng, p, act, x.comp); } catch (err) { halted = { step: s + 1, name: err.name, message: err.message, where: typeof err.where === 'function' ? err.where() : null }; break; }
    S.push(observe(eng, p));
  }
  const ms = (Date.now() - t0) / Math.max(1, S.length);
  if (halted) { out[id] = { halted, ms, halt: halted.step, haltWhy: halted.message }; continue; }
  if (x.steps) { out[id] = { ms, Ebits: S[n - 1].Eb, EbitsPrev: S[n - 2].Eb }; continue; }
  const at = s => S[s - 1], avgOf = (k, a, b) => S.slice(a, b).reduce((t, o) => t + o[k], 0) / (b - a);
  const avgP = (a, b) => { const r = new Array(16).fill(0); for (let i = a; i < b; i++) S[i].prob.forEach((v, k) => r[k] += v / (b - a)); return r; };
  const supp = w => { const o = [...w.keys()].sort((a, b) => w[b] - w[a]); const R = []; let c = 0; for (const v of o) { R.push(v); c += w[v]; if (c >= 0.8) break; } return R.sort((a, b) => a - b).map(v => v.toString(2).padStart(4, '0')); };
  const L1 = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
  let Hflips = 0; for (let i = 1; i < N; i++) if (Math.sign(S[i].H) !== Math.sign(S[i - 1].H) && S[i].H !== 0 && S[i - 1].H !== 0) Hflips++;
  let rhoMax = 0, rhoArg = 0; S.forEach((o, i) => { if (o.rho > rhoMax) { rhoMax = o.rho; rhoArg = i + 1; } });
  let Emin = Infinity, Emax = 0; S.forEach(o => { Emin = Math.min(Emin, o.E); Emax = Math.max(Emax, o.E); });
  const late = avgP(N - 60, N);
  out[id] = { ms, lateCount: supp(late).length, spreadEarly: avgOf('spread', 0, 20), spreadLate: avgOf('spread', N - 60, N), maxShare: Math.max(...late), E0, E1: at(1).E, E60: at(60).E, E300: at(300).E, E600: at(N).E, Emin, Emax,
    flux1: at(1).flux, flux60: at(60).flux, flux300: at(300).flux, flux600: at(N).flux, hole600: at(N).hole, tube600: at(N).tube,
    H60: at(60).H, H300: at(300).H, H600: at(N).H, Hflips, rec: S.reduce((t, o) => t + o.rec, 0),
    symEarly: avgOf('sym', 0, 20), symLate: avgOf('sym', N - 60, N), goldEarly: avgOf('gold', 0, 20), goldLate: avgOf('gold', N - 60, N),
    drEarly: avgOf('dr', 0, 20), drLate: avgOf('dr', N - 60, N), absEarly: avgOf('absM', 0, 20), absLate: avgOf('absM', N - 60, N),
    m7Early: avgOf('m7', 0, 20), m7Late: avgOf('m7', N - 60, N), nz7Late: avgOf('nz7', N - 60, N), intEarly: avgOf('int', 0, 20), intLate: avgOf('int', N - 60, N),
    rho1: at(1).rho, rhoMax, rhoArg, rho600: at(N).rho, early: supp(avgP(20, 40)), late: supp(late), moving: L1(avgP(N - 300, N - 240), late) };
}
parentPort.postMessage(out);
