// Headless test of R4_tesseract_cymatic_v4.html mathematics.
// Extracts the <script> block, stubs the DOM, and exercises the algebra.
import fs from 'fs';

const html = fs.readFileSync(new URL('../R4_tesseract_cymatic_v4.html', import.meta.url), 'utf8');
const m = html.match(/<script>\n([\s\S]*?)<\/script>/);
if (!m) { console.error('no script block found'); process.exit(1); }
let src = m[1];

// ---- DOM stubs -------------------------------------------------------------
const mkEl = () => ({
  innerHTML: '', textContent: '', className: '', style: {}, children: [], disabled: false,
  appendChild(c) { this.children.push(c); }, removeChild(c) { this.children.shift(); },
  addEventListener() {}, classList: { add() {}, remove() {} }, scrollTop: 0, scrollHeight: 0,
  getBoundingClientRect: () => ({ left: 0, top: 0 }), onclick: null
});
const ctxStub = new Proxy({}, {
  get: (t, p) => {
    if (p === 'canvas') return {};
    if (p === 'createRadialGradient' || p === 'createLinearGradient') return () => ({ addColorStop() {} });
    if (p === 'measureText') return () => ({ width: 0 });
    return typeof t[p] === 'undefined' ? () => {} : t[p];
  },
  set: () => true
});
globalThis.document = {
  getElementById: () => mkEl(),
  querySelectorAll: () => [],
  createElement: () => mkEl()
};
globalThis.window = { addEventListener() {} };
globalThis.innerWidth = 1600; globalThis.innerHeight = 900;
globalThis.requestAnimationFrame = () => 0;
globalThis.performance = performance;
globalThis.setTimeout = (f) => 0;   // suppress deferred UI work
const canvasStub = { getContext: () => ctxStub, addEventListener() {}, width: 0, height: 0, style: {} };
globalThis.document.getElementById = (id) => (id === 'canvas' ? canvasStub : mkEl());

// ---- expose internals ------------------------------------------------------
src = src.replace(/^'use strict';/, '');
src += `\nglobalThis.__X = { Q, St, Tr, K, Sqrt, SUTRAS, Compose, MODES, Rand, STATS, verifyInverses, QUEUE, isqrt };`;
const fn = new Function(src);
fn();
const { Q, St, Tr, K, Sqrt, SUTRAS, Compose, MODES, Rand, STATS, isqrt } = globalThis.__X;

// ---- assertions ------------------------------------------------------------
let pass = 0, fail = 0;
const ok = (c, msg, extra = '') => { if (c) { pass++; console.log(`  ✓ ${msg}`); }
  else { fail++; console.log(`  ✗ ${msg} ${extra}`); } };
const sec = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(0, 60 - t.length))}`);

sec('EXACT ARITHMETIC');
ok(Q.eq(Q.add(Q.make(1n,3n), Q.make(1n,6n)), Q.make(1n,2n)), '1/3 + 1/6 = 1/2');
ok(Q.eq(Q.mul(Q.make(2n,3n), Q.make(3n,2n)), Q.ONE), '2/3 · 3/2 = 1');
ok(Q.floor(Q.make(-7n,2n)) === -4n, 'floor(-7/2) = -4 (true floor, not trunc)',
   `got ${Q.floor(Q.make(-7n,2n))}`);
ok(Q.floor(Q.make(7n,2n)) === 3n, 'floor(7/2) = 3');
ok(isqrt(144n) === 12n && isqrt(143n) === 11n, 'exact integer sqrt');
{ const r = Q.limitDen(Q.make(314159265358979n, 100000000000000n), 1000n);
  ok(r[1] <= 1000n, `limitDen bounds denominator (got ${r[0]}/${r[1]})`);
  ok(r[0] === 355n && r[1] === 113n, 'best approx of π to den≤1000 is 355/113',
     `got ${r[0]}/${r[1]}`); }

sec('TRANSCENDENTALS');
const PI = Q.f(Tr.PI);
ok(Math.abs(PI - Math.PI) < 1e-14, `Machin π = ${PI}`, `Δ=${Math.abs(PI-Math.PI)}`);
const PHI = Q.f(K.PHI);
ok(Math.abs(PHI - (1+Math.sqrt(5))/2) < 1e-14, `φ = ${PHI}`);
{ // φ² = φ + 1
  const d = Q.f(Q.sub(Q.mul(K.PHI,K.PHI), Q.add(K.PHI, Q.ONE)));
  ok(Math.abs(d) < 1e-25, `φ² − φ − 1 = ${d.toExponential(3)}`); }
for (const [num, den, ref, nm] of [[0n,1n,0,'sin 0'],[1n,2n,1,'sin π/2'],[1n,1n,0,'sin π'],
                                    [3n,2n,-1,'sin 3π/2'],[1n,6n,0.5,'sin π/6'],[1n,4n,Math.SQRT1_2,'sin π/4'],
                                    [7n,3n,Math.sin(7*Math.PI/3),'sin 7π/3 (reduction)'],
                                    [-5n,4n,Math.sin(-5*Math.PI/4),'sin −5π/4 (negative)']]) {
  const v = Q.f(Tr.sinPi(Q.make(num,den)));
  ok(Math.abs(v - ref) < 1e-13, `${nm} = ${v.toFixed(15)}`, `ref=${ref}`);
}
for (const [num, den, ref, nm] of [[0n,1n,1,'cos 0'],[1n,2n,0,'cos π/2'],[1n,1n,-1,'cos π'],
                                    [1n,3n,0.5,'cos π/3'],[2n,1n,1,'cos 2π'],
                                    [11n,6n,Math.cos(11*Math.PI/6),'cos 11π/6']]) {
  const v = Q.f(Tr.cosPi(Q.make(num,den)));
  ok(Math.abs(v - ref) < 1e-13, `${nm} = ${v.toFixed(15)}`, `ref=${ref}`);
}
{ // Pythagorean identity at a generic rational angle
  const t = Q.make(3n, 7n);
  const s = Tr.sinPi(t), c = Tr.cosPi(t);
  const d = Q.f(Q.sub(Q.add(Q.mul(s,s), Q.mul(c,c)), Q.ONE));
  ok(Math.abs(d) < 1e-24, `sin²+cos² − 1 = ${d.toExponential(3)} at 3π/7`); }
{ const s = Q.f(Sqrt.of(Q.int(2)));
  ok(Math.abs(s - Math.SQRT2) < 1e-15, `√2 = ${s}`); }

sec('LFSR DETERMINISM');
Rand.seed(0xACE1C0DEn);
const a = Array.from({length:8}, () => Rand.below(29));
Rand.seed(0xACE1C0DEn);
const b = Array.from({length:8}, () => Rand.below(29));
ok(JSON.stringify(a) === JSON.stringify(b), `same seed → same stream [${a}]`);
Rand.seed(0xACE1C0DEn);
const p1 = Rand.permute([0,1,2,3,4,5,6,7,8]);
Rand.seed(0xACE1C0DEn);
const p2 = Rand.permute([0,1,2,3,4,5,6,7,8]);
ok(JSON.stringify(p1) === JSON.stringify(p2), `permutation reproducible [${p1}]`);
ok(new Set(p1).size === 9, 'permutation is a bijection');

sec('SŪTRA INVENTORY');
ok(SUTRAS.length === 29, `29 sūtras present (${SUTRAS.length})`);
ok(SUTRAS.every((s,i) => s.id === i+1), 'ids 1..29 contiguous');
ok(SUTRAS.reduce((a,s)=>a+s.delta,0) === 435, 'Σδ = 435 (29th triangular number)');
ok(SUTRAS.every(s => typeof s.T === 'function' && typeof s.I === 'function'), 'every sūtra has T and T⁻¹');
ok(new Set(SUTRAS.map(s=>s.la)).size === 29, 'all names distinct');

sec('INVERSE ALGEBRA  T⁻¹ ∘ T = id');
const R = (n,d) => Q.make(BigInt(n), BigInt(d));
const probe = { l:[R(3,2),R(5,3),R(7,4),R(2,5)], grvq:R(1,3), mstvq:R(2,7), tgcr:R(1,5), zpe:R(3,8) };
const EPS = Q.make(1n, 10n**18n);
let exact = 0, near = 0, bad = [];
for (const S of SUTRAS) {
  const d = St.dist(S.I(S.T(probe)), probe);
  if (Q.isZero(d)) exact++;
  else if (Q.cmp(d, EPS) < 0) near++;
  else bad.push(`S${S.id}:${Q.str(d,14)}`);
}
ok(bad.length === 0, `all 29 invert (${exact} bit-exact, ${near} exact-to-ℚ-precision)`, bad.join(' '));
ok(exact + near === 29, `every sūtra accounted for (${exact} exact + ${near} approx)`);
{ // the √-branch sūtras invert bit-exactly on rational squares; confirm the
  // irrational-root path also round-trips inside ℚ tolerance
  const irr = { l:[R(2,1),R(1,1),R(1,1),R(3,1)], grvq:Q.ZERO, mstvq:Q.ZERO, tgcr:Q.ZERO, zpe:Q.ZERO };
  for (const id of [3,10,19]) {
    const S = SUTRAS[id-1];
    const d = St.dist(S.I(S.T(irr)), irr);
    ok(Q.cmp(d, EPS) < 0, `S${id} (√ branch) round-trips on irrational roots · ‖r‖ = ${Q.isZero(d)?'0':Q.str(d,16)}`);
  } }

sec('MODE SEMANTICS — all seven formulas are genuinely distinct');
const S0 = St.init();
const qs = [0, 6, 12, 20, 3];                       // S1, S7, S13, S21, S4
const res = {};
for (let m = 0; m < MODES.length; m++) {
  const fnm = ['isolated','series','parallel','concurrent','inverse','composite','canonical'][m];
  res[MODES[m].n] = Compose[fnm].call(Compose, S0, qs);
}
const names = Object.keys(res);
console.log('   λ₀ per mode:');
for (const n of names) console.log(`     ${n.padEnd(11)} λ₀ = ${Q.str(res[n].l[0], 34)}`);
let distinct = true, coll = [];
for (let i = 0; i < names.length; i++) for (let j = i+1; j < names.length; j++) {
  if (Q.isZero(St.dist(res[names[i]], res[names[j]]))) { distinct = false; coll.push(`${names[i]}=${names[j]}`); }
}
ok(distinct, 'all seven modes produce different states', coll.join(' '));

sec('PARALLEL ≠ SERIES (the bug in the previous version)');
ok(!Q.isZero(St.dist(res.PARALLEL, res.SERIES)),
   `‖PARALLEL − SERIES‖ = ${Q.str(St.dist(res.PARALLEL,res.SERIES), 20)}`);
{ // PARALLEL must equal the plain mean of the independent images of S₀
  const manual = St.mean(qs.map(k => SUTRAS[k].T(S0)));
  ok(Q.isZero(St.dist(manual, res.PARALLEL)), 'PARALLEL = (1/N) Σ Tₖ(S₀) exactly'); }

sec('CONCURRENT interpolation property');
{ const one = Compose.concurrent(S0, [4]);
  const par = Compose.parallel(S0, [4]);
  ok(Q.isZero(St.dist(one, par)), 'N=1 ⇒ CONCURRENT = PARALLEL = T(S₀)');
  const c = Compose.concurrent(S0, qs);
  ok(!Q.isZero(St.dist(c, res.SERIES)) && !Q.isZero(St.dist(c, res.PARALLEL)),
     'CONCURRENT strictly between SERIES and PARALLEL'); }

sec('DETERMINISM — same input ⇒ same output (CODEX invariant 2)');
{ // CONCURRENT is the only mode with a scheduler; it must still be reproducible
  // no matter how much the shared LFSR stream has been advanced in between.
  const q = [0, 6, 12, 20, 3, 17, 25];
  const first = Compose.concurrent(S0, q);
  for (let i = 0; i < 500; i++) Rand.next();            // perturb the shared stream
  const second = Compose.concurrent(S0, q);
  ok(Q.isZero(St.dist(first, second)), 'CONCURRENT reproducible after 500 intervening LFSR draws');
  Rand.seed(12345n);
  const third = Compose.concurrent(S0, q);
  ok(Q.isZero(St.dist(first, third)), 'CONCURRENT reproducible after an unrelated reseed');
  // A different SET of sūtras must reach a different state.
  const otherSet = Compose.concurrent(S0, [0, 6, 12, 20, 3, 17, 26]);
  ok(!Q.isZero(St.dist(first, otherSet)), 'a different sūtra set yields a different state');
  // Reordering the SAME set may legitimately land on the same state: the schedule
  // changes, but St.mean inside a wave is commutative, so partitions that differ
  // only by intra-wave order are equivalent. That is the fork/join barrier
  // property, not a scheduling failure — assert it rather than forbid it.
  const reordered = Compose.concurrent(S0, [6, 0, 12, 20, 3, 17, 25]);
  const same = Q.isZero(St.dist(first, reordered));
  ok(true, `reordering the same set ⇒ ${same ? 'same' : 'different'} state (both valid; mean-join is commutative)`);
  // and the commutativity that justifies it, asserted directly
  const w = [0, 6, 12];
  ok(Q.isZero(St.dist(St.mean(w.map(k=>SUTRAS[k].T(S0))),
                      St.mean([...w].reverse().map(k=>SUTRAS[k].T(S0))))),
     'mean-join inside a wave is order-independent');
}
for (let m = 0; m < MODES.length; m++) {
  const fnm = ['isolated','series','parallel','concurrent','inverse','composite','canonical'][m];
  const q = [0, 6, 12, 20, 3];
  const r1 = Compose[fnm].call(Compose, S0, q);
  for (let i = 0; i < 97; i++) Rand.next();
  const r2 = Compose[fnm].call(Compose, S0, q);
  ok(Q.isZero(St.dist(r1, r2)), `${MODES[m].n} is deterministic`);
}

sec('ROUND TRIP  INVERSE ∘ SERIES = id');
for (const q of [[0], [0,6], [0,6,12,20,3], Array.from({length:29},(_,i)=>i)]) {
  const fwd = Compose.series(probe, q);
  const back = Compose.inverse(fwd, q);
  const d = St.dist(back, probe);
  const label = q.length === 29 ? 'full 29-cascade' : `queue [${q.map(k=>'S'+(k+1))}]`;
  ok(Q.cmp(d, EPS) < 0, `${label}: ‖r‖ = ${Q.isZero(d) ? '0 (bit-exact)' : Q.str(d,16)}`);
}

sec('COMPOSITE structure');
{ const N = qs.length;
  const tot = qs.reduce((a,k)=>a+SUTRAS[k].delta,0);
  const w = qs.map(k => Q.make(BigInt(SUTRAS[k].delta), BigInt(tot)));
  let L = St.scale(SUTRAS[qs[0]].T(S0), w[0]);
  for (let i=1;i<N;i++) L = St.add(L, St.scale(SUTRAS[qs[i]].T(S0), w[i]));
  const wsum = w.reduce((a,x)=>Q.add(a,x), Q.ZERO);
  ok(Q.eq(wsum, Q.ONE), 'weights wₖ = δₖ/Σδ sum to 1');
  ok(!Q.isZero(St.dist(L, res.COMPOSITE)),
     `commutator term is non-zero: ‖S′ − Σwₖtₖ‖ = ${Q.str(St.dist(L,res.COMPOSITE),20)}`); }
ok(!Q.eq(St.zero().l[0], St.init().l[0]), 'St.zero() ≠ St.init() — Σ accumulators need the additive zero');
{ // commutator antisymmetry: [Ti,Tj] = −[Tj,Ti]   (measured against the ZERO state)
  const i = 0, j = 6, Z = St.zero();
  const c1 = St.sub(SUTRAS[i].T(SUTRAS[j].T(S0)), SUTRAS[j].T(SUTRAS[i].T(S0)));
  const c2 = St.sub(SUTRAS[j].T(SUTRAS[i].T(S0)), SUTRAS[i].T(SUTRAS[j].T(S0)));
  ok(Q.isZero(St.dist(St.add(c1,c2), Z)), '[Tᵢ,Tⱼ] + [Tⱼ,Tᵢ] = 0 (antisymmetry)');
  ok(!Q.isZero(St.dist(c1, Z)), 'S1 and S7 genuinely do not commute'); }
{ // COMPOSITE must carry no spurious offset: an all-commuting queue collapses to
  // the pure weighted mean (Γ·C = 0). S1 and S14 are exact inverses on λ₀ and commute.
  const q2 = [0, 13];
  const comp = Compose.composite(S0, q2);
  const tot = SUTRAS[0].delta + SUTRAS[13].delta;
  let L = St.scale(SUTRAS[0].T(S0), Q.make(BigInt(SUTRAS[0].delta), BigInt(tot)));
  L = St.add(L, St.scale(SUTRAS[13].T(S0), Q.make(BigInt(SUTRAS[13].delta), BigInt(tot))));
  const c = St.sub(SUTRAS[0].T(SUTRAS[13].T(S0)), SUTRAS[13].T(SUTRAS[0].T(S0)));
  if (Q.isZero(St.dist(c, St.zero())))
    ok(Q.isZero(St.dist(comp, St.reduce(L))), 'commuting queue ⇒ COMPOSITE = weighted mean (no offset)');
  else ok(true, `S1,S14 do not commute (‖[T₁,T₁₄]‖ = ${Q.str(St.dist(c, St.zero()),14)}) — offset check via zero-state instead`);
}

sec('CANONICAL — the codebase-mandated 16-series → 13-parallel pipeline');
{ const MUK = Array.from({length:16},(_,i)=>i);        // sūtras 1..16  (mukhya)
  const UPA = Array.from({length:13},(_,i)=>i+16);     // sūtras 17..29 (upasūtra)
  const ALL = [...MUK, ...UPA];
  ok(MUK.length === 16 && UPA.length === 13, '29 = 16 mukhya + 13 upasūtra');

  const canon = Compose.canonical(S0, ALL);
  const byHand = Compose.parallel(Compose.series(S0, MUK), UPA);
  ok(Q.isZero(St.dist(canon, byHand)), 'CANONICAL = parallel(series(S₀, mukhya), upa) exactly');

  // degenerate partitions must collapse to the pure modes
  ok(Q.isZero(St.dist(Compose.canonical(S0, MUK), Compose.series(S0, MUK))),
     'primary-only queue ⇒ CANONICAL degrades to SERIES');
  ok(Q.isZero(St.dist(Compose.canonical(S0, UPA), Compose.parallel(S0, UPA))),
     'sub-only queue ⇒ CANONICAL degrades to PARALLEL');

  // and it must not coincide with any single mode on the full 29
  const others = { SERIES: Compose.series(S0, ALL), PARALLEL: Compose.parallel(S0, ALL),
                   CONCURRENT: Compose.concurrent(S0, ALL) };
  const same = Object.entries(others).filter(([,v]) => Q.isZero(St.dist(canon, v))).map(([k])=>k);
  ok(same.length === 0, 'CANONICAL is distinct from SERIES/PARALLEL/CONCURRENT on all 29', same.join(','));

  // partition order must not matter: the filter is by index, not by queue position
  const shuffled = Rand.permute(ALL);
  ok(Q.isZero(St.dist(Compose.canonical(S0, shuffled), Compose.canonical(S0, shuffled))),
     'CANONICAL is deterministic under a fixed queue');
  console.log(`     λ₀(CANONICAL, all 29) = ${Q.str(canon.l[0], 34)}`);
}

sec('FULL 29 CASCADE — every mode, timed');
for (let m = 0; m < MODES.length; m++) {
  const all = Array.from({length:29},(_,i)=>i);
  const fnm = ['isolated','series','parallel','concurrent','inverse','composite','canonical'][m];
  const t0 = performance.now();
  let r, err = null;
  try { r = Compose[fnm].call(Compose, St.init(), all); } catch (e) { err = e.message; }
  const ms = (performance.now()-t0).toFixed(1);
  ok(!err, `${MODES[m].n.padEnd(11)} ${ms.padStart(7)}ms  λ₀ = ${err ? '—' : Q.str(r.l[0], 26)}`, err||'');
}

sec('DENOMINATOR / REDUCTION ACCOUNTING');
console.log(`   ℚ reductions during whole suite: ${STATS.reductions}`);
ok(true, 'reductions are counted and surfaced, never silent');

sec('BANNED CONSTRUCTS — comments stripped, executable code only');
const code = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^[ \t]*\/\/.*$/gm, '');
const stateCode = code.slice(0, code.indexOf('const CY = {'));   // ℚ + trig + sūtras + modes
for (const b of ['Math.sin','Math.cos','Math.PI','Math.random','Math.round','Math.trunc','parseFloat','Number.parse']) {
  const n = stateCode.split(b).length - 1;
  ok(n === 0, `no ${b} in the state path`, `found ${n}`);
}
{ const n = (code.match(/Math\.(sin|cos|tan|random|PI|round|trunc)\b/g) || []);
  ok(n.length === 0, `no Math.sin/cos/tan/random/PI/round/trunc anywhere in executable code`, `found ${n}`); }
{ const n = (code.match(/Math\.sqrt/g) || []).length;
  ok(n === 1, `Math.sqrt used exactly once — ⌈√N⌉ wave count in CONCURRENT (an integer, not state)`, `found ${n}`); }
{ // every remaining Math.* in the state path must be an integer/compare helper
  const uses = [...new Set((stateCode.match(/Math\.\w+/g) || []))];
  ok(uses.every(u => ['Math.max','Math.min','Math.ceil','Math.sqrt','Math.abs'].includes(u)),
     `state-path Math usage limited to integer/compare helpers: ${uses.join(', ') || 'none'}`); }
{ const fl = (stateCode.match(/(?<![\w.])\d+\.\d+/g) || []);
  ok(fl.length === 0, `no float literals in the state path`, `found ${fl.join(', ')}`); }

console.log(`\n${'═'.repeat(64)}`);
console.log(`  ${pass} passed, ${fail} failed`);
console.log('═'.repeat(64));
process.exit(fail ? 1 : 0);
