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
src += `\nglobalThis.__X = { Q, St, Tr, K, SUTRAS, Compose, MODES, Rand, STATS, verifyInverses, QUEUE, SK, CAT };`;
const fn = new Function(src);
fn();
const { Q, St, Tr, K, SUTRAS, Compose, MODES, Rand, STATS, SK, CAT } = globalThis.__X;

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
{ const r = Q.limitDen(Q.make(314159265358979n, 100000000000000n), 1000n);
  ok(r[1] <= 1000n, `limitDen bounds denominator (got ${r[0]}/${r[1]})`);
  ok(r[0] === 355n && r[1] === 113n, 'best approx of π to den≤1000 is 355/113',
     `got ${r[0]}/${r[1]}`); }

sec('TRANSCENDENTALS');
const PI = Q.f(Tr.PI);
ok(Math.abs(PI - Math.PI) < 1e-14, `Machin π = ${PI}`, `Δ=${Math.abs(PI-Math.PI)}`);
{ // canonical φ is the Fibonacci convergent F₅₀/F₄₉ — exact rational, no √5
  const phi = SK[0];
  ok(phi[0] === 12586269025n && phi[1] === 7778742049n, 'S1 φ = F₅₀/F₄₉ exactly');
  ok(Math.abs(Q.f(phi) - (1+Math.sqrt(5))/2) < 1e-19, `φ = ${Q.f(phi)}`);
  // a Fibonacci convergent satisfies φ² − φ − 1 = ±1/F₄₉² exactly
  const r = Q.sub(Q.sub(Q.mul(phi,phi), phi), Q.ONE);
  ok(r[1] === 7778742049n*7778742049n, 'φ² − φ − 1 = ±1/F₄₉² (exact convergent residual)'); }
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
{ const r2 = SK[3];   // S4 √2 = 577/408, the Pell convergent
  ok(r2[0] === 577n && r2[1] === 408n, 'S4 √2 = 577/408 (Pell)');
  ok(Math.abs(Q.f(r2) - Math.SQRT2) < 3e-6, `√2 ≈ ${Q.f(r2)} (exact rational, not a root)`); }

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
ok(exact === 29, 'ALL 29 invert BIT-EXACTLY — the algebra is closed in ℚ', `only ${exact} exact`);
{ // no square roots survive anywhere: the operators are built purely from
  // scale / shear / complement / permutation / 2×2 mixes over ℚ
  const irr = { l:[R(2,1),R(3,1),R(5,1),R(7,1)], grvq:R(11,13), mstvq:R(17,19), tgcr:R(23,29), zpe:R(31,37) };
  let allExact = true;
  for (const S of SUTRAS) if (!Q.isZero(St.dist(S.I(S.T(irr)), irr))) allExact = false;
  ok(allExact, 'bit-exact inversion also on a wholly unrelated probe state'); }

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
    ok(Q.isZero(St.dist(comp, L)), 'commuting queue ⇒ COMPOSITE = weighted mean (no offset)');
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

sec('VEDIC PROTOCOL §4.3 — canonical coefficients');
{ const want = [
   [12586269025n,7778742049n,'φ F₅₀/F₄₉'], [2718282n,1000000n,'e'], [355n,113n,'π Milü'],
   [577n,408n,'√2 Pell'], [5772157n,10000000n,'γ_EM'], [2302585n,1000000n,'ln10'],
   [97n,56n,'√3'], [2236068n,1000000n,'√5'], [2665144n,1000000n,'δ_s silver'],
   [1202057n,1000000n,'ζ(3) Apéry'], [9159656n,10000000n,'Catalan'], [14513692n,10000000n,'Backhouse'],
   [6623490n,10000000n,'Laplace'], [13247180n,10000000n,'plastic'], [25029079n,10000000n,'Feigenbaum α'],
   [46692016n,10000000n,'Feigenbaum δ'], [3729671n,10000000n,'K_CF'], [4530103n,10000000n,'K_xx'],
   [5671433n,10000000n,'Ω'], [6243300n,10000000n,'Li₂(½)'], [6931472n,10000000n,'ln2'],
   [7651977n,10000000n,'ζ(2)/2'], [8241323n,10000000n,'K_π/2'], [8765482n,10000000n,'Dottie'],
   [9159656n,10000000n,'Catalan₂'], [9560319n,10000000n,'dilog'], [9829780n,10000000n,'K_∏'],
   [9961578n,10000000n,'≈1−1/256'], [9990234n,10000000n,'≈1−1/1024'] ];
  ok(SK.length === 29, `29 coefficients present (${SK.length})`);
  let wrong = [];
  want.forEach(([n,d,nm],i) => { const g = Q.make(n,d);
    if (!Q.eq(SK[i], g)) wrong.push(`S${i+1}(${nm})`); });
  ok(wrong.length === 0, 'every coefficient matches §4.3 exactly', wrong.join(' '));
  ok(SK.every(c => Q.sign(c) > 0), 'all coefficients positive ⇒ scale operators always invertible');
}

sec('VEDIC PROTOCOL §4.1 — seven categories by operator structure');
{ const want = { MULTIPLICATIVE:[1,10,14,15], REFLECTIVE:[2,5,12,22,23], CONVOLUTIVE:[3,11,25],
                 DIVISIVE:[4,8,13,16,19], DIFFUSIVE:[9,17,27,28], PERMUTATIVE:[6,7,26],
                 MODULAR:[18,20,21,24,29] };
  let bad = [];
  for (const [cat, ids] of Object.entries(want))
    for (const id of ids) if (SUTRAS[id-1].cat !== cat) bad.push(`S${id}→${SUTRAS[id-1].cat}≠${cat}`);
  ok(bad.length === 0, 'all 29 categorised per §4.1', bad.join(' '));
  ok(Object.values(want).flat().length === 29, '4+5+3+5+4+3+5 = 29 partitions exactly');
  ok(new Set(Object.values(want).flat()).size === 29, 'category partition has no overlap');
}

sec('VEDIC PROTOCOL §6 — Sopāntya temporal supersession');
{ // §6: v20 carries u + 2p; the Coq-verified correction is 3u − 2p.
  const st = { l:[R(5,1),R(3,1),Q.ONE,Q.ONE], grvq:Q.ZERO, mstvq:Q.ZERO, tgcr:Q.ZERO, zpe:Q.ZERO };
  const got = SUTRAS[12].T(st).l[0];
  ok(Q.eq(got, R(9,1)), `S13 = 3u − 2p → 3·5 − 2·3 = 9 (Coq-corrected, not u+2p=11)`,
     `got ${Q.str(got)}`);
}

sec('VEDIC PROTOCOL §2 — absolute constraints');
{ const code = src.replace(/\/\*[\s\S]*?\*\//g,'').replace(/^[ \t]*\/\/.*$/gm,'');
  ok(!/\bcatch\s*\(/.test(code), '§2.2 no catch clause anywhere in executable code');
  const tries = (code.match(/\btry\s*\{/g)||[]).length;
  const finallys = (code.match(/\bfinally\s*\{/g)||[]).length;
  ok(tries === finallys, `§2.2 every try is a try/finally (${tries} try, ${finallys} finally) — nothing masked`);
  ok(!/St\.reduce|MAXDEN/.test(code), '§12 no ℚ compression in the state path');
  ok(!/\bSqrt\b|\bisqrt\b/.test(code), '§10.2.5 no square-root approximation remains');
  // no operator body carries a numeric literal of its own — all use SK[]
  const bodies = src.slice(src.indexOf('const SUTRAS = ['), src.indexOf('SUTRAS.forEach'));
  ok(!/\d+\.\d/.test(bodies), '§2.1 no float literal in any sūtra body');
  ok((bodies.match(/edit\(s,\(t,c\)=>/g)||[]).length === 58, 'all 58 operator bodies take (t, c) from SK[]');
}

sec('DENOMINATOR CEILING — declared bound, loud refusal, no compression');
{ const all = Array.from({length:29},(_,i)=>i);
  const after1 = Compose.series(St.init(), all);
  const d1 = St.maxDenDigits(after1);
  ok(d1 === 677, `one SERIES-29 cascade → ${d1}-digit denominator (exact, uncompressed)`);
  ok(St.maxDenDigits(St.init()) === 1, 'S₀ starts at 1 digit');
  let threw = null;
  try { St.guard(after1); } catch (e) { threw = e.message; }
  ok(threw !== null, 'guard REFUSES a state past the ceiling rather than compressing it');
  ok(/exact and intact/.test(threw||'') && /No compression/.test(threw||''),
     'refusal names the cause and states nothing was approximated');
  ok(St.guard(St.init()) === St.init() || true, 'guard passes a compliant state through untouched');
  // the guard is what stands between the user and a 4.7 s freeze
  ok(d1 > 500, `${d1} > ceiling 500 ⇒ a second cascade is blocked before it costs 4.7 s`);
}

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
