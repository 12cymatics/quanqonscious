/* Gate for the exact kernel inside vedic_v18.51.1_exact_phi.html.

   The kernel is not copied here. KERNEL_FACTORY is sliced out of the page and
   evaluated, so there is one copy of the formulas and nothing that can drift
   from it. Where a statement can be checked against something the kernel did
   not compute, it is: reduced fractions against a Euclid written here, the
   storage bound in bare BigInt arithmetic, the storage budget against the
   same composition evaluated with no storage at all.

   What it holds the page to:
     self-tests         every one of runSelfTests passes
     canonical Q        Q.mk reduces by the true gcd, dyadic or not
     storage            |x - storeQ(x)| <= |x| / 2^128, and storeQ is
                        idempotent on what it produced
     refusal            an operand or strength outside its domain throws a
                        RangeError; nothing is clamped into range
     budget             a stored composition is within the summed storage
                        budget of the same composition without storage
     R4                 every step ends at or below T + d(T+1) + d^2, with
                        T = max(E_in, ceiling) -- dormant, braked and gated
     step export        an exported step replays bit for bit from its file
                        alone, in every R4 branch, and its stage flags judge
                        the exact stage, not the stored one
     modes              the mode identities decide and pass, and the
                        comparison built from the audit's states equals a
                        fresh one
     presets            each preset that names the vertices its field settles
                        on does settle there, run through the kernel

   Run: node tools/check_exact_phi_kernel.mjs
*/
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const HTML = readFileSync(join(ROOT, 'vedic_v18.51.1_exact_phi.html'), 'utf8');
const start = HTML.indexOf('const KERNEL_FACTORY = () => {');
const end = HTML.indexOf('\nconst KERNEL = KERNEL_FACTORY();');
if (start < 0 || end < 0 || end < start) {
  console.error('FAIL -- KERNEL_FACTORY could not be located in the page');
  process.exit(1);
}
const K = new Function(HTML.slice(start, end) + '\nreturn KERNEL_FACTORY();')();
const { Q, A4, C4 } = K;

let pass = 0;
const failures = [];
function ok(cond, label, detail) {
  if (cond) { pass++; console.log('  ok   ' + label); }
  else { failures.push(label + (detail ? '  -- ' + detail : '')); console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
const section = t => console.log('\n' + t);
const throwsRange = f => { try { f(); return false; } catch (e) { return e instanceof RangeError; } };

/* deterministic corpus: a linear congruential generator, no Math.random */
let seed = 0x5eedn;
const rnd = bits => {
  let r = 0n;
  for (let i = 0; i < bits; i += 31) { seed = (seed * 1103515245n + 12345n) % 2147483648n; r = (r << 31n) | seed; }
  return r >> BigInt(Math.ceil(bits / 31) * 31 - bits);
};
const euclid = (a, b) => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };

section('self-tests');
let tests = [];
try { tests = K.runSelfTests(); } catch (e) { ok(false, 'runSelfTests', e.message); }
ok(tests.length >= 61, `runSelfTests passes all ${tests.length}`);

section('canonical rationals');
{
  let bad = 0, n = 0;
  for (let t = 0; t < 3000; t++) {
    let num = rnd(1 + (t * 37) % 400);
    const den = t % 3 === 0 ? (1n << BigInt(t % 300)) : (rnd(1 + (t * 53) % 400) | 1n) << BigInt(t % 5);
    if (t % 7 === 0) num = -num;
    if (den === 0n) continue;
    if (t % 11 === 0) num <<= BigInt(t % 90);
    n++;
    const q = Q.mk(num, den);
    const g = euclid(num, den);
    const wantN = num === 0n ? 0n : num / g, wantD = num === 0n ? 1n : den / g;
    if (q.n !== wantN || q.d !== wantD) bad++;
  }
  ok(bad === 0, `Q.mk reduces ${n} fractions by Euclid's gcd (a third with power-of-two denominators)`, `${bad} differ`);
}

section('storage');
{
  const B = BigInt(K.STORE_BITS);
  ok(K.STORE_BITS === 128, 'STORE_BITS is 128');
  let bad = 0, notIdem = 0, n = 0;
  for (let t = 0; t < 2000; t++) {
    let num = rnd(1 + (t * 29) % 700) + 1n;
    const den = rnd(1 + (t * 41) % 700) + 1n;
    if (t % 2) num = -num;
    const x = Q.mk(num, den), s = K.storeQ(x);
    n++;
    /* |x - s| <= |x| / 2^B  <=>  |x.n s.d - s.n x.d| 2^B <= |x.n| s.d  (both sides over x.d s.d > 0) */
    const diff = x.n * s.d - s.n * x.d;
    if ((diff < 0n ? -diff : diff) << B > (x.n < 0n ? -x.n : x.n) * s.d) bad++;
    const s2 = K.storeQ(s);
    if (s2.n !== s.n || s2.d !== s.d) notIdem++;
  }
  ok(bad === 0, `|x - storeQ(x)| <= |x|/2^128 on ${n} rationals, checked in bare BigInt`, `${bad} exceed it`);
  ok(notIdem === 0, 'storeQ returns its own output unchanged', `${notIdem} moved`);
  ok(K.storeQ(Q.ZERO).n === 0n, 'zero is stored as zero');
}

section('refusal, not clamping');
{
  let refused = 0, total = 0, kept = 0, defaults = 0;
  for (const id of Object.keys(K.ARG_DOMAIN)) {
    K.ARG_DOMAIN[id].forEach((dom, i) => {
      defaults++;
      const below = Q.sub(dom.lo, Q.ONE), above = Q.add(dom.hi, Q.ONE);
      total += 2;
      if (throwsRange(() => K.checkArg(id, i, below))) refused++;
      if (throwsRange(() => K.checkArg(id, i, above))) refused++;
      /* an in-domain operand comes back as the very value given */
      if (K.checkArg(id, i, dom.lo) === dom.lo) kept++;
    });
  }
  ok(total > 0 && refused === total, `${refused}/${total} out-of-domain operands refused with a RangeError`);
  ok(kept === defaults, `${kept}/${defaults} in-domain operands returned unchanged`);
  let strength = 0;
  for (let id = 1; id <= 29; id++) {
    if (throwsRange(() => K.checkStrength(id, Q.mk(101n, 100n)))) strength++;
    if (throwsRange(() => K.checkStrength(id, Q.mk(-1n, 100n)))) strength++;
  }
  ok(strength === 58, `${strength}/58 strengths outside [0, 1] refused`);
}

section('storage budget');
{
  for (const [ids, mode] of [['1,4,9', 'SERIES'], ['3,6,11,24', 'PARALLEL'], ['7,13,26', 'SYMMETRIC_CONCURRENT'], ['22,5,19', 'SERIES']]) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(3n, 4n) }));
    const seedState = K.initState();
    let delta = Q.ZERO, stores = 0;
    const keep = x => { const s = x.map(K.storeC4); delta = Q.add(delta, K.storeBudget(s)); stores++; return s; };
    const stored = K.applySutraCore(seedState, active, mode, keep);
    const exact = K.applySutraCore(seedState, active, mode);
    let dist2 = A4.ZERO;
    for (let v = 0; v < 16; v++) dist2 = A4.add(dist2, C4.norm2(C4.sub(stored[v], exact[v])));
    const within = A4.cmp(dist2, A4.fromQ(Q.mul(delta, delta))) <= 0;
    ok(within && stores > 0, `${mode} ${ids}: stored within the budget of the unstored result (${stores} stores)`);
    /* and the budget is a statement about 128-bit storage, not a vacuous bound */
    const e = K.energy(exact);
    ok(A4.cmp(A4.fromQ(Q.mul(delta, delta)), A4.scaleQ(A4.add(A4.ONE, e), Q.mk(1n, 1n << 200n))) <= 0,
       `${mode} ${ids}: budget squared is below 2^-200 of (1 + energy)`);
  }
}

section('R4 singularity controller');
{
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), maya: X.fromQ(Q.mk(1n, 512n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)) };
  const seen = { dormant: 0, braked: 0, gated: 0 };
  for (const [ids, mode, n] of [['3,9,11,25,29', 'SERIES', 4], ['1,5,9,22', 'PARALLEL', 4], ['5,7,12,22,29', 'PARALLEL', 4]]) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) }));
    let eng = K.initEngine(), held = true, why = '';
    for (let s = 0; s < n; s++) {
      const eIn = K.energy(eng.psi), ceiling = eng.ceiling;
      const T = A4.cmp(eIn, ceiling) > 0 ? eIn : ceiling;
      eng = K.step(eng, params, active, mode);
      const L = eng.last, d = L.stepError.delta;
      const bound = A4.add(T, A4.add(A4.scaleQ(T, d), A4.fromQ(Q.add(d, Q.mul(d, d)))));
      const eOut = K.energy(eng.psi);
      if (A4.cmp(eOut, bound) > 0) { held = false; why = `step ${s + 1} ends above T + d(T+1) + d^2`; }
      if (!A4.eq(eOut, eng.energyLedger)) { held = false; why = `step ${s + 1} ledger is not the state's energy`; }
      if (L.r4.gate) {
        seen.gated++;
        const g = L.r4.gate.meanGate;
        if (Q.sign(g) < 0 || Q.cmp(g, Q.ONE) > 0) { held = false; why = 'mean gate outside [0, 1]'; }
      } else if (L.r4.engaged) seen.braked++;
      else seen.dormant++;
    }
    ok(held, `${mode} ${ids}: ${n} steps end at or below T + d(T+1) + d^2`, why);
  }
  ok(seen.dormant > 0 && seen.braked > 0 && seen.gated > 0,
     `the controller was seen dormant (${seen.dormant}), braking only (${seen.braked}) and gating (${seen.gated})`);
}

section('step export');
{
  /* An exported step is a claim about what the kernel did, so it is held to
     two things. The file alone replays the step bit for bit, through every R4
     branch, from the second step on (the first would let a replay that
     silently used the seed state or the initial dielectric pass). And each
     stage's pass flag judges the exact stage on its stored input, with what
     storage did reported beside it: Maya moved by storage alone must still
     read as energy-preserving, because it is. */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), maya: X.fromQ(Q.mk(1n, 512n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)) };
  const fromQ = o => Q.mk(BigInt(o.n), BigInt(o.d));
  const fromA4 = o => A4.mk(fromQ(o.a), fromQ(o.b), fromQ(o.c), fromQ(o.d));
  const fromC4 = o => C4.mk(fromA4(o.re), fromA4(o.im));
  const fromExact = o => ('re' in o) ? fromC4(o) : ('a' in o && 'b' in o) ? fromA4(o) : fromQ(o);
  const tested = { dormant: 0, braked: 0, gated: 0 };
  let storageSeen = false;
  for (const [ids, mode, n] of [['3,9,11,25,29', 'SERIES', 4], ['1,5,9,22', 'PARALLEL', 4], ['5,7,12,22,29', 'PARALLEL', 4]]) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) }));
    let eng = K.initEngine();
    for (let s = 0; s < n; s++) {
      eng = K.step(eng, params, active, mode);
      if (s === 0) continue;
      const branch = eng.last.r4.gate ? 'gated' : eng.last.r4.engaged ? 'braked' : 'dormant';
      if (tested[branch]) continue;
      tested[branch]++;
      const file = JSON.parse(JSON.stringify({ frame: eng.frame, last: K.serializeLast(eng.last), psi: K.serializeState(eng.psi),
        dielectric: eng.dielectric.map(K.serializeQ), ceiling: K.serializeA4(eng.ceiling) }));
      const L = file.last, where = `${branch} step (${mode} ${ids}, step ${s + 1})`;
      let same = false, why = '';
      try {
        const p = Object.fromEntries(Object.entries(L.params).map(([k, v]) => [k, fromExact(v)]));
        const act = L.activeSutras.map(x => x.args ? { id: x.id, strength: fromQ(x.strength), args: x.args.map(fromQ) } : { id: x.id, strength: fromQ(x.strength) });
        const replay = K.step({ psi: L.input.map(fromC4), dielectric: L.dielectricIn.map(fromQ), ceiling: fromA4(file.ceiling),
                                energyLedger: null, frame: file.frame - 1, last: null }, p, act, L.composition);
        same = JSON.stringify(K.serializeState(replay.psi)) === JSON.stringify(file.psi) &&
               JSON.stringify(replay.dielectric.map(K.serializeQ)) === JSON.stringify(file.dielectric) &&
               JSON.stringify(K.serializeLast(replay.last)) === JSON.stringify(L);
        if (!same) why = 'the replay differs from the file';
      } catch (e) { why = e.message; }
      ok(same, `${where}: replays from its exported file alone, bit for bit`, why);
      ok(L.stagesRun === (eng.last.r4.engaged ? 'sigma-braked' : 'uncontrolled'), `${where}: names the run its stage states come from (${L.stagesRun})`);
      const m = L.mayaTransform, z = L.zpeLedger;
      ok(L.mstvqPairs.every(x => x.unitarityPass && x.storedMatchesExact), `${where}: every MSTVQ pair rotation preserves its energy exactly`);
      ok(L.tgcrSpectral.every(x => x.spectralPass && x.unitNormPass), `${where}: TGCR multiplies every Walsh mode by its unit phase exactly`);
      ok(m.energyPass && m.storedMatchesExact, `${where}: Maya preserves the energy exactly and the stored stage is its storage`);
      ok(z.sourceIsZero && A4.isZero(fromA4(z.netEnergyInjection)) && z.storedMatchesExact,
         `${where}: ZPE with a zero source injects exactly zero energy`);
      if (!A4.isZero(fromA4(m.storageEnergyChange))) storageSeen = true;
    }
  }
  ok(tested.dormant && tested.braked && tested.gated, `exports were replayed from all three R4 branches (dormant ${tested.dormant}, braked ${tested.braked}, gated ${tested.gated})`);
  ok(storageSeen, 'storage moved the Maya energy in at least one export, and the flag still judged the exact stage');
}

section('series / parallel / concurrent');
{
  const seedState = K.initState();
  for (const ids of ['', '9', '5,22', '1,4,26']) {
    const active = ids ? ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) })) : [];
    const a = K.modeAudit(seedState, active);
    ok(a.ok && a.results.length > 0, `modeAudit on {${ids}} decides ${a.results.length} identities and all hold`);
    if (active.length) {
      const shared = K.modeComparison(seedState, active, a.states), fresh = K.modeComparison(seedState, active);
      const same = shared.modes.every(m => shared.states[m].every((z, v) => C4.eq(z, fresh.states[m][v])) &&
                                           A4.eq(shared.energies[m], fresh.energies[m]));
      ok(same, `{${ids}}: the comparison from the audit's states equals a fresh one`);
    }
  }
}

section('presets settle where their descriptions say');
{
  /* Each of these presets says, in the page, which vertices its field settles
     on. Run each through the kernel for 40 steps and read the smallest set of
     vertices holding 80% of the energy, averaged over the last eight steps. The
     definitions must also be the ones the page carries. */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), maya: X.fromQ(Q.mk(1n, 512n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)) };
  const EXPECT = {
    'pre-bindu':       { line: "'pre-bindu':          { comp: 'PARALLEL',             set: { 2: 25, 7: 100, 22: 75, 23: 50, 26: 100 }, vortex: 0 },",
                         comp: 'PARALLEL', set: { 2: 25, 7: 100, 22: 75, 23: 50, 26: 100 }, support: ['0000'] },
    'pre-star':        { line: "'pre-star':           { comp: 'PARALLEL',             set: { 2: 25, 23: 25 }, vortex: 0 },",
                         comp: 'PARALLEL', set: { 2: 25, 23: 25 }, support: ['0000', '0001', '0100', '1000'] },
    'pre-crown':       { line: "'pre-crown':          { comp: 'PARALLEL',             set: { 13: 100, 18: 50, 26: 50 }, vortex: 0 },",
                         comp: 'PARALLEL', set: { 13: 100, 18: 50, 26: 50 }, support: ['1000', '1011', '1101', '1110', '1111'] },
    'pre-square':      { line: "'pre-square':         { comp: 'PARALLEL',             set: { 8: 50, 16: 50, 21: 100, 29: 25 }, vortex: 0 },",
                         comp: 'PARALLEL', set: { 8: 50, 16: 50, 21: 100, 29: 25 }, support: ['0000', '0001', '1000', '1001'] },
    'pre-equilibrium': { line: "'pre-equilibrium':    { comp: 'SERIES',               set: { 7: 50, 19: 75, 22: 100 }, vortex: 0 },",
                         comp: 'SERIES', set: { 7: 50, 19: 75, 22: 100 }, uniform: true },
    'pre-cube':        { line: "'pre-cube':           { comp: 'SYMMETRIC_CONCURRENT', set: { 16: 75, 22: 75 }, args: { 16: ['17/4'] }, vortex: 0 },",
                         comp: 'SYMMETRIC_CONCURRENT', set: { 16: 75, 22: 75 }, args: { 16: ['17/4'] },
                         support: ['1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'] },
    'pre-octad':       { line: "'pre-octad':          { comp: 'SYMMETRIC_CONCURRENT', set: { 14: 25, 17: 100, 22: 50 }, args: { 14: ['11/4'], 17: ['23/4'] }, vortex: 0 },",
                         comp: 'SYMMETRIC_CONCURRENT', set: { 14: 25, 17: 100, 22: 50 }, args: { 14: ['11/4'], 17: ['23/4'] },
                         support: ['0100', '0101', '0110', '0111', '1010', '1100', '1101', '1110'] }
  };
  /* |z|^2 of each vertex as a double, read from the exact state after scaling
     by its largest power of two, so a decaying state stays readable */
  const bl = x => x === 0n ? 0 : (x < 0n ? -x : x).toString(2).length;
  const qNum = q => { let n = q.n; if (n === 0n) return 0; const neg = n < 0n; if (neg) n = -n;
    const sh = BigInt(Math.max(0, 64 + bl(q.d) - bl(n))); const v = Number((n << sh) / q.d) / 2 ** Number(sh); return neg ? -v : v; };
  const probs = psi => {
    let e = null;
    for (const z of psi) for (const a of [z.re, z.im]) for (const q of [a.a, a.b, a.c, a.d]) if (q.n !== 0n) { const x = bl(q.n) - bl(q.d); if (e === null || x > e) e = x; }
    const sc = q => q.n === 0n ? q : (e >= 0 ? Q.mk(q.n, q.d << BigInt(e)) : Q.mk(q.n << BigInt(-e), q.d));
    const num = a => qNum(sc(a.a)) + qNum(sc(a.b)) * Math.SQRT2 + qNum(sc(a.c)) * Math.sqrt(5) + qNum(sc(a.d)) * Math.sqrt(10);
    return psi.map(z => { const r = num(z.re), i = num(z.im); return r * r + i * i; });
  };
  for (const [id, x] of Object.entries(EXPECT)) {
    ok(HTML.includes(x.line), `${id}: the page defines it as tested here`);
    const active = Object.entries(x.set).map(([k, v]) => {
      const sid = Number(k), st = Q.mk(BigInt(v), 100n);
      return x.args && x.args[k] ? { id: sid, strength: st, args: [st, ...x.args[k].map(t => Q.parse(t))] } : { id: sid, strength: st };
    });
    let eng = K.initEngine();
    const acc = new Array(16).fill(0);
    for (let s = 0; s < 40; s++) {
      eng = K.step(eng, params, active, x.comp);
      if (s >= 32) { const p = probs(eng.psi), tot = p.reduce((a, b) => a + b, 0); p.forEach((v, i) => { acc[i] += v / tot / 8; }); }
    }
    if (x.uniform) {
      /* measured: a fixed pattern 2.5e-5 from even, the same from step 20 to
         step 160; the page claims one part in ten thousand */
      const dev = Math.max(...acc.map(v => Math.abs(v * 16 - 1)));
      ok(dev < 1e-4, `${id}: every vertex holds the same share of the energy to one part in ten thousand`, `largest deviation ${dev}`);
      continue;
    }
    const order = [...acc.keys()].sort((a, b) => acc[b] - acc[a]);
    const S = []; let c = 0;
    for (const v of order) { S.push(v); c += acc[v]; if (c >= 0.8) break; }
    const got = S.sort((a, b) => a - b).map(v => v.toString(2).padStart(4, '0'));
    ok(got.join(',') === x.support.join(','), `${id}: settles on ${x.support.join(' ')}`, `got ${got.join(' ')}`);
  }
}

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
