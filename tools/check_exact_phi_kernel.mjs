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

const R4_CASES = [['3,9,11,25,29', 'SERIES', 4, Q.TWO], ['1,5,9,22', 'PARALLEL', 4, Q.TWO],
                  ['5,7,12,22,29', 'PARALLEL', 4, Q.TWO], ['2,23', 'PARALLEL', 4, Q.mk(1n, 10n)]];

section('R4 singularity controller');
{
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), maya: X.fromQ(Q.mk(1n, 512n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)) };
  const seen = { dormant: 0, braked: 0, gated: 0 };
  /* The gate acts only when the sigma-braked step still exceeds T. With every
     strength scaling its sutra, r = 2 makes sigma small enough that braking
     alone always suffices on these sets (measured: no gated step in 4 steps of
     any of the first three). At r = 1/10, the R4 slider's minimum, sigma is
     close to 1, and the parallel S2 + S23 at 1/2 reaches the gate on steps 1
     and 3. Before the strength fix the gate was reached at r = 2 only
     because S7, S12 and S2's swap ignored the brake. */
  for (const [ids, mode, n, r] of R4_CASES) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) }));
    const p = { ...params, r4R: r };
    let eng = K.initEngine(), held = true, why = '';
    for (let s = 0; s < n; s++) {
      const eIn = K.energy(eng.psi), ceiling = eng.ceiling;
      const T = A4.cmp(eIn, ceiling) > 0 ? eIn : ceiling;
      eng = K.step(eng, p, active, mode);
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
    ok(held, `${mode} ${ids} at r = ${Q.toString(r)}: ${n} steps end at or below T + d(T+1) + d^2`, why);
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
  for (const [ids, mode, n, r] of R4_CASES) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) }));
    let eng = K.initEngine();
    for (let s = 0; s < n; s++) {
      eng = K.step(eng, { ...params, r4R: r }, active, mode);
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

section('every strength is the strength');
{
  /* Strength 0 is the identity for all 29 sutras -- the v18 guarantee
     (SutraWS/Contracts.lean, identity_preserved) -- and strength 1 is the
     formula each sutra had: nine of them used to ignore their strength (S2,
     S6, S7, S12, S18, S25, S26 entirely, S3's transform and S23's swap), so the
     R4 brake could not slow them. The full-strength formulas of those nine
     are written out here, not taken from the kernel. */
  const hwf = v => (v & 1) + ((v >> 1) & 1) + ((v >> 2) & 1) + ((v >> 3) & 1);
  const cayQ = t => { const t2 = Q.mul(t, t), d = Q.add(Q.ONE, t2); return C4.mk(A4.fromQ(Q.div(Q.sub(Q.ONE, t2), d)), A4.fromQ(Q.div(Q.mul(Q.TWO, t), d))); };
  const scaleC = (z, c) => C4.mul(c, z);
  const permOut = (x, P) => { const o = new Array(16); for (let v = 0; v < 16; v++) o[P(v)] = x[v]; return o; };
  const iter = (f, k) => v => { for (let i = 0; i < k; i++) v = f(v); return v; };
  const comp = v => v ^ 15, rot = v => ((v << 1) & 15) | ((v >> 3) & 1);
  const rev = v => ((v & 1) << 3) | ((v & 2) << 1) | ((v & 4) >> 1) | ((v & 8) >> 3);
  const sw03 = v => ((v & 1) << 3) | (v & 6) | ((v & 8) >> 3);
  const aff = v => { const b = i => (v >> i) & 1; return (b(0) ^ b(1)) | ((b(1) ^ b(2)) << 1) | ((b(2) ^ b(3)) << 2) | (b(3) << 3); };
  const whtG = x => Array.from({ length: 16 }, (_, u) => { let acc = C4.ZERO; for (let v = 0; v < 16; v++) acc = (hwf(u & v) & 1) ? C4.sub(acc, x[v]) : C4.add(acc, x[v]); return C4.scaleQ(acc, Q.mk(1n, 4n)); });
  const FULL = {
    2: (x, a) => permOut(x, iter(comp, Number(a[0].n))),
    3: (x, a) => whtG(x).map((z, v) => scaleC(z, cayQ(Q.mul(Q.mk(BigInt(hwf(v) + 1), 16n), a[0])))),
    6: (x, a) => permOut(x, iter(rot, Number(a[0].n))),
    7: (x, a) => permOut(x, iter(rev, Number(a[0].n))),
    12: (x, a) => permOut(x, iter(sw03, Number(a[0].n))),
    18: (x, a) => permOut(x, iter(aff, Number(a[0].n))),
    23: (x, a) => { const p = cayQ(Q.div(a[0], Q.mk(10n))), pc = C4.conj(p), o = x.slice(); for (let v = 0; v < 8; v++) { o[v] = C4.mul(p, x[comp(v)]); o[comp(v)] = C4.mul(pc, x[v]); } return o; },
    25: (x, a) => { const cs = cayQ(a[0]), c = cs.re, s = cs.im, o = x.slice(); for (let v = 0; v < 8; v++) { const u = comp(v); o[v] = C4.add(C4.scaleA4(x[v], c), C4.scaleA4(x[u], s)); o[u] = C4.add(C4.scaleA4(x[v], A4.neg(s)), C4.scaleA4(x[u], c)); } return o; },
    26: (x, a) => permOut(x, v => v ^ (1 << Number(a[0].n)))
  };
  const seedS = K.initState();
  const states = [seedS, whtG(seedS), seedS.map((z, v) => C4.scaleQ(z, Q.mk(BigInt(v + 3), 7n)))];
  const same = (a, b) => a.every((z, v) => C4.eq(z, b[v]));
  const opVecs = id => {
    const spec = K.SUTRA_ARG_SPEC[id] || [], vecs = [spec.map(x => x.def)];
    spec.forEach((x, i) => { for (let k = Math.ceil(x.lo); k <= Math.floor(x.hi); k++) { const v = spec.map(y => y.def); v[i] = Q.mk(BigInt(k)); vecs.push(v); } });
    return vecs;
  };
  let notId = [], cases0 = 0, full = 0, fullBad = [], ignores = [], notUnit = [], unitCases = 0;
  for (let id = 1; id <= 29; id++) {
    for (const ops of opVecs(id)) for (const st of states) {
      let out0; try { out0 = K.applySutra(id, st, [Q.ZERO, ...ops]); } catch (e) { continue; }
      cases0++;
      if (!same(out0, st)) notId.push(id);
      if (FULL[id]) { full++; if (!same(K.applySutra(id, st, [Q.ONE, ...ops]), FULL[id](st, ops))) fullBad.push(id); }
      if (![5, 22, 29].includes(id)) for (const s of [Q.mk(1n, 4n), Q.HALF, Q.mk(3n, 4n)]) {
        unitCases++;
        if (!A4.eq(K.energy(K.applySutra(id, st, [s, ...ops])), K.energy(st))) notUnit.push(id);
      }
    }
    if (same(K.applySutra(id, seedS, Q.HALF), K.applySutra(id, seedS, Q.ONE))) ignores.push(id);
  }
  const u = a => [...new Set(a)].join(' ');
  ok(notId.length === 0 && cases0 > 1000, `every sutra is the identity at strength 0 (${cases0} sutra/operand/state cases)`, `not at S${u(notId)}`);
  ok(fullBad.length === 0 && full > 100, `at strength 1 the nine are exactly the formulas they had (${full} cases)`, `differ at S${u(fullBad)}`);
  ok(ignores.length === 0, 'no sutra gives the same result at strength 1/2 as at strength 1', `S${u(ignores)}`);
  ok(notUnit.length === 0, `every sutra but the contractions S5, S22, S29 preserves the energy exactly at 1/4, 1/2, 3/4 (${unitCases} cases)`, `S${u(notUnit)}`);
  /* the operands are arguments too: for every sutra and every operand, some
     other value in its domain (quarter grid) gives a different result at
     strength 1/2. S19 ignoring its multiplier or S20 its scale passed every
     other check here, so this one is what holds them. */
  const ignoredOps = []; let opCount = 0;
  for (let id = 1; id <= 29; id++) {
    const spec = K.SUTRA_ARG_SPEC[id] || [];
    spec.forEach((x, i) => {
      opCount++;
      const base = K.applySutra(id, seedS, [Q.HALF, ...spec.map(y => y.def)]);
      let moves = false;
      for (let q = Math.ceil(x.lo * 4); q <= Math.floor(x.hi * 4) && !moves; q++) {
        const v = spec.map(y => y.def); v[i] = Q.mk(BigInt(q), 4n);
        let o; try { o = K.applySutra(id, seedS, [Q.HALF, ...v]); } catch (e) { continue; }
        if (!same(o, base)) moves = true;
      }
      if (!moves) ignoredOps.push(`S${id}.${x.name}`);
    });
  }
  ok(ignoredOps.length === 0 && opCount >= 27, `no sutra ignores an operand: each of the ${opCount} has a value that changes the result`, ignoredOps.join(' '));
  /* the principal fractional power, checked on eigenvectors: for a cycle
     v0 -> P v0 -> ... of the permutation, y = sum_m mu^-m e_{P^m v0} with
     mu^L = 1 is an eigenvector of x -> x o P^-1 with eigenvalue mu, and the
     sutra at strength s must multiply it by 1, u, u^2 or conj(u) for
     mu = 1, i, -1, -i, with u = cayley(s) computed here */
  const I = C4.mk(A4.ZERO, A4.ONE), mus = [C4.ONE, I, C4.neg(C4.ONE), C4.neg(I)];
  let eigBad = [], eigCases = 0;
  for (const [id, P, ops] of [[2, comp, [Q.ONE]], [6, rot, [Q.ONE]], [6, iter(rot, 3), [Q.mk(3n)]], [7, rev, [Q.ONE]], [12, sw03, [Q.ONE]],
                              [18, aff, [Q.ONE]], [18, iter(aff, 2), [Q.TWO]], [26, v => v ^ 4, [Q.TWO]]]) {
    for (const s of [Q.mk(1n, 4n), Q.mk(3n, 5n), Q.mk(9n, 10n)]) {
      const uu = cayQ(s), phi = [C4.ONE, uu, C4.mul(uu, uu), C4.conj(uu)];
      const seen = new Set();
      for (let v0 = 0; v0 < 16; v0++) {
        if (seen.has(v0)) continue;
        const cyc = [v0]; for (let w = P(v0); w !== v0; w = P(w)) cyc.push(w);
        cyc.forEach(w => seen.add(w));
        mus.forEach((mu, j) => {
          let muL = C4.ONE; for (let m = 0; m < cyc.length; m++) muL = C4.mul(muL, mu);
          if (!C4.eq(muL, C4.ONE)) return;
          const y = Array.from({ length: 16 }, () => C4.ZERO); let w = C4.ONE;
          const muInv = C4.conj(mu);
          cyc.forEach(v => { y[v] = w; w = C4.mul(w, muInv); });
          eigCases++;
          if (!same(K.applySutra(id, y, [s, ...ops]), y.map(z => C4.mul(phi[j], z)))) eigBad.push(`S${id}`);
        });
      }
    }
  }
  ok(eigBad.length === 0 && eigCases > 100, `the permutation sutras act as the principal fractional power on every cycle eigenvector (${eigCases} cases)`, u(eigBad));
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
    'pre-bindu':       { line: "'pre-bindu':          { comp: 'PARALLEL',             set: { 5: 25, 12: 50, 28: 75 }, vortex: 0 },",
                         comp: 'PARALLEL', set: { 5: 25, 12: 50, 28: 75 }, support: ['0000'], dormant: true },
    'pre-star':        { line: "'pre-star':           { comp: 'SYMMETRIC_CONCURRENT', set: { 13: 100, 20: 75, 22: 50 }, vortex: 0 },",
                         comp: 'SYMMETRIC_CONCURRENT', set: { 13: 100, 20: 75, 22: 50 }, support: ['0000', '0001', '0100', '1000'], dormant: true },
    'pre-crown':       { line: "'pre-crown':          { comp: 'SERIES',               set: { 5: 30, 7: 5, 13: 40, 18: 35, 19: 55, 20: 90, 22: 20 }, args: { 13: ['5/2'], 19: ['23/4'], 20: ['7'] }, vortex: 0 },",
                         comp: 'SERIES', set: { 5: 30, 7: 5, 13: 40, 18: 35, 19: 55, 20: 90, 22: 20 }, args: { 13: ['5/2'], 19: ['23/4'], 20: ['7'] },
                         support: ['0110', '1000', '1011', '1101', '1110', '1111'], peaks: ['1011', '1110'], dormant: true },
    'pre-square':      { line: "'pre-square':         { comp: 'PARALLEL',             set: { 8: 50, 16: 50, 21: 100, 29: 25 }, vortex: 0 },",
                         comp: 'PARALLEL', set: { 8: 50, 16: 50, 21: 100, 29: 25 }, support: ['0000', '0001', '1000', '1001'], dormant: true },
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
    let eng = K.initEngine(), engaged = 0;
    const acc = new Array(16).fill(0);
    for (let s = 0; s < 40; s++) {
      eng = K.step(eng, params, active, x.comp);
      if (eng.last.r4.engaged) engaged++;
      if (s >= 32) { const p = probs(eng.psi), tot = p.reduce((a, b) => a + b, 0); p.forEach((v, i) => { acc[i] += v / tot / 8; }); }
    }
    if (x.dormant) ok(engaged === 0, `${id}: R4 stays dormant, as its description says`, `engaged on ${engaged} of 40 steps`);
    if (x.uniform) {
      /* measured: a fixed pattern 2.5e-5 from even, the same from step 20 to
         step 160; the page claims one part in ten thousand */
      const dev = Math.max(...acc.map(v => Math.abs(v * 16 - 1)));
      ok(dev < 1e-4, `${id}: every vertex holds the same share of the energy to one part in ten thousand`, `largest deviation ${dev}`);
      continue;
    }
    const order = [...acc.keys()].sort((a, b) => acc[b] - acc[a]);
    if (x.peaks) {
      const top = order.slice(0, x.peaks.length).map(v => v.toString(2).padStart(4, '0')).sort();
      ok(top.join(',') === x.peaks.join(','), `${id}: highest at ${x.peaks.join(' and ')}`, `highest at ${top.join(' and ')}`);
    }
    const S = []; let c = 0;
    for (const v of order) { S.push(v); c += acc[v]; if (c >= 0.8) break; }
    const got = S.sort((a, b) => a - b).map(v => v.toString(2).padStart(4, '0'));
    ok(got.join(',') === x.support.join(','), `${id}: settles on ${x.support.join(' ')}`, `got ${got.join(' ')}`);
  }
  /* pre-symmetry's description: uncontrolled, the set grows about twofold per
     step, 1.2x to 3.6x, and R4 holds it at the ceiling. Measured from the seed
     over steps 9 to 40 (the first eight are the approach): median 2.0, least
     1.2, most 3.6. The description used to say 35x, which nothing measured. */
  {
    const line = "'pre-symmetry':       { comp: 'PARALLEL',             set: { 5: 50, 7: 50, 12: 50, 22: 50, 29: 50 }, vortex: 0 },";
    ok(HTML.includes(line), 'pre-symmetry: the page defines it as tested here');
    const act = [5, 7, 12, 22, 29].map(id => ({ id, strength: Q.mk(1n, 2n) }));
    const aF = x => qNum(x.a) + qNum(x.b) * Math.SQRT2 + qNum(x.c) * Math.sqrt(5) + qNum(x.d) * Math.sqrt(10);
    let eng = K.initEngine(), engaged = 0; const r = [];
    for (let s = 0; s < 40; s++) {
      eng = K.step(eng, params, act, 'PARALLEL');
      if (s >= 8) { r.push(aF(eng.last.r4.eUncontrolled) / aF(eng.last.r4.target)); if (eng.last.r4.engaged) engaged++; }
    }
    const sorted = r.slice().sort((a, b) => a - b), med = sorted[sorted.length >> 1];
    ok(sorted[0] >= 1.15 && sorted[sorted.length - 1] <= 3.65 && med >= 1.5 && med <= 2.5,
       `pre-symmetry: uncontrolled it grows about twofold per step, 1.2x to 3.6x (median ${med.toFixed(2)}, ${sorted[0].toFixed(2)} to ${sorted[sorted.length - 1].toFixed(2)})`);
    ok(engaged === r.length, `pre-symmetry: R4 holds it on every one of those steps (${engaged}/${r.length})`);
  }
}

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
