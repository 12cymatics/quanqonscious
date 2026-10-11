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
     Maya               pi, e, phi and sin inside the kernel's certified
                        enclosures (independent 600-bit arithmetic), the
                        layer inside its bracket, the flow within its error
                        bound of the exact Euler map
     magnetism          induction, Faraday potential, potential and
                        reconnections recomputed here; no flux from Faraday,
                        flux closed on every cube, the memory bounded
     parameters         vortex, drive and polarisation each change the run
     presets            each preset does, over 600 steps, what its
                        description says: flux growth, R4, where the field
                        goes, still moving after step 300, all distinct

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

/* [sutras at 1/2, composition, steps, r, drive]. The drive is energy the
   sigma-brake cannot slow, because it scales strengths: with S7 + S12 and a
   drive of 4 every step from the third on reaches the gate (measured). With
   no drive the gate is reached only on the first step from the seed now that
   Maya no longer shuffles the state, and an export is not tested there. */
const R4_CASES = [['3,9,11,25,29', 'SERIES', 4, Q.TWO, 0n], ['1,5,9,22', 'PARALLEL', 4, Q.TWO, 0n],
                  ['5,7,12,22,29', 'PARALLEL', 4, Q.TWO, 0n], ['2,23', 'PARALLEL', 4, Q.mk(1n, 10n), 0n],
                  ['7,12', 'SERIES', 4, Q.TWO, 4n]];

section('R4 singularity controller');
{
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ZERO };
  const seen = { dormant: 0, braked: 0, gated: 0 };
  /* The gate acts only when the sigma-braked step still exceeds T. With every
     strength scaling its sutra, r = 2 makes sigma small enough that braking
     alone always suffices on these sets (measured: no gated step in 4 steps of
     any of the first three). At r = 1/10, the R4 slider's minimum, sigma is
     close to 1, and the parallel S2 + S23 at 1/2 reaches the gate on steps 1
     and 3. Before the strength fix the gate was reached at r = 2 only
     because S7, S12 and S2's swap ignored the brake. */
  for (const [ids, mode, n, r, drive] of R4_CASES) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) }));
    const p = { ...params, r4R: r, zpeSource: X.fromQ(Q.mk(drive)) };
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
    ok(held, `${mode} ${ids} at r = ${Q.toString(r)}, drive ${drive}: ${n} steps end at or below T + d(T+1) + d^2`, why);
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
     storage did reported beside it: the magnetism stage moved by storage
     alone must still read as energy-preserving, because it is. The medium the
     step read -- Maya's phase, the vector potential and the Faraday
     potential -- is in the file too. */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ZERO };
  const fromQ = o => Q.mk(BigInt(o.n), BigInt(o.d));
  const fromA4 = o => A4.mk(fromQ(o.a), fromQ(o.b), fromQ(o.c), fromQ(o.d));
  const fromC4 = o => C4.mk(fromA4(o.re), fromA4(o.im));
  const fromExact = o => ('re' in o) ? fromC4(o) : ('a' in o && 'b' in o) ? fromA4(o) : fromQ(o);
  const tested = { dormant: 0, braked: 0, gated: 0 };
  let storageSeen = false;
  for (const [ids, mode, n, r, drive] of R4_CASES) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) }));
    let eng = K.initEngine();
    for (let s = 0; s < n; s++) {
      eng = K.step(eng, { ...params, r4R: r, zpeSource: X.fromQ(Q.mk(drive)) }, active, mode);
      if (s === 0) continue;
      const branch = eng.last.r4.gate ? 'gated' : eng.last.r4.engaged ? 'braked' : 'dormant';
      if (tested[branch]) continue;
      tested[branch]++;
      const file = JSON.parse(JSON.stringify({ frame: eng.frame, last: K.serializeLast(eng.last), psi: K.serializeState(eng.psi),
        dielectric: eng.dielectric.map(K.serializeQ), ceiling: K.serializeA4(eng.ceiling),
        mayaPhase: K.serializeQ(eng.mayaPhase), mayaErr: K.serializeQ(eng.mayaErr),
        magA: eng.magA.map(K.serializeQ), magChi: eng.magChi.map(K.serializeQ) }));
      const L = file.last, where = `${branch} step (${mode} ${ids}, step ${s + 1})`;
      let same = false, why = '';
      try {
        const p = Object.fromEntries(Object.entries(L.params).map(([k, v]) => [k, fromExact(v)]));
        const act = L.activeSutras.map(x => x.args ? { id: x.id, strength: fromQ(x.strength), args: x.args.map(fromQ) } : { id: x.id, strength: fromQ(x.strength) });
        const replay = K.step({ psi: L.input.map(fromC4), dielectric: L.dielectricIn.map(fromQ), ceiling: fromA4(file.ceiling),
                                energyLedger: null, frame: file.frame - 1, last: null,
                                mayaPhase: fromQ(L.mayaPhaseIn), mayaErr: fromQ(L.mayaErrIn),
                                magA: L.magAIn.map(fromQ), magChi: L.magChiIn.map(fromQ) }, p, act, L.composition);
        same = JSON.stringify(K.serializeState(replay.psi)) === JSON.stringify(file.psi) &&
               JSON.stringify(replay.dielectric.map(K.serializeQ)) === JSON.stringify(file.dielectric) &&
               JSON.stringify(K.serializeQ(replay.mayaPhase)) === JSON.stringify(file.mayaPhase) &&
               JSON.stringify(replay.magA.map(K.serializeQ)) === JSON.stringify(file.magA) &&
               JSON.stringify(replay.magChi.map(K.serializeQ)) === JSON.stringify(file.magChi) &&
               JSON.stringify(K.serializeLast(replay.last)) === JSON.stringify(L);
        if (!same) why = 'the replay differs from the file';
      } catch (e) { why = e.message; }
      ok(same, `${where}: replays from its exported file alone, bit for bit`, why);
      ok(L.stagesRun === (eng.last.r4.engaged ? 'sigma-braked' : 'uncontrolled'), `${where}: names the run its stage states come from (${L.stagesRun})`);
      const m = L.magnetismStage, y = L.maya, z = L.zpeLedger;
      ok(L.mstvqPairs.every(x => x.unitarityPass && x.storedMatchesExact), `${where}: every MSTVQ pair rotation preserves its energy exactly`);
      ok(L.tgcrSpectral.every(x => x.spectralPass && x.unitNormPass), `${where}: TGCR multiplies every Walsh mode by its unit phase exactly`);
      ok(m.energyPass && m.storedMatchesExact, `${where}: the magnetism memory's phase turns keep the energy exactly and the stored stage is their storage`);
      ok(L.magnetismIn.closed, `${where}: the flux it starts from closes on every cube`);
      const fM = fromQ(y.factor);
      ok(Q.cmp(fM, Q.mk(19n, 20n)) > 0 && Q.cmp(fM, Q.mk(21n, 20n)) < 0 && Q.cmp(fromQ(y.phaseNext), fromQ(y.phase)) > 0,
         `${where}: Maya's factor is in (19/20, 21/20) and its phase rises`);
      ok(z.storedMatchesExact && A4.eq(fromA4(z.energyAfterPhase), fromA4(z.energyBefore)) &&
         (!z.sourceIsZero || A4.isZero(fromA4(z.netEnergyInjection))),
         `${where}: ZPE's phase keeps the energy exactly${z.sourceIsZero ? ' and its zero source injects none' : '; the drive is the only injection'}`);
      if (!A4.isZero(fromA4(m.storageEnergyChange))) storageSeen = true;
    }
  }
  ok(tested.dormant && tested.braked && tested.gated, `exports were replayed from all three R4 branches (dormant ${tested.dormant}, braked ${tested.braked}, gated ${tested.gated})`);
  ok(storageSeen, 'storage moved the magnetism stage energy in at least one export, and the flag still judged the exact stage');
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

section('Maya coherence flow');
{
  /* Everything Maya rests on is checked against values this gate computes in
     its own fixed-point arithmetic, not the kernel's: pi by Machin, e by its
     series, sqrt5 by Newton, sin by its series to far more terms, and the
     layer by bisection on that independent beta. */
  const P = 600n, ONE = 1n << P;
  const fx = q => (q.n << P) / q.d;                       /* rational -> fixed point, truncated */
  const inQ = (x, br) => (fx(br[0]) <= x + 1n) && (x <= fx(br[1]) + 1n);
  const atanInv = x => { let s = 0n, t = ONE / x, n = 1n, sg = 1n; const x2 = x * x; while (t > 0n) { s += sg * t / n; t /= x2; n += 2n; sg = -sg; } return s; };
  const PI = 16n * atanInv(5n) - 4n * atanInv(239n);
  let E = 0n; { let t = ONE; for (let k = 1n; t > 0n; k++) { E += t; t /= k; } }
  const isqrt = n => { let x = 1n << BigInt((n.toString(2).length + 1) >> 1); for (;;) { const y = (x + n / x) >> 1n; if (y >= x) return x; x = y; } };
  const SQ5 = isqrt(5n << (2n * P));
  const PHI = (ONE + SQ5) / 2n;
  ok(inQ(PI, K.PI_BRACKET), 'pi (Machin, 600 bits) lies in the kernel\'s certified bracket');
  ok(inQ(E, K.E_BRACKET), 'e (series, 600 bits) lies in the kernel\'s bracket');
  ok(inQ(PHI, K.PHI_BRACKET), 'phi (Newton sqrt5, 600 bits) lies in the kernel\'s bracket');
  const mulF = (a, b) => (a * b) >> P, divF = (a, b) => (a << P) / b;
  const sinF = x => { let s = 0n, t = x; for (let k = 1n; t !== 0n; k += 2n) { s += t; t = -mulF(mulF(t, x), x) / ((k + 1n) * (k + 2n)); } return s; };
  const cosF = x => { let s = 0n, t = ONE; for (let k = 0n; t !== 0n; k += 2n) { s += t; t = -mulF(mulF(t, x), x) / ((k + 1n) * (k + 2n)); } return s; };
  /* at 6 terms the tail is what keeps the enclosure honest; at 40 it is
     below the grid step, so both are checked */
  let trigOk = true;
  for (const q of ['1/16', '-3/7', '22/29', '1', '-1', '5/9']) for (const n of [6, 40]) {
    const qq = Q.parse(q), t = K.trigEnclose(qq, n);
    if (!inQ(sinF(fx(qq)), t.sin) || !inQ(cosF(fx(qq)), t.cos)) trigOk = false;
  }
  ok(trigOk, 'sin and cos enclosures, at 6 terms and at 40, contain the independent 600-bit series at six rationals in [-1, 1]');
  ok((() => { try { K.trigEnclose(Q.mk(9n, 8n), 40); return false; } catch (e) { return e instanceof RangeError; } })(),
     'outside [-1, 1], where the enclosure is proved, it refuses');
  const A = divF(PHI, E), B = divF(E, PI);
  const betaF = x => mulF(A, sinF(x)) - mulF(B, mulF(mulF(x, x), x));
  let lo = fx(Q.mk(3n, 4n)), hi = ONE;
  for (let i = 0; i < 200; i++) { const m = (lo + hi) >> 1n; if (betaF(m) > 0n) lo = m; else hi = m; }
  ok(fx(K.MAYA_LAYER[0]) < lo && hi < fx(K.MAYA_LAYER[1]), `the layer phi* = ${(Number(lo >> (P - 50n)) / 2 ** 50).toFixed(12)} (independent bisection) lies inside the kernel's bracket`);
  /* the flow: 400 kernel steps of the phase against the same Euler map in
     600-bit fixed point; the distance must be within the kernel's own bound */
  let x = K.MAYA_PHASE0, err = Q.ZERO, xr = fx(K.MAYA_PHASE0), H = fx(K.MAYA_H), within = true, rising = true, below = true, Mok = true;
  let worst = 0;
  for (let s = 0; s < 400; s++) {
    const st = K.mayaStep(x, err);
    for (let j = 0; j < 5; j++) xr += mulF(H, betaF(xr));
    const d = fx(st.next) - xr, bound = fx(st.err) + 2n;
    if ((d < 0n ? -d : d) > bound) within = false;
    worst = Math.max(worst, Number(d < 0n ? -d : d) / Number(ONE));
    if (Q.cmp(st.next, st.phaseIn) <= 0) rising = false;
    if (Q.cmp(st.next, K.MAYA_LAYER[1]) >= 0) below = false;
    if (!(Q.cmp(st.M, Q.mk(19n, 20n)) > 0 && Q.cmp(st.M, Q.mk(21n, 20n)) < 0 && Q.cmp(st.M, Q.ONE) > 0)) Mok = false;
    x = st.next; err = st.err;
  }
  ok(within, `over 400 steps the stored phase stays within its certified error of the exact Euler map (largest gap ${worst.toExponential(1)}, bound ${K.Q.toString(err).length > 40 ? 'below 2^-100' : K.Q.toString(err)})`);
  ok(Q.cmp(err, Q.mk(1n, 1n << 100n)) < 0, 'and that error bound is below 2^-100');
  ok(rising && below, 'the phase rises every step and never reaches the top of the layer bracket (toward_layer)');
  ok(Mok, 'the factor stays in (1, 21/20) while the phase rises (factor_mem, factor_gt_one_iff)');
  /* measured: 0.5131 after 400 steps, 0.7417 after 600; the S-curve of
     dphi = beta/100 a step from 1/16 */
  ok(Q.cmp(x, Q.HALF) > 0 && Q.cmp(x, Q.mk(13n, 25n)) < 0, `after 400 steps the phase has flowed from 1/16 to ${(Number(fx(x) >> (P - 40n)) / 2 ** 40).toFixed(4)}, between 1/2 and 0.52`);
}

section('magnetism memory');
{
  /* Induction recomputed here from the definitions, independently of the
     kernel's indexing, and compared with what the kernel stored. */
  const X = A4;
  const base = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ONE,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.mk(3n, 2n) };
  const act = [3, 9, 11, 25, 29].map(id => ({ id, strength: Q.mk(1n, 2n) }));
  const hwf = v => (v & 1) + ((v >> 1) & 1) + ((v >> 2) & 1) + ((v >> 3) & 1);
  const edges = []; for (let a = 0; a < 4; a++) for (let v = 0; v < 16; v++) if (!(v & (1 << a))) edges.push([v, a, v | (1 << a)]);
  const cyc = [Q.ONE, Q.mk(13n, 10n), Q.ONE, Q.mk(7n, 10n)];
  const within = (got, want) => Q.cmp(Q.abs(Q.sub(got, want)), Q.mul(Q.abs(want), Q.mk(1n, 1n << 127n))) <= 0;
  let eng = K.initEngine(), inductOk = true, chiOk = true, closedOk = true, gradOk = true, maxA = Q.ZERO, recOk = true, potOk = true;
  for (let s = 0; s < 60; s++) {
    const prev = eng;
    eng = K.step(eng, base, act, 'SERIES');
    const L = eng.last, rIn = L.rhoIn, rOut = L.rhoOut;
    const Etot = rOut.reduce((t, x) => Q.add(t, x), Q.ZERO);
    edges.forEach(([v, a, h], e) => {
      const share = Q.div(Q.mul(Q.mk(8n), Q.add(rOut[v], rOut[h])), Etot);
      const k = a >= 2 ? Q.mul(Q.mk(6n, 100n), base.vortex) : Q.mul(Q.mul(Q.mk(3n, 100n), Q.HALF), base.vortex);
      const want = Q.add(Q.mul(Q.mk(499n, 500n), prev.magA[e]), Q.mul(Q.mul(k, cyc[(hwf(v) + a) % 4]), share));
      if (!within(eng.magA[e], want)) inductOk = false;
      if (Q.cmp(Q.abs(eng.magA[e]), maxA) > 0) maxA = Q.abs(eng.magA[e]);
    });
    for (let v = 0; v < 16; v++) if (!within(eng.magChi[v], Q.add(Q.mul(Q.mk(499n, 500n), prev.magChi[v]), Q.sub(rOut[v], rIn[v])))) chiOk = false;
    const md = K.magData(eng.magA, eng.magChi);
    if (!md.closed) closedOk = false;
    if (!K.curlA(K.fullA(Array(32).fill(Q.ZERO), eng.magChi)).every(Q.isZero)) gradOk = false;
    /* the potential each vertex sees: in along edges it heads, out along those it tails, minus (1/10) of chi's Laplacian */
    for (let v = 0; v < 16; v++) {
      let want = Q.ZERO;
      for (let a = 0; a < 4; a++) { const e = edges.findIndex(([t, ax]) => ax === a && t === (v & ~(1 << a))); want = (v & (1 << a)) ? Q.add(want, eng.magA[e]) : Q.sub(want, eng.magA[e]);
        want = Q.sub(want, Q.mul(Q.mk(1n, 10n), Q.sub(eng.magChi[v], eng.magChi[v ^ (1 << a)]))); }
      if (!Q.eq(md.potential[v], want)) potOk = false;
    }
    const fl = q => Number(q.n) / Number(q.d), bv = K.bvecAt(md.flux);
    edges.forEach(([v, , h], e) => {
      const a = bv[v].map(fl), b = bv[h].map(fl), na = Math.hypot(...a), nb = Math.hypot(...b);
      if (na * na <= 1e-6 || nb * nb <= 1e-6) { if (md.reconnections.includes(e)) recOk = false; return; }
      const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
      if (Math.abs(c + 0.7) > 1e-9 && (c < -0.7) !== md.reconnections.includes(e)) recOk = false;
    });
  }
  ok(inductOk, 'A = (499/500) A + omega (6/100 | eta 3/100) m_e share_e on all 32 edges for 60 steps, recomputed here');
  ok(chiOk, 'the Faraday potential chi = (499/500) chi + (rho_out - rho_in), recomputed here');
  ok(gradOk, 'the Faraday part of A carries no flux on any of those steps (faraday_flux_free)');
  ok(closedOk, 'the flux out of every one of the 8 cubes is zero on every step (flux_closed)');
  ok(potOk, 'the potential each vertex turns by is div A - (1/10) L chi, recomputed here');
  ok(recOk, 'reconnections are exactly the edges where cos(B_i, B_j) < -0.7 (reconnect_iff)');
  /* retention_bound: sources are at most |omega| (6/100)(13/10) 16 per edge
     (shares are at most 16), so |A| never exceeds 500 times that */
  const smax = Q.mul(Q.mul(Q.abs(base.vortex), Q.mk(6n * 13n * 16n, 1000n)), Q.mk(1001n, 1000n));
  ok(Q.cmp(maxA, Q.mul(smax, Q.mk(500n))) <= 0, `|A| stays within max |A_0|, s/(1 - r) (retention_bound): ${(Number(maxA.n) / Number(maxA.d)).toFixed(3)} <= ${(Number(smax.n) / Number(smax.d) * 500).toFixed(1)}`);
  ok(Q.isZero(K.helicityOf(Array(32).fill(Q.ZERO), eng.magChi)), 'a potential that is all Faraday has no helicity (helicity_of_gradient)');
  const sh = K.edgeShares(eng.last.rhoOut);
  ok(Q.eq(sh.reduce((t, x) => Q.add(t, x), Q.ZERO), Q.mk(32n)), 'the edge shares sum to 32 (edge_share_total)');
  ok(eng.last && JSON.parse(JSON.stringify(K.serializeLast(eng.last))).magnetismStage.energyPass, 'the magnetism stage keeps the energy exactly (phase_energy)');
  ok((() => { try { K.edgeShares(Array(16).fill(Q.ZERO)); return false; } catch (e) { return e instanceof RangeError; } })(),
     'a field with no energy has no shares: refused, not divided by zero');
}

section('no parameter is ignored');
{
  /* the vortex, the drive and the polarisation each change the run: run 8
     steps with the parameter at a value and at another, from the same seed */
  const X = A4;
  const base = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ZERO };
  const act = [3, 7, 9].map(id => ({ id, strength: Q.mk(1n, 2n) }));
  const run = p => { let e = K.initEngine(); for (let s = 0; s < 8; s++) e = K.step(e, p, act, 'SERIES'); return e; };
  const ref = run(base), key = e => JSON.stringify(K.serializeState(e.psi)) + JSON.stringify(e.dielectric.map(K.serializeQ));
  ok(key(run({ ...base, vortex: Q.ONE })) !== key(ref), 'vortex omega changes the field');
  ok(key(run({ ...base, zpeSource: X.ONE })) !== key(ref), 'the drive changes the field');
  ok(key(run({ ...base, polarisation: Q.ZERO })) !== key(ref), 'the polarisation changes the dielectric and the field');
  const d0 = K.evolveDielectric(K.initDielectric(), act, base, Array(16).fill(Q.ZERO));
  const rho = K.initState().map(z => K.C4.norm2(z)).map(x => Q.add(x.a, Q.ZERO));
  const d1 = K.evolveDielectric(K.initDielectric(), act, { ...base, polarisation: Q.mk(3n, 2n) }, rho);
  ok(d1.every((q, v) => Q.eq(Q.sub(q, d0[v]), Q.mul(Q.mul(Q.mk(3n, 2n), rho[v]), Q.mk(1n, 8n)))),
     'polarisation adds exactly chi rho / 8 to each vertex of the dielectric');
  ok((() => { try { K.evolveDielectric(K.initDielectric(), act, base); return false; } catch (e) { return e instanceof RangeError; } })(),
     'the dielectric update without the field energies is refused');
}

section('R4 gate weights');
{
  /* stored a_v, b_v = 1 - a_v: in [0, 1] and summing to 1 (store_mem_unit) */
  const field = K.initDielectric();
  let good = true;
  for (const s4 of [Q.mk(1n, 1000n), Q.ONE, Q.mk(77n, 3n)]) for (const w of K.gateABField(field, s4))
    if (!(Q.sign(w.a) >= 0 && Q.sign(w.b) >= 0 && Q.eq(Q.add(w.a, w.b), Q.ONE) && Q.cmp(w.a, Q.ONE) <= 0)) good = false;
  ok(good, 'the gate weights lie in [0, 1] and sum to 1 at three scales');
}

section('presets develop over the run as their descriptions say');
{
  /* Each preset's description, its button title, states what it does over
     600 steps from the seed: how much the flux grows from step 60 to step
     600, whether R4 stays dormant or when it takes over, the vertices holding
     80% of the energy over steps 20-40 and over steps 540-600, and -- for all
     but SQUARE, which R4 holds -- that the field is still moving after step
     300 (L1 at least 0.05 between the mean shares of steps 300-360 and
     540-600). Every one is run here for 600 steps, four at a time. */
  const EXPECT = {
    "pre-cymatic": { line: "  'pre-cymatic':        { comp: 'PARALLEL',             set: { 9: 25, 17: 25, 27: 50, 28: 50 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 28, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0011", "0100", "0111", "1000", "1011", "1100", "1110", "1111"], "late": ["0001", "0010", "0011", "0101", "1010", "1100", "1110"], "moving": true} },
    "pre-bindu": { line: "  'pre-bindu':          { comp: 'PARALLEL',             set: { 5: 25, 12: 50, 28: 75 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 216, "r4": 0, "r4first": -1, "early": ["0000", "0010", "0011", "0100", "0101", "0110", "1001", "1010", "1011", "1100"], "late": ["0011", "1010", "1011", "1110", "1111"], "moving": true} },
    "pre-star": { line: "  'pre-star':           { comp: 'SYMMETRIC_CONCURRENT', set: { 13: 100, 20: 75, 22: 50 }, omega: 10, drive: 1, chi: 10 },",
      claim: {"growth": 20, "r4": 0, "r4first": -1, "early": ["0000", "1111"], "late": ["1001", "1010", "1100"], "moving": true} },
    "pre-crown": { line: "  'pre-crown':          { comp: 'SERIES',               set: { 5: 30, 7: 5, 13: 40, 18: 35, 19: 55, 20: 90, 22: 20 }, args: { 13: ['5/2'], 19: ['23/4'], 20: ['7'] }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 160, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0100", "0101", "0110", "0111", "1000", "1001", "1110", "1111"], "late": ["0001", "0010", "0011", "0100", "0110", "1011", "1100", "1110"], "moving": true} },
    "pre-square": { line: "  'pre-square':         { comp: 'PARALLEL',             set: { 8: 50, 16: 50, 21: 100, 29: 25 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 41, "r4": 368, "r4first": 208, "early": ["0000", "1111"], "late": ["0000", "1111"], "moving": false} },
    "pre-equilibrium": { line: "  'pre-equilibrium':    { comp: 'SERIES',               set: { 7: 50, 19: 75, 22: 100 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 265, "r4": 0, "r4first": -1, "early": ["0000", "0011", "0100", "0101", "0110", "1001", "1010", "1100", "1111"], "late": ["0001", "0101", "0110", "0111", "1000", "1001", "1010", "1110"], "moving": true} },
    "pre-cube": { line: "  'pre-cube':           { comp: 'SYMMETRIC_CONCURRENT', set: { 16: 75, 22: 75 }, args: { 16: ['17/4'] }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 7.1, "r4": 0, "r4first": -1, "early": ["1000", "1001", "1010", "1011", "1100"], "late": ["1100"], "moving": true} },
    "pre-octad": { line: "  'pre-octad':          { comp: 'SYMMETRIC_CONCURRENT', set: { 14: 25, 17: 100, 22: 50 }, args: { 14: ['11/4'], 17: ['23/4'] }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 23, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0010", "0101", "1000", "1010", "1111"], "late": ["0000", "0011", "0101", "0110", "1001", "1010", "1111"], "moving": true} },
    "pre-symmetry": { line: "  'pre-symmetry':       { comp: 'PARALLEL',             set: { 5: 50, 7: 50, 12: 50, 22: 50, 29: 50 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 27, "r4": 597, "r4first": 1, "early": ["0000", "0100", "0101", "0110", "1001", "1010", "1101", "1110", "1111"], "late": ["0001", "0101", "0111", "1000", "1110"], "moving": true} },
    "pre-wormhole": { line: "  'pre-wormhole':       { comp: 'SERIES',               set: { 5: 50, 9: 50, 22: 50, 23: 50, 26: 50, 29: 50 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 16, "r4": 0, "r4first": -1, "early": ["0001", "0010", "0011", "0100", "0101", "0110", "0111", "1001", "1011", "1101", "1111"], "late": ["0010", "0011", "0100", "0101", "1101"], "moving": true} },
    "pre-singularity": { line: "  'pre-singularity':    { comp: 'SERIES',               set: { 1: 50, 10: 50, 14: 50, 15: 50 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 43, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0100", "0101", "0110", "1001", "1010", "1101", "1110", "1111"], "late": ["0000", "0100", "0110", "1000", "1001", "1010", "1101", "1110", "1111"], "moving": true} },
    "pre-magnetic": { line: "  'pre-magnetic':       { comp: 'SERIES',               set: { 3: 50, 9: 50, 11: 50, 25: 50, 29: 50 }, omega: 10, drive: 4, chi: 10 },",
      claim: {"growth": 28, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0010", "1001", "1010", "1100", "1111"], "late": ["0000", "0101", "0110", "1001", "1010", "1100"], "moving": true} },
    "pre-cancel": { line: "  'pre-cancel':         { comp: 'SERIES',               set: { 5: 85, 7: 35, 8: 15, 9: 65, 11: 20, 22: 80, 28: 30, 29: 55 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 3.4, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0011", "0101", "0110", "0111", "1001", "1010", "1111"], "late": ["0101", "0110", "1010"], "moving": true} },
    "pre-dynamo": { line: "  'pre-dynamo':         { comp: 'SYMMETRIC_CONCURRENT', set: { 3: 50, 7: 50, 9: 50, 11: 50, 25: 50, 29: 50 }, omega: 10, drive: 4, chi: 10 },",
      claim: {"growth": 83, "r4": 0, "r4first": -1, "early": ["0000", "0010", "0011", "0100", "0101", "0110", "1001", "1010", "1100"], "late": ["0000", "0011", "0101", "1000", "1001", "1010", "1100", "1110"], "moving": true} },
    "pre-reconnect": { line: "  'pre-reconnect':      { comp: 'SYMMETRIC_CONCURRENT', set: { 5: 50, 9: 50, 22: 50, 26: 50 }, omega: 10, drive: 4, chi: 10 },",
      claim: {"growth": 42, "r4": 0, "r4first": -1, "early": ["0100", "0101", "0110", "0111", "1000", "1001", "1010", "1011", "1110", "1111"], "late": ["0011", "0101", "0110", "1001", "1010", "1100", "1101"], "moving": true} },
    "pre-inverse-growth": { line: "  'pre-inverse-growth': { comp: 'SYMMETRIC_CONCURRENT', set: { 1: 30, 5: 70, 14: 30, 29: 70 }, omega: 10, drive: 0, chi: 10 },",
      claim: {"growth": 5.5, "r4": 0, "r4first": -1, "early": ["0000", "0001", "0011", "0100", "0101", "0111", "1001", "1010", "1110"], "late": ["0011", "0100", "1011"], "moving": true} },
    "pre-all29": { line: "  'pre-all29':          { comp: 'SYMMETRIC_CONCURRENT', set: Object.fromEntries(SUTRAS.map(s => [s.id, 25])), omega: 10, drive: 4, chi: 10 }",
      claim: {"growth": 40, "r4": 0, "r4first": -1, "early": ["0001", "0011", "0101", "0110", "1001", "1011", "1100", "1101", "1111"], "late": ["0011", "0100", "0101", "0110", "0111", "1001", "1010", "1100"], "moving": true} }
  };

  const { Worker } = await import('node:worker_threads');
  const KSRC = HTML.slice(start, end);
  const WORKER = `
    const { parentPort, workerData } = require('node:worker_threads');
    const K = new Function(workerData.src + '\\nreturn KERNEL_FACTORY();')();
    const { Q, A4 } = K, X = A4;
    const bl = x => x === 0n ? 0 : (x < 0n ? -x : x).toString(2).length;
    const qNum = q => { let n = q.n; if (n === 0n) return 0; const neg = n < 0n; if (neg) n = -n; const sh = BigInt(Math.max(0, 64 + bl(q.d) - bl(n))); const v = Number((n << sh) / q.d) / 2 ** Number(sh); return neg ? -v : v; };
    const probs = psi => { let e = null;
      for (const z of psi) for (const a of [z.re, z.im]) for (const q of [a.a, a.b, a.c, a.d]) if (q.n !== 0n) { const x = bl(q.n) - bl(q.d); if (e === null || x > e) e = x; }
      const sc = q => q.n === 0n ? q : (e >= 0 ? Q.mk(q.n, q.d << BigInt(e)) : Q.mk(q.n << BigInt(-e), q.d));
      const num = a => qNum(sc(a.a)) + qNum(sc(a.b)) * Math.SQRT2 + qNum(sc(a.c)) * Math.sqrt(5) + qNum(sc(a.d)) * Math.sqrt(10);
      const p = psi.map(z => { const r = num(z.re), i = num(z.im); return r * r + i * i; }); const t = p.reduce((a, b) => a + b, 0); return p.map(v => v / t); };
    const out = {};
    for (const [id, x] of workerData.presets) {
      const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
        gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
        tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.fromQ(Q.mk(BigInt(x.drive), 4n)),
        r4R: Q.TWO, geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
        polarisation: Q.mk(BigInt(x.chi), 10n), vortex: Q.mk(BigInt(x.omega), 10n) };
      const active = Object.entries(x.set).map(([k, v]) => { const sid = Number(k), st = Q.mk(BigInt(v), 100n);
        return x.args && x.args[k] ? { id: sid, strength: st, args: [st, ...x.args[k].map(t => Q.parse(t))] } : { id: sid, strength: st }; });
      let eng = K.initEngine(); const H = []; let r4 = 0, r4first = -1, UB60 = 0, UB600 = 0;
      for (let s = 0; s < 600; s++) {
        eng = K.step(eng, params, active, x.comp);
        H.push(probs(eng.psi));
        if (eng.last.r4.engaged) { r4++; if (r4first < 0) r4first = s + 1; }
        if (s + 1 === 60) UB60 = qNum(K.magData(eng.magA, eng.magChi).fluxEnergy);
        if (s + 1 === 600) UB600 = qNum(K.magData(eng.magA, eng.magChi).fluxEnergy);
      }
      const avg = (a, b) => { const r = new Array(16).fill(0); for (let i = a; i < b; i++) H[i].forEach((v, k) => r[k] += v / (b - a)); return r; };
      const supp = w => { const o = [...w.keys()].sort((a, b) => w[b] - w[a]); const S = []; let c = 0; for (const v of o) { S.push(v); c += w[v]; if (c >= 0.8) break; } return S.sort((a, b) => a - b).map(v => v.toString(2).padStart(4, '0')); };
      const L1 = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
      const late = avg(540, 600);
      out[id] = { r4, r4first, growth: UB600 / UB60, early: supp(avg(20, 40)), late: supp(late), moving: L1(avg(300, 360), late), lateVec: late };
    }
    parentPort.postMessage(out);`;
  const defs = Object.entries(EXPECT).map(([id, x]) => {
    const m = x.line.match(/comp: '(\w+)',\s*set: (\{[^}]*\}|Object\.fromEntries\(SUTRAS[^)]*\)\)),(?: args: (\{[^}]*\]\s*\}),)? omega: (-?\d+), drive: (\d+), chi: (\d+) \}/);
    if (!m) throw new Error(`${id}: cannot read its definition`);
    /* ALL 29 is every sutra at 25%, written with the page's SUTRAS table */
    const set = m[2].startsWith('Object') ? Object.fromEntries(Array.from({ length: 29 }, (_, i) => [i + 1, 25])) : Function('return ' + m[2])();
    return [id, { comp: m[1], set, args: m[3] ? Function('return ' + m[3])() : null, omega: +m[4], drive: +m[5], chi: +m[6] }];
  });
  const groups = [[], [], [], []];
  defs.forEach((d, i) => groups[i % 4].push(d));
  const results = Object.assign({}, ...await Promise.all(groups.map(g => new Promise((res, rej) => {
    const w = new Worker(WORKER, { eval: true, workerData: { src: KSRC, presets: g } });
    w.once('message', res); w.once('error', rej);
  }))));
  const vsList = l => l.length > 1 ? l.slice(0, -1).join(', ') + ' and ' + l[l.length - 1] : l[0];
  for (const [id, x] of Object.entries(EXPECT)) {
    const c = x.claim, r = results[id];
    ok(HTML.includes(x.line), `${id}: the page defines it as tested here`);
    const tm = HTML.match(new RegExp(`id="${id}" title="([^"]*)"`)), title = tm ? tm[1].replace(/&amp;/g, '&') : '';
    const gs = c.growth >= 10 ? `×${c.growth}` : `×${c.growth.toFixed(1)}`;
    ok(title.includes(gs) && title.includes(vsList(c.late)) && (c.r4 === 0 ? title.includes('R4 stays dormant') : title.includes(`step ${c.r4first}`)),
       `${id}: its description states these claims`, title);
    ok(r.growth >= c.growth && r.growth < (c.growth >= 10 ? c.growth + 1 : c.growth + 0.1), `${id}: the flux grows ${gs} from step 60 to step 600 (${r.growth.toFixed(2)})`);
    ok(r.r4 === c.r4 && r.r4first === c.r4first, c.r4 === 0 ? `${id}: R4 stays dormant for 600 steps` : `${id}: R4 takes over at step ${c.r4first} and engages on ${c.r4} of 600 steps`, `R4 ${r.r4} from ${r.r4first}`);
    ok(r.early.join(' ') === c.early.join(' ') && r.late.join(' ') === c.late.join(' '),
       `${id}: the field moves from ${c.early.length} vertices onto ${c.late.join(' ')}`, `${r.early.join(' ')} -> ${r.late.join(' ')}`);
    if (c.moving) ok(r.moving >= 0.05, `${id}: still moving after step 300 (L1 ${r.moving.toFixed(3)})`);
  }
  let closest = Infinity, pair = '';
  const ids = Object.keys(results);
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const d = results[ids[i]].lateVec.reduce((s, v, k) => s + Math.abs(v - results[ids[j]].lateVec[k]), 0);
    if (d < closest) { closest = d; pair = ids[i] + ' and ' + ids[j]; }
  }
  /* measured 0.437, all29 and reconnect */
  ok(closest >= 0.4, `every two presets end at least 0.4 apart in L1 (closest ${pair}, ${closest.toFixed(3)})`);
}


console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
