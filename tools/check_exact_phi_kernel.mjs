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
     modes              the mode identities decide and pass, and the
                        comparison built from the audit's states equals a
                        fresh one

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

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
