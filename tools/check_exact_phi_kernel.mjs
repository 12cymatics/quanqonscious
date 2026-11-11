/* Gate for the exact kernel inside vedic_v18.51.1_exact_phi.html.

   The kernel is not copied here. KERNEL_FACTORY is sliced out of the page and
   evaluated, so there is one copy of the formulas and nothing that can drift
   from it. Where a statement can be checked against something the kernel did
   not compute, it is: the 29 formulas and GRVQ against
   tools/vedic_reference.py, a separate transcription of the formulas
   in Python's exact Fraction; reduced fractions against a
   Euclid written here; the certified arithmetic in bare BigInt arithmetic;
   the TGCR geometry, the stages and the couplings recomputed here from their
   definitions. Where a run carries the field in certified intervals, an
   identity recomputed here is checked to the width of the intervals: the
   two enclosures must overlap and each be narrower than 2^-200 of the value.

   What it holds the page to:
     self-tests         every one of runSelfTests passes
     canonical Q        Q.mk reduces by the true gcd, dyadic or not
     storage            storeQ returns the value it is given: nothing is rounded
     certified Q        beyond 4096 bits a value is an interval that contains
                        it; every operation's interval contains the exact
                        result; an undecided decision is refused, never
                        guessed, and is decided by replaying at more bits
     refusal            an operand or strength outside its domain throws
     formulas           each of the 29 equals the reference exactly, at its
                        default and its extreme operands, at two gains and
                        several times, on exact states of every sign, with
                        no limit on exact size, and refuses exactly where the
                        reference divides by zero; the shipped intervals
                        contain every exact result; GRVQ's increment and
                        MSTVQ's B likewise
     statements         every statement the sutra channel decides holds on
                        the corpus and on a run
     couplings          lambda is inverse psi; each GRVQ feed moves exactly
                        its lambda by GRVQ x its sutra's alpha; MSTVQ's
                        feedback; kappa
     stages             the magnetism, MSTVQ, TGCR and ZPE rotations keep
                        the energy exactly; ZPE adds only its source; the
                        TGCR geometry recomputed by walking faces
     step export        an exported step replays bit for bit from its file
     the formula        calculated once at the settings, put in over the
                        glide in equal shares, then none applied; the
                        energy then does not move
     glide              each strength rises in equal steps from where it
                        stood to its setting over the glide, then holds
     modes              the mode identities decide and pass
     Maya, magnetism, torus, sources, parameters

   Run: node tools/check_exact_phi_kernel.mjs
*/
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import vm from 'node:vm';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PAGE = join(ROOT, 'vedic_v18.51.1_exact_phi.html');
const HTML = readFileSync(PAGE, 'utf8');
const start = HTML.indexOf('const KERNEL_FACTORY = () => {');
const end = HTML.indexOf('\nconst KERNEL = KERNEL_FACTORY();');
if (start < 0 || end < 0 || end < start) {
  console.error('FAIL -- KERNEL_FACTORY could not be located in the page');
  process.exit(1);
}
const K = new vm.Script(HTML.slice(start, end) + '\nKERNEL_FACTORY();',
  { filename: PAGE, lineOffset: HTML.slice(0, start).split('\n').length - 1 }).runInThisContext();
const { Q, A4 } = K;

let pass = 0;
const failures = [];
function ok(cond, label, detail) {
  if (cond) { pass++; console.log('  ok   ' + label); }
  else { failures.push(label + (detail ? '  -- ' + detail : '')); console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
}
const section = t => console.log('\n' + t);
const throwsRange = f => { try { f(); return false; } catch (e) { return e instanceof RangeError; } };

let seed = 0x5eedn;
const rnd = bits => {
  let r = 0n;
  for (let i = 0; i < bits; i += 31) { seed = (seed * 1103515245n + 12345n) % 2147483648n; r = (r << 31n) | seed; }
  return r >> BigInt(Math.ceil(bits / 31) * 31 - bits);
};
const euclid = (a, b) => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
const X = A4;
const PARAMS = (o = {}) => ({ dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
  appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
  tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
  geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
  polarisation: Q.ONE, vortex: Q.ZERO, ...o });
const S = x => {
  if (!Q.isExact(x)) throw new Error(`an interval where an exact value is compared: ${Q.toString(x)}`);
  return { n: x.n.toString(), d: x.d.toString() };
};
const modelRun = (act, mode, glide = 10) => {
  let formula = null, ramp = [], k = 0;
  return (eng, p) => {
    if (!formula) formula = K.formulaOf(eng.psi, act, mode, K.timeOf(eng.frame));
    const rs = K.rampStep(ramp, act, glide); ramp = rs.ramp;
    const next = K.step(eng, p, rs.active, mode, formula, k < glide ? Q.mk(1n, BigInt(glide)) : Q.ZERO);
    if (k < glide) k++;
    return next;
  };
};
const fromQ = o => 'lo' in o ? Object.freeze({ lo: fromQ(o.lo), hi: fromQ(o.hi) }) : Q.mk(BigInt(o.n), BigInt(o.d));
const ident = (x, y) => Q.isExact(x) ? Q.isExact(y) && x.n === y.n && x.d === y.d : !Q.isExact(y) && ident(x.lo, y.lo) && ident(x.hi, y.hi);
const sameQ = (a, b) => a.length === b.length && a.every((x, i) => ident(x, b[i]));
const absB = v => v < 0n ? -v : v;
const TIGHT = 200n;
const tight = x => {
  if (Q.isExact(x)) return true;
  const wn = x.hi.n * x.lo.d - x.lo.n * x.hi.d, wd = x.hi.d * x.lo.d;
  return [[1n, 1n], [absB(x.lo.n), x.lo.d], [absB(x.hi.n), x.hi.d]].some(([mn, md]) => (wn << TIGHT) * md <= mn * wd);
};
const near = (x, y) => Q.agree(x, y) && tight(x) && tight(y);
const nearAll = (a, b) => a.length === b.length && a.every((x, i) => near(x, b[i]));
const zeroish = x => Q.mayZero(x) && tight(x);
const apart = (a, b) => a.some((x, i) => !Q.agree(x, b[i]));
const mid = q => Q.isExact(q) ? q : Q.mul(Q.add(q.lo, q.hi), Q.HALF);
const exactly = f => { K.setExactBits(null); try { return f(); } finally { K.setExactBits(4096); } };
const energyQ = p => p.reduce((s, x) => Q.add(s, Q.mul(x, x)), Q.ZERO);
const reference = cases => JSON.parse(execFileSync('python3', [join(ROOT, 'tools', 'vedic_reference.py')],
  { input: JSON.stringify({ cases }), maxBuffer: 1 << 28 }).toString()).results;

section('self-tests');
let tests = [];
try { tests = K.runSelfTests(); } catch (e) { ok(false, 'runSelfTests', e.message); }
ok(tests.length >= 90, `runSelfTests passes all ${tests.length}`);


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
  ok(K.STORE_BITS === null, 'STORE_BITS is null: no value is rounded where it is stored');
  let moved = 0, n = 0;
  for (let t = 0; t < 2000; t++) {
    let num = rnd(1 + (t * 29) % 700) + 1n;
    const den = rnd(1 + (t * 41) % 700) + 1n;
    if (t % 2) num = -num;
    const x = Q.mk(num, den);
    n++;
    if (K.storeQ(x) !== x) moved++;
  }
  const wide = Q.add(Q.mk(rnd(4200) | 1n, rnd(4200) | 1n), Q.ZERO);
  ok(moved === 0 && !Q.isExact(wide) && K.storeQ(wide) === wide,
     `storeQ returns each of ${n} rationals, and an interval, as the very value it was given`, `${moved} moved`);
  const refuses = v => { try { K.storeQ(v); return false; } catch (e) { return e instanceof TypeError; } };
  ok([1.5, NaN, 3n, null, { n: 1n }, { n: 1n, d: 0n }, { lo: Q.ONE, hi: Q.ZERO }].every(refuses),
     'storeQ refuses anything that is not an exact rational or an interval with its ends in order');
  ok(Q.isExact(K.storeBudget()) && K.storeBudget().n === 0n, 'the rounding a storage boundary adds is exactly zero');
}


section('certified arithmetic');
{
  ok(K.workBits() === 256 && K.exactBits() === 4096 && K.MAX_WORK_BITS === 65536,
     'a value is exact up to 4096 bits, then an interval at 256 bits, replayed at up to 65536');
  const W = BigInt(K.workBits());
  const le = (a, b) => a.n * b.d <= b.n * a.d;
  const bare = {
    add: (a, b) => ({ n: a.n * b.d + b.n * a.d, d: a.d * b.d }),
    sub: (a, b) => ({ n: a.n * b.d - b.n * a.d, d: a.d * b.d }),
    mul: (a, b) => ({ n: a.n * b.n, d: a.d * b.d }),
    div: (a, b) => b.n < 0n ? { n: -a.n * b.d, d: a.d * -b.n } : { n: a.n * b.d, d: a.d * b.n }
  };
  const holds = (v, x) => Q.isExact(v) ? v.n * x.d === x.n * v.d : le(v.lo, x) && le(x, v.hi);
  const pow2 = d => (d & (d - 1n)) === 0n;
  const sgnB = x => x.n > 0n ? 1 : x.n < 0n ? -1 : 0;
  const floorB = x => { const z = x.n / x.d; return x.n % x.d !== 0n && x.n < 0n ? z - 1n : z; };
  const asQ = x => Q.mk(x.n, x.d);
  const xs = [];
  let enclosed = 0, dyadic = 0, narrow = 0;
  for (let t = 0; t < 120; t++) {
    const n = (rnd(2100 + (t * 13) % 900) + 1n) * (t % 3 ? 1n : -1n), d = rnd(2100 + (t * 7) % 900) + 1n;
    const x = { n, d }, v = Q.add(asQ(x), Q.ZERO);
    if (Q.isExact(v)) continue;
    xs.push([x, v]);
    if (le(v.lo, x) && le(x, v.hi) && !le(v.hi, v.lo)) enclosed++;
    if (pow2(v.lo.d) && pow2(v.hi.d)) dyadic++;
    const wn = v.hi.n * v.lo.d - v.lo.n * v.hi.d, wd = v.hi.d * v.lo.d;
    if ((wn << (W - 1n)) * x.d <= absB(x.n) * wd) narrow++;
  }
  const n = xs.length;
  ok(n >= 100 && enclosed === n && dyadic === n && narrow === n,
     `${n} rationals of more than 4096 bits are each carried as an interval with dyadic ends that contains it and is at most 2^-255 of it wide, checked in bare BigInt`,
     `${enclosed} enclosed, ${dyadic} dyadic, ${narrow} narrow`);
  let opsOk = 0, opsN = 0;
  const bad = [];
  for (let t = 0; t + 1 < xs.length; t += 2) {
    const [x, u] = xs[t], [y, v] = xs[t + 1], small = { n: BigInt(t) - 51n, d: 7n };
    for (const op of ['add', 'sub', 'mul', 'div']) {
      opsN += 2;
      if (holds(Q[op](u, v), bare[op](x, y))) opsOk++; else bad.push(op);
      if (holds(Q[op](u, asQ(small)), bare[op](x, small))) opsOk++; else bad.push(op + ' by a small rational');
    }
    opsN += 2;
    if (holds(Q.neg(u), { n: -x.n, d: x.d })) opsOk++; else bad.push('neg');
    if (holds(Q.abs(u), { n: absB(x.n), d: x.d })) opsOk++; else bad.push('abs');
  }
  ok(opsOk === opsN, `${opsN} interval sums, differences, products, quotients, negations and absolute values each contain the exact result, computed in bare BigInt`,
     bad.slice(0, 4).join(', '));
  let exactOk = 0, exactN = 0;
  for (let t = 0; t < 400; t++) {
    const x = { n: rnd(1 + t % 900) - rnd(1 + (t * 3) % 900), d: rnd(1 + (t * 5) % 900) + 1n };
    const y = { n: rnd(1 + (t * 7) % 900) + 1n, d: rnd(1 + (t * 11) % 900) + 1n };
    for (const op of ['add', 'sub', 'mul', 'div']) {
      exactN++;
      const got = Q[op](asQ(x), asQ(y)), w = bare[op](x, y);
      if (Q.isExact(got) && got.n * w.d === w.n * got.d && euclid(got.n, got.d) === 1n) exactOk++;
    }
  }
  ok(exactOk === exactN, `${exactN} operations on rationals that stay under 4096 bits give the exact reduced result, checked in bare BigInt`);
  let decided = 0, decN = 0, floorBits = 0, replayed = 0;
  for (let t = 0; t + 1 < xs.length; t += 2) {
    const [x, u] = xs[t], [y, v] = xs[t + 1];
    decN += 3;
    if (Q.sign(u) === sgnB(x)) decided++;
    if (Q.cmp(u, v) === sgnB(bare.sub(x, y))) decided++;
    const f = K.certify(() => Q.floor(Q.add(asQ(x), Q.ZERO)), 256);
    if (f.value === floorB(x)) decided++;
    if (f.bits > 256) replayed++;
    floorBits = Math.max(floorBits, f.bits);
  }
  ok(decided === decN && replayed > 0 && K.workBits() === 256,
     `${decN} signs, comparisons and floors decided from intervals are the exact ones, computed in bare BigInt; ${replayed} floors of values with long integer parts were replayed, at up to ${floorBits} bits`);
  const [x0, u0] = xs[0];
  const z = Q.sub(u0, u0);
  const undecided = f => { try { f(); return false; } catch (e) { return e instanceof K.NeedPrecision; } };
  ok(!Q.isExact(z) && [() => Q.sign(z), () => Q.isZero(z), () => Q.eq(z, Q.ZERO), () => Q.cmp(z, Q.ZERO), () => Q.floor(z), () => Q.div(Q.ONE, z)].every(undecided),
     'u - u for an interval u reaches across zero, and its sign, zero test, equality, comparison, floor and reciprocal are each refused as undecided');
  ok(Q.agree(z, Q.ZERO) && Q.mayZero(z) && !Q.agree(u0, Q.add(u0, Q.ONE)) && !Q.mayZero(u0) && throwsRange(() => Q.div(u0, Q.ZERO)),
     'a claim is refuted only by disjoint ends: u - u may be zero, u and u + 1 cannot agree, and a division by an exact zero is a RangeError');
  const x0q = asQ(x0);
  const nudged = k => () => Q.sign(Q.sub(Q.add(x0q, asQ({ n: x0.n, d: x0.d << BigInt(k) })), x0q));
  const c = K.certify(nudged(900), 256);
  ok(c.value === sgnB(x0) && c.bits === 1024 && K.workBits() === 256,
     `the sign of (x + x 2^-900) - x, undecided at 256 bits, is decided by replaying at 512 and then 1024 bits (${c.bits}), and the working precision is restored`);
  let refusedAt = null;
  try { K.certify(nudged(70000), 256); } catch (e) { refusedAt = e instanceof RangeError ? K.workBits() : 'not a RangeError: ' + e.message; }
  ok(refusedAt === 256, 'the sign of (x + x 2^-70000) - x is refused with a RangeError once 65536 bits do not decide it, and the working precision is restored', String(refusedAt));
  const kept = exactly(() => Q.add(x0q, Q.ZERO));
  ok(Q.isExact(kept) && kept.n * x0.d === x0.n * kept.d && K.exactBits() === 4096,
     'with no limit on exact size the same value stays an exact rational, and the limit is restored');
}


section('refusal, not clamping');
{
  ok(Object.keys(K.SUTRA_ARG_SPEC).length === 29 && Object.values(K.SUTRA_ARG_SPEC).every(a => a.length === 0),
     'no method takes an operand from a control: every operand is read from the field');
  let strength = 0;
  for (let id = 1; id <= 29; id++) {
    if (throwsRange(() => K.checkStrength(id, Q.mk(101n, 100n)))) strength++;
    if (throwsRange(() => K.checkStrength(id, Q.mk(-1n, 100n)))) strength++;
  }
  ok(strength === 58, `${strength}/58 strengths outside [0, 1] refused`);
}


section('the 29 formulas against an independent transcription');
{
  const states = [['the seed', K.initState()]];
  for (let k = 0; k < 6; k++) states.push([`state ${k}`, Array.from({ length: 16 }, () => {
    const n = rnd(6) - 32n + BigInt(k), d = rnd(2) + 1n; return Q.mk(n === 0n ? 1n : n, d); })]);
  states.push(['odd denominators', Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(v * v * 37 - 4000), BigInt(300 + 200 * (v % 5))))]);
  states.push(['inside (-1, 1)', Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v * 7919) % 199) - 99), 101n))]);
  states.push(['tens', Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v + 3) * 104729) % 91 - 45), 7n))]);
  states.push(['all negative', K.initState().map(x => Q.neg(Q.abs(x)))]);
  const zeroAt = (p, v) => p.map((x, i) => i === v ? Q.ZERO : x);
  const seedS = K.initState();
  states.push(['a zero base vertex (S6)', zeroAt(seedS, 0)]);
  states.push(['a zero complement numerator (S13)', zeroAt(seedS, 12)]);
  states.push(['a zero root (S19)', zeroAt(seedS, 9)]);
  const evenSum = seedS.filter((_, i) => !(i & 1)).reduce((s, x) => Q.add(s, x), Q.ZERO);
  states.push(['a zero half-sum (S25)', seedS.map((x, v) => v === 0 ? Q.sub(x, evenSum) : x)]);
  states.push(['lambda_0 = lambda_2 = 0 (GRVQ)', K.forwardL([Q.ZERO, Q.mk(3n), Q.ZERO, Q.mk(5n)]).map((x, v) => Q.add(x, K.initDielectric()[v]))]);
  const times = [Q.ZERO, Q.mk(1500n), Q.mk(2000n), Q.mk(9001n, 3n)];
  const cases = [], meta = [];
  states.forEach(([name, psi], si) => {
    for (let id = 1; id <= 29; id++) {
      const spec = K.SUTRA_ARG_SPEC[id], dom = K.ARG_DOMAIN[id];
      const vecs = [[Q.ONE, ...spec.map(a => a.def)], [Q.mk(3n, 7n), ...spec.map(a => a.def)]];
      spec.forEach((a, i) => { for (const b of [dom[i].lo, dom[i].hi]) { const v = [Q.ONE, ...spec.map(x => x.def)]; v[i + 1] = b; vecs.push(v); } });
      vecs.forEach((a, vi) => {
        const gain = vi % 2 ? Q.HALF : Q.ONE, t = times[(si + id + vi) % times.length];
        let js;
        try { const c = K.certify(() => K.applySutra(id, psi, a, { t, gain })); js = { q: c.value, bits: c.bits }; }
        catch (e) { js = { refused: e instanceof RangeError ? e.message : 'not a RangeError: ' + e.message }; }
        cases.push({ kind: 'sutra', id, psi: psi.map(S), args: a.map(S), gain: S(gain), t: S(t) });
        meta.push({ id, name, js, psi, a, gain, t });
      });
    }
  });
  const out = reference(cases);
  let agree = 0, refusedBoth = 0, exactAlso = 0, widest = 0;
  const bad = [], refusals = {};
  const within = (v, x) => Q.isExact(v) ? ident(v, x) : Q.mayLe(v.lo, x) && Q.mayLe(x, v.hi) && tight(v);
  out.forEach((r, i) => {
    const m = meta[i];
    if (r.refused || m.js.refused) {
      if (r.refused && m.js.refused && !m.js.refused.startsWith('not a')) { refusedBoth++; (refusals[m.name] = refusals[m.name] || new Set()).add(m.id); }
      else bad.push(`S${m.id} on ${m.name}: reference ${r.refused ? 'refuses (' + r.refused + ')' : 'computes'}, kernel ${m.js.refused || 'computes'}`);
      return;
    }
    widest = Math.max(widest, m.js.bits);
    if (m.js.q.every(Q.isExact)) exactAlso++;
    if (r.psi.every((x, j) => within(m.js.q[j], fromQ(x)))) agree++;
    else bad.push(`S${m.id} on ${m.name}: values differ`);
  });
  ok(bad.length === 0 && agree + refusedBoth === cases.length,
     `${cases.length} applications (${states.length} fields x 29 methods, two strengths, two gains, four times): ${agree} are the reference's exact field (${exactAlso} exact outright, the rest certified intervals that contain it, decided at up to ${widest} bits), ${refusedBoth} refused by both`, bad.slice(0, 4).join('; '));
  const refusedIds = new Set(Object.values(refusals).flatMap(x => [...x]));
  ok([...refusedIds].every(id => id === 1 || id === 12), 'the only refusals are repetends over three digits (S1, S12)', [...refusedIds].join(','));
  /* GRVQ's increment and MSTVQ's B on their own */
  const gcases = [], gmeta = [];
  for (let k = 0; k < 40; k++) {
    const lam = [0, 1, 2, 3].map(j => Q.mk(rnd(9) - 256n, rnd(4) + 1n));
    if (k % 5 === 0) lam[(k / 5) % 4] = Q.ZERO;
    const kk = k % 4, t = Q.mk(BigInt(k * 777), 1n);
    gcases.push({ kind: 'grvq', psi: [], lambda: lam.map(S), k: kk, t: S(t) });
    let js; try { js = S(exactly(() => K.grvqDelta(lam, kk, t))); } catch (e) { js = e instanceof RangeError ? 'refused' : 'not a RangeError: ' + e.message; }
    gmeta.push(js);
  }
  const gout = reference(gcases);
  ok(gout.every((r, i) => (r.refused && gmeta[i] === 'refused') || (r.delta && r.delta.n === gmeta[i].n && r.delta.d === gmeta[i].d)),
     'GRVQ.evolveExact (s9 + s7 + s10 + s6 + s14 + s15 + s28 + s29) equals the reference on 40 lambdas, zeros included');
  const bcases = states.slice(0, 8).map(([, psi], i) => ({ kind: 'setB', psi: psi.map(S), s: S(Q.mk(BigInt(i + 1), 13n)) }));
  const bout = reference(bcases);
  ok(bout.every((r, i) => exactly(() => K.mstvqB(states[i][1], Q.mk(BigInt(i + 1), 13n))).every((x, j) => S(x).n === r.B[j].n && S(x).d === r.B[j].d)),
     'MSTVQ.setB equals the reference on 8 states');
}

section('the statements each formula carries');
{
  const corpus = K.formulaCorpusAudit(Array.from({ length: 29 }, (_, i) => i + 1));
  let held = 0, refused = 0, failed = [];
  for (const entry of corpus) for (const r of entry.sutras) {
    if (r.refused) { refused++; continue; }
    if (r.ok) held++; else failed.push(`S${r.id} on state ${entry.corpus}: ${r.claims.filter(c => !c.ok).map(c => c.name).join(',')}`);
  }
  ok(failed.length === 0 && held + refused === 87, `on the corpus: ${held} of 87 hold every statement, ${refused} refused`, failed.join('; '));
  const names = new Set();
  for (const entry of corpus) for (const r of entry.sutras) if (r.claims) for (const c of r.claims) names.add(c.name);
  ok(names.has('vedic_update_zero') && names.has('rot_pair_energy') && names.size === 2, 'two statements are decided for every method: it is the identity at strength zero, and it keeps the energy of the field');
  const page = HTML.match(/sutra: \{\n\s*draws:[^\n]*\n\s*theorems: \[([^\]]*)\]/);
  const cited = page ? new Set([...page[1].matchAll(/'([^']+)'/g)].map(m => m[1])) : new Set();
  const missing = [...names].filter(n => !cited.has(n));
  ok(page && missing.length === 0, 'every statement decided is cited by the sutra channel', missing.join(', '));
  const act = [2, 9, 12, 22, 29].map(id => ({ id, strength: Q.HALF }));
  const go = modelRun(act, 'SERIES');
  let eng = K.initEngine(), runOk = true;
  for (let s = 0; s < 6; s++) { eng = go(eng, PARAMS({ vortex: Q.ONE }));
    for (const a of act) if (!K.formulaClaims(a.id, eng.psi, a.strength).ok) runOk = false; }
  ok(runOk, 'no statement is refuted for S2, S9, S12, S22, S29 on six steps of a run, at the live strength, with the field in certified intervals');
}

section('the couplings: lambda, GRVQ, MSTVQ, kappa');
{
  const sgn = (i, k) => ((i >> k) & 1) ? 1n : -1n;
  const inv = p => [0, 1, 2, 3].map(k => Q.mul(p.reduce((s, x, i) => sgn(i, k) > 0n ? Q.add(s, x) : Q.sub(s, x), Q.ZERO), Q.mk(1n, 16n)));
  const fwd = l => Array.from({ length: 16 }, (_, i) => l.reduce((s, x, k) => sgn(i, k) > 0n ? Q.add(s, x) : Q.sub(s, x), Q.ZERO));
  const seedS = K.initState();
  ok(sameQ(K.inverseL(seedS), [3n, 4n, 5n, 6n].map(n => Q.mk(n))) && sameQ(inv(seedS), K.inverseL(seedS)),
     "the seed's lambda is (3, 4, 5, 6): lambda_k = sum_i s_ik psi_i / 16, recomputed here");
  ok(sameQ(fwd([3n, 4n, 5n, 6n].map(n => Q.mk(n))).map((x, v) => Q.add(x, K.initDielectric()[v])), seedS) && seedS.every(x => x.n !== 0n),
     'the seed is forward(lambda) plus the dielectric seed, and no vertex is zero');
  let feedOk = true;
  for (const [id, k] of [[1, 0], [5, 1], [9, 2], [14, 3]]) {
    const t = Q.mk(2500n), gain = Q.mk(3n, 4n), rec = [], al = Q.mul(Q.mul(Q.mk(BigInt(id), 435n), Q.HALF), gain);
    const out = K.applySutra(id, seedS, Q.HALF, { t, gain, record: rec });
    const field = K.vedicField(id, seedS, K.sutraArgs(id, Q.HALF), al).p;
    const d = Q.mul(K.grvqDelta(inv(seedS), k, t), al);
    const lam = inv(out), lamF = inv(field);
    if (!lam.every((x, j) => near(x, j === k ? Q.add(lamF[j], d) : lamF[j])) || !near(rec[0].grvq.delta, d)) feedOk = false;
  }
  ok(feedOk, 'S1, S5, S9 and S14 add GRVQ(lambda of their input) x their alpha to lambda_0, 1, 2, 3 of their result and change no other lambda');
  const act = [{ id: 1, strength: Q.mk(3n, 10n) }, { id: 3, strength: Q.mk(1n, 5n) }, { id: 22, strength: Q.HALF }, { id: 27, strength: Q.HALF }];
  const p = PARAMS({ vortex: Q.ONE });
  const run = K.certify(() => {
    const go = modelRun(act, 'SERIES');
    let eng = K.initEngine();
    for (let s = 0; s < 3; s++) eng = go(eng, p);
    const L = K.formulaOf(eng.psi, act, 'SERIES', K.timeOf(eng.frame)), tenth = Q.mk(1n, 10n);
    const nxt = K.step(eng, p, act, 'SERIES', L, tenth);
    let shares = L.grvqRecord.length === 3, x = L.afterS;
    for (const g of L.grvqRecord) {
      const s = act.find(a => a.id === g.id), k = (g.id - 1) % 4;
      const want = Q.mul(K.grvqDelta(inv(x), k, L.time), Q.mul(Q.mk(BigInt(g.id), 435n), s.strength));
      if (g.k !== k || !near(g.delta, want)) shares = false;
      x = K.injectLambda(x, k, g.delta);
    }
    const a1 = Q.mul(Q.mk(1n, 435n), act[0].strength);
    const B = K.mstvqB(L.afterD, a1), b2 = B.reduce((s, y) => Q.add(s, Q.mul(y, y)), Q.ZERO);
    const T = [0, 1, 2].map(i => Q.sub(Q.mul(B[i], B[i]), Q.mul(Q.HALF, b2)));
    let fb = L.afterD;
    for (let k = 0; k < 3; k++) fb = K.injectLambda(fb, k, Q.mul(T[k], Q.mk(1n, 200000000n)));
    const trace = Q.add(Q.add(T[0], T[1]), T[2]);
    const gc = K.grvqClaims(L.record.filter(r => r.grvq).map(r => r.grvq).concat(L.grvqRecord), L.time);
    return { shares, stage: nearAll(x, L.afterD), mstvq: nearAll(fb, L.afterB) && sameQ(L.mstvq.B, B),
             kappa: near(nxt.tgcrKappa, Q.add(eng.tgcrKappa, Q.mul(Q.mul(trace, Q.mk(1n, 500n)), tenth))) && near(trace, Q.neg(Q.mul(Q.HALF, b2))),
             setters: L.record.filter(r => r.B).map(r => r.id), gc };
  });
  const r = run.value;
  ok(r.shares && r.stage, `each other active sutra adds GRVQ x its alpha to lambda_(id-1 mod 4), in order, and the stage is their sum (certified at ${run.bits} bits)`);
  ok(r.mstvq, "MSTVQ: B from psi_0+psi_1-psi_14-psi_15 etc. times S1's |alpha|, T_kk / 2e8 into lambda_0..2, recomputed here");
  ok(r.kappa, 'the TGCR stiffness kappa moves by tr T / 500 times the share of the formula put in, and tr T = -|B|^2 / 2 (stress_trace)');
  ok(r.setters.length === 1 && r.setters[0] === 3, 'of the active sutras only S3 sets MSTVQ B in the core (S2, S3 and S29 do)');
  ok(r.gc.ok && r.gc.results.length === 3 * 4, `no coupling statement is refuted on the four feeds (${r.gc.results.length} decided)`);
}

section('the stages of the real field');
{
  const p = PARAMS({ vortex: Q.mk(3n, 2n), zpeSource: X.fromQ(Q.mk(5n, 4n)) });
  const act = [{ id: 9, strength: Q.HALF }, { id: 29, strength: Q.HALF }];
  const id = x => x;
  const run = K.certify(() => {
    const go = modelRun(act, 'SERIES');
    let eng = K.initEngine();
    for (let s = 0; s < 4; s++) eng = go(eng, p);
    const L = eng.last;
    const en = (a, b) => near(energyQ(a), energyQ(b));
    const out = {};
    out.mag = en(K.edgeRotate(L.afterF, L.ctx.magTau, id), L.afterF);
    out.mstvq = en(K.complementRotate(L.afterG, K.mstvqTau(L.wheeler, p)), L.afterG);
    out.tgcr = en(K.edgeRotate(L.afterM, K.tgcrTau(L.afterM, p), id), L.afterM);
    const zr = K.complementRotate(L.afterT, K.zpeTau(L.wheeler, p));
    out.zpe = en(zr, L.afterT);
    const src = Q.mul(Q.mk(1n, 64n), Q.mk(5n, 4n));
    out.drive = sameQ(zr.map((x, v) => [3, 5, 6, 9, 10, 12].includes(v) ? Q.add(x, src) : x), L.afterZ);
    const tauWant = K.MAG_EDGES.map(([v, , h], e) => Q.mul(Q.mk(1n, 64n), Q.sub(L.magAIn[e], Q.mul(Q.mk(1n, 10n), Q.sub(L.magChiIn[h], L.magChiIn[v])))));
    out.tau = nearAll(L.ctx.magTau, tauWant);
    const psi = eng.psi, w = p.vortex;
    const vEdge = (u, h) => Q.mul(w, Q.mul(psi[u], psi[h]));
    const curl = (b, a, c) => Q.sub(Q.add(vEdge(b, b | (1 << a)), vEdge(b | (1 << a), b | (1 << a) | (1 << c))), Q.add(vEdge(b | (1 << c), b | (1 << a) | (1 << c)), vEdge(b, b | (1 << c))));
    const pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    const sd = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, -1, 0], [1, 0, 0]], asd = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [-1, 0, 0]];
    const G = K.tgcrGeometry(psi, w, eng.tgcrKappa);
    let pont = Q.ZERO, fsq = Q.ZERO, dens = true;
    for (let i = 0; i < 16; i++) {
      const loc = pairs.map(([a, c]) => curl(i & ~((1 << a) | (1 << c)), a, c));
      const s3 = [0, 1, 2].map(k => loc.reduce((t, f, q) => Q.add(t, Q.mul(f, Q.mk(BigInt(sd[q][k]), 2n))), Q.ZERO));
      const a3 = [0, 1, 2].map(k => loc.reduce((t, f, q) => Q.add(t, Q.mul(f, Q.mk(BigInt(asd[q][k]), 2n))), Q.ZERO));
      const ss = energyQ(s3), aa = energyQ(a3);
      pont = Q.add(pont, Q.sub(ss, aa)); fsq = Q.add(fsq, Q.add(ss, aa));
      if (!near(G.density[i], Q.sub(ss, aa))) dens = false;
    }
    out.geometry = dens && near(G.pontryagin, pont) && near(G.FSq, fsq);
    out.energy = near(G.energy, Q.add(Q.add(Q.mul(Q.HALF, fsq), Q.mul(Q.mul(Q.HALF, eng.tgcrKappa), energyQ(G.v))), Q.mul(Q.HALF, pont)));
    out.closes = G.cubeFlux.every(zeroish) && G.density.some(x => !Q.mayZero(x));
    out.claims = K.stageClaims(L, p);
    const m = a => a.map(mid), eqE = (a, b) => Q.eq(energyQ(a), energyQ(b));
    const mw = { magneticTraceTotal: mid(L.wheeler.magneticTraceTotal), dielectricTraceTotal: mid(L.wheeler.dielectricTraceTotal) };
    out.exact = exactly(() => {
      const fF = m(L.afterF), fG = m(L.afterG), fM = m(L.afterM), fT = m(L.afterT), ps = m(psi), kap = mid(eng.tgcrKappa);
      const Gx = K.tgcrGeometry(ps, w, kap);
      return { mag: eqE(K.edgeRotate(fF, m(L.ctx.magTau), id), fF), mstvq: eqE(K.complementRotate(fG, K.mstvqTau(mw, p)), fG),
               tgcr: eqE(K.edgeRotate(fM, K.tgcrTau(fM, p), id), fM), zpe: eqE(K.complementRotate(fT, K.zpeTau(mw, p)), fT),
               closes: Gx.cubeFlux.every(Q.isZero) && Gx.density.some(x => !Q.isZero(x)) };
    });
    return out;
  });
  const r = run.value, x = r.exact;
  ok(r.mag && x.mag, `the magnetism stage, every edge rotated by dt A_e axis by axis, keeps the energy (rot_pair_energy): exactly on the exact midpoints of its inputs, and to the width of the intervals on the run (certified at ${run.bits} bits)`);
  ok(r.mstvq && x.mstvq, 'the MSTVQ stage keeps the energy, exactly on exact inputs and to the width of the intervals on the run');
  ok(r.tgcr && x.tgcr, 'the TGCR stage keeps the energy, exactly on exact inputs and to the width of the intervals on the run');
  ok(r.zpe && x.zpe, "ZPE's rotation keeps the energy, exactly on exact inputs and to the width of the intervals on the run");
  ok(r.drive, 'and the drive dt x source is added to the six inertial vertices only');
  ok(r.tau, "the magnetism stage turns each edge by dt times the whole potential on it, Faraday part included");
  let rotOk = true;
  for (const [x, y, t] of [[3, -2, 1 / 7], [-5, 11, -2], [1, 0, 1 / 3]]) {
    const [u, w] = K.rotPair(Q.mk(BigInt(x)), Q.mk(BigInt(y)), Q.parse(t === 1 / 7 ? '1/7' : t === -2 ? '-2' : '1/3'));
    const th = 2 * Math.atan(t), c = Math.cos(th), s = Math.sin(th);
    if (Math.abs(Number(u.n) / Number(u.d) - (c * x + s * y)) > 1e-12 || Math.abs(Number(w.n) / Number(w.d) - (c * y - s * x)) > 1e-12) rotOk = false;
  }
  ok(rotOk, 'a pair rotation by t is the rotation by 2 arctan t, against floats');
  ok(r.geometry, "TGCR: F = d1 v, v = omega psi_t psi_h, split at each vertex by the Hodge tables; F^F and |F|^2 recomputed here");
  ok(r.energy, 'TGCR energy = |F|^2/2 + kappa |v|^2/2 + eta F^F with eta = 1/2');
  ok(r.closes && x.closes, 'the vortex curvature closes on every cube and is not zero: exactly on the exact midpoints of the field, and to the width of the intervals on the run');
  ok(r.claims.ok, `no stage statement is refuted (${r.claims.results.map(c => c.name).join(', ')})`);
}

const EXPORT_CASES = [['9,11,29', 'PARALLEL', Q.ONE, 0n, Q.ONE], ['3,9,11,25,29', 'SYMMETRIC_CONCURRENT', Q.mk(3n, 10n), 1n, Q.TWO],
                      ['1', 'SERIES', Q.HALF, 0n, Q.ONE]];
section('step export');
{
  for (const [ids, mode, strength, drive, vortex] of EXPORT_CASES) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength }));
    const run = K.certify(() => {
      const go = modelRun(active, mode);
      let e = K.initEngine();
      for (let s = 0; s < 2; s++) e = go(e, PARAMS({ zpeSource: X.fromQ(Q.mk(drive)), vortex }));
      return e;
    });
    const eng = run.value;
    const file = JSON.parse(JSON.stringify({ frame: eng.frame, last: K.serializeLast(eng.last), psi: K.serializeState(eng.psi),
      dielectric: eng.dielectric.map(K.serializeQ), tgcrKappa: K.serializeQ(eng.tgcrKappa),
      mayaPhase: K.serializeQ(eng.mayaPhase), mayaErr: K.serializeQ(eng.mayaErr),
      magA: eng.magA.map(K.serializeQ), magChi: eng.magChi.map(K.serializeQ) }));
    const L = file.last, where = `${mode} ${ids}, step 2`;
    let same = false, why = '';
    try {
      const fromA = o => A4.mk(fromQ(o.a), fromQ(o.b), fromQ(o.c), fromQ(o.d));
      const p = Object.fromEntries(Object.entries(L.params).map(([k, v]) => [k, ('a' in v && 'b' in v) ? fromA(v) : fromQ(v)]));
      const actOf = list => list.map(x => x.args ? { id: x.id, strength: fromQ(x.strength), args: x.args.map(fromQ) } : { id: x.id, strength: fromQ(x.strength) });
      const act = actOf(L.activeSutras), FF = L.formula;
      if (L.workBits !== run.bits || L.exactBits !== 4096) throw new Error(`the file records ${L.workBits} and ${L.exactBits} bits, the run used ${run.bits} and 4096`);
      K.setWorkBits(L.workBits);
      const formula = K.formulaOf(FF.input.map(fromQ), actOf(FF.activeSutras), FF.composition, fromQ(FF.time));
      const replay = K.step({ psi: L.input.map(fromQ), dielectric: L.dielectricIn.map(fromQ),
                              energyLedger: null, frame: L.frameIn, last: null, tgcrKappa: fromQ(L.tgcrKappaIn),
                              mayaPhase: fromQ(L.mayaPhaseIn), mayaErr: fromQ(L.mayaErrIn),
                              magA: L.magAIn.map(fromQ), magChi: L.magChiIn.map(fromQ) }, p, act, L.composition, formula, fromQ(L.formulaPart));
      same = JSON.stringify(K.serializeState(replay.psi)) === JSON.stringify(file.psi) &&
             JSON.stringify(replay.dielectric.map(K.serializeQ)) === JSON.stringify(file.dielectric) &&
             JSON.stringify(K.serializeQ(replay.tgcrKappa)) === JSON.stringify(file.tgcrKappa) &&
             JSON.stringify(replay.magA.map(K.serializeQ)) === JSON.stringify(file.magA) &&
             JSON.stringify(replay.magChi.map(K.serializeQ)) === JSON.stringify(file.magChi) &&
             JSON.stringify(K.serializeLast(replay.last)) === JSON.stringify(L);
      if (!same) why = 'the replay differs from the file';
    } catch (e) { why = e.message; }
    finally { K.setWorkBits(256); }
    ok(same, `${where}: replays from its exported file alone, bit for bit, at the ${L.workBits} bits it records`, why);
    ok(L.magnetismIn.closed && L.tgcr.cubeFlux.map(fromQ).every(zeroish),
       `${where}: the flux it starts from and the vortex curvature it ends with close on every cube, to the width of their intervals`);
    ok(L.lambdaIn.length === 4 && L.lambdaOut.length === 4 && L.sutraRecord.length === active.length * (mode === 'SYMMETRIC_CONCURRENT' ? 2 : 1),
       `${where}: carries lambda in and out and one record per formula applied`);
  }
}

section('the formula');
{
  const red = (n, d) => { const g = euclid(n, d) || 1n; return [n / g, d / g]; };
  const eqF = (q, [n, d]) => q.n === n && q.d === d;
  const tg = [{ id: 9, strength: Q.mk(3n, 10n) }];
  const f = K.formulaOf(K.initState(), tg, 'SERIES', K.timeOf(0));
  ok(f.active.length === 1 && eqF(f.active[0].strength, [3n, 10n]) && f.record.length === 1 && eqF(f.record[0].alpha, red(9n * 3n, 435n * 10n)) &&
     sameQ(f.delta, f.afterB.map((x, i) => Q.sub(x, K.initState()[i]))),
     'the formula runs each sutra at its setting, alpha = id/435 x setting, and its change is the field after it less the field it was calculated from');
  const quartet = [2, 6, 23, 26].map(id => ({ id, strength: Q.ONE }));
  const p = PARAMS({ vortex: Q.ONE });
  const run = K.certify(() => {
    const go = modelRun(quartet, 'PARALLEL');
    let e = K.initEngine(), first = null, shareOk = true, sameF = true, heldOk = true, ledgerZero = true;
    const parts = [];
    for (let s = 1; s <= 14; s++) {
      e = go(e, p);
      const L = e.last;
      if (!first) first = L.formula;
      if (L.formula !== first) sameF = false;
      parts.push(Q.toString(L.part));
      if (!(ident(L.part, s <= 10 ? Q.mk(1n, 10n) : Q.ZERO) && sameQ(L.afterF, L.input.map((x, i) => Q.add(x, Q.mul(first.delta[i], L.part)))))) shareOk = false;
      if (!(Q.isExact(L.stepError.delta) && L.stepError.delta.n === 0n)) ledgerZero = false;
      if (s > 10 && !near(L.eFinal, L.e0)) heldOk = false;
    }
    return { sameF, shareOk, heldOk, ledgerZero, parts };
  });
  const r = run.value;
  ok(r.sameF && r.shareOk, `the formula is calculated once and put in over the glide in equal shares of 1/10, then none of it is applied (certified at ${run.bits} bits)`, r.parts.join(' '));
  ok(r.heldOk && r.ledgerZero, 'after the glide, with no drive, no step moves the energy of the field: no step rounds anything, and the energy before and after each step agree to the width of their intervals');
}

section('glide');
{
  const red = (n, d) => { if (d < 0n) { n = -n; d = -d; } const g = euclid(n, d) || 1n; return [n / g, d / g]; };
  const F = (n, d = 1n) => red(n, d);
  const add = ([a, b], [c, d]) => red(a * d + c * b, b * d), sub = ([a, b], [c, d]) => red(a * d - c * b, b * d), mul = ([a, b], [c, d]) => red(a * c, b * d);
  const eqF = (q, [n, d]) => q.n === n && q.d === d;
  const at = (from, to, k, n) => add(from, mul(sub(to, from), F(BigInt(Math.min(k, n)), BigInt(n))));
  const s9 = q => [{ id: 9, strength: q }];
  let r = [], up = true;
  const seen = [];
  for (let j = 1; j <= 12; j++) {
    const x = K.rampStep(r, s9(Q.mk(1n, 2n)), 10); r = x.ramp;
    const want = at(F(0n), F(1n, 2n), j, 10);
    seen.push(x.active.length ? Q.toString(x.active[0].strength) : 'none');
    if (!(x.active.length === 1 && x.active[0].id === 9 && eqF(x.active[0].strength, want) && eqF(x.active[0].args[0], want))) up = false;
  }
  ok(up, 'switched on, a formula rises from 0 by the same share each step, reaches its setting at the end of the glide and holds it (ramp_steady, ramp_end, ramp_hold)', seen.join(' '));
  r = [];
  for (let j = 0; j < 5; j++) r = K.rampStep(r, s9(Q.mk(1n, 2n)), 10).ramp;
  const ch = K.rampStep(r, s9(Q.mk(1n, 10n)), 10);
  ok(ch.active.length === 1 && eqF(ch.active[0].strength, at(F(1n, 4n), F(1n, 10n), 1, 10)),
     'a setting changed mid-glide glides on from where the strength stood (ramp_zero, ramp_restart)', Q.toString(ch.active[0].strength));
  let d = ch.ramp, down = [];
  for (let j = 1; j <= 10; j++) { const x = K.rampStep(d, [], 10); d = x.ramp; down.push(x.active.length); }
  ok(down.slice(0, 9).every(n => n === 1) && down[9] === 0 && d.length === 0,
     'a formula switched off glides down to nothing over the glide and is then gone', down.join(','));
  r = [];
  for (let j = 0; j < 4; j++) r = K.rampStep(r, s9(Q.mk(1n, 2n)), 10).ramp;
  const re = [];
  for (let j = 1; j <= 5; j++) { const x = K.rampStep(r, s9(Q.mk(1n, 2n)), 4); r = x.ramp; re.push(x.active[0].strength); }
  ok(re.every((q, i) => eqF(q, at(F(1n, 5n), F(1n, 2n), i + 1, 4))),
     'a glide length changed mid-glide goes on from where the strength stood, over the new number of steps', re.map(Q.toString).join(' '));
  r = [];
  for (let j = 0; j < 12; j++) r = K.rampStep(r, s9(Q.mk(3n, 10n)), 10).ramp;
  const held = [];
  for (let j = 0; j < 3; j++) { const x = K.rampStep(r, s9(Q.mk(3n, 10n)), 25); r = x.ramp; held.push(x.active[0].strength); }
  ok(held.every(q => eqF(q, F(3n, 10n))), 'a formula held at its setting stays there exactly when the glide length changes', held.map(Q.toString).join(' '));
  const mix = K.rampStep([], [{ id: 29, strength: Q.ONE }, { id: 3, strength: Q.ONE }, { id: 11, strength: Q.ONE }], 3);
  ok(mix.active.map(a => a.id).join(',') === '3,11,29' && mix.active.every(a => eqF(a.strength, F(1n, 3n))),
     'the formulas run in id order, each at its share of the glide', mix.active.map(a => `S${a.id}·${Q.toString(a.strength)}`).join(' '));
  const throws = f => { try { f(); return false; } catch { return true; } };
  ok(throws(() => K.rampStep([], s9(Q.ONE), 0)) && throws(() => K.rampStep([], s9(Q.ONE), 1.5)) &&
     throws(() => K.rampStep([], [...s9(Q.ONE), ...s9(Q.HALF)], 10)) && throws(() => K.rampStep([], s9(Q.mk(3n, 2n)), 10)),
     'a glide that is not a whole number of steps of at least 1, a formula set twice, or a strength outside [0, 1] is refused');
}

section('series / parallel / concurrent');
{
  const seedState = K.initState();
  for (const ids of ['', '9', '5,22', '1,4,26', '3,9,11,25,29']) {
    const active = ids ? ids.split(',').map(Number).map(id => ({ id, strength: Q.mk(1n, 2n) })) : [];
    const a = K.modeAudit(seedState, active);
    ok(a.ok && a.results.length > 0, `modeAudit on {${ids}} decides ${a.results.length} identities and all hold`);
    if (active.length) {
      const shared = K.modeComparison(seedState, active, a.states), fresh = K.modeComparison(seedState, active);
      ok(shared.modes.every(m => sameQ(shared.states[m], fresh.states[m]) && ident(shared.energies[m], fresh.energies[m])),
         `{${ids}}: the comparison from the audit's states equals a fresh one`);
      if (active.length > 1) ok(shared.allDistinct, `{${ids}}: series, parallel and concurrent give three different fields`);
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
  const base = PARAMS({ vortex: Q.ONE });
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  const hwf = v => (v & 1) + ((v >> 1) & 1) + ((v >> 2) & 1) + ((v >> 3) & 1);
  const edges = []; for (let a = 0; a < 4; a++) for (let v = 0; v < 16; v++) if (!(v & (1 << a))) edges.push([v, a, v | (1 << a)]);
  const cyc = [Q.ONE, Q.mk(13n, 10n), Q.ONE, Q.mk(7n, 10n)];
  const fl = q => { const m = mid(q); return Number(m.n) / Number(m.d); };
  const run = K.certify(() => {
    let eng = K.initEngine(), inductOk = true, chiOk = true, closedOk = true, gradOk = true, maxA = Q.ZERO, recOk = true, potOk = true, shareOk = true;
    const go = modelRun(act, 'PARALLEL');
    for (let s = 0; s < 40; s++) {
      const prev = eng;
      eng = go(eng, base);
      const L = eng.last, rOut = eng.psi.map(x => Q.mul(x, x));
      const Etot = rOut.reduce((t, x) => Q.add(t, x), Q.ZERO);
      edges.forEach(([v, a, h], e) => {
        const share = Q.div(Q.mul(Q.mk(8n), Q.add(rOut[v], rOut[h])), Etot);
        const k = a >= 2 ? Q.mul(Q.mk(6n, 100n), base.vortex) : Q.mul(Q.mul(Q.mk(3n, 100n), Q.HALF), base.vortex);
        const want = Q.add(Q.mul(Q.mk(499n, 500n), prev.magA[e]), Q.mul(Q.mul(k, cyc[(hwf(v) + a) % 4]), share));
        if (!near(eng.magA[e], want)) inductOk = false;
        maxA = Q.max(maxA, Q.abs(eng.magA[e]));
      });
      for (let v = 0; v < 16; v++) if (!near(eng.magChi[v], Q.add(Q.mul(Q.mk(499n, 500n), prev.magChi[v]), Q.sub(eng.psi[v], prev.psi[v])))) chiOk = false;
      const Ein = prev.psi.reduce((t, x) => Q.add(t, Q.mul(x, x)), Q.ZERO);
      if (!L.shareIn.every((x, v) => near(x, Q.div(Q.mul(Q.mk(16n), Q.mul(prev.psi[v], prev.psi[v])), Ein)))) shareOk = false;
      const md = K.magData(eng.magA, eng.magChi);
      if (!md.closed) closedOk = false;
      if (!K.curlA(K.fullA(Array(32).fill(Q.ZERO), eng.magChi)).every(zeroish)) gradOk = false;
      for (let v = 0; v < 16; v++) {
        let want = Q.ZERO;
        for (let a = 0; a < 4; a++) { const e = edges.findIndex(([t, ax]) => ax === a && t === (v & ~(1 << a))); want = (v & (1 << a)) ? Q.add(want, eng.magA[e]) : Q.sub(want, eng.magA[e]);
          want = Q.sub(want, Q.mul(Q.mk(1n, 10n), Q.sub(eng.magChi[v], eng.magChi[v ^ (1 << a)]))); }
        if (!near(md.potential[v], want)) potOk = false;
      }
      const bv = K.bvecAt(md.flux);
      edges.forEach(([v, , h], e) => {
        const a = bv[v].map(fl), b = bv[h].map(fl), na = Math.hypot(...a), nb = Math.hypot(...b);
        if (na * na <= 1e-6 || nb * nb <= 1e-6) { if (md.reconnections.includes(e)) recOk = false; return; }
        const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
        if (Math.abs(c + 0.7) > 1e-9 && (c < -0.7) !== md.reconnections.includes(e)) recOk = false;
      });
    }
    const smax = Q.mul(Q.mul(Q.abs(base.vortex), Q.mk(6n * 13n * 16n, 1000n)), Q.mk(1001n, 1000n));
    return { inductOk, chiOk, shareOk, gradOk, closedOk, potOk, recOk, smax, maxA, bounded: Q.cmp(maxA, Q.mul(smax, Q.mk(500n))) <= 0,
             helicity: zeroish(K.helicityOf(Array(32).fill(Q.ZERO), eng.magChi)),
             shares: near(K.edgeShares(eng.last.rhoOut).reduce((t, x) => Q.add(t, x), Q.ZERO), Q.mk(32n)) };
  });
  const r = run.value;
  ok(r.inductOk, `A = (499/500) A + omega (6/100 | eta 3/100) m_e share_e on all 32 edges for 40 steps, recomputed here (certified at ${run.bits} bits)`);
  ok(r.chiOk, "the Faraday potential chi = (499/500) chi + (psi_out - psi_in), recomputed here");
  ok(r.shareOk, 'the dielectric is polarised by each vertex share 16 psi_v^2 / E of the field energy, recomputed here');
  ok(r.gradOk, 'the Faraday part of A carries no flux on any of those steps (faraday_flux_free), to the width of the intervals');
  ok(r.closedOk, 'the flux out of every one of the 8 cubes is zero on every step (flux_closed), to the width of the intervals');
  ok(r.potOk, 'the potential at each vertex is div A - (1/10) L chi, recomputed here');
  ok(r.recOk, 'reconnections are exactly the edges where cos(B_i, B_j) < -0.7 (reconnect_iff)');
  ok(r.bounded, `|A| stays within max |A_0|, s/(1 - r) (retention_bound): ${fl(r.maxA).toFixed(3)} <= ${(fl(r.smax) * 500).toFixed(1)}`);
  ok(r.helicity, 'a potential that is all Faraday has no helicity (helicity_of_gradient), to the width of the intervals');
  ok(r.shares, 'the edge shares sum to 32 (edge_share_total)');
  ok(throwsRange(() => K.edgeShares(Array(16).fill(Q.ZERO))) && throwsRange(() => K.energyShare(Array(16).fill(Q.ZERO))),
     'a field with no energy has no shares: refused, not divided by zero');
}

section('no parameter is ignored');
{
  const base = PARAMS();
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  const run = p => K.certify(() => { const go = modelRun(act, 'PARALLEL'); let e = K.initEngine(); for (let s = 0; s < 6; s++) e = go(e, p); return e; }).value;
  const ref = run(base), moved = (a, b) => apart(a.psi, b.psi) || apart(a.dielectric, b.dielectric);
  ok(moved(run({ ...base, vortex: Q.ONE }), ref), 'vortex omega changes the field: some value certainly differs');
  ok(moved(run({ ...base, zpeSource: X.ONE }), ref), 'the drive changes the field');
  ok(moved(run({ ...base, polarisation: Q.ZERO }), ref), 'the polarisation changes the dielectric and the field');
  ok(moved(run({ ...base, tgcr: X.fromQ(Q.mk(1n, 64n)), vortex: Q.ONE }), run({ ...base, vortex: Q.ONE })), 'the TGCR rate changes the field');
  const share = K.energyShare(K.rhoHat(K.initState()));
  const d0 = K.evolveDielectric(K.initDielectric(), act, base, Array(16).fill(Q.ZERO));
  const d1 = K.evolveDielectric(K.initDielectric(), act, { ...base, polarisation: Q.mk(3n, 2n) }, share);
  ok(d1.every(Q.isExact) && d0.every(Q.isExact) && d1.every((q, v) => Q.eq(Q.sub(q, d0[v]), Q.mul(Q.mul(Q.mk(3n, 2n), share[v]), Q.mk(1n, 8n)))),
     'polarisation adds exactly chi share / 8 to each vertex of the dielectric');
  ok(throwsRange(() => K.evolveDielectric(K.initDielectric(), act, base)), 'the dielectric update without the field shares is refused');
}

section('the engine on the torus');
{
  /* Where each vertex is on the drawn torus is read here from
     tessVertexToTorus's own angles, theta = atan2(w, x) + pi and
     phi = atan2(z, y) + pi with x, y, z, w the bits as -1/+1, not from the
     kernel's Gray table. Every circulation is recomputed by walking a face's
     corners and summing A along each side, with no face or cube table. */
  const quarter = t => ((Math.round((t - Math.PI / 4) / (Math.PI / 2)) % 4) + 4) % 4;
  const sgnb = (k, b) => ((k >> b) & 1) ? 1 : -1;
  const at = k => [quarter(Math.atan2(sgnb(k, 3), sgnb(k, 0)) + Math.PI), quarter(Math.atan2(sgnb(k, 2), sgnb(k, 1)) + Math.PI)];
  const P = Array.from({ length: 16 }, (_, k) => at(k));
  ok(P.every(([r, s], k) => K.ringThetaOf(k) === r && K.ringPhiOf(k) === s) && new Set(P.map(([r, s]) => r * 4 + s)).size === 16,
     'every vertex sits at the lattice position tessVertexToTorus\'s angles give it, each position once');
  const step4 = (a, b) => (b - a + 4) % 4 === 1 ? 1 : (a - b + 4) % 4 === 1 ? -1 : 0;
  /* A on the oriented side u -> w of the cube, from the potential on its edges */
  const edgeOf = (u, w) => { const a = Math.log2(u ^ w), base = Math.min(u, w); return [K.MAG_EDGES.findIndex(([v, b]) => v === base && b === a), u < w ? 1 : -1]; };
  const side = (A, u, w) => { const [e, sg] = edgeOf(u, w); return sg > 0 ? A[e] : Q.neg(A[e]); };
  const circ = (A, cs) => cs.reduce((t, u, k) => Q.add(t, side(A, u, cs[(k + 1) % 4])), Q.ZERO);
  /* the 24 squares of the 4-cube, each as its four corners in a loop */
  const squares = [];
  for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) for (let v = 0; v < 16; v++) if (!(v & ((1 << a) | (1 << b))))
    squares.push([v, v ^ (1 << a), v ^ (1 << a) ^ (1 << b), v ^ (1 << b)]);
  const classify = cs => {
    const th = cs.map(k => P[k][0]), ph = cs.map(k => P[k][1]);
    if (ph.every(x => x === ph[0]) && new Set(th).size === 4) return { kind: 'hole', ring: ph[0], dir: step4(th[0], th[1]) };
    if (th.every(x => x === th[0]) && new Set(ph).size === 4) return { kind: 'tube', ring: th[0], dir: step4(ph[0], ph[1]) };
    /* a cell: unwrap the corners around the first and take the loop's signed area */
    const u = cs.map(k => [(P[k][0] - th[0] + 5) % 4 - 1, (P[k][1] - ph[0] + 5) % 4 - 1]);
    let area = 0; for (let k = 0; k < 4; k++) area += u[k][0] * u[(k + 1) % 4][1] - u[(k + 1) % 4][0] * u[k][1];
    const lo = (xs) => { const m = [...new Set(xs)]; return (m[1] - m[0] + 4) % 4 === 1 ? m[0] : m[1]; };
    return { kind: 'cell', r: lo(th), s: lo(ph), orient: Math.sign(area), unit: Math.abs(area) === 2 };
  };
  const C = squares.map(classify);
  ok(C.filter(c => c.kind === 'hole').length === 4 && C.filter(c => c.kind === 'tube').length === 4 &&
     new Set(C.filter(c => c.kind === 'cell' && c.unit).map(c => c.r * 4 + c.s)).size === 16,
     'the 24 squares are 4 loops round the torus, 4 round the tube and the 16 unit cells, each once (hole_loop, tube_loop, cells_tile)');
  /* the gate's own reading of a potential on the torus */
  const read = (A, chi) => {
    const AF = K.MAG_EDGES.map(([v, , h], e) => Q.sub(A[e], Q.mul(Q.mk(1n, 10n), Q.sub(chi[h], chi[v]))));
    const hole = [], tube = [], cells = [[], [], [], []];
    squares.forEach((cs, i) => { const c = C[i], x = circ(A, cs);
      if (c.kind === 'hole') hole[c.ring] = c.dir > 0 ? x : Q.neg(x);
      else if (c.kind === 'tube') tube[c.ring] = c.dir > 0 ? x : Q.neg(x);
      else cells[c.r][c.s] = c.orient > 0 ? x : Q.neg(x); });
    const flow = K.MAG_EDGES.map(([v, a, h], e) => {
      const d = (a === 0 || a === 3) ? step4(P[v][0], P[h][0]) : step4(P[v][1], P[h][1]);
      return d > 0 ? AF[e] : Q.neg(AF[e]); });
    return { hole, tube, cells, flow };
  };
  const same = (a, b) => a.length === b.length && a.every((x, i) => Array.isArray(x) ? same(x, b[i]) : near(x, b[i]));
  const lawsHold = g => [0, 1, 2, 3].every(s =>
      near(Q.sub(g.hole[(s + 1) % 4], g.hole[s]), Q.neg([0, 1, 2, 3].reduce((t, r) => Q.add(t, g.cells[r][s]), Q.ZERO)))) &&
    [0, 1, 2, 3].every(r =>
      near(Q.sub(g.tube[(r + 1) % 4], g.tube[r]), [0, 1, 2, 3].reduce((t, s) => Q.add(t, g.cells[r][s]), Q.ZERO)));
  let agree = true, laws = true, balanced = true, exact = true, where = '';
  for (let trial = 0; trial < 40; trial++) {
    const A = K.MAG_EDGES.map((_, e) => Q.mk(BigInt(((e + 3) * (trial + 7) * 7919) % 2001 - 1000), BigInt(1 + ((e * trial) % 9))));
    const chi = Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v + 1) * (trial + 2) * 104729) % 301 - 150), BigInt(1 + v % 5)));
    const g = read(A, chi), t = K.torusData(A, chi);
    if (![g.hole, g.tube, g.flow, t.hole, t.tube, t.flow, ...g.cells, ...t.cells].every(a => a.every(Q.isExact))) exact = false;
    if (!(same(g.hole, t.hole) && same(g.tube, t.tube) && same(g.cells, t.cells) && same(g.flow, t.flow))) { agree = false; where = 'trial ' + trial; }
    if (!lawsHold(g)) laws = false;
    if (!t.balanced) balanced = false;
  }
  ok(agree && exact, 'the kernel\'s hole, tube and cell fluxes and its edge flows are exactly the gate\'s own circulations on 40 exact potentials', where);
  ok(laws, 'from one loop to the next the flux steps by exactly the circulation of the cells between (hole_flux_step_y/z, tube_flux_step_x/w)');
  ok(balanced, 'the kernel decides all eight strip balances on each of them (cube_strip)');
  /* and on a run: the S9 · S11 · S29 preset, every step */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ONE };
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  const run = K.certify(() => {
    const go = modelRun(act, 'PARALLEL');
    let eng = K.initEngine(), runAgree = true, runLaws = true, nonzero = false;
    for (let s = 0; s < 40; s++) {
      eng = go(eng, params);
      const g = read(eng.magA, eng.magChi), t = K.magData(eng.magA, eng.magChi).torus;
      if (!(same(g.hole, t.hole) && same(g.tube, t.tube) && same(g.cells, t.cells) && same(g.flow, t.flow) && t.balanced)) runAgree = false;
      if (!lawsHold(g)) runLaws = false;
      if (g.hole.some(x => !Q.mayZero(x)) && g.cells.some(r => r.some(x => !Q.mayZero(x)))) nonzero = true;
    }
    return { runAgree, runLaws, nonzero };
  });
  ok(run.value.runAgree && run.value.runLaws && run.value.nonzero,
     `on 40 steps of the S9 · S11 · S29 preset the torus fluxes are the gate's, no balance is refuted, and the loops and cells carry flux, to the width of the intervals (certified at ${run.bits} bits)`);
}


section('the sources of the fields around the construct');
{
  /* The current J = (d1)^T B and the charge rho = (d0)^T d0 q are recomputed
     here from the 4-cube itself: the gate walks each square's corners
     base -> +a -> +a+b -> +b to find which edges its boundary runs along and
     which way, with no edge or face table of the kernel's; the neighbours of a
     vertex are the four that differ from it in one bit. Only which square a
     face index names (its base and two axes) is read from the kernel. */
  const edgeAt = (u, w) => { const a = Math.log2(u ^ w); return K.MAG_EDGES.findIndex(([v, b]) => v === Math.min(u, w) && b === a); };
  const walkJ = B => {
    const J = new Array(32).fill(Q.ZERO);
    K.MAG_FACES.forEach((f, i) => {
      const v = f.base, cs = [v, v ^ (1 << f.a), v ^ (1 << f.a) ^ (1 << f.b), v ^ (1 << f.b)];
      for (let k = 0; k < 4; k++) { const u = cs[k], w = cs[(k + 1) % 4], e = edgeAt(u, w);
        J[e] = u < w ? Q.add(J[e], B[i]) : Q.sub(J[e], B[i]); }
    });
    return J;
  };
  /* net current into each vertex, from J alone */
  const inflow = J => Array.from({ length: 16 }, (_, v) => [0, 1, 2, 3].reduce((t, a) => {
    const w = v ^ (1 << a), e = edgeAt(v, w);
    return w < v ? Q.add(t, J[e]) : Q.sub(t, J[e]); }, Q.ZERO));
  const lap = q => Array.from({ length: 16 }, (_, v) => [0, 1, 2, 3].reduce((t, a) => Q.add(t, Q.sub(q[v], q[v ^ (1 << a)])), Q.ZERO));
  const eqv = (a, b) => a.length === b.length && a.every((x, i) => near(x, b[i]));
  let jSame = true, kirch = true, rSame = true, neutral = true, flags = true, where = '';
  for (let trial = 0; trial < 40; trial++) {
    const B = Array.from({ length: 24 }, (_, f) => Q.mk(BigInt(((f + 5) * (trial + 3) * 7907) % 1999 - 999), BigInt(1 + ((f + trial) % 7))));
    const q = Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v + 2) * (trial + 11) * 6151) % 401 - 200), BigInt(1 + ((v * trial) % 6))));
    const J = K.currentOf(B);
    if (!(J.every(Q.isExact) && eqv(J, walkJ(B)))) { jSame = false; where = 'trial ' + trial; }
    /* these fluxes are not the curl of anything: conservation is d1 d0 = 0,
       not a property of the potential */
    if (!inflow(J).every(Q.isZero)) kirch = false;
    const rho = K.laplacianV(q);
    if (!(rho.every(Q.isExact) && eqv(rho, lap(q)))) rSame = false;
    if (!Q.isZero(rho.reduce((t, x) => Q.add(t, x), Q.ZERO))) neutral = false;
    /* the kernel's flags on a potential whose curl is this trial's */
    const A = K.MAG_EDGES.map((_, e) => Q.mk(BigInt(((e + 1) * (trial + 5) * 3571) % 801 - 400), BigInt(1 + e % 4)));
    const sd = K.sourcesData(A, q);
    if (!(sd.kirchhoff === true && sd.neutral === true && eqv(sd.current, walkJ(K.curlA(A))) && eqv(sd.charge, lap(q)))) flags = false;
  }
  ok(jSame, 'the lattice current is the sum, over the faces round each edge, of their flux the way their boundary runs (J = (d1)^T B), on 40 fluxes', where);
  ok(kirch, 'as much current flows into every vertex as out of it, for any flux (kirchhoff, recomputed from J alone)');
  ok(rSame, 'each vertex\'s charge is the dielectric there less its four neighbours\', summed (rho = (d0)^T d0 q, charge_laplacian), on 40 dielectrics');
  ok(neutral, 'the charges sum to zero (charge_neutral, recomputed)');
  ok(flags, 'sourcesData gives exactly these, and decides both laws true, on 40 potentials and dielectrics');
  /* and on a run: the S9 · S11 · S29 preset's sources are the gate's and not empty */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ONE };
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  const run = K.certify(() => {
    const go = modelRun(act, 'PARALLEL');
    let eng = K.initEngine(), runOk = true, live = 0;
    for (let s = 0; s < 30; s++) {
      eng = go(eng, params);
      const sd = K.sourcesData(eng.magA, eng.dielectric);
      if (!(sd.kirchhoff && sd.neutral && eqv(sd.current, walkJ(K.curlA(eng.magA))) && eqv(sd.charge, lap(eng.dielectric)) && inflow(sd.current).every(zeroish))) runOk = false;
      if (sd.current.some(x => !Q.mayZero(x)) && sd.charge.some(x => !Q.mayZero(x))) live++;
    }
    return { runOk, live };
  });
  const { runOk, live } = run.value;
  ok(runOk && live === 30, `on 30 steps of the S9 · S11 · S29 preset the sources are the gate's, both laws hold to the width of the intervals, and current and charge are both certainly present (${live}/30 steps, certified at ${run.bits} bits)`);
}


console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
