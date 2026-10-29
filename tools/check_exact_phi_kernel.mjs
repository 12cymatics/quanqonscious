/* Gate for the exact kernel inside vedic_v18.51.1_exact_phi.html.

   The kernel is not copied here. KERNEL_FACTORY is sliced out of the page and
   evaluated, so there is one copy of the formulas and nothing that can drift
   from it. Where a statement can be checked against something the kernel did
   not compute, it is: the 29 v18.24 formulas and GRVQ against
   tools/vedic_reference.py, a separate transcription of v18.24's
   SUTRAS[].evolve in Python's exact Fraction; reduced fractions against a
   Euclid written here; exact storage equality in bare BigInt arithmetic; the TGCR
   geometry, the stages and the couplings recomputed here from their
   definitions.

   What it holds the page to:
     self-tests         every one of runSelfTests passes
     canonical Q        Q.mk reduces by the true gcd, dyadic or not
     storage            storeQ(x) = x, canonical and idempotent, without rounding
     refusal            an operand or strength outside its domain throws
     formulas           each of the 29 equals the reference exactly, at its
                        default and its extreme operands, at two gains and
                        several times, on stored, unstored and signed states,
                        and refuses exactly where the reference divides by
                        zero; GRVQ's increment and MSTVQ's B likewise
     statements         every statement the sutra channel decides holds on
                        the corpus and on a run
     couplings          lambda is inverse psi; each GRVQ feed moves exactly
                        its lambda; the 1/N shares; MSTVQ's feedback; kappa
     stages             the magnetism, MSTVQ, TGCR and ZPE rotations keep
                        the energy exactly; ZPE adds only its source; the
                        TGCR geometry recomputed by walking faces
     step export        an exported step replays bit for bit from its file
     strengths          a step from the seed runs each formula at its
                        setting
     glide              each strength rises in equal steps from where it
                        stood to its setting over the glide, then holds
     modes              the mode identities decide and pass
     Maya, magnetism, torus, sources, parameters

   Run: node tools/check_exact_phi_kernel.mjs
   Separate required full-set run (one GitHub job per composition):
     node tools/check_exact_phi_kernel.mjs --full-29=SERIES
     node tools/check_exact_phi_kernel.mjs --full-29=PARALLEL
     node tools/check_exact_phi_kernel.mjs --full-29=SYMMETRIC_CONCURRENT
   These runs use the native seed, all 29 at 50%, and the literal operands.
   They require an actual completed step; a timeout or domain refusal is a
   failure, never a smaller substitute run. The default corpus remains intact.
*/
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const HTML = readFileSync(join(ROOT, 'vedic_v18.51.1_exact_phi.html'), 'utf8');
const start = HTML.indexOf('const KERNEL_FACTORY = () => {');
const end = HTML.indexOf('\nconst KERNEL = KERNEL_FACTORY();');
if (start < 0 || end < 0 || end < start) {
  console.error('FAIL -- KERNEL_FACTORY could not be located in the page');
  process.exit(1);
}
const K = new Function(HTML.slice(start, end) + '\nreturn KERNEL_FACTORY();')();
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
  gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
  tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
  geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
  polarisation: Q.ONE, vortex: Q.ZERO, ...o });
/* Exact hexadecimal integer transport avoids Python's decimal digit limit;
   it changes no value and preserves every numerator and denominator bit. */
const S = x => ({ n: x.n.toString(16), d: x.d.toString(16), base: 16 });
const integerOf = (s, base = 10) => {
  if (base === 10) return BigInt(s);
  if (base !== 16 || typeof s !== 'string' || !/^-?[0-9a-f]+$/i.test(s))
    throw new TypeError('invalid exact integer encoding');
  return s[0] === '-' ? -BigInt('0x' + s.slice(1)) : BigInt('0x' + s);
};
const fromQ = o => Q.mk(integerOf(o.n, o.base), integerOf(o.d, o.base));
const sameQ = (a, b) => a.length === b.length && a.every((x, i) => Q.eq(x, b[i]));
const energyQ = p => p.reduce((s, x) => Q.add(s, Q.mul(x, x)), Q.ZERO);
const reference = cases => {
  let raw;
  try {
    raw = execFileSync('python3', ['-I', join(ROOT, 'tools', 'vedic_reference.py')],
      { input: JSON.stringify({ cases }), maxBuffer: 1 << 28 }).toString();
  } catch (e) {
    throw new Error('exact Fraction oracle did not complete: code=' + e.code +
      ', status=' + e.status + ', signal=' + e.signal + '. ' + e.message);
  }
  const response = JSON.parse(raw);
  if (response.arithmetic !== 'exact-rational' || !Array.isArray(response.results) ||
      response.results.length !== cases.length)
    throw new Error('Fraction oracle omitted a result or did not declare exact rational arithmetic');
  return response.results;
};

/* Frozen input contract, independent of K.SUTRA_ARG_SPEC. The phi operand is
   the specified rational 12586269025/7778742049, not an assertion that it is
   the exact algebraic golden ratio. Changing a default must not silently
   change both the calculation and its expected input at the same time. */
const literal = (n, d = 1n) => Object.freeze({ n, d });
const literalPhi = literal(12586269025n, 7778742049n);
const LITERAL_DEFAULTS = Object.freeze({
  1: [literal(0n), literal(1n)], 2: [literal(10n)], 3: [literal(16n)],
  4: [literal(4n)], 5: [], 6: [literalPhi], 7: [], 8: [literal(1n, 4n)],
  9: [], 10: [literal(10n)], 11: [], 12: [literal(9n)], 13: [literal(8n)],
  14: [literal(3n), literal(1n)], 15: [], 16: [], 17: [literalPhi],
  18: [literal(5n), literal(7n)], 19: [literalPhi, literal(1n, 10n)],
  20: [literal(7n)], 21: [literal(5n), literal(7n)], 22: [],
  23: [literal(10n)], 24: [literal(2n)],
  25: [literal(1n, 4n), literal(1n, 20n)], 26: [literal(16n)], 27: [],
  28: [literal(1n, 20n)], 29: [literal(1n, 5n)]
});
function checkLiteralDefaults() {
  const bad = [];
  for (let id = 1; id <= 29; id++) {
    const actual = K.SUTRA_ARG_SPEC[id], expected = LITERAL_DEFAULTS[id];
    if (!Array.isArray(actual) || actual.length !== expected.length ||
        actual.some((arg, i) => arg.def.n * expected[i].d !== expected[i].n * arg.def.d)) bad.push(id);
  }
  ok(Object.keys(K.SUTRA_ARG_SPEC).length === 29 && bad.length === 0,
    'all 29 default operand lists equal the independent literal rational input contract',
    bad.length ? 'different defaults: S' + bad.join(', S') : '');
}

const cli = process.argv.slice(2);
const fullArg = cli.length === 1 && /^--full-29=(SERIES|PARALLEL|SYMMETRIC_CONCURRENT)$/.exec(cli[0]);
if (cli.length && !fullArg) {
  console.error('unknown arguments: use no arguments, or --full-29=SERIES|PARALLEL|SYMMETRIC_CONCURRENT');
  process.exit(2);
}
if (fullArg) {
  const mode = fullArg[1];
  const reportDir = process.env.EXACT29_REPORT_DIR || join(ROOT, 'artifacts', 'exact29');
  mkdirSync(reportDir, { recursive: true });
  const reportPath = join(reportDir, mode + '.json');
  const report = { schema: 'vedic-full29-verification/1', mode,
    head: process.env.GITHUB_SHA || null,
    pageSha256: createHash('sha256').update(HTML).digest('hex'),
    scope: 'unrounded literal rational vertex formulas and their consumption by one completed implemented step; Maya enclosures and full-model coupling fidelity are separate claims',
    integerEncoding: 'hexadecimal numerator and denominator, base=16',
    completed: false, stage: 'initialization' };
  const checkpoint = (stage, values = {}) => {
    Object.assign(report, values, { stage, updatedAt: new Date().toISOString() });
    writeFileSync(reportPath, JSON.stringify(report));
    console.log('FULL29 ' + mode + ': ' + stage);
  };
  const freeze = value => {
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
      for (const child of Object.values(value)) freeze(child);
      Object.freeze(value);
    }
    return value;
  };
  const rawJson = value => JSON.stringify(value, (_, x) => typeof x === 'bigint' ? { bigintHex: x.toString(16) } : x);
  const canonical = value => value && typeof value.n === 'bigint' && typeof value.d === 'bigint' &&
    value.d > 0n && euclid(value.n, value.d) === 1n;
  try {
    checkpoint('checking literal inputs');
    checkLiteralDefaults();
    ok(K.STORE_BITS === null, 'the full-set calculation has no finite precision storage setting');
    const active = freeze(Array.from({ length: 29 }, (_, i) => ({ id: i + 1,
      strength: literal(1n, 2n), args: [literal(1n, 2n), ...LITERAL_DEFAULTS[i + 1]] })));
    const initial = freeze(K.initEngine()), params = freeze(PARAMS({ vortex: Q.ONE }));
    const initialJson = rawJson(initial), activeJson = rawJson(active), paramsJson = rawJson(params);
    let nativeSeed = initial.psi.length === 16;
    for (let i = 0; i < 16; i++) {
      const hw = (i & 1) + ((i >> 1) & 1) + ((i >> 2) & 1) + ((i >> 3) & 1);
      const den = 1n + BigInt((hw - 2) * (hw - 2));
      const axis = [3n, 4n, 5n, 6n].reduce((sum, x, k) => sum + ((i & (1 << k)) ? x : -x), 0n);
      if (!initial.psi[i] || initial.psi[i].n * den !== (axis * den + 1n) * initial.psi[i].d) nativeSeed = false;
    }
    ok(nativeSeed, 'all29 starts from the complete native forwarded lambda plus dielectric seed');
    if (failures.length) throw new Error('full-set input contract failed: ' + failures.join('; '));
    checkpoint('calculating complete production step', { active: active.map(a => ({ id: a.id, strength: S(a.strength), args: a.args.map(S) })),
      input: initial.psi.map(S), parameters: JSON.parse(rawJson(params)) });
    const progress = event => checkpoint(event.stage + (event.id ? ' S' + event.id : '') + ': ' + event.phase,
      { progress: event });
    const engine = K.stepComplete29(initial, params, active, mode, progress), last = engine.last;
    checkpoint('production step returned; calculating independent complete composition');
    const oracle = reference([{ kind: 'composition', psi: initial.psi.map(S), mode,
      active: active.map(a => ({ id: a.id, args: a.args.map(S) })), t: S(Q.ZERO) }])[0];
    if (oracle.refused) throw new Error('full29 reference composition is undefined: ' + oracle.refused);
    const expectedIds = Array.from({ length: 29 }, (_, i) => i + 1);
    if (mode === 'SYMMETRIC_CONCURRENT') expectedIds.push(...expectedIds.slice().reverse());
    const records = last && last.sutraRecord;
    ok(engine.frame === 1 && last && last.frameIn === 0 && last.mode === mode &&
       last.active.length === 29 && last.active.every((a, i) => a.id === i + 1 && a.strength.n * 2n === a.strength.d),
       'one complete production step returned with all 29 at their actual 50% setting');
    ok(Array.isArray(records) && records.length === expectedIds.length &&
       records.every((r, i) => r.id === expectedIds[i]),
       'every required formula application is present in order (' + expectedIds.length + ' records)');
    ok(last && last.execution && last.execution.required === 29 && last.execution.completed === 29 &&
       last.execution.applications === expectedIds.length &&
       Array.isArray(last.execution.order) && last.execution.order.length === expectedIds.length &&
       last.execution.order.every((id, i) => id === expectedIds[i]),
       'the production all29 entry point records complete execution before returning');
    ok(Array.isArray(oracle.records) && oracle.records.length === expectedIds.length &&
       oracle.records.every((r, i) => r.id === expectedIds[i]),
       'the independent reference completed every required application in the same specified order');
    let exactRecords = Array.isArray(records) && records.length === expectedIds.length &&
      Array.isArray(oracle.records) && oracle.records.length === expectedIds.length;
    if (exactRecords) for (let i = 0; i < expectedIds.length; i++) {
      const actual = records[i], expected = oracle.records[i];
      if (!actual.info || !Array.isArray(actual.info.p) || actual.info.p.length !== 16 ||
          !actual.info.p.every(canonical) || !sameQ(actual.info.p, expected.psi.map(fromQ)) ||
          !Q.eq(actual.alpha, fromQ(expected.alpha))) exactRecords = false;
    }
    ok(exactRecords, 'each actual formula output and alpha equals the independent unrounded Fraction calculation');
    ok(last && Array.isArray(last.afterS) && last.afterS.length === 16 &&
       sameQ(last.afterS, oracle.psi.map(fromQ)),
       'the completed step consumes the full independently composed field as afterS');
    ok(Array.isArray(engine.psi) && engine.psi.length === 16 && engine.psi.every(canonical) &&
       last && sameQ(engine.psi, last.afterZ),
       'the returned state is the canonical rational output of the completed final stage');
    ok(last && last.arithmetic === 'exact-rational' && last.storeBits === null &&
       last.stepError && Q.isZero(last.stepError.delta),
       'the completed step reports exact rational storage with zero quantization error');
    ok(rawJson(initial) === initialJson && rawJson(active) === activeJson && rawJson(params) === paramsJson,
       'the calculation leaves its complete input state, arguments and parameters unchanged');
    if (failures.length) throw new Error(failures.join('; '));
    checkpoint('completed and compared', { completed: true, checksPassed: pass,
      execution: last.execution, output: engine.psi.map(S), afterS: last.afterS.map(S),
      formulaRecords: records.map(r => ({ id: r.id, alpha: S(r.alpha), output: r.info.p.map(S) })),
      referenceAfterS: oracle.psi, referenceRecords: oracle.records });
    console.log(pass + ' full29 checks passed; complete exact outputs: ' + reportPath);
    process.exit(0);
  } catch (error) {
    checkpoint('failed', { completed: false, error: error.stack || String(error), checksPassed: pass,
      failures: failures.slice() });
    console.error('FAIL full29 ' + mode + ': ' + (error.stack || error));
    process.exit(1);
  }
}

section('literal formula operands');
checkLiteralDefaults();

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
  ok(K.STORE_BITS === null, 'STORE_BITS is null: rational storage has no precision truncation');
  let bad = 0, notIdem = 0, n = 0;
  for (let t = 0; t < 2000; t++) {
    let num = rnd(1 + (t * 29) % 700) + 1n;
    const den = rnd(1 + (t * 41) % 700) + 1n;
    if (t % 2) num = -num;
    const x = Q.mk(num, den), s = K.storeQ(x);
    n++;
    /* Equality is decided independently in bare BigInt arithmetic. A small
       nonzero difference is a failure, regardless of its relative size. */
    if (s.d <= 0n || x.n * s.d !== s.n * x.d || euclid(s.n, s.d) !== 1n) bad++;
    const s2 = K.storeQ(s);
    if (s2.n !== s.n || s2.d !== s.d) notIdem++;
  }
  ok(bad === 0, `storeQ(x) = x exactly and remains canonical on ${n} rationals, checked in bare BigInt`, `${bad} differ`);
  ok(notIdem === 0, 'storeQ returns its own output unchanged', `${notIdem} moved`);
  ok(K.storeQ(Q.ZERO).n === 0n, 'zero is stored as zero');
  ok(K.storeBudget([Q.mk(1n, 3n), Q.mk(-7n, 11n), Q.mk((1n << 700n) + 1n, 17n)]).n === 0n,
     'the rational storage quantization error is zero, including non-dyadic values beyond 128 bits');
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


section('S13 undefined denominator is visible');
{
  const psi = K.initState().slice(); psi[12] = Q.ZERO;
  const before = psi.map(S), args = [Q.HALF, ...LITERAL_DEFAULTS[13]];
  let error = null;
  try { K.applySutra(13, psi, args, { t: Q.ZERO, gain: Q.ONE }); }
  catch (e) { error = e; }
  const expected = reference([{ kind: 'sutra', id: 13, psi: before, args: args.map(S),
    t: S(Q.ZERO), gain: S(Q.ONE) }])[0];
  ok(error instanceof RangeError && /S13/.test(error.message) && /12/.test(error.message),
     'S13 names its zero denominator pair instead of silently skipping it', error ? error.message : 'calculation returned');
  ok(typeof expected.refused === 'string' && /S13/.test(expected.refused) && /12/.test(expected.refused),
     'the independent Fraction oracle refuses the same explicitly undefined S13 pair', expected.refused || 'calculation returned');
  ok(JSON.stringify(psi.map(S)) === JSON.stringify(before), 'an undefined S13 application leaves its caller input unchanged');
}

section('the 29 v18.24 formulas against an independent unrounded transcription');
{
  const states = [['the seed', K.initState()], ['the seed, stored', K.initState().map(K.storeQ)]];
  for (let k = 0; k < 6; k++) states.push([`stored state ${k}`, Array.from({ length: 16 }, () => {
    const n = rnd(12 + 5 * k) - (1n << BigInt(11 + 5 * k)), d = rnd(7) + 1n; return K.storeQ(Q.mk(n === 0n ? 1n : n, d)); })]);
  states.push(['unstored, odd denominators', Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(v * v * 37 - 400), BigInt(3 + 2 * (v % 5))))]);
  states.push(['inside (-1, 1)', Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v * 7919) % 199) - 99), 101n))]);
  states.push(['thousands', Array.from({ length: 16 }, (_, v) => K.storeQ(Q.mk(BigInt(((v + 3) * 104729) % 90001 - 45000), 7n)))]);
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
        try { js = { psi: K.applySutra(id, psi, a, { t, gain }).map(S) }; }
        catch (e) { js = { refused: e instanceof RangeError ? e.message : 'not a RangeError: ' + e.message }; }
        cases.push({ kind: 'sutra', id, psi: psi.map(S), args: a.map(S), gain: S(gain), t: S(t) });
        meta.push({ id, name, js });
      });
    }
  });
  const out = reference(cases);
  let agree = 0, refusedBoth = 0;
  const bad = [], refusals = {};
  out.forEach((r, i) => {
    const m = meta[i];
    if (r.refused || m.js.refused) {
      if (r.refused && m.js.refused && !m.js.refused.startsWith('not a')) { refusedBoth++; (refusals[m.name] = refusals[m.name] || new Set()).add(m.id); }
      else bad.push(`S${m.id} on ${m.name}: reference ${r.refused ? 'refuses' : 'computes'}, kernel ${m.js.refused || 'computes'}`);
      return;
    }
    if (Array.isArray(r.psi) && r.psi.length === 16 && m.js.psi.length === 16 &&
        r.psi.every((x, j) => x.n === m.js.psi[j].n && x.d === m.js.psi[j].d && x.base === m.js.psi[j].base)) agree++;
    else bad.push(`S${m.id} on ${m.name}: values differ`);
  });
  ok(bad.length === 0 && agree + refusedBoth === cases.length,
     `${cases.length} applications (${states.length} states x 29 formulas x default and extreme operands, two gains, four times): ${agree} equal the reference exactly, ${refusedBoth} refused by both`, bad.slice(0, 4).join('; '));
  /* S13 divides by each complement vertex 8..15. Determine this expectation
     from the supplied field, independently of either implementation's refusal. */
  const want = Object.fromEntries(states.filter(([, psi]) => psi.slice(8).some(x => x.n === 0n))
    .map(([name]) => [name, 13]));
  for (const [name, id] of Object.entries(want))
    ok(refusals[name] && refusals[name].has(+id), `${name}: S${id} refuses, in the kernel and the reference`, refusals[name] ? [...refusals[name]].join(',') : 'none');
  const unexpected = Object.entries(refusals).flatMap(([name, ids]) => [...ids]
    .filter(id => want[name] !== id).map(id => `${name}: S${id}`));
  ok(unexpected.length === 0, 'nothing refuses where nothing divides by zero', unexpected.join('; '));
  /* GRVQ's increment and MSTVQ's B on their own */
  const gcases = [], gmeta = [];
  for (let k = 0; k < 40; k++) {
    const lam = [0, 1, 2, 3].map(j => Q.mk(rnd(9) - 256n, rnd(4) + 1n));
    if (k % 5 === 0) lam[(k / 5) % 4] = Q.ZERO;
    const kk = k % 4, t = Q.mk(BigInt(k * 777), 1n);
    gcases.push({ kind: 'grvq', psi: [], lambda: lam.map(S), k: kk, t: S(t) });
    let js; try { js = S(K.grvqDelta(lam, kk, t)); } catch (e) { js = 'refused'; }
    gmeta.push(js);
  }
  const gout = reference(gcases);
  ok(gout.every((r, i) => (r.refused && gmeta[i] === 'refused') || (r.delta && r.delta.n === gmeta[i].n && r.delta.d === gmeta[i].d)),
     'GRVQ.evolveExact (s9 + s7 + s10 + s6 + s14 + s15 + s28 + s29) equals the reference on 40 lambdas, zeros included');
  const bcases = states.slice(0, 8).map(([, psi], i) => ({ kind: 'setB', psi: psi.map(S), s: S(Q.mk(BigInt(i + 1), 13n)) }));
  const bout = reference(bcases);
  ok(bout.every((r, i) => K.mstvqB(states[i][1], Q.mk(BigInt(i + 1), 13n)).every((x, j) => S(x).n === r.B[j].n && S(x).d === r.B[j].d)),
     'MSTVQ.setB equals the reference on 8 states');
}

/* This retained compatibility check deliberately transforms the legacy
   rounding and S15 residual and stubs GRVQ/MSTVQ/USO/R4. Its float tolerance
   checks only that transformed vertex-formula harness, not the complete
   original coupled model or exact equality. The Fraction oracle above is
   the independent exact rational comparison. */
section('the 29 formulas against the transformed v18.24 floating-point compatibility harness');
{
  const V18 = readFileSync(join(ROOT, 'vedic_v18.24_full_kernel.html'), 'utf8');
  const cut = (from, to) => { const a = V18.indexOf(from); const b = V18.indexOf(to, a); if (a < 0 || b < 0) throw new Error('v18.24: cannot find ' + from); return V18.slice(a, b + to.length); };
  const qsrc = cut('const gcd = (a, b) =>', "flStrict: r => { if (r.d === 0n) return NaN; const v = Number(r.n) / Number(r.d); return isFinite(v) ? v : NaN; },\n};");
  const vtxsrc = cut('const VTX = {', 'auditEnergyEnvelope(maxEnergy) {\n    return { strict: true, mutated: false, maxEnergy };\n  }\n};');
  let sut = cut('const SUTRAS = [', '\n];');
  const exactBi = src => { let out = '', i = 0, n = 0;
    for (;;) { const j = src.indexOf('Q.mk(Bi(', i); if (j < 0) { out += src.slice(i); break; }
      out += src.slice(i, j); let k = j + 8, depth = 1; while (depth) { const c = src[k++]; if (c === '(') depth++; else if (c === ')') depth--; }
      const m = /^\s*,\s*(\d+)n\)/.exec(src.slice(k));
      if (!m) { out += src.slice(j, k); i = k; continue; }
      out += `QD((${src.slice(j + 8, k - 1).trim()}) / ${m[1]})`; i = k + m[0].length; n++; }
    return { out, n }; };
  const conv1 = exactBi(sut); sut = conv1.out;
  const s15a = "        const residual = Q.fl(Q.sub(prodSum, sumProd));\n        if (isFinite(residual) && abs(residual) > 1e-6) {\n          const corr = -residual;";
  const s15b = "        const residual = Q.sub(prodSum, sumProd);\n        if (bigAbs(residual.n) * 1000000n > residual.d) {\n          const corr = Q.mk(-residual.n, residual.d), aQ = Q.mul(Q.mk(15n, 435n), Q.mk(BigInt(this.strength), 100n));";
  const s15c = ['Q.add(Q.ONE, QD((α * corr * 10000) / 10000))', 'Q.add(Q.ONE, Q.mul(aQ, corr))'], s15d = ['Q.add(Q.ONE, QD((-α * corr * 5000) / 10000))', 'Q.sub(Q.ONE, Q.mul(Q.mul(aQ, corr), Q.HALF))'];
  const s15ok = sut.split(s15a).length === 2 && sut.split(s15c[0]).length === 2 && sut.split(s15d[0]).length === 2;
  sut = sut.replace(s15a, s15b).replace(s15c[0], s15c[1]).replace(s15d[0], s15d[1]);
  const harness = `const abs = Math.abs, max = Math.max, min = Math.min, floor = Math.floor, sqrt = Math.sqrt, round = Math.round, PI = Math.PI, sin = Math.sin, cos = Math.cos, exp = Math.exp, log = Math.log, pow = Math.pow;
    const sf = v => String(v); function bigAbs(n) { return n < 0n ? -n : n; }
    ${qsrc}
    const Bi = strictBi, SUTRA_SUM = 435n;
    const QD = x => { if (!Number.isFinite(x)) throw new RangeError('non-finite ' + x); let v = x, d = 1n; while (!Number.isInteger(v)) { v *= 2; d *= 2n; } return Q.mk(BigInt(v), d); };
    const C = { phi: Q.mk(12586269025n, 7778742049n), phi_f: 12586269025 / 7778742049, alpha: Q.mk(1000000n, 137035999n), alpha_f: 1 / 137.035999 };
    let lambda = [Q.mk(3n), Q.mk(4n), Q.mk(5n), Q.mk(6n)], simTime = 0;
    const R4val = Q.ONE, SULBA = { PI_SULBA: 3 }, USO = { unified: () => 1 };
    const GRVQ = { evolve: () => 0 }, MSTVQ = { setB() {}, compute() {}, trace: () => 0 }, INTER_SUTRA = { recordInteraction() {} };
    const ALPHA = { computeQ(sid, strength) { return Q.mul(Q.mk(BigInt(sid), SUTRA_SUM), Q.mk(BigInt(Math.round(strength || 0)), 100n)); } };
    const SUPPRESS = { alpha(sid) { const s = SUTRAS[sid - 1]; return Q.fl(ALPHA.computeQ(sid, s ? s.strength : 0)); }, alphaPhi(sid) { return this.alpha(sid); } };
    ${vtxsrc}
    ${sut}
    return { Q, VTX, SUTRAS };`;
  const V = new Function(harness)();
  const bl = x => x === 0n ? 0 : (x < 0n ? -x : x).toString(2).length;
  const magLog = q => q.n === 0n ? -Infinity : bl(q.n) - bl(q.d);
  const states = [K.initEngine().psi];
  for (let t = 0; t < 5; t++) states.push(Array.from({ length: 16 }, () => Q.mk(rnd(14) - 8192n + 1n, 100n)));
  let worstBits = Infinity, where = '', compared = 0, refusedBoth = 0, mismatchRefusal = [];
  const perId = {};
  for (let id = 1; id <= 29; id++) for (const pct of [5, 50, 100]) for (const [si, s0] of states.entries()) {
    const sutra = V.SUTRAS[id - 1]; sutra.strength = pct;
    for (const k of Object.keys(sutra)) if (k.startsWith('_')) sutra[k] = k === '_cachedOrder' ? -1 : null;
    V.VTX.psi = s0.map(q => V.Q.mk(q.n, q.d));
    let old = null, port = null, oldErr = null, portErr = null;
    try { sutra.evolve(1); old = V.VTX.psi.map(q => q.d === 0n ? null : Q.mk(q.n, q.d)); if (old.some(q => q === null)) { oldErr = 'a 0/0'; old = null; } } catch (e) { oldErr = e.message; }
    try { const st = Q.mk(BigInt(pct), 100n); port = K.vedicField(id, s0, K.sutraArgs(id, st), Q.mul(Q.mk(BigInt(id), 435n), st)).p; } catch (e) { portErr = e; }
    if (!old || !port) {
      if (!old && portErr instanceof RangeError) refusedBoth++;
      else mismatchRefusal.push(`S${id} ${pct}% state ${si}: v18.24 ${oldErr || 'ok'}, port ${portErr ? portErr.message : 'ok'}`);
      continue;
    }
    compared++;
    let disp = -Infinity, diff = -Infinity;
    for (let v = 0; v < 16; v++) { disp = Math.max(disp, magLog(Q.sub(port[v], s0[v])), magLog(Q.sub(old[v], s0[v]))); diff = Math.max(diff, magLog(Q.sub(old[v], port[v]))); }
    const bits = disp - diff;
    perId[id] = Math.min(perId[id] ?? Infinity, bits);
    if (bits < worstBits) { worstBits = bits; where = `S${id} at ${pct}% on state ${si}`; }
  }
  ok(conv1.n >= 40 && s15ok, `v18.24's SUTRAS array, Q and VTX are read from vedic_v18.24_full_kernel.html; each of its ${conv1.n} roundings Q.mk(Bi(x · N), N) is made the exact value of the double x, and S15's float residual is taken exactly`);
  ok(compared === 29 * 3 * 6 - refusedBoth && mismatchRefusal.length === 0 && worstBits >= 32,
     `on ${compared} applications (29 formulas x 5%, 50%, 100% x 6 states) the transformed v18.24 harness and the kernel agree to within 2^-${worstBits === Infinity ? 'inf' : worstBits} of the displacement (worst ${where}); ${refusedBoth} refused by both`,
     mismatchRefusal.slice(0, 3).join('; ') + ' ' + Object.entries(perId).filter(([, b]) => b < 32).map(([i, b]) => `S${i}:${b}`).join(' '));
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
  ok(names.has('vedic_update_zero') && names.size >= 30, `${names.size} distinct statements are decided`);
  const page = HTML.match(/sutra: \{\n\s*draws:[^\n]*\n\s*theorems: \[([^\]]*)\]/);
  const cited = page ? new Set([...page[1].matchAll(/'([^']+)'/g)].map(m => m[1])) : new Set();
  const missing = [...names].filter(n => !cited.has(n));
  ok(page && missing.length === 0, 'every statement decided is cited by the sutra channel', missing.join(', '));
  /* a run: each step's formulas, on the field they act on */
  const act = [2, 9, 12, 22, 29].map(id => ({ id, strength: Q.HALF }));
  let eng = K.initEngine(), runOk = true;
  for (let s = 0; s < 6; s++) { eng = K.step(eng, PARAMS({ vortex: Q.ONE }), act, 'SERIES');
    for (const a of act) if (!K.formulaClaims(a.id, eng.psi, a.strength).ok) runOk = false; }
  ok(runOk, 'every statement holds for S2, S9, S12, S22, S29 on six steps of a run, at the live strength');
}

section('the couplings: lambda, GRVQ, MSTVQ, kappa');
{
  const sgn = (i, k) => ((i >> k) & 1) ? 1n : -1n;
  const inv = p => [0, 1, 2, 3].map(k => Q.mul(p.reduce((s, x, i) => sgn(i, k) > 0n ? Q.add(s, x) : Q.sub(s, x), Q.ZERO), Q.mk(1n, 16n)));
  const fwd = l => Array.from({ length: 16 }, (_, i) => l.reduce((s, x, k) => sgn(i, k) > 0n ? Q.add(s, x) : Q.sub(s, x), Q.ZERO));
  const seedS = K.initState();
  ok(sameQ(K.inverseL(seedS), [3n, 4n, 5n, 6n].map(n => Q.mk(n))) && sameQ(inv(seedS), K.inverseL(seedS)),
     "the seed's lambda is v18.24's (3, 4, 5, 6): lambda_k = sum_i s_ik psi_i / 16, recomputed here");
  ok(sameQ(fwd([3n, 4n, 5n, 6n].map(n => Q.mk(n))).map((x, v) => Q.add(x, K.initDielectric()[v])), seedS) && seedS.every(x => x.n !== 0n),
     'the seed is v18.24 forward(lambda) plus the dielectric seed, and no vertex is zero');
  const st = x => x.n === 0n ? x : K.storeQ(x);
  let feedOk = true;
  for (const [id, k] of [[1, 0], [5, 1], [9, 2], [14, 3]]) {
    const t = Q.mk(2500n), gain = Q.mk(3n, 4n), rec = [], reg = inv(seedS), reg0 = reg.slice();
    const out = K.applySutra(id, seedS, Q.HALF, { t, gain, record: rec, lam: reg });
    const field = K.vedicField(id, seedS, K.sutraArgs(id, Q.HALF), Q.mul(Q.mul(Q.mk(BigInt(id), 435n), Q.HALF), gain)).p;
    const d = Q.mul(K.grvqDelta(reg0, k, t), gain);
    if (!sameQ(out, field) || !reg.every((x, j) => Q.eq(x, j === k ? st(Q.add(reg0[j], d)) : reg0[j])) || !Q.eq(rec[0].grvq.delta, d)) feedOk = false;
  }
  ok(feedOk, 'S1, S5, S9 and S14 move psi by their own formula only, and add GRVQ(lambda) x gain to lambda_0, 1, 2, 3 of the step register without quantization, as v18.24 adds it to lambda[k]');
  /* the other active sutras' 1/N shares, and the MSTVQ feedback, in a step */
  const act = [{ id: 1, strength: Q.mk(3n, 10n) }, { id: 3, strength: Q.mk(1n, 5n) }, { id: 22, strength: Q.HALF }, { id: 27, strength: Q.HALF }];
  const p = PARAMS({ vortex: Q.ONE });
  let eng = K.initEngine();
  for (let s = 0; s < 3; s++) eng = K.step(eng, p, act, 'SERIES');
  const L = eng.last;
  const N = BigInt(act.length);
  let shares = L.grvqRecord.length === 3;
  const reg = inv(L.input);
  for (const r of L.sutraRecord) if (r.grvq) reg[r.grvq.k] = st(Q.add(reg[r.grvq.k], r.grvq.delta));
  for (const g of L.grvqRecord) {
    const k = (g.id - 1) % 4, want = Q.mul(K.grvqDelta(reg, k, L.time), Q.mk(1n, N));
    if (g.k !== k || !Q.eq(g.delta, want)) shares = false;
    reg[k] = st(Q.add(reg[k], g.delta));
  }
  ok(shares && sameQ(L.afterD, L.afterS), 'each other active sutra adds GRVQ / N to lambda_(id-1 mod 4) of the register, in order, and psi is left as the formulas made it');
  const a1 = Q.mul(Q.mk(1n, 435n), act[0].strength);
  const B = K.mstvqB(L.afterS, a1), b2 = B.reduce((s, y) => Q.add(s, Q.mul(y, y)), Q.ZERO);
  const T = [0, 1, 2].map(i => Q.sub(Q.mul(B[i], B[i]), Q.mul(Q.HALF, b2)));
  for (let k = 0; k < 3; k++) reg[k] = st(Q.add(reg[k], Q.mul(T[k], Q.mk(1n, 200000000n))));
  ok(sameQ(L.afterB, L.afterS) && sameQ(L.mstvq.B, B) && sameQ(L.lambdaReg, reg),
     "MSTVQ: B from psi_0+psi_1-psi_14-psi_15 etc. times S1's |alpha|, T_kk / 2e8 into lambda_0..2 of the register, psi left as it was, recomputed here");
  const trace = Q.add(Q.add(T[0], T[1]), T[2]);
  ok(Q.eq(eng.tgcrKappa, K.storeQ(Q.add(L.tgcrKappaIn, Q.mul(trace, Q.mk(1n, 500n))))) && Q.eq(trace, Q.neg(Q.mul(Q.HALF, b2))),
     'the TGCR stiffness kappa moves by tr T / 500, and tr T = -|B|^2 / 2 (stress_trace)');
  const setters = L.sutraRecord.filter(r => r.B).map(r => r.id);
  ok(setters.length === 1 && setters[0] === 3, 'of the active sutras only S3 sets MSTVQ B in the core (S2, S3 and S29 do)');
  const gc = K.grvqClaims(L.sutraRecord.filter(r => r.grvq).map(r => r.grvq).concat(L.grvqRecord), L.time);
  ok(gc.ok && gc.results.length === 3 * 4, `the coupling statements hold on all four feeds (${gc.results.length} decided)`);
}

section('the stages of the real field');
{
  const p = PARAMS({ vortex: Q.mk(3n, 2n), zpeSource: X.fromQ(Q.mk(5n, 4n)) });
  const act = [{ id: 9, strength: Q.HALF }, { id: 29, strength: Q.HALF }];
  let eng = K.initEngine();
  for (let s = 0; s < 4; s++) eng = K.step(eng, p, act, 'SERIES');
  const L = eng.last, id = x => x;
  const eq = (a, b) => Q.eq(energyQ(a), energyQ(b));
  ok(eq(K.edgeRotate(L.afterB, L.ctx.magTau, id), L.afterB), 'the magnetism stage, every edge rotated by dt A_e axis by axis, keeps the energy exactly (rot_pair_energy)');
  ok(eq(K.complementRotate(L.afterG, K.mstvqTau(L.wheeler, p)), L.afterG), 'the MSTVQ stage keeps the energy exactly');
  ok(eq(K.edgeRotate(L.afterM, K.tgcrTau(L.afterM, p), id), L.afterM), 'the TGCR stage keeps the energy exactly');
  const zr = K.complementRotate(L.afterT, K.zpeTau(L.wheeler, p));
  ok(eq(zr, L.afterT), "ZPE's rotation keeps the energy exactly");
  const src = Q.mul(Q.mk(1n, 64n), Q.mk(5n, 4n));
  ok(sameQ(K.complementRotate(L.afterT, K.zpeTau(L.wheeler, p)).map((x, v) => [3, 5, 6, 9, 10, 12].includes(v) ? Q.add(x, src) : x).map(K.storeQ), L.afterZ),
     'and the drive dt x source is added to the six inertial vertices only');
  /* the magnetism stage's angles: dt (A_e - (1/10)(chi_head - chi_tail)) */
  const tauWant = K.MAG_EDGES.map(([v, , h], e) => Q.mul(Q.mk(1n, 64n), Q.sub(L.magAIn[e], Q.mul(Q.mk(1n, 10n), Q.sub(L.magChiIn[h], L.magChiIn[v])))));
  ok(sameQ(L.ctx.magTau, tauWant), "the magnetism stage turns each edge by dt times the whole potential on it, Faraday part included");
  /* the rotation itself, against cos/sin of 2 arctan t in floats */
  let rotOk = true;
  for (const [x, y, t] of [[3, -2, 1 / 7], [-5, 11, -2], [1, 0, 1 / 3]]) {
    const [u, w] = K.rotPair(Q.mk(BigInt(x)), Q.mk(BigInt(y)), Q.parse(t === 1 / 7 ? '1/7' : t === -2 ? '-2' : '1/3'));
    const th = 2 * Math.atan(t), c = Math.cos(th), s = Math.sin(th);
    if (Math.abs(Number(u.n) / Number(u.d) - (c * x + s * y)) > 1e-12 || Math.abs(Number(w.n) / Number(w.d) - (c * y - s * x)) > 1e-12) rotOk = false;
  }
  ok(rotOk, 'a pair rotation by t is the rotation by 2 arctan t, against floats');
  /* TGCR geometry walked here: bond current, curl by corners, v18.24 Hodge tables */
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
    if (!Q.eq(G.density[i], Q.sub(ss, aa))) dens = false;
  }
  ok(dens && Q.eq(G.pontryagin, pont) && Q.eq(G.FSq, fsq), "TGCR: F = d1 v, v = omega psi_t psi_h, split at each vertex by v18.24's tables; F^F and |F|^2 recomputed here");
  ok(Q.eq(G.energy, Q.add(Q.add(Q.mul(Q.HALF, fsq), Q.mul(Q.mul(Q.HALF, eng.tgcrKappa), energyQ(G.v))), Q.mul(Q.HALF, pont))),
     'TGCR energy = |F|^2/2 + kappa |v|^2/2 + eta F^F with eta = 1/2');
  ok(G.cubeFlux.every(Q.isZero) && G.density.some(x => x.n !== 0n), 'the vortex curvature closes on every cube and is not zero');
  const sc = K.stageClaims(L, p);
  ok(sc.ok, `the stage statements hold (${sc.results.map(r => r.name).join(', ')})`);
}

const EXPORT_CASES = [['9,11,29', 'PARALLEL', Q.ONE, 0n, Q.ONE], ['3,9,11,25,29', 'SYMMETRIC_CONCURRENT', Q.mk(3n, 10n), 1n, Q.TWO],
                      ['1', 'SERIES', Q.HALF, 0n, Q.ONE]];
section('step export');
{
  for (const [ids, mode, strength, drive, vortex] of EXPORT_CASES) {
    const active = ids.split(',').map(Number).map(id => ({ id, strength }));
    let eng = K.initEngine();
    for (let s = 0; s < 2; s++) eng = K.step(eng, PARAMS({ zpeSource: X.fromQ(Q.mk(drive)), vortex }), active, mode);
    const file = JSON.parse(JSON.stringify({ frame: eng.frame, last: K.serializeLast(eng.last), psi: K.serializeState(eng.psi),
      dielectric: eng.dielectric.map(K.serializeQ), tgcrKappa: K.serializeQ(eng.tgcrKappa),
      mayaPhase: K.serializeQ(eng.mayaPhase), mayaErr: K.serializeQ(eng.mayaErr),
      magA: eng.magA.map(K.serializeQ), magChi: eng.magChi.map(K.serializeQ) }));
    const L = file.last, where = `${mode} ${ids}, step 2`;
    let same = false, why = '';
    try {
      const fromA = o => A4.mk(fromQ(o.a), fromQ(o.b), fromQ(o.c), fromQ(o.d));
      const p = Object.fromEntries(Object.entries(L.params).map(([k, v]) => [k, ('a' in v && 'b' in v) ? fromA(v) : fromQ(v)]));
      const act = L.activeSutras.map(x => x.args ? { id: x.id, strength: fromQ(x.strength), args: x.args.map(fromQ) } : { id: x.id, strength: fromQ(x.strength) });
      const replay = K.step({ psi: L.input.map(fromQ), dielectric: L.dielectricIn.map(fromQ),
                              energyLedger: null, frame: L.frameIn, last: null, tgcrKappa: fromQ(L.tgcrKappaIn),
                              mayaPhase: fromQ(L.mayaPhaseIn), mayaErr: fromQ(L.mayaErrIn),
                              magA: L.magAIn.map(fromQ), magChi: L.magChiIn.map(fromQ) }, p, act, L.composition);
      same = JSON.stringify(K.serializeState(replay.psi)) === JSON.stringify(file.psi) &&
             JSON.stringify(replay.dielectric.map(K.serializeQ)) === JSON.stringify(file.dielectric) &&
             JSON.stringify(K.serializeQ(replay.tgcrKappa)) === JSON.stringify(file.tgcrKappa) &&
             JSON.stringify(replay.magA.map(K.serializeQ)) === JSON.stringify(file.magA) &&
             JSON.stringify(replay.magChi.map(K.serializeQ)) === JSON.stringify(file.magChi) &&
             JSON.stringify(K.serializeLast(replay.last)) === JSON.stringify(L);
      if (!same) why = 'the replay differs from the file';
    } catch (e) { why = e.message; }
    ok(same, `${where}: replays from its exported file alone, bit for bit`, why);
    ok(L.magnetismIn.closed && L.tgcr.cubeFlux.every(x => x.n === '0'), `${where}: the flux it starts from and the vortex curvature it ends with close on every cube`);
    ok(L.lambdaIn.length === 4 && L.lambdaOut.length === 4 && L.sutraRecord.length === active.length * (mode === 'SYMMETRIC_CONCURRENT' ? 2 : 1),
       `${where}: carries lambda in and out and one record per formula applied`);
  }
}

section('strengths');
{
  const red = (n, d) => { const g = euclid(n, d) || 1n; return [n / g, d / g]; };
  const eqF = (q, [n, d]) => q.n === n && q.d === d;
  const tg = [{ id: 9, strength: Q.mk(3n, 10n) }];
  let e = K.initEngine(), stepsOk = true;
  for (let s2 = 1; s2 <= 4; s2++) { e = K.step(e, PARAMS(), tg, 'SERIES');
    const al = red(9n * 3n, 435n * 10n);
    if (!(e.last.active.length === 1 && eqF(e.last.active[0].strength, [3n, 10n]) && e.last.sutraRecord.length === 1 && eqF(e.last.sutraRecord[0].alpha, al))) stepsOk = false; }
  ok(stepsOk, 'a step from the seed runs each formula at its setting from the first step: alpha = id/435 x setting');
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
  r = [];
  const s2 = radix => [{ id: 2, strength: Q.mk(1n, 2n), args: [Q.mk(1n, 2n), Q.mk(BigInt(radix))] }];
  for (let j = 0; j < 3; j++) r = K.rampStep(r, s2(10), 10).ramp;
  const op = K.rampStep(r, s2(12), 10);
  ok(eqF(op.active[0].strength, at(F(0n), F(1n, 2n), 4, 10)) && op.active[0].args[1].n === 12n,
     'an operand changed mid-glide applies at once and the strength glides on', `${Q.toString(op.active[0].strength)} radix ${op.active[0].args[1].n}`);
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
      ok(shared.modes.every(m => sameQ(shared.states[m], fresh.states[m]) && Q.eq(shared.energies[m], fresh.energies[m])),
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
  const exact = (got, want) => got.n * want.d === want.n * got.d;
  let eng = K.initEngine(), inductOk = true, chiOk = true, closedOk = true, gradOk = true, maxA = Q.ZERO, recOk = true, potOk = true, shareOk = true;
  for (let s = 0; s < 40; s++) {
    const prev = eng;
    eng = K.step(eng, base, act, 'PARALLEL');
    const L = eng.last, rOut = eng.psi.map(x => Q.mul(x, x));
    const Etot = rOut.reduce((t, x) => Q.add(t, x), Q.ZERO);
    edges.forEach(([v, a, h], e) => {
      const share = Q.div(Q.mul(Q.mk(8n), Q.add(rOut[v], rOut[h])), Etot);
      const k = a >= 2 ? Q.mul(Q.mk(6n, 100n), base.vortex) : Q.mul(Q.mul(Q.mk(3n, 100n), Q.HALF), base.vortex);
      const want = Q.add(Q.mul(Q.mk(499n, 500n), prev.magA[e]), Q.mul(Q.mul(k, cyc[(hwf(v) + a) % 4]), share));
      if (!exact(eng.magA[e], want)) inductOk = false;
      if (Q.cmp(Q.abs(eng.magA[e]), maxA) > 0) maxA = Q.abs(eng.magA[e]);
    });
    for (let v = 0; v < 16; v++) if (!exact(eng.magChi[v], Q.add(Q.mul(Q.mk(499n, 500n), prev.magChi[v]), Q.sub(eng.psi[v], prev.psi[v])))) chiOk = false;
    const Ein = prev.psi.reduce((t, x) => Q.add(t, Q.mul(x, x)), Q.ZERO);
    if (!L.shareIn.every((x, v) => Q.eq(x, Q.div(Q.mul(Q.mk(16n), Q.mul(prev.psi[v], prev.psi[v])), Ein)))) shareOk = false;
    const md = K.magData(eng.magA, eng.magChi);
    if (!md.closed) closedOk = false;
    if (!K.curlA(K.fullA(Array(32).fill(Q.ZERO), eng.magChi)).every(Q.isZero)) gradOk = false;
    for (let v = 0; v < 16; v++) {
      let want = Q.ZERO;
      for (let a = 0; a < 4; a++) { const e = edges.findIndex(([t, ax]) => ax === a && t === (v & ~(1 << a))); want = (v & (1 << a)) ? Q.add(want, eng.magA[e]) : Q.sub(want, eng.magA[e]);
        want = Q.sub(want, Q.mul(Q.mk(1n, 10n), Q.sub(eng.magChi[v], eng.magChi[v ^ (1 << a)]))); }
      if (!Q.eq(md.potential[v], want)) potOk = false;
    }
    const fl = q => Number(q.n) / Number(q.d), bv = K.bvecAt(md.flux);
    edges.forEach(([v, , h], e) => {
      const a = bv[v].map(fl), b = bv[h].map(fl), na = Math.hypot(...a), nb = Math.hypot(...b);
      if (na * na < 1e-6 || nb * nb < 1e-6) { if (md.reconnections.includes(e)) recOk = false; return; }
      const c = a.reduce((dot, value, k) => dot + value * b[k], 0) / (na * nb);
      if (Math.abs(c + 0.7) > 1e-9 && (c < -0.7) !== md.reconnections.includes(e)) recOk = false;
    });
  }
  ok(inductOk, 'A = (499/500) A + omega (6/100 | eta 3/100) m_e share_e on all 32 edges for 40 steps, recomputed here');
  ok(chiOk, "the Faraday potential chi = (499/500) chi + (psi_out - psi_in), v18.24's dPsi, recomputed here");
  ok(shareOk, 'the dielectric is polarised by each vertex share 16 psi_v^2 / E of the field energy, recomputed here');
  ok(gradOk, 'the Faraday part of A carries no flux on any of those steps (faraday_flux_free)');
  ok(closedOk, 'the flux out of every one of the 8 cubes is zero on every step (flux_closed)');
  ok(potOk, 'the potential at each vertex is div A - (1/10) L chi, recomputed here');
  ok(recOk, 'reconnections are exactly the edges where cos(B_i, B_j) < -0.7 (reconnect_iff)');
  const smax = Q.mul(Q.mul(Q.abs(base.vortex), Q.mk(6n * 13n * 16n, 1000n)), Q.mk(1001n, 1000n));
  ok(Q.cmp(maxA, Q.mul(smax, Q.mk(500n))) <= 0, `|A| stays within max |A_0|, s/(1 - r) (retention_bound): ${(Number(maxA.n) / Number(maxA.d)).toFixed(3)} <= ${(Number(smax.n) / Number(smax.d) * 500).toFixed(1)}`);
  ok(Q.isZero(K.helicityOf(Array(32).fill(Q.ZERO), eng.magChi)), 'a potential that is all Faraday has no helicity (helicity_of_gradient)');
  ok(Q.eq(K.edgeShares(eng.last.rhoOut).reduce((t, x) => Q.add(t, x), Q.ZERO), Q.mk(32n)), 'the edge shares sum to 32 (edge_share_total)');
  ok(throwsRange(() => K.edgeShares(Array(16).fill(Q.ZERO))) && throwsRange(() => K.energyShare(Array(16).fill(Q.ZERO))),
     'a field with no energy has no shares: refused, not divided by zero');
}

section('no parameter is ignored');
{
  const base = PARAMS();
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  const run = p => { let e = K.initEngine(); for (let s = 0; s < 6; s++) e = K.step(e, p, act, 'PARALLEL'); return e; };
  const ref = run(base), key = e => JSON.stringify(K.serializeState(e.psi)) + JSON.stringify(e.dielectric.map(K.serializeQ));
  ok(key(run({ ...base, vortex: Q.ONE })) !== key(ref), 'vortex omega changes the field');
  ok(key(run({ ...base, zpeSource: X.ONE })) !== key(ref), 'the drive changes the field');
  ok(key(run({ ...base, polarisation: Q.ZERO })) !== key(ref), 'the polarisation changes the dielectric and the field');
  ok(key(run({ ...base, tgcr: X.fromQ(Q.mk(1n, 64n)), vortex: Q.ONE })) !== key(run({ ...base, vortex: Q.ONE })), 'the TGCR rate changes the field');
  const share = K.energyShare(K.rhoHat(K.initState()));
  const d0 = K.evolveDielectric(K.initDielectric(), act, base, Array(16).fill(Q.ZERO));
  const d1 = K.evolveDielectric(K.initDielectric(), act, { ...base, polarisation: Q.mk(3n, 2n) }, share);
  ok(d1.every((q, v) => Q.eq(Q.sub(q, d0[v]), Q.mul(Q.mul(Q.mk(3n, 2n), share[v]), Q.mk(1n, 8n)))),
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
  const same = (a, b) => a.length === b.length && a.every((x, i) => Array.isArray(x) ? same(x, b[i]) : Q.eq(x, b[i]));
  const lawsHold = g => [0, 1, 2, 3].every(s =>
      Q.eq(Q.sub(g.hole[(s + 1) % 4], g.hole[s]), Q.neg([0, 1, 2, 3].reduce((t, r) => Q.add(t, g.cells[r][s]), Q.ZERO)))) &&
    [0, 1, 2, 3].every(r =>
      Q.eq(Q.sub(g.tube[(r + 1) % 4], g.tube[r]), [0, 1, 2, 3].reduce((t, s) => Q.add(t, g.cells[r][s]), Q.ZERO)));
  let agree = true, laws = true, balanced = true, where = '';
  for (let trial = 0; trial < 40; trial++) {
    const A = K.MAG_EDGES.map((_, e) => Q.mk(BigInt(((e + 3) * (trial + 7) * 7919) % 2001 - 1000), BigInt(1 + ((e * trial) % 9))));
    const chi = Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v + 1) * (trial + 2) * 104729) % 301 - 150), BigInt(1 + v % 5)));
    const g = read(A, chi), t = K.torusData(A, chi);
    if (!(same(g.hole, t.hole) && same(g.tube, t.tube) && same(g.cells, t.cells) && same(g.flow, t.flow))) { agree = false; where = 'trial ' + trial; }
    if (!lawsHold(g)) laws = false;
    if (!t.balanced) balanced = false;
  }
  ok(agree, 'the kernel\'s hole, tube and cell fluxes and its edge flows are the gate\'s own circulations on 40 potentials', where);
  ok(laws, 'from one loop to the next the flux steps by exactly the circulation of the cells between (hole_flux_step_y/z, tube_flux_step_x/w)');
  ok(balanced, 'the kernel decides all eight strip balances on each of them (cube_strip)');
  /* and on a run: the damper preset, every step */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ONE };
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  let eng = K.initEngine(), runAgree = true, runLaws = true, nonzero = false;
  for (let s = 0; s < 40; s++) {
    eng = K.step(eng, params, act, 'PARALLEL');
    const g = read(eng.magA, eng.magChi), t = K.magData(eng.magA, eng.magChi).torus;
    if (!(same(g.hole, t.hole) && same(g.tube, t.tube) && same(g.cells, t.cells) && same(g.flow, t.flow) && t.balanced)) runAgree = false;
    if (!lawsHold(g)) runLaws = false;
    if (g.hole.some(x => !Q.isZero(x)) && g.cells.some(r => r.some(x => !Q.isZero(x)))) nonzero = true;
  }
  ok(runAgree && runLaws && nonzero, 'on 40 steps of the damper preset the torus fluxes are the gate\'s, every balance holds, and the loops and cells carry flux');
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
  const eqv = (a, b) => a.length === b.length && a.every((x, i) => Q.eq(x, b[i]));
  let jSame = true, kirch = true, rSame = true, neutral = true, flags = true, where = '';
  for (let trial = 0; trial < 40; trial++) {
    const B = Array.from({ length: 24 }, (_, f) => Q.mk(BigInt(((f + 5) * (trial + 3) * 7907) % 1999 - 999), BigInt(1 + ((f + trial) % 7))));
    const q = Array.from({ length: 16 }, (_, v) => Q.mk(BigInt(((v + 2) * (trial + 11) * 6151) % 401 - 200), BigInt(1 + ((v * trial) % 6))));
    const J = K.currentOf(B);
    if (!eqv(J, walkJ(B))) { jSame = false; where = 'trial ' + trial; }
    /* these fluxes are not the curl of anything: conservation is d1 d0 = 0,
       not a property of the potential */
    if (!inflow(J).every(Q.isZero)) kirch = false;
    const rho = K.laplacianV(q);
    if (!eqv(rho, lap(q))) rSame = false;
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
  /* and on a run: the damper preset's sources are the gate's and not empty */
  const X = A4;
  const params = { dt: X.fromQ(Q.mk(1n, 64n)), dielectricScale: X.fromQ(Q.mk(1n, 8n)), coherence: Q.mk(1n, 4n),
    gammaW: X.fromQ(Q.mk(424923n, 10000n)), appliedH: X.fromQ(Q.mk(1n, 64n)), mstvq: X.fromQ(Q.mk(1n, 128n)),
    tgcr: X.fromQ(Q.mk(1n, 256n)), zpe: X.fromQ(Q.mk(1n, 512n)), zpeSource: X.ZERO,
    geoD: X.fromQ(Q.mk(1n, 16n)), geoM: X.fromQ(Q.mk(1n, 8n)), geoP: X.fromQ(Q.mk(1n, 128n)),
    polarisation: Q.ONE, vortex: Q.ONE };
  const act = [9, 11, 29].map(id => ({ id, strength: Q.ONE }));
  let eng = K.initEngine(), runOk = true, live = 0;
  for (let s = 0; s < 30; s++) {
    eng = K.step(eng, params, act, 'PARALLEL');
    const sd = K.sourcesData(eng.magA, eng.dielectric);
    if (!(sd.kirchhoff && sd.neutral && eqv(sd.current, walkJ(K.curlA(eng.magA))) && eqv(sd.charge, lap(eng.dielectric)) && inflow(sd.current).every(Q.isZero))) runOk = false;
    if (sd.current.some(x => !Q.isZero(x)) && sd.charge.some(x => !Q.isZero(x))) live++;
  }
  ok(runOk && live === 30, `on 30 steps of the damper preset the sources are the gate's, both laws hold, and current and charge are both present (${live}/30 steps)`);
}


console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) { for (const f of failures) console.log('  ' + f); process.exit(1); }
