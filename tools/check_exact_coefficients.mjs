/* Fast, scoped checks of the production kernel's repaired rational arithmetic
   and Hodge coefficients. No runSelfTests(), full simulation step, or nonlinear
   all29 corpus is called here. Required full29 jobs remain separate.

   Hodge convention: oriented Euclidean pairs (01,02,03,12,13,23), with
     SD  = (F01+F23, F02-F13, F03+F12)/2,
     ASD = (F01-F23, F02+F13, F03-F12)/2.
   bvecAt retains both triples and the inherited vertex scale 1/6. Thus the
   six-coordinate squared norm and mixed dot are the raw six-pair quantities
   divided by 72. Expected arithmetic below uses its own bare BigInt fractions.

   Mutants exist only in memory: the former lossy projection, dropped ASD, and
   a reconnection consumer that ignores ASD must all be detected by this gate.

   Run on GitHub: node tools/check_exact_coefficients.mjs
   To additionally require the full formulation's unresolved status to clear:
     node tools/check_exact_coefficients.mjs --require-complete
*/
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const HTML = readFileSync(join(ROOT, 'vedic_v18.51.1_exact_phi.html'), 'utf8');
const cli = process.argv.slice(2);
if (cli.length > 1 || (cli.length === 1 && cli[0] !== '--require-complete')) {
  console.error('Usage: node tools/check_exact_coefficients.mjs [--require-complete]');
  process.exit(2);
}
const requireComplete = cli.length === 1;
let checks = 0;
function check(value, label) {
  if (!value) throw new Error(label);
  checks++;
  console.log('  ok   ' + label);
}
function instantiate(source) {
  const start = source.indexOf('const KERNEL_FACTORY = () => {');
  const end = source.indexOf('\nconst KERNEL = KERNEL_FACTORY();');
  if (start < 0 || end <= start) throw new Error('Cannot locate production KERNEL_FACTORY');
  return new Function(source.slice(start, end) + '\nreturn KERNEL_FACTORY();')();
}

const abs = x => x < 0n ? -x : x;
const gcd = (a, b) => { a = abs(a); b = abs(b); while (b) [a, b] = [b, a % b]; return a; };
function r(n, d = 1n) {
  n = BigInt(n); d = BigInt(d);
  if (!d) throw new RangeError('Independent rational denominator is zero');
  if (d < 0n) { n = -n; d = -d; }
  const g = gcd(n, d);
  return Object.freeze({ n: n / g, d: d / g });
}
const ZERO = r(0n), ONE = r(1n), HALF = r(1n, 2n);
const add = (a, b) => r(a.n * b.d + b.n * a.d, a.d * b.d);
const sub = (a, b) => r(a.n * b.d - b.n * a.d, a.d * b.d);
const mul = (a, b) => r(a.n * b.n, a.d * b.d);
const div = (a, b) => r(a.n * b.d, a.d * b.n);
const equal = (a, b) => a && b && typeof a.n === 'bigint' && typeof a.d === 'bigint' &&
  typeof b.n === 'bigint' && typeof b.d === 'bigint' && a.n * b.d === b.n * a.d;
const compare = (a, b) => { const d = a.n * b.d - b.n * a.d; return d < 0n ? -1 : d > 0n ? 1 : 0; };
const canonical = q => q && typeof q.n === 'bigint' && typeof q.d === 'bigint' && q.d > 0n && gcd(q.n, q.d) === 1n;
const sum = values => values.reduce(add, ZERO);
const dot = (a, b) => sum(a.map((value, i) => mul(value, b[i])));
const same = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => equal(x, b[i]));
const sameIds = (a, b) => Array.isArray(a) && a.length === b.length && a.every((x, i) => x === b[i]);
const encode = value => JSON.stringify(value, (_, x) => typeof x === 'bigint' ? x.toString(16) : x);
const a4Coefficients = x => [x.a, x.b, x.c, x.d];
// The basis index is a bit mask for sqrt(2) and sqrt(5). Shared factors
// contribute 2 or 5 and the remaining factors combine by xor.
function independentA4Product(x, y) {
  const out = Array(4).fill(ZERO);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const overlap = i & j, factor = r(((overlap & 1) ? 2 : 1) * ((overlap & 2) ? 5 : 1));
    out[i ^ j] = add(out[i ^ j], mul(mul(x[i], y[j]), factor));
  }
  return out;
}

// Independent topology: each oriented two-face has one value; a vertex
// touches exactly one face for each of the six ordered axis pairs.
const PAIRS = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const EDGES = [];
for (let a = 0; a < 4; a++) for (let v = 0; v < 16; v++)
  if (!(v & (1 << a))) EDGES.push([v, a, v | (1 << a)]);
const FACES = [];
for (let p = 0; p < PAIRS.length; p++) {
  const [a, b] = PAIRS[p];
  for (let base = 0; base < 16; base++)
    if (!(base & ((1 << a) | (1 << b)))) FACES.push({ p, a, b, base });
}
const localRaw = (B, vertex) => PAIRS.map(([a, b]) =>
  B[FACES.findIndex(f => f.a === a && f.b === b && f.base === (vertex & ~((1 << a) | (1 << b))))]);
const splitScaled = f => [
  add(f[0], f[5]), sub(f[1], f[4]), add(f[2], f[3]),
  sub(f[0], f[5]), add(f[1], f[4]), sub(f[2], f[3])
].map(x => div(x, r(12n)));
const reconstruct = b => [
  add(b[0], b[3]), add(b[1], b[4]), add(b[2], b[5]),
  sub(b[2], b[5]), sub(b[4], b[1]), sub(b[0], b[3])
].map(x => mul(x, r(6n)));
const expectedSD = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, -1, 0], [1, 0, 0]].map(row => row.map(n => r(n, 2n)));
const expectedASD = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [-1, 0, 0]].map(row => row.map(n => r(n, 2n)));
const faceCases = Array.from({ length: 24 }, (_, face) => FACES.map((_, i) => i === face ? ONE : ZERO));
faceCases.push(FACES.map((_, i) => r((i % 7 - 3) * (i + 2), i % 5 + 2)));
faceCases.push(FACES.map((_, i) => r((i % 2 ? -1 : 1) * (i + 3), i % 4 + 1)));
faceCases.push(FACES.map(f => f.p === 0 ? ONE : f.p === 5 ? r(-1n) : ZERO)); // Pure ASD.
faceCases.push(FACES.map(f => f.p === 0 || f.p === 5 ? ONE : ZERO)); // Pure SD.

function hodgeProblems(K) {
  const problems = new Set();
  if (!Array.isArray(K.HODGE_SD) || K.HODGE_SD.length !== 6 ||
      !K.HODGE_SD.every((row, i) => same(row, expectedSD[i]))) problems.add('self-dual coefficient matrix');
  if (!Array.isArray(K.HODGE_ASD) || K.HODGE_ASD.length !== 6 ||
      !K.HODGE_ASD.every((row, i) => same(row, expectedASD[i]))) problems.add('anti-self-dual coefficient matrix');
  const observed = [];
  for (const B of faceCases) {
    const input = Object.freeze(B.slice()), before = encode(input), vectors = K.bvecAt(input);
    if (encode(input) !== before) problems.add('mutated face input');
    observed.push(vectors);
    if (!Array.isArray(vectors) || vectors.length !== 16 ||
        !vectors.every(row => Array.isArray(row) && row.length === 6 && row.every(canonical))) {
      problems.add('not sixteen canonical six-coordinate vectors');
      continue;
    }
    for (let v = 0; v < 16; v++) {
      const raw = localRaw(B, v), actual = vectors[v];
      if (!same(actual, splitScaled(raw))) problems.add('Hodge values or inherited 1/6 scale');
      if (!same(reconstruct(actual), raw)) problems.add('lossless reconstruction');
      if (!equal(dot(actual, actual), div(dot(raw, raw), r(72n)))) problems.add('half norm with 1/6 scale');
    }
  }
  const left = observed[24], right = observed[25];
  if (left?.every(row => Array.isArray(row) && row.length === 6) &&
      right?.every(row => Array.isArray(row) && row.length === 6)) {
    for (let v = 0; v < 16; v++) {
      const want = div(dot(localRaw(faceCases[24], v), localRaw(faceCases[25], v)), r(72n));
      if (!equal(dot(left[v], right[v]), want)) problems.add('mixed dot over both triples');
    }
  } else problems.add('mixed dot lacks six coordinates');
  return [...problems];
}

const edgeIndex = (v, a) => EDGES.findIndex(edge => edge[0] === v && edge[1] === a);
const independentCurl = A => FACES.map(({ a, b, base }) =>
  sub(add(A[edgeIndex(base, a)], A[edgeIndex(base ^ (1 << a), b)]),
      add(A[edgeIndex(base ^ (1 << b), a)], A[edgeIndex(base, b)])));
function independentReconnections(A) {
  const B = independentCurl(A), vectors = Array.from({ length: 16 }, (_, v) => splitScaled(localRaw(B, v)));
  const result = [];
  for (let e = 0; e < EDGES.length; e++) {
    const [v, , h] = EDGES[e], a = vectors[v], b = vectors[h];
    const na = dot(a, a), nb = dot(b, b), ab = dot(a, b);
    if (compare(na, r(1n, 1000000n)) < 0 || compare(nb, r(1n, 1000000n)) < 0) continue;
    if (ab.n < 0n && compare(mul(ab, ab), mul(r(49n, 100n), mul(na, nb))) > 0) result.push(e);
  }
  return result;
}
// On edge (0,0,1), these exact curls are respectively
// F(0)=(1,0,0,0,0,f) and F(1)=(1,0,0,0,0,-f).
// f=2 gives full cosine -3/5 but SD-only cosine -1.
// f=3 gives full cosine -4/5 but a three-coordinate dot/full norm -2/5.
const witnessPotential = f => EDGES.map(([v, a]) => a === 1 ? r(v & 1) :
  a === 3 ? r(f * (1 - 2 * (v & 1)) * ((v >> 2) & 1)) : ZERO);
const potentialCases = Array.from({ length: 32 }, (_, edge) => EDGES.map((_, i) => i === edge ? ONE : ZERO));
potentialCases.push(EDGES.map((_, i) => r((i % 9 - 4) * (i + 1), i % 5 + 1)));
potentialCases.push(EDGES.map((_, i) => r((i % 2 ? -1 : 1) * (i + 3), i % 7 + 1)));
potentialCases.push(witnessPotential(2), witnessPotential(3));
function reconnectionProblems(K) {
  const problems = new Set();
  for (let i = 0; i < potentialCases.length; i++) {
    const A = Object.freeze(potentialCases[i].slice()), before = encode(A);
    if (!same(K.curlA(A), independentCurl(A))) problems.add('oriented curl at fixture ' + i);
    if (!sameIds(K.reconnectionsOf(A), independentReconnections(A))) problems.add('six-coordinate reconnection decision at fixture ' + i);
    if (encode(A) !== before) problems.add('mutated edge potential');
  }
  return [...problems];
}

function replaceBlock(source, startMarker, endMarker, replacement) {
  const start = source.indexOf(startMarker), end = source.indexOf(endMarker, start);
  if (start < 0 || end <= start || source.indexOf(startMarker, start + 1) >= 0)
    throw new Error('Mutation anchor must identify exactly one source block: ' + startMarker);
  return source.slice(0, start) + replacement + source.slice(end);
}
function replaceOnce(source, before, after) {
  const at = source.indexOf(before);
  if (at < 0 || source.indexOf(before, at + before.length) >= 0) throw new Error('Mutation must replace exactly one source expression');
  return source.slice(0, at) + after + source.slice(at + before.length);
}
const oldProjection = `  const bvecAt = B => {
    const legacy = [[0,0,1],[0,-1,0],[0,-1,1],[1,0,0],[1,0,-1],[-1,1,0]]
      .map((r,p) => r.map(c => Q.mk(BigInt(c), p===2||p===4||p===5 ? 2n : 1n)));
    return Array.from({length:16}, (_,v) => [
      ...[0,1,2].map(k => Q.mul(FACES_AT[v].reduce((s,i) => Q.add(s,Q.mul(B[i],legacy[MAG_FACES[i].p][k])),Q.ZERO),Q.mk(1n,6n))),
      Q.ZERO,Q.ZERO,Q.ZERO]);
  };
`;
const droppedASD = `  const bvecAt = B => Array.from({length:16}, (_,v) => [
    ...[0,1,2].map(k => Q.mul(FACES_AT[v].reduce((s,i) => Q.add(s,Q.mul(B[i],HODGE_SD[MAG_FACES[i].p][k])),Q.ZERO),Q.mk(1n,6n))),
    Q.ZERO,Q.ZERO,Q.ZERO]);
`;

try {
  const K = instantiate(HTML);
  check(encode(K.MAG_EDGES) === encode(EDGES) && K.MAG_FACES.length === FACES.length &&
    K.MAG_FACES.every((f, i) => ['p', 'a', 'b', 'base'].every(key => f[key] === FACES[i][key])),
    'oriented faces and edges match an independently constructed four-cube');
  const hodgeErrors = hodgeProblems(K);
  check(hodgeErrors.length === 0, 'every face basis and mixed SD/ASD field reconstructs exactly, with norm and dot scaled by 1/72' +
    (hodgeErrors.length ? ': ' + hodgeErrors.join('; ') : ''));
  check(!independentReconnections(witnessPotential(2)).includes(0) && independentReconnections(witnessPotential(3)).includes(0),
    'analytic reconnection witnesses distinguish a lost ASD norm and a lost ASD dot');
  const reconnectionErrors = reconnectionProblems(K);
  check(reconnectionErrors.length === 0, 'actual reconnection consumer agrees with exact six-coordinate decisions on every edge basis and mixed fixture' +
    (reconnectionErrors.length ? ': ' + reconnectionErrors.join('; ') : ''));

  // Restore the former projection in memory, padded with zeros to make its
  // interface six-dimensional: rejection must concern values, not dimensions.
  for (const [label, replacement] of [['old lossy projection', oldProjection], ['dropped anti-self-dual triple', droppedASD]]) {
    const mutant = instantiate(replaceBlock(HTML, '  const bvecAt =', '  const RECONNECT_FLOOR =', replacement));
    const errors = hodgeProblems(mutant);
    check(errors.includes('lossless reconstruction') && errors.includes('half norm with 1/6 scale'),
      'in-memory mutant rejected: ' + label);
  }
  const dotText = 'const dot = a.reduce((s, c, k) => Q.add(s, Q.mul(c, b[k])), Q.ZERO);';
  const threeDot = replaceOnce(HTML, dotText, 'const dot = a.slice(0,3).reduce((s, c, k) => Q.add(s, Q.mul(c, b[k])), Q.ZERO);');
  check(reconnectionProblems(instantiate(threeDot)).length > 0,
    'in-memory mutant rejected: reconnection dot ignores ASD');
  const threeNormAndDot = replaceOnce(threeDot,
    'n2 = u => u.reduce((s, c) => Q.add(s, Q.mul(c, c)), Q.ZERO)',
    'n2 = u => u.slice(0,3).reduce((s, c) => Q.add(s, Q.mul(c, c)), Q.ZERO)');
  check(reconnectionProblems(instantiate(threeNormAndDot)).length > 0,
    'in-memory mutant rejected: reconnection norm and dot both ignore ASD');

  const storageInputs = [ZERO, r(1n, 3n), r(-7n, 11n), r((1n << 700n) + 1n, 17n),
    r(1n, (1n << 521n) + 3n), { n: -42n, d: 70n }];
  check(K.STORE_BITS === null && K.ARITHMETIC === 'exact-rational' && storageInputs.every(input => {
    const out = K.storeQ(input);
    return canonical(out) && equal(input, out) && equal(K.storeQ(out), input);
  }), 'storage preserves every rational bit, including odd denominators, and canonicalizes without a precision cap');
  check(equal(K.storeBudget(storageInputs.map(x => r(x.n, x.d))), ZERO), 'storage contributes exactly zero quantization error');

  const domainInput = Object.freeze(Array.from({ length: 16 }, (_, i) => i === 12 ? ZERO : r(i + 1, 7n)));
  const domainBefore = encode(domainInput);
  let refusal = null;
  try { K.applySutra(13, domainInput, HALF); } catch (error) { refusal = error; }
  check(refusal instanceof RangeError && /S13/.test(refusal.message) && /12/.test(refusal.message) &&
    encode(domainInput) === domainBefore, 'S13 refuses its named zero denominator without mutating the supplied state');

  const normalization = Array.from({ length: 29 }, (_, i) => BigInt(i + 1)).reduce((a, b) => a + b, 0n);
  check(normalization === 435n && K.SUTRA_SUM === normalization, 'the sutra normalization is derived from 1 + ... + 29 = 435');
  let alphaExact = typeof K.alphaOf === 'function';
  if (alphaExact) for (let id = 1; id <= 29; id++) for (const strength of [ZERO, HALF, r(3n, 4n), ONE])
    for (const gain of [HALF, ONE]) {
      const expected = mul(mul(r(id, normalization), strength), gain);
      if (!equal(K.alphaOf(id, [strength], gain), expected)) alphaExact = false;
    }
  check(alphaExact, 'the actual alpha function uses id/435 times strength times gain for every sutra');
  const p29 = Object.freeze(Array.from({ length: 16 }, (_, i) => r((i - 7) * (i + 2), i % 5 + 2)));
  const p29Before = encode(p29), record = [], mean = div(sum(p29), r(16n));
  const output = K.applySutra(29, p29, HALF, { record });
  const a29 = mul(mul(r(29n, normalization), HALF), r(1n, 5n));
  const expected29 = p29.map(value => add(value, mul(sub(mean, value), a29)));
  check(equal(K.SUTRA_ARG_SPEC[29][0].def, r(1n, 5n)) && same(output, expected29) && record.length === 1 &&
    equal(record[0].alpha, r(1n, 30n)) && equal(record[0].info.alpha29, r(1n, 150n)) &&
    same(record[0].info.p, output) && encode(p29) === p29Before,
    'S29 actual output consumes the independent literal 1/5 coefficient, giving alpha29 = 1/150 at 50%');

  const valid = Object.freeze(Array.from({ length: 29 }, (_, i) => Object.freeze({ id: i + 1, strength: HALF })));
  check(K.requireComplete29(valid) === valid, 'the complete-set input guard accepts all 29 positive strengths');
  const rejected = [
    ['missing sutra', valid.slice(0, 28)],
    ['duplicate sutra', valid.map((s, i) => i === 28 ? { id: 1, strength: HALF } : s)],
    ['zero strength', valid.map(s => s.id === 13 ? { ...s, strength: ZERO } : s)],
    ['zero gain', valid.map(s => s.id === 13 ? { ...s, gain: ZERO } : s)],
    ['conflicting strength operand', valid.map(s => s.id === 29 ? { ...s, args: [r(1n, 4n), r(1n, 5n)] } : s)]
  ];
  for (const [label, active] of rejected) {
    let error = null;
    try { K.requireComplete29(active); } catch (caught) { error = caught; }
    check(error instanceof RangeError, 'complete-set input guard rejects ' + label);
  }

  const status = K.FORMULATION_STATUS;
  check(status && typeof status.complete === 'boolean' && Array.isArray(status.blockers) &&
    status.blockers.every(x => typeof x === 'string' && x.length > 0) &&
    (status.complete ? status.blockers.length === 0 : status.blockers.length > 0),
    'formulation status names unresolved blockers and cannot advertise completion while they remain');

  const algebraicPhi = a4Coefficients(K.A4.PHI), algebraicSquare = independentA4Product(algebraicPhi, algebraicPhi);
  const algebraicResidual = algebraicSquare.map((value, i) => sub(sub(value, algebraicPhi[i]), i === 0 ? ONE : ZERO));
  check(same(algebraicPhi, [HALF, ZERO, HALF, ZERO]) && algebraicResidual.every(value => equal(value, ZERO)) &&
    same(a4Coefficients(K.A4.mul(K.A4.PHI, K.A4.PHI)), algebraicSquare),
    'the separate A4.PHI is (1 + sqrt(5))/2 and satisfies its polynomial in independent exact algebra');
  const literalPhi = r(12586269025n, 7778742049n), literalInverseOperand = sub(literalPhi, ONE);
  check(equal(K.PHI_Q, literalPhi) && equal(K.PHI_INV_Q, literalInverseOperand),
    'the active rational phi and inverse-named operands retain the declared literal input values');
  const phiResidual = sub(sub(mul(K.PHI_Q, K.PHI_Q), K.PHI_Q), ONE);
  const inverseResidual = sub(mul(K.PHI_Q, K.PHI_INV_Q), ONE);
  const cassiniResidual = r(-1n, 7778742049n * 7778742049n);
  check(equal(phiResidual, cassiniResidual) && equal(inverseResidual, cassiniResidual) &&
    equal(status.literalPhi, literalPhi) && equal(status.literalPhiPolynomialResidual, phiResidual) &&
    equal(status.literalPhiReciprocalResidual, inverseResidual),
    'reported literal phi and reciprocal defects both equal the independently computed nonzero value -1/7778742049^2');

  const measuredBlockers = [];
  if (!equal(phiResidual, ZERO)) measuredBlockers.push('The active rational phi has phi^2 - phi - 1 = ' + phiResidual.n + '/' + phiResidual.d + ', not zero.');
  if (!equal(inverseResidual, ZERO)) measuredBlockers.push('The active inverse-named operand has phi * inverse - 1 = ' + inverseResidual.n + '/' + inverseResidual.d + ', not zero.');
  for (const [name, interval] of [['MAYA_A', K.MAYA_A], ['MAYA_B', K.MAYA_B]]) {
    check(Array.isArray(interval) && interval.length === 2 && interval.every(canonical) && compare(interval[0], interval[1]) <= 0,
      name + ' is an ordered exact rational enclosure');
    const width = sub(interval[1], interval[0]);
    if (!equal(width, ZERO)) measuredBlockers.push(name + ' has nonzero enclosure width ' + width.n + '/' + width.d + '; this coefficient is enclosed, not evaluated exactly.');
  }
  check(!status.complete || measuredBlockers.length === 0,
    'a complete status must also satisfy independently measured phi, inverse and Maya obligations');
  console.log('\n' + checks + ' fast arithmetic/coefficient checks passed. Full29 completion and full-model fidelity are separate required claims.');
  if (!status.complete || measuredBlockers.length) {
    console.log('FORMULATION INCOMPLETE:');
    for (const blocker of measuredBlockers) console.log('  - Measured: ' + blocker);
    for (const blocker of status.blockers) console.log('  - ' + blocker);
    if (requireComplete) process.exitCode = 1;
  }
} catch (error) {
  console.error('FAIL exact coefficients: ' + (error.stack || error));
  process.exitCode = 1;
}
