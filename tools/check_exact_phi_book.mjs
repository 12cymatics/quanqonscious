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
if (start < 0 || end < start) { console.error('FAIL -- KERNEL_FACTORY could not be located in the page'); process.exit(1); }
const K = new vm.Script(HTML.slice(start, end) + '\nKERNEL_FACTORY();',
  { filename: PAGE, lineOffset: HTML.slice(0, start).split('\n').length - 1 }).runInThisContext();
const { Q, BOOK } = K;

let pass = 0;
const failures = [];
const ok = (cond, label, detail) => {
  if (cond) { pass++; console.log('  ok   ' + label); }
  else { failures.push(label); console.log('  FAIL ' + label + (detail ? '  -- ' + detail : '')); }
};
const S = x => { if (!Q.isExact(x)) throw new Error('an interval where an exact value is compared'); return `${x.n}/${x.d}`; };
const enc = v => Array.isArray(v) ? v.map(enc) : S(v);
const f = (n, d = 1) => Q.mk(BigInt(n), BigInt(d)), l = xs => xs.map(v => Array.isArray(v) ? f(v[0], v[1]) : f(v));
const reference = cases => JSON.parse(execFileSync('python3', [join(ROOT, 'tools', 'vedic_book_reference.py')],
  { input: JSON.stringify({ cases }), maxBuffer: 1 << 28 }).toString()).results;
const mine = (id, operands) => {
  try { return { terms: BOOK.terms(id, operands).map(S) }; }
  catch (e) { return { refused: e instanceof RangeError ? e.message : 'not a RangeError: ' + e.message }; }
};

const VARIANTS = {
  1: [{ denominator: f(29) }, { denominator: f(139) }, { denominator: f(9) }, { denominator: f(21) }, { denominator: f(19, 2) }],
  2: [{ a: f(88), b: f(113), base: f(100) }, { a: f(-7, 3), b: f(999), base: f(1000) }, { a: f(1), b: f(1), base: f(0) }],
  3: [{ a: l([1, -2, 3, 4]), b: l([5, 0, -1]) }, { a: l([[1, 2]]), b: l([[3, 7], 2]) }, { a: [], b: l([1]) }],
  4: [{ dividend: l([-6, 11, -6, 1]), divisor: l([-1, 1]) }, { dividend: l([3, 0, 2, 5, 7]), divisor: l([1, 2, 3]) }, { dividend: l([1]), divisor: l([1, 1]) },
      { dividend: l([1, 2]), divisor: l([1, 0]) }],
  5: [{ a: f(2), b: f(6), c: f(3), d: f(4) }, { a: f(1), b: f(12), c: f(3), d: f(4) }, { a: f(1), b: f(2), c: f(3), d: f(4) }],
  6: [{ a: f(3), b: f(4), c: f(6), d: f(5), e: f(9), f: f(10) }, { a: f(0), b: f(4), c: f(8), d: f(3), e: f(2), f: f(4) },
      { a: f(1), b: f(2), c: f(3), d: f(2), e: f(4), f: f(5) }, { a: f(1), b: f(2), c: f(7), d: f(3), e: f(5), f: f(1) }],
  7: [{ a: f(7), b: f(3), c: f(17), d: f(13) }, { a: f(1), b: f(1), c: f(2), d: f(3) }, { a: f(2, 3), b: f(-5), c: f(1), d: f(0) }],
  8: [{ a: f(2), b: f(7), c: f(-4) }, { a: f(0), b: f(1), c: f(1) }, { a: f(-3, 5), b: f(1, 7), c: f(2) }],
  9: [{ p: l([1, 0, -3, 0, 5]) }, { p: l([4]) }, { p: l([[1, 2], [1, 3], [1, 4]]) }],
  10: [{ n: f(1012), base: f(1000) }, { n: f(7, 2), base: f(10) }, { n: f(5), base: f(-10) }],
  11: [{ a: f(3), b: f(1), total: f(82) }, { a: f(-2, 3), b: f(5), total: f(1, 9) }],
  12: [{ denominator: f(13) }, { denominator: f(19) }, { denominator: f(49) }, { denominator: f(15) }, { denominator: f(1) }],
  13: [{ first: l([3, 2]), difference: l([1, 1]) }, { first: l([1, 0]), difference: l([1, 0]) }, { first: l([0, 1]), difference: l([0, -1]) },
       { first: l([1, 2, 3]), difference: l([1, 0]) }, { first: l([5, 1]), difference: l([0, 0]) }],
  14: [{ n: f(7), places: f(1) }, { n: f(99999), places: f(5) }, { n: f(4), places: f(0) }, { n: f(5, 2), places: f(2) }],
  15: [{ factors: [l([1, 2, 3]), l([4, 5]), l([-6, 7])] }, { factors: [l([[1, 2], 3])] }, { factors: [] }],
  16: [{ factors: [l([2, 1]), l([5, 1])] }, { factors: [l([1, 2, 3]), l([4, 0, 1]), l([7])] }, { factors: [l([3])] }],
  17: [{ a: f(46), b: f(44), reference_base: f(100), ratio: f(1, 2) }, { a: f(23), b: f(21), reference_base: f(10), ratio: f(2) },
       { a: f(1), b: f(1), reference_base: f(10), ratio: f(0) }],
  18: [{ dividend: f(121), divisors: l([2, 3, 4, 5, 6]) }, { dividend: f(-7), divisors: l([4, 8]) }, { dividend: f(10), divisors: l([3, 4]) },
       { dividend: f(10), divisors: l([0]) }],
  19: [{ quadratic: l([6, 13, 6]), factor: l([3, 2]) }, { quadratic: l([2, 5, 2]), factor: l([1, 1]) }, { quadratic: l([2, 5, 0]), factor: l([2, 1]) }],
  20: [1, 2, 3, 4, 5, 6, 7, 0].map(n => ({ numerator: f(n) })),
  21: [{ n: f(2774), divisor: f(19) }, { n: f(5293240096), divisor: f(139) }, { n: f(100), divisor: f(7) }, { n: f(0), divisor: f(3) },
       { n: f(10), divisor: f(5) }, { n: f(91), divisor: f(13) }],
  22: [{ n: f(104), base: f(100) }, { n: f(7, 3), base: f(10) }, { n: f(1), base: f(0) }],
  23: [{ n: f(96), base: f(100) }, { n: f(1, 2), base: f(1) }],
  24: [{ a: f(116), b: f(114) }, { a: f(999), b: f(991) }, { a: f(1996), b: f(1904) }, { a: f(43), b: f(46) }, { a: f(-3), b: f(7) }, { a: f(5), b: f(5) }],
  25: [{ a: l([2, 1]), b: l([5, 1]), common: l([1, 1]), numerator_last: f(3), denominator_last: f(7) },
       { a: l([1, 1]), b: l([2, 2]), common: l([4]), numerator_last: f(1), denominator_last: f(2) },
       { a: l([1, 1]), b: l([0, 0]), common: l([1]), numerator_last: f(1), denominator_last: f(2) },
       { a: l([1, 1]), b: l([1, 0]), common: l([1]), numerator_last: f(1), denominator_last: f(-1) }],
  26: [{ a: l([1, 2, 3]), b: l([-4, 5]) }],
  27: [{ a: f(3), b: f(2), c: f(-1), xy: f(7), yz: f(-1), zx: f(2) }, { a: f(1), b: f(1), c: f(1), xy: f(1), yz: f(1), zx: f(1) },
       { a: f(0), b: f(1), c: f(1), xy: f(1), yz: f(1), zx: f(1) }, { a: f(1), b: f(6), c: f(2), xy: f(5), yz: f(7), zx: f(3) }],
  28: [{ total: f(10, 3), split: f(3) }, { total: f(5, 6), split: f(3, 2), sign: f(-1) }, { total: f(26, 5), split: f(5), numerator: l([1, 2]), denominator: l([3, -1]) },
       { total: f(3), split: f(2) }, { total: f(2), split: f(1), numerator: l([0, 0]) }, { total: f(0), split: f(0) }],
  29: [{ factors: [l([1, 1]), l([1, 1]), l([1, 1]), l([1, 1])] }]
};

console.log('\nthe 29 methods of the book, against vedic_book.py');
{
  const cases = [], got = [], examples = [];
  for (let id = 1; id <= 29; id++) {
    examples.push(cases.length);
    cases.push({ id, operands: null }); got.push(mine(id));
    cases.push({ id, operands: Object.fromEntries(Object.entries(BOOK.EXAMPLES[id]).map(([k, v]) => [k, enc(v)])) }); got.push(mine(id, BOOK.EXAMPLES[id]));
    for (const v of VARIANTS[id]) { cases.push({ id, operands: Object.fromEntries(Object.entries(v).map(([k, x]) => [k, enc(x)])) }); got.push(mine(id, v)); }
  }
  const want = reference(cases);
  let same = 0, refusedBoth = 0;
  const bad = [];
  want.forEach((w, i) => {
    const g = got[i];
    if (w.refused || g.refused) {
      if (w.refused && g.refused && !g.refused.startsWith('not a')) refusedBoth++;
      else bad.push(`S${cases[i].id} case ${i}: reference ${w.refused ? 'refuses' : 'computes'}, kernel ${g.refused || 'computes'}`);
    } else if (JSON.stringify(w.terms) === JSON.stringify(g.terms)) same++;
    else bad.push(`S${cases[i].id} case ${i}: ${g.terms.join(' ')} is not ${w.terms.join(' ')}`);
  });
  ok(bad.length === 0 && same + refusedBoth === cases.length && refusedBoth > 20,
     `${cases.length} applications of the 29 methods: ${same} give the reference's terms exactly, ${refusedBoth} are refused by both`, bad.slice(0, 5).join('; '));
  const perId = Array.from({ length: 29 }, (_, i) => cases.filter((c, j) => c.id === i + 1 && !want[j].refused).length);
  ok(perId.every(n => n >= 3), `every one of the 29 is computed, not refused, on at least three operand sets (fewest ${Math.min(...perId)})`);
  ok(examples.every(i => got[i].terms && JSON.stringify(got[i]) === JSON.stringify(got[i + 1]) && JSON.stringify(want[i]) === JSON.stringify(want[i + 1])),
     "the kernel's example operands are the book's worked examples the reference carries");
}

console.log('\nthe book methods on the vertex field');
{
  const edges = new Set(BOOK.MATCHINGS.flat().map(([u, v]) => u * 16 + v));
  ok(BOOK.MATCHINGS.every(m => new Set(m.flat()).size === 2 * m.length) && BOOK.MATCHINGS.flat().length === 64 && edges.size === 32,
     `the tesseract's 32 edges, each met from both ends, fall into ${BOOK.MATCHINGS.length} matchings in which no two edges share a vertex`);
  const states = [K.initState(), Array.from({ length: 16 }, (_, v) => f(v * v * 37 - 400, 3 + 2 * (v % 5))), Array.from({ length: 16 }, (_, v) => f(((v * 7919) % 199) - 99, 101))];
  const strengths = [f(1, 1000), f(1, 2), f(-3, 7)];
  const cases = [], got = [];
  K.setExactBits(null);
  try {
    for (let id = 1; id <= 29; id++) states.forEach((psi, si) => {
      const strength = strengths[(id + si) % 3];
      cases.push({ id, operands: null, psi: psi.map(S), strength: S(strength) });
      got.push(BOOK.apply(id, psi, strength).map(S));
    });
  } finally { K.setExactBits(4096); }
  const want = reference(cases);
  const bad = cases.filter((c, i) => JSON.stringify(want[i].psi) !== JSON.stringify(got[i])).map(c => 'S' + c.id);
  ok(bad.length === 0, `${cases.length} fields (29 methods x 3 states, three strengths) rotated on the matched edges by strength x each term equal vedic_field.py's exactly`, [...new Set(bad)].join(' '));
  const energy = p => p.reduce((s, x) => Q.add(s, Q.mul(x, x)), Q.ZERO);
  K.setExactBits(null);
  let kept = true, moved = 0;
  try {
    for (let id = 1; id <= 29; id++) {
      const out = BOOK.apply(id, states[0], f(1, 8));
      if (!Q.eq(energy(out), energy(states[0]))) kept = false;
      if (out.some((x, i) => !Q.eq(x, states[0][i]))) moved++;
    }
  } finally { K.setExactBits(4096); }
  ok(kept && moved === 29, `each of the 29 keeps the field's energy exactly and moves the field (${moved} of 29 move it)`);
  ok(states[0].every((x, i) => BOOK.apply(7, states[0], Q.ZERO)[i] === x || Q.eq(BOOK.apply(7, states[0], Q.ZERO)[i], x)), 'at strength zero the field is unchanged');
}

console.log(`\n${pass} passed, ${failures.length} failed`);
if (failures.length) process.exit(1);
