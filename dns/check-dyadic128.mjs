import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const D = require('./dyadic128.js');

let passed = 0, failed = 0;
function ok(cond, name, detail = ''){
  if (cond){ passed++; console.log(`  ok   ${name}`); }
  else { failed++; console.log(`  FAIL ${name}${detail ? '\n       ' + detail : ''}`); }
}

const gcd = (a, b) => { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) [a, b] = [b, a % b]; return a; };
const rat = (n, d) => { if (d < 0n){ n = -n; d = -d; } const g = gcd(n, d) || 1n; return { n: n/g, d: d/g }; };
const ofDyadic = x => x.e >= 0 ? rat(x.m << BigInt(x.e), 1n) : rat(x.m, 1n << BigInt(-x.e));
const rsub = (a, b) => rat(a.n*b.d - b.n*a.d, a.d*b.d);
const rabs = a => ({ n: a.n < 0n ? -a.n : a.n, d: a.d });
const rcmp = (a, b) => { const z = a.n*b.d - b.n*a.d; return z < 0n ? -1 : z > 0n ? 1 : 0; };
const halfUlp = x => x.e - 1 >= 0 ? rat(1n << BigInt(x.e - 1), 1n) : rat(1n, 1n << BigInt(1 - x.e));
const within = (x, exact, ulps) => rcmp(rabs(rsub(ofDyadic(x), exact)), rat(BigInt(2*ulps), 1n).n === 0n ? exact : { n: halfUlp(x).n*BigInt(2*ulps), d: halfUlp(x).d }) <= 0;
const nearest = (x, exact) => rcmp(rabs(rsub(ofDyadic(x), exact)), halfUlp(x)) <= 0;
const normal = x => x.m === 0n || (D.bitLength(x.m) === D.BITS);

let seed = 0x9e3779b9;
const rnd = () => { seed ^= seed << 13; seed >>>= 0; seed ^= seed >>> 17; seed ^= seed << 5; seed >>>= 0; return BigInt(seed); };
const big = words => { let x = 0n; for (let i = 0; i < words; i++) x = (x << 32n) | rnd(); return x; };
const randRat = () => { const n = big(1 + Number(rnd() % 6n)) * (rnd() & 1n ? -1n : 1n); const d = big(1 + Number(rnd() % 5n)) + 1n; return rat(n, d); };

console.log('1. every stored value is the nearest 128-bit dyadic to its exact rational');
{
  let bad = 0, abnormal = 0;
  for (let t = 0; t < 2000; t++){
    const q = randRat(), x = D.fromRational(q.n, q.d);
    if (!nearest(x, q)) bad++;
    if (!normal(x)) abnormal++;
  }
  ok(bad === 0, `2000 random rationals rounded to nearest (${bad} not)`);
  ok(abnormal === 0, `every mantissa is exactly ${D.BITS} bits (${abnormal} not)`);
}

console.log('2. ties round to even');
{
  const B = BigInt(D.BITS);
  const evenBase = (1n << (B - 1n)) + 2n, oddBase = (1n << (B - 1n)) + 3n;
  const tieUp = D.store((oddBase << 1n) + 1n, -1, 1n), tieDown = D.store((evenBase << 1n) + 1n, -1, 1n);
  ok(tieUp.m === oddBase + 1n, 'an odd mantissa at a tie rounds up to even');
  ok(tieDown.m === evenBase, 'an even mantissa at a tie stays');
}

console.log('3. add, sub, mul, div and rational scaling are correctly rounded');
{
  let bad = 0;
  for (let t = 0; t < 1500; t++){
    const a = randRat(), b = randRat(), x = D.fromRational(a.n, a.d), y = D.fromRational(b.n, b.d);
    const xr = ofDyadic(x), yr = ofDyadic(y);
    const sum = rat(xr.n*yr.d + yr.n*xr.d, xr.d*yr.d), dif = rat(xr.n*yr.d - yr.n*xr.d, xr.d*yr.d);
    const pro = rat(xr.n*yr.n, xr.d*yr.d), quo = rat(xr.n*yr.d, xr.d*yr.n);
    const p = (rnd() % 1000n) - 500n || 7n, q = (rnd() % 997n) + 1n;
    const sca = rat(xr.n*p, xr.d*q);
    const pairs = [[D.add(x, y), sum], [D.sub(x, y), dif], [D.mul(x, y), pro], [D.div(x, y), quo], [D.mulRat(x, p, q), sca]];
    for (const [got, ex] of pairs) if (ex.n !== 0n && (!nearest(got, ex) || !normal(got))) bad++;
  }
  ok(bad === 0, `7500 operations on random operands, every one the nearest 128-bit value (${bad} not)`);
}

console.log('4. square root is correctly rounded');
{
  let bad = 0;
  for (let t = 0; t < 800; t++){
    const a = randRat(), x = D.fromRational(a.n < 0n ? -a.n : a.n, a.d), r = D.sqrt(x), xr = ofDyadic(x), rr = ofDyadic(r), h = halfUlp(r);
    const lo = rsub(rr, h), hi = rat(rr.n*h.d + h.n*rr.d, rr.d*h.d);
    const sq = v => rat(v.n*v.n, v.d*v.d);
    if (rcmp(sq(lo), xr) > 0 || rcmp(sq(hi), xr) < 0 || !normal(r)) bad++;
  }
  ok(bad === 0, `800 random square roots bracketed by half an ulp either side (${bad} not)`);
  let threw = false; try { D.sqrt(D.fromBigInt(-4n)); } catch (e){ threw = /negative/.test(e.message); }
  ok(threw, 'the square root of a negative number is refused');
}

console.log('5. pi, ln 2 and e agree with their published digits to the last bit');
{
  const PI = rat(31415926535897932384626433832795028841971693993751058209749445923n, 10n**64n);
  const LN2 = rat(6931471805599453094172321214581765680755001343602552541206800094933n, 10n**67n);
  const E = rat(27182818284590452353602874713526624977572470936999595749669676277n, 10n**64n);
  ok(within(D.pi(), PI, 1), 'pi within one ulp of 64 published digits');
  ok(within(D.ln2(), LN2, 1), 'ln 2 within one ulp of 67 published digits');
  ok(within(D.exp(D.ONE), E, 2), 'exp(1) within two ulps of 64 published digits of e');
}

console.log('6. sine, cosine and exp satisfy their identities to a few ulps');
{
  const one = rat(1n, 1n);
  let worstTrig = 0, worstExp = 0;
  for (const [n, d] of [[1n, 7n], [-3n, 2n], [355n, 113n], [22n, 7n], [1000n, 3n], [-77n, 5n], [1n, 1000000n]]){
    const x = D.fromRational(n, d), s = D.sin(x), c = D.cos(x);
    const sum = D.add(D.mul(s, s), D.mul(c, c));
    const err = rabs(rsub(ofDyadic(sum), one));
    if (rcmp(err, rat(8n, 1n << 128n)) > 0) worstTrig++;
    const pe = D.mul(D.exp(x), D.exp(D.neg(x)));
    const relE = rabs(rsub(ofDyadic(pe), one));
    if (rcmp(relE, rat(8n, 1n << 128n)) > 0) worstExp++;
  }
  ok(worstTrig === 0, 'sin^2 + cos^2 = 1 within 8 units of 2^-128 at seven arguments, including 1000/3 and 355/113');
  ok(worstExp === 0, 'exp(x) exp(-x) = 1 within 8 units of 2^-128 at the same arguments');
  const half = rat(1n, 2n);
  const sixth = D.mulRat(D.pi(), 1n, 6n), third = D.mulRat(D.pi(), 1n, 3n);
  ok(within(D.sin(sixth), half, 4), 'sin(pi/6) = 1/2 within four ulps');
  ok(within(D.cos(third), half, 4), 'cos(pi/3) = 1/2 within four ulps');
  ok(within(D.exp(D.ln2()), rat(2n, 1n), 4), 'exp(ln 2) = 2 within four ulps');
  const signs = [[D.sin(D.fromRational(355n, 113n)), -1, 'sin(355/113) < 0, since 355/113 exceeds pi by 2.7e-7'], [D.sin(D.fromRational(-3n, 2n)), -1, 'sin(-3/2) < 0'],
                 [D.cos(D.fromRational(22n, 7n)), -1, 'cos(22/7) < 0'], [D.cos(D.fromRational(1000n, 3n)), 1, 'cos(1000/3) > 0'],
                 [D.sin(D.fromRational(9n, 2n)), -1, 'sin(9/2) < 0'], [D.cos(D.fromRational(-5n, 2n)), -1, 'cos(-5/2) < 0']];
  for (const [v, want, name] of signs) ok(D.sign(v) === want, `${name} (quadrant signs)`);
}

console.log('7. the representation never grows');
{
  let x = D.fromRational(1n, 3n), worst = 0;
  for (let k = 0; k < 5000; k++){ x = D.add(D.mulRat(x, 7n, 23n), D.fromRational(BigInt(k % 11), 13n)); worst = Math.max(worst, D.bitLength(x.m)); }
  ok(worst === D.BITS, `after 5000 rational updates the mantissa is still ${D.BITS} bits (largest ${worst})`);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
