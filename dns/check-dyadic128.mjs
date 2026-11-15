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

console.log('7. expm1 and tanh against published digits and their series, with no cancellation near zero');
{
  const E = rat(27182818284590452353602874713526624977572470936999595749669676277n, 10n**64n);
  const E2 = rat(E.n*E.n, E.d*E.d);
  ok(within(D.expm1(D.ONE), rsub(E, rat(1n, 1n)), 2), 'expm1(1) = e - 1 within two ulps');
  ok(within(D.tanh(D.ONE), rat(E2.n - E2.d, E2.n + E2.d), 4), 'tanh(1) = (e^2 - 1)/(e^2 + 1) within four ulps');
  ok(within(D.tanh(D.fromRational(1n, 2n)), rat(E.n - E.d, E.n + E.d), 4), 'tanh(1/2) = (e - 1)/(e + 1) within four ulps');
  const x = D.fromRational(1n, 1n << 40n), xr = ofDyadic(x);
  const pow = k => rat(xr.n**BigInt(k), xr.d**BigInt(k));
  const series = (terms) => terms.reduce((acc, [c, d, k]) => { const t = pow(k); return rat(acc.n*t.d*d + c*t.n*acc.d, acc.d*t.d*d); }, rat(0n, 1n));
  ok(within(D.expm1(x), series([[1n, 1n, 1], [1n, 2n, 2], [1n, 6n, 3], [1n, 24n, 4], [1n, 120n, 5]]), 1), 'expm1(2^-40) matches its series to one ulp, where exp(x) - 1 would lose 40 bits');
  ok(within(D.tanh(x), series([[1n, 1n, 1], [-1n, 3n, 3], [2n, 15n, 5]]), 1), 'tanh(2^-40) matches x - x^3/3 + 2x^5/15 to one ulp');
  const y = D.fromRational(-7n, 5n);
  ok(D.eq(D.tanh(y), D.neg(D.tanh(D.neg(y)))), 'tanh is exactly odd');
}

console.log('8. the natural logarithm inverts exp, adds, and keeps its precision next to 1');
{
  ok(D.eq(D.ln(D.fromBigInt(2n)), D.ln2()), 'ln 2 is the series value of ln 2 itself');
  ok(D.isZero(D.ln(D.ONE)), 'ln 1 is exactly zero');
  let bad = 0;
  const ulpOf = v => v.e >= 0 ? rat(1n << BigInt(v.e), 1n) : rat(1n, 1n << BigInt(-v.e));
  for (const [n, d] of [[10n, 1n], [1n, 3n], [355n, 113n], [7n, 1000n], [123456789n, 1n], [2n, 3n], [3n, 2n]]){
    const y = D.fromRational(n, d), l = D.ln(y);
    const cond = BigInt(4 + Math.ceil(Math.abs(D.toNumberForDisplay(l))));
    const u = ulpOf(y), errY = rabs(rsub(ofDyadic(D.exp(l)), ofDyadic(y)));
    if (rcmp(errY, rat(u.n*cond, u.d)) > 0) bad++;
    const x = D.fromRational(n, 7n*d), ux = ulpOf(x), errX = rabs(rsub(ofDyadic(D.ln(D.exp(x))), ofDyadic(x)));
    const boundX = rat(4n*ux.n*(1n << 125n) + ux.d, ux.d*(1n << 125n));
    if (rcmp(errX, boundX) > 0) bad++;
  }
  ok(bad === 0, `exp(ln y) = y within 4 + |ln y| ulps, the conditioning of exp, and ln(exp x) = x within 4 ulps of x plus 2^-125, `
    + `the rounding of exp(x) near 1 carried through ln, at seven arguments each (${bad} not)`);
  const a = D.fromRational(17n, 5n), b = D.fromRational(9n, 11n);
  ok(within(D.add(D.ln(a), D.ln(b)), ofDyadic(D.ln(D.mul(a, b))), 8), 'ln a + ln b = ln ab within eight ulps');
  const t = D.add(D.ONE, D.fromRational(1n, 1n << 50n)), x = rat(1n, 1n << 50n);
  const series = rat(x.n*x.d*x.d*6n - 3n*x.n*x.n*x.d + 2n*x.n*x.n*x.n, 6n*x.d*x.d*x.d);
  ok(within(D.ln(t), series, 1), 'ln(1 + 2^-50) matches x - x^2/2 + x^3/3 to one ulp, where a fixed-point series would keep only 118 bits');
  let refused = false; try { D.ln(D.neg(D.ONE)); } catch (e){ refused = /positive/.test(e.message); }
  ok(refused, 'the logarithm of a negative number is refused');
}

console.log('9. decimal strings enter exactly, and nothing else is accepted');
{
  const cases = [['0.1', rat(1n, 10n)], ['-1.0035510043427237e-6', rat(-10035510043427237n, 10n**22n)], ['998.2041322005837', rat(9982041322005837n, 10n**13n)],
                 ['6.02214076e23', rat(602214076n*10n**15n, 1n)], ['9.80665', rat(980665n, 100000n)], ['7', rat(7n, 1n)]];
  let bad = 0;
  for (const [t, q] of cases) if (!nearest(D.fromDecimal(t), q) || !normal(D.fromDecimal(t))) bad++;
  ok(bad === 0, `${cases.length} decimal strings, each the nearest 128-bit value to its exact rational (${bad} not)`);
  let refused = 0;
  for (const t of ['1e', 'abc', '', '1.2.3', ' 1', '0x10', '.5', 'Infinity', 'NaN']){ try { D.fromDecimal(t); } catch (e){ if (e instanceof RangeError) refused++; } }
  let typeRefused = false; try { D.fromDecimal(0.1); } catch (e){ typeRefused = e instanceof TypeError; }
  ok(refused === 9 && typeRefused, `nine malformed strings and a float argument are all refused (${refused} of 9)`);
}

console.log('10. the representation never grows');
{
  let x = D.fromRational(1n, 3n), worst = 0;
  for (let k = 0; k < 5000; k++){ x = D.add(D.mulRat(x, 7n, 23n), D.fromRational(BigInt(k % 11), 13n)); worst = Math.max(worst, D.bitLength(x.m)); }
  ok(worst === D.BITS, `after 5000 rational updates the mantissa is still ${D.BITS} bits (largest ${worst})`);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exitCode = failed ? 1 : 0;
