'use strict';

const BITS = 128;
const GUARD = 40;
const POW2 = Array.from({ length: 8193 }, (_, k) => 1n << BigInt(k));

function bitLength(x){
  if (x < 0n) x = -x;
  if (x === 0n) return 0;
  if (x >= POW2[8192]) return x.toString(2).length;
  let lo = 0, hi = 8192;
  while (lo < hi){ const mid = (lo + hi + 1) >> 1; if (POW2[mid] <= x) lo = mid; else hi = mid - 1; }
  return lo + 1;
}

function requireBig(x, name){
  if (typeof x !== 'bigint') throw new TypeError(`${name} must be a BigInt.`);
  return x;
}
function requireInt(x, name){
  if (!Number.isSafeInteger(x)) throw new TypeError(`${name} must be a safe integer exponent.`);
  return x;
}

function divRoundEven(num, den){
  const q = num/den, r = num - q*den, t = r << 1n;
  return t > den || (t === den && (q & 1n) === 1n) ? q + 1n : q;
}

const ZERO = Object.freeze({ m: 0n, e: 0 });

function store(N, e, q = 1n){
  requireBig(N, 'N'); requireInt(e, 'e'); requireBig(q, 'q');
  if (q <= 0n) throw new RangeError('store: the divisor must be positive.');
  if (N === 0n) return ZERO;
  const neg = N < 0n, n = neg ? -N : N;
  let k = BITS - (bitLength(n) - bitLength(q));
  let num = k >= 0 ? n << BigInt(k) : n, den = k >= 0 ? q : q << BigInt(-k);
  if (num >= den*POW2[BITS]){
    k -= 1;
    if (k >= 0){ num = n << BigInt(k); den = q; } else { num = n; den = q << BigInt(-k); }
  }
  let m = divRoundEven(num, den);
  if (m === POW2[BITS]){ m = POW2[BITS - 1]; k -= 1; }
  return Object.freeze({ m: neg ? -m : m, e: e - k });
}

const fromBigInt = n => store(requireBig(n, 'n'), 0, 1n);
const fromRational = (n, d) => {
  requireBig(n, 'n'); requireBig(d, 'd');
  if (d === 0n) throw new RangeError('fromRational: zero denominator.');
  return d < 0n ? store(-n, 0, -d) : store(n, 0, d);
};

function align(a, b){
  const lo = Math.min(a.e, b.e);
  return [a.m << BigInt(a.e - lo), b.m << BigInt(b.e - lo), lo];
}
function add(a, b){
  if (a.m === 0n) return b;
  if (b.m === 0n) return a;
  if (a.e - b.e > BITS + 1) return a;
  if (b.e - a.e > BITS + 1) return b;
  const [x, y, lo] = align(a, b);
  return store(x + y, lo);
}
function sub(a, b){
  if (b.m === 0n) return a;
  return add(a, neg(b));
}
const neg = a => a.m === 0n ? ZERO : Object.freeze({ m: -a.m, e: a.e });
const abs = a => a.m < 0n ? neg(a) : a;
const mul = (a, b) => (a.m === 0n || b.m === 0n) ? ZERO : store(a.m*b.m, a.e + b.e);
function div(a, b){
  if (b.m === 0n) throw new RangeError('div: division by zero.');
  if (a.m === 0n) return ZERO;
  return b.m < 0n ? store(-a.m, a.e - b.e, -b.m) : store(a.m, a.e - b.e, b.m);
}
function mulRat(a, p, q){
  requireBig(p, 'p'); requireBig(q, 'q');
  if (q === 0n) throw new RangeError('mulRat: zero denominator.');
  if (a.m === 0n || p === 0n) return ZERO;
  return q < 0n ? store(-a.m*p, a.e, -q) : store(a.m*p, a.e, q);
}
const scale2 = (a, s) => a.m === 0n ? ZERO : Object.freeze({ m: a.m, e: a.e + requireInt(s, 's') });
const sign = a => a.m < 0n ? -1 : a.m > 0n ? 1 : 0;
function cmp(a, b){
  const sa = sign(a), sb = sign(b);
  if (sa !== sb) return sa < sb ? -1 : 1;
  if (sa === 0) return 0;
  const [x, y] = align(a, b);
  return x < y ? -1 : x > y ? 1 : 0;
}
const isZero = a => a.m === 0n;
const eq = (a, b) => a.m === b.m && (a.m === 0n || a.e === b.e);

function isqrt(n){
  if (n < 0n) throw new RangeError('isqrt: negative argument.');
  if (n < 2n) return n;
  let x = POW2[(bitLength(n) + 1) >> 1];
  while (true){
    const y = (x + n/x) >> 1n;
    if (y >= x) return x;
    x = y;
  }
}
function sqrt(a){
  if (a.m < 0n) throw new RangeError('sqrt: negative argument.');
  if (a.m === 0n) return ZERO;
  let s = 2*BITS - bitLength(a.m);
  if ((((a.e - s) % 2) + 2) % 2 !== 0) s -= 1;
  const M = s >= 0 ? a.m << BigInt(s) : a.m >> BigInt(-s);
  if (s < 0 && (M << BigInt(-s)) !== a.m) throw new RangeError('sqrt: mantissa wider than its normal form.');
  let r = isqrt(M), e = (a.e - s)/2;
  if (4n*M > (2n*r + 1n)*(2n*r + 1n)) r += 1n;
  if (r === POW2[BITS]){ r = POW2[BITS - 1]; e += 1; }
  return Object.freeze({ m: r, e });
}

const FIX = BITS + GUARD;
function atanInv(k){
  const K = BigInt(k), one = POW2[FIX];
  let term = one/K, sum = term, n = 1n, k2 = K*K;
  while (term !== 0n){
    term /= k2; n += 2n;
    const t = term/n;
    sum += (n & 3n) === 3n ? -t : t;
  }
  return sum;
}
let PI_FIX = null, LN2_FIX = null;
function piFix(){
  if (!PI_FIX) PI_FIX = 16n*atanInv(5) - 4n*atanInv(239);
  return PI_FIX;
}
function ln2Fix(){
  if (!LN2_FIX){
    const one = POW2[FIX];
    let sum = 0n, k = 1n, p = 2n;
    while (true){ const t = one/(k*p); if (t === 0n) break; sum += t; k += 1n; p <<= 1n; }
    LN2_FIX = sum;
  }
  return LN2_FIX;
}
const pi = () => store(piFix(), -FIX);
const ln2 = () => store(ln2Fix(), -FIX);

function toFix(a){
  const sh = a.e + FIX;
  return sh >= 0 ? a.m << BigInt(sh) : (a.m >= 0n ? a.m >> BigInt(-sh) : -((-a.m) >> BigInt(-sh)));
}

function sinCosFix(xf){
  const P = piFix(), halfPi = P >> 1n, twoPi = P << 1n;
  let r = xf % twoPi;
  if (r < 0n) r += twoPi;
  const qFull = (r + (halfPi >> 1n))/halfPi;
  r -= qFull*halfPi;
  const quadrant = Number(qFull & 3n);
  const one = POW2[FIX], r2 = (r*r) >> BigInt(FIX);
  let s = r, c = one, ts = r, tc = one, n = 1n;
  while (ts !== 0n || tc !== 0n){
    ts = -((ts*r2) >> BigInt(FIX))/((2n*n)*(2n*n + 1n));
    tc = -((tc*r2) >> BigInt(FIX))/((2n*n - 1n)*(2n*n));
    s += ts; c += tc; n += 1n;
  }
  if (quadrant === 0) return [s, c];
  if (quadrant === 1) return [c, -s];
  if (quadrant === 2) return [-s, -c];
  return [-c, s];
}
function sin(a){ return store(sinCosFix(toFix(a))[0], -FIX); }
function cos(a){ return store(sinCosFix(toFix(a))[1], -FIX); }

function exp(a){
  const xf = toFix(a), L = ln2Fix();
  const k = (xf >= 0n ? xf + (L >> 1n) : xf - (L >> 1n))/L;
  const r = xf - k*L, one = POW2[FIX];
  let sum = one, term = one, n = 1n;
  while (term !== 0n){ term = ((term*r) >> BigInt(FIX))/n; sum += term; n += 1n; }
  return store(sum, Number(k) - FIX);
}

const HALF = store(1n, -1, 1n);

function expm1(a){
  if (a.m === 0n) return ZERO;
  if (cmp(abs(a), HALF) > 0) return sub(exp(a), fromBigInt(1n));
  const xf = toFix(a), one = POW2[FIX];
  let sum = one, term = one, n = 2n;
  while (term !== 0n){ term = ((term*xf) >> BigInt(FIX))/n; sum += term; n += 1n; }
  return store(a.m*sum, a.e - FIX);
}

function tanh(a){
  if (a.m === 0n) return ZERO;
  if (a.m < 0n) return neg(tanh(neg(a)));
  const t = expm1(scale2(a, 1));
  return div(t, add(t, fromBigInt(2n)));
}

const DECIMAL = /^([+-]?)(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/;
function fromDecimal(text){
  if (typeof text !== 'string') throw new TypeError('fromDecimal: a decimal string is required, so no float is read on the way in.');
  const g = DECIMAL.exec(text);
  if (!g) throw new RangeError(`fromDecimal: ${JSON.stringify(text)} is not a decimal number.`);
  const frac = g[3] || '', exp10 = (g[4] ? Number(g[4]) : 0) - frac.length;
  if (!Number.isSafeInteger(exp10) || Math.abs(exp10) > 100000) throw new RangeError(`fromDecimal: exponent of ${JSON.stringify(text)} is out of range.`);
  let n = BigInt(g[2] + frac);
  if (g[1] === '-') n = -n;
  return exp10 >= 0 ? store(n*10n**BigInt(exp10), 0, 1n) : store(n, 0, 10n**BigInt(-exp10));
}

function toNumberForDisplay(a){
  if (a.m === 0n) return 0;
  const bl = bitLength(a.m), sh = Math.max(0, bl - 53);
  const top = Number(a.m < 0n ? -((-a.m) >> BigInt(sh)) : a.m >> BigInt(sh));
  return top*Math.pow(2, a.e + sh);
}

const DYADIC128 = Object.freeze({
  BITS, ZERO, ONE: fromBigInt(1n), store, fromBigInt, fromRational, add, sub, neg, abs, mul, div,
  mulRat, scale2, cmp, sign, isZero, eq, sqrt, isqrt, pi, ln2, sin, cos, exp, expm1, tanh, fromDecimal, bitLength,
  toNumberForDisplay
});
if (typeof module !== 'undefined' && module.exports) module.exports = DYADIC128;
if (typeof globalThis !== 'undefined') globalThis.DYADIC128 = DYADIC128;
